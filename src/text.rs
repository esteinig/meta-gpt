use std::path::{Path, PathBuf};
use clap::{Args, ValueEnum};
use hf_hub::api::sync::Api;
use std::io::Write;

use tokenizers::Tokenizer;

use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_core::utils::{cuda_is_available, metal_is_available};

use candle_transformers::models::quantized_llama as llama;
use candle_transformers::models::quantized_qwen2 as qwen2;
use candle_transformers::models::quantized_qwen3 as qwen3;


use candle_transformers::utils::apply_repeat_penalty;
use candle_transformers::generation::{LogitsProcessor, Sampling};

use crate::error::GptError;


pub fn device(gpu: usize) -> Result<Device, GptError> {

    if cuda_is_available() {
        Ok(Device::new_cuda(gpu)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(gpu)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

// Define a trait that abstracts over model weight types
pub trait InferenceModel {
    /// run a single forward pass at a given position
    fn forward(&mut self, input: &Tensor, position: usize) -> Result<Tensor, GptError>;
}

impl InferenceModel for llama::ModelWeights {
    fn forward(&mut self, input: &Tensor, position: usize) -> Result<Tensor, GptError> {
        // delegate to the inherent method on ModelWeights
        Ok(llama::ModelWeights::forward(self, input, position)?)
    }
}

impl InferenceModel for qwen2::ModelWeights {
    fn forward(&mut self, input: &Tensor, position: usize) -> Result<Tensor, GptError> {
        // delegate to the inherent method on ModelWeights
        Ok(qwen2::ModelWeights::forward(self, input, position)?)
    }
}

impl InferenceModel for qwen3::ModelWeights {
    fn forward(&mut self, input: &Tensor, position: usize) -> Result<Tensor, GptError> {
        // delegate to the inherent method on ModelWeights
        Ok(qwen3::ModelWeights::forward(self, input, position)?)
    }
}

pub struct TextGenerator {
    pub model: Box<dyn InferenceModel>,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub config: GeneratorConfig
}

impl TextGenerator {
    pub fn new(config: GeneratorConfig) -> Result<Self, GptError> {

        match config.model {
            GeneratorModel::DeepseekR1Llama8bQ4KM => {
                log::info!("Activate reduced precision GEMM kernels for quantized Llama.");
                candle_core::cuda::set_gemm_reduced_precision_f16(true);
                candle_core::cuda::set_gemm_reduced_precision_bf16(true);
            },
            _ => {},
        }

        let (model_path, tokenizer_path) = Self::get_model(&config)?;

        let device = device(config.gpu)?;
        let mut file = std::fs::File::open(&model_path)?;

        let start_tensor_load = std::time::Instant::now();
        let gguf = gguf_file::Content::read(&mut file)
            .map_err(|e| e.with_path(&model_path))?;
        Self::gguf_info(&gguf, &start_tensor_load);

        let tokenizer = Tokenizer::from_file(tokenizer_path)?;

        let model: Box<dyn InferenceModel>  = match config.model {
            GeneratorModel::DeepseekR1Llama8bQ4KM => {

                let model = llama::ModelWeights::from_gguf(
                    gguf, 
                    &mut file, 
                    &device
                )?;
                Box::new(model)
            },
            GeneratorModel::DeepseekR1Qwen7bQ2KL 
                | GeneratorModel::DeepseekR1Qwen7bQ4KM 
                | GeneratorModel::DeepseekR1Qwen7bQ80
                | GeneratorModel::DeepseekR1Qwen14bQ2KL
                | GeneratorModel::DeepseekR1Qwen14bQ4KM
                | GeneratorModel::DeepseekR1Qwen14bQ80
                | GeneratorModel::DeepseekR1Qwen32bQ2KL
                | GeneratorModel::DeepseekR1Qwen32bQ4KM
                | GeneratorModel::DeepseekR1Qwen32bQ80
             => {

                let model = qwen2::ModelWeights::from_gguf(
                    gguf, 
                    &mut file, 
                    &device
                )?;
                Box::new(model)
            },
            GeneratorModel::Qwen4bQ80 
            | GeneratorModel::Qwen8bQ80 
            | GeneratorModel::Qwen14bQ80 
            | GeneratorModel::Qwen32bQ80 
            | GeneratorModel::Qwen4bQ4KXL 
            | GeneratorModel::Qwen8bQ4KXL 
            | GeneratorModel::Qwen14bQ40 
            | GeneratorModel::Qwen32bQ40 => {

                let model = qwen3::ModelWeights::from_gguf(
                    gguf, 
                    &mut file, 
                    &device
                )?;
                Box::new(model)
            }
        };

        log::info!("Inference model weights loaded on GPU.");

        Ok(Self {
            model,
            tokenizer,
            device,
            config
        })
    }
    pub fn run(
        &mut self,
        prompt: &str,
        disable_thinking: bool
    ) -> Result<(String, String), GptError> {

        let mut tos = TokenOutputStream::new(
            self.tokenizer.clone()
        );

        let prompt = if self.config.raw_prompt {
            prompt.to_string()
        } else {
            self.config.model.format_prompt(prompt, disable_thinking)
        };

        if self.config.log_info {
            log::info!("Prompt is: \n\n{prompt}\n\n");
        }

        let tokens = tos
            .tokenizer()
            .encode(prompt, true)?;

        let mut tokens = tokens.get_ids().to_vec();
        let to_sample = self.config.sample_len.saturating_sub(1);

        match self.config.model {
            GeneratorModel::DeepseekR1Llama8bQ4KM => {
                tokens = if tokens.len() + to_sample > llama::MAX_SEQ_LEN - 10 {
                    let to_remove = tokens.len() + to_sample + 10 - llama::MAX_SEQ_LEN;
                    tokens[tokens.len().saturating_sub(to_remove)..].to_vec()
                } else {
                    tokens.to_vec()
                };
            },

            GeneratorModel::DeepseekR1Qwen7bQ2KL 
                | GeneratorModel::DeepseekR1Qwen7bQ4KM 
                | GeneratorModel::DeepseekR1Qwen7bQ80
                | GeneratorModel::DeepseekR1Qwen14bQ2KL
                | GeneratorModel::DeepseekR1Qwen14bQ4KM
                | GeneratorModel::DeepseekR1Qwen14bQ80
                | GeneratorModel::DeepseekR1Qwen32bQ2KL
                | GeneratorModel::DeepseekR1Qwen32bQ4KM
                | GeneratorModel::DeepseekR1Qwen32bQ80
                | GeneratorModel::Qwen4bQ80 
                | GeneratorModel::Qwen8bQ80 
                | GeneratorModel::Qwen32bQ80
                | GeneratorModel::Qwen14bQ80 
                | GeneratorModel::Qwen4bQ4KXL 
                | GeneratorModel::Qwen8bQ4KXL 
                | GeneratorModel::Qwen14bQ40 
                | GeneratorModel::Qwen32bQ40
            => {}
        }

        let mut logits_processor = TextGenerator::build_logits_processor(
            self.config.temperature, 
            self.config.seed, 
            self.config.top_k, 
            self.config.top_p
        );

        log::info!("Build model architecture with weights.");
        
        let eos_token = self.config.model.get_eos_token(&tos)?;

        log::info!("Start generative processing and sampling tokens.");
        let (thoughts, answer) = TextGenerator::generate(
            self.model.as_mut(),
            &self.device,
            &tokens,
            &mut tos,
            &mut logits_processor,
            self.config.sample_len,
            self.config.repeat_penalty,
            self.config.repeat_last_n,
            eos_token,
            self.config.split_prompt,
            self.config.log_info
        )?;

        Ok((thoughts, answer))
        
    }
    pub fn generate(
        model: &mut dyn InferenceModel,
        device: &Device,
        tokens: &[u32],
        tos: &mut TokenOutputStream,
        logits_processor: &mut LogitsProcessor,
        to_sample: usize,
        repeat_penalty: f32,
        repeat_last_n: usize,
        eos_token: u32,
        split_prompt: bool,
        log_info: bool,
    ) -> Result<(String, String), GptError> {

        // Holds all tokens generated
        let mut all_tokens = vec![];
        
        // Process the prompt input
        let (first_token, prompt_dt) = TextGenerator::process_prompt(
            &mut *model,
            &tokens,
            split_prompt,
            &device,
            logits_processor,
        )?;

        all_tokens.push(first_token);
        if let Some(t) = tos.next_token(first_token)? {
            if log_info {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }

        // Process the main sample loop
        let (mut text, sampled, gen_dt) = TextGenerator::sample_tokens(
            &mut *model,
            first_token,
            &mut all_tokens,
            logits_processor,
            to_sample,
            tokens.len(),
            repeat_penalty,
            repeat_last_n,
            eos_token,
            tos,
            &device,
            log_info
        )?;

        // Flush any trailing subwords and final newline
        if let Some(rest) = tos.decode_rest().map_err(candle_core::Error::msg)? {
            if log_info {
                print!("{rest}");
                std::io::stdout().flush()?;
                println!("\n");
            }
            text.push_str(&rest)
        }


        if log_info {
            log::info!(
                "{:4} prompt tokens processed @ {:.2} token/s",
                tokens.len(),
                tokens.len() as f64 / prompt_dt.as_secs_f64(),
            );
            log::info!(
                "{:4} tokens generated @ {:.2} token/s",
                sampled,
                sampled as f64 / gen_dt.as_secs_f64(),
            );
        }

        Ok(split_think(&text))
    }
    /// Run the post-prompt generation loop
    fn sample_tokens(
        model: &mut dyn InferenceModel,
        mut next_token: u32,
        all_tokens: &mut Vec<u32>,
        logits_processor: &mut LogitsProcessor,
        to_sample: usize,
        prompt_len: usize,
        repeat_penalty: f32,
        repeat_last_n: usize,
        eos_token: u32,
        tos: &mut TokenOutputStream,
        device: &Device,
        log_info: bool
    ) -> Result<(String, u32, std::time::Duration), GptError> {

        let start = std::time::Instant::now();

        let mut sampled = 0;
        let mut generated = String::new();
        
        for i in 0..to_sample {

            // Forward one token
            let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
            let mut logits = model.forward(&input, prompt_len + i)?.squeeze(0)?;

            // Optional repeat-penalty
            if repeat_penalty != 1.0 {
                let begin = all_tokens.len().saturating_sub(repeat_last_n);
                logits = apply_repeat_penalty(&logits, repeat_penalty, &all_tokens[begin..])?;
            }

            // Sample, record, print & break on EOS
            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);

            if let Some(text) = tos.next_token(next_token)? {
                if log_info {
                    print!("{}", text);
                    std::io::stdout().flush()?;
                }
                generated.push_str(&text)
            }

            sampled += 1;
            if next_token == eos_token {
                break;
            }
        }

        Ok((generated, sampled, start.elapsed()))
    }
    pub fn process_prompt(
        model: &mut dyn InferenceModel,
        tokens: &[u32],
        split_prompt: bool,
        device: &Device,
        logits_processor: &mut LogitsProcessor,
    ) -> Result<(u32, std::time::Duration), GptError> {
        let start = std::time::Instant::now();
        
        let tok = if !split_prompt {
            let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0)?.squeeze(0)?;
            logits_processor.sample(&logits)?
        } else {
            let mut last = 0;
            for (pos, &tok) in tokens.iter().enumerate() {
                let single = Tensor::new(&[tok], device)?.unsqueeze(0)?;
                let logits = model.forward(&single, pos)?.squeeze(0)?;
                last = logits_processor.sample(&logits)?;
            }
            last
        };

        Ok((tok, start.elapsed()))
    }
    pub fn build_logits_processor(temperature: f64, seed: u64, top_k: Option<usize>, top_p: Option<f64>) -> LogitsProcessor {

        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (top_k, top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        LogitsProcessor::from_sampling(seed, sampling)
    }
    pub fn gguf_info(gguf: &gguf_file::Content, start: &std::time::Instant) {

        let mut total_size_in_bytes = 0;
        for (_, tensor) in gguf.tensor_infos.iter() {
            let elem_count = tensor.shape.elem_count();
            total_size_in_bytes +=
                elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
        }
        log::info!(
            "loaded {:?} tensors ({}) in {:.2}s",
            gguf.tensor_infos.len(),
            &format_size(total_size_in_bytes),
            start.elapsed().as_secs_f32(),
        );
    }
    pub fn get_model(config: &GeneratorConfig) -> Result<(PathBuf, PathBuf), GptError> {

        let model_path = config.model_dir.join(&config.model.model_file());
        let tokenizer_path = config.model_dir.join(&config.model.tokenizer_file());
        
        let model_file = if model_path.exists() & !config.force_download {
            model_path.clone()
        } else {
            config.model.save_model(&config.model_dir)?
        };

        let tokenizer_file = if tokenizer_path.exists() & !config.force_download {
            tokenizer_path.clone()
        } else {
            config.model.save_tokenizer(&config.model_dir)?
        };

        Ok((model_file, tokenizer_file))
    }
}

fn split_think(text: &str) -> (String, String) {
    // Split at most once on the delimiter
    let mut parts = text.splitn(2, "</think>");

    // Always get the text before the first (or only) chunk
    let first = parts.next().unwrap_or("");

    if let Some(second) = parts.next() {
        // We had exactly two parts
        let thought  = first.to_string();
        let answer  = second.to_string();
        (thought, answer)
    } else {
        // No delimiter found → return the whole thing as `answer`
        let thought = String::new();
        let answer = text.to_string();
        (thought, answer)
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
pub enum GeneratorModel {
    #[value(name = "deepseekr1-llama8b-q4-km")]
    DeepseekR1Llama8bQ4KM,
    
    #[value(name = "deepseekr1-qwen7b-q2-kl")]
    DeepseekR1Qwen7bQ2KL,
    #[value(name = "deepseekr1-qwen7b-q4-km")]
    DeepseekR1Qwen7bQ4KM,
    #[value(name = "deepseekr1-qwen7b-q8-0")]
    DeepseekR1Qwen7bQ80,

    #[value(name = "deepseekr1-qwen14b-q2-kl")]
    DeepseekR1Qwen14bQ2KL,
    #[value(name = "deepseekr1-qwen14b-q4-km")]
    DeepseekR1Qwen14bQ4KM,
    #[value(name = "deepseekr1-qwen14b-q8-0")]
    DeepseekR1Qwen14bQ80,

    #[value(name = "deepseekr1-qwen32b-q2-kl")]
    DeepseekR1Qwen32bQ2KL,
    #[value(name = "deepseekr1-qwen32b-q4-km")]
    DeepseekR1Qwen32bQ4KM,
    #[value(name = "deepseekr1-qwen32b-q8-0")]
    DeepseekR1Qwen32bQ80,

    #[value(name = "qwen3-4b-q4-kxl")]
    Qwen4bQ4KXL,
    #[value(name = "qwen3-8b-q4-kxl")]
    Qwen8bQ4KXL,
    #[value(name = "qwen3-14b-q4-0")]
    Qwen14bQ40,
    #[value(name = "qwen3-32b-q4-0")]
    Qwen32bQ40,

    #[value(name = "qwen3-4b-q8-0")]
    Qwen4bQ80,
    #[value(name = "qwen3-8b-q8-0")]
    Qwen8bQ80,
    #[value(name = "qwen3-14b-q8-0")]
    Qwen14bQ80,
    #[value(name = "qwen3-32b-q8-0")]
    Qwen32bQ80,
}

impl GeneratorModel {

    /// Download and save the GGUF model file as `{model_name}.{ext}`
    pub fn save_model(&self, outdir: &Path) -> Result<PathBuf, GptError> {
        let src = self.download_model()?;
        std::fs::create_dir_all(outdir)?;
        let dest = outdir.join(&self.model_file());
        std::fs::copy(&src, &dest)?;
        Ok(dest)
    }
    /// Download and save the tokenizer to `{model_name}.tokenizer.json`
    pub fn save_tokenizer(&self, outdir: &Path) -> Result<PathBuf, GptError> {
        let src = self.download_tokenizer()?;
        std::fs::create_dir_all(outdir)?;
        let dest = outdir.join(&self.tokenizer_file());
        std::fs::copy(&src, &dest)?;
        Ok(dest)
    }
    pub fn download_tokenizer(&self) -> Result<PathBuf, GptError> {
        log::info!("Downloading tokenizer...");
        let api = Api::new()?;
        let repo = self.tokenizer_repository();
        let api = api.model(repo.to_string());
        let tokenizer_path = api.get("tokenizer.json")?;
        Ok(tokenizer_path)
    }
    pub fn download_model(&self) -> Result<PathBuf, GptError> {
        log::info!("Downloading model weights...");
        let api = Api::new()?;
        let repo = hf_hub::Repo::with_revision(
            self.model_repository().to_string(),
            hf_hub::RepoType::Model,
            self.model_revision().to_string(),
        );
        let model_path = api.repo(repo).get(
            self.model_config()
        )?;
        Ok(model_path)
    }
    pub fn tokenizer_repository(&self) -> &'static str {
        match self {
            GeneratorModel::DeepseekR1Llama8bQ4KM 
                => "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            GeneratorModel::DeepseekR1Qwen7bQ2KL  
                | GeneratorModel::DeepseekR1Qwen7bQ4KM  
                | GeneratorModel::DeepseekR1Qwen7bQ80
                => "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            GeneratorModel::DeepseekR1Qwen14bQ2KL
                | GeneratorModel::DeepseekR1Qwen14bQ4KM
                | GeneratorModel::DeepseekR1Qwen14bQ80
                => "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            GeneratorModel::DeepseekR1Qwen32bQ2KL
                | GeneratorModel::DeepseekR1Qwen32bQ4KM
                | GeneratorModel::DeepseekR1Qwen32bQ80
                => "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            GeneratorModel::Qwen4bQ80  | GeneratorModel::Qwen4bQ4KXL => "Qwen/Qwen3-4B",
            GeneratorModel::Qwen8bQ80  | GeneratorModel::Qwen8bQ4KXL  => "Qwen/Qwen3-8B",
            GeneratorModel::Qwen14bQ80 | GeneratorModel::Qwen14bQ40 => "Qwen/Qwen3-14B",
            GeneratorModel::Qwen32bQ80 | GeneratorModel::Qwen32bQ40 => "Qwen/Qwen3-32B",
        }
    }
    pub fn model_repository(&self) -> &'static str {
        match self {
            GeneratorModel::DeepseekR1Llama8bQ4KM 
                => "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
            GeneratorModel::DeepseekR1Qwen7bQ2KL
                | GeneratorModel::DeepseekR1Qwen7bQ4KM 
                | GeneratorModel::DeepseekR1Qwen7bQ80
                => "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
            GeneratorModel::DeepseekR1Qwen14bQ2KL
                | GeneratorModel::DeepseekR1Qwen14bQ4KM
                | GeneratorModel::DeepseekR1Qwen14bQ80
                => "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
            GeneratorModel::DeepseekR1Qwen32bQ2KL
                | GeneratorModel::DeepseekR1Qwen32bQ4KM
                | GeneratorModel::DeepseekR1Qwen32bQ80
                => "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF",
            GeneratorModel::Qwen4bQ80  | GeneratorModel::Qwen4bQ4KXL => "unsloth/Qwen3-4B-GGUF",
            GeneratorModel::Qwen8bQ80  | GeneratorModel::Qwen8bQ4KXL => "unsloth/Qwen3-8B-GGUF",
            GeneratorModel::Qwen14bQ80 | GeneratorModel::Qwen14bQ40 => "unsloth/Qwen3-14B-GGUF",
            GeneratorModel::Qwen32bQ80 | GeneratorModel::Qwen32bQ40=> "unsloth/Qwen3-32B-GGUF",
        }
    }
    pub fn model_config(&self) -> &'static str {
        match self {
            GeneratorModel::DeepseekR1Llama8bQ4KM => "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
            GeneratorModel::DeepseekR1Qwen7bQ2KL => "DeepSeek-R1-Distill-Qwen-7B-Q2_K_L.gguf",
            GeneratorModel::DeepseekR1Qwen7bQ4KM => "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
            GeneratorModel::DeepseekR1Qwen7bQ80 => "DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf",
            GeneratorModel::DeepseekR1Qwen14bQ2KL => "DeepSeek-R1-Distill-Qwen-14B-Q2_K_L.gguf",
            GeneratorModel::DeepseekR1Qwen14bQ4KM => "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
            GeneratorModel::DeepseekR1Qwen14bQ80 => "DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf",
            GeneratorModel::DeepseekR1Qwen32bQ2KL => "DeepSeek-R1-Distill-Qwen-32B-Q2_K_L.gguf",
            GeneratorModel::DeepseekR1Qwen32bQ4KM => "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
            GeneratorModel::DeepseekR1Qwen32bQ80 => "DeepSeek-R1-Distill-Qwen-32B-Q8_0.gguf",
            GeneratorModel::Qwen4bQ80 => "Qwen3-4B-Q8_0.gguf",
            GeneratorModel::Qwen8bQ80 => "Qwen3-8B-Q8_0.gguf",
            GeneratorModel::Qwen14bQ80 => "Qwen3-14B-Q8_0.gguf",
            GeneratorModel::Qwen32bQ80 => "Qwen3-32B-Q8_0.gguf",
            GeneratorModel::Qwen4bQ4KXL => "Qwen3-4B-Q4_0.gguf",
            GeneratorModel::Qwen8bQ4KXL => "Qwen3-8B-Q4_0.gguf",
            GeneratorModel::Qwen14bQ40 => "Qwen3-14B-Q4_0.gguf",
            GeneratorModel::Qwen32bQ40 => "Qwen3-32B-Q4_0.gguf",
        }
    }
    pub fn model_revision(&self) -> &'static str {
        "main"
    }
    // For writing files to disk
    pub fn model_name(&self) -> &'static str {
        match self {
            GeneratorModel::DeepseekR1Llama8bQ4KM  => "deepseekr1-llama8b-q4-km",
            GeneratorModel::DeepseekR1Qwen7bQ2KL => "deepseekr1-qwen7b-q2-kl",
            GeneratorModel::DeepseekR1Qwen7bQ4KM => "deepseekr1-qwen7b-q4-km",
            GeneratorModel::DeepseekR1Qwen7bQ80 => "deepseekr1-qwen7b-q8-0",
            GeneratorModel::DeepseekR1Qwen14bQ2KL => "deepseekr1-qwen14b-q2-kl",
            GeneratorModel::DeepseekR1Qwen14bQ4KM => "deepseekr1-qwen14b-q4-km",
            GeneratorModel::DeepseekR1Qwen14bQ80 => "deepseekr1-qwen14b-q8-0",
            GeneratorModel::DeepseekR1Qwen32bQ2KL => "deepseekr1-qwen32b-q2-kl",
            GeneratorModel::DeepseekR1Qwen32bQ4KM => "deepseekr1-qwen32b-q4-km",
            GeneratorModel::DeepseekR1Qwen32bQ80 => "deepseekr1-qwen32b-q8-0",
            GeneratorModel::Qwen4bQ80 => "qwen3-4b-q8-0",
            GeneratorModel::Qwen8bQ80 => "qwen3-8b-q8-0",
            GeneratorModel::Qwen14bQ80 => "qwen3-14b-q8-0",
            GeneratorModel::Qwen32bQ80 => "qwen3-32b-q8-0",
            GeneratorModel::Qwen4bQ4KXL => "qwen3-4b-q4-kxl",
            GeneratorModel::Qwen8bQ4KXL => "qwen3-8b-q4-kxl",
            GeneratorModel::Qwen14bQ40 => "qwen3-14b-q4-0",
            GeneratorModel::Qwen32bQ40 => "qwen3-32b-q4-0",
        }
    }
    pub fn model_file(&self) -> PathBuf {
        let model_config = PathBuf::from(
            self.model_config()
        );
        let ext = model_config
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("config");
        PathBuf::from(format!("{}.{}", self.model_name(), ext))
    }

    pub fn tokenizer_file(&self) -> PathBuf {
        PathBuf::from(format!("{}.tokenizer.json", self.model_name()))
    }

    pub fn get_eos_token(
        &self, 
        tos: &TokenOutputStream,
    ) -> Result<u32, GptError> {

        // Get the  end of sentence token
        let eos_token = match self {
            GeneratorModel::DeepseekR1Llama8bQ4KM 
                | GeneratorModel::DeepseekR1Qwen7bQ2KL 
                | GeneratorModel::DeepseekR1Qwen7bQ4KM 
                | GeneratorModel::DeepseekR1Qwen7bQ80
                | GeneratorModel::DeepseekR1Qwen14bQ2KL
                | GeneratorModel::DeepseekR1Qwen14bQ4KM
                | GeneratorModel::DeepseekR1Qwen14bQ80
                | GeneratorModel::DeepseekR1Qwen32bQ2KL
                | GeneratorModel::DeepseekR1Qwen32bQ4KM
                | GeneratorModel::DeepseekR1Qwen32bQ80  => "<｜end▁of▁sentence｜>",
            GeneratorModel::Qwen4bQ80 
                | GeneratorModel::Qwen8bQ80
                | GeneratorModel::Qwen14bQ80 
                | GeneratorModel::Qwen32bQ80
                | GeneratorModel::Qwen4bQ4KXL 
                | GeneratorModel::Qwen8bQ4KXL 
                | GeneratorModel::Qwen14bQ40 
                | GeneratorModel::Qwen32bQ40 => "<|im_end|>"
        };
        let eos = *tos.tokenizer()
            .get_vocab(true)
            .get(eos_token)
            .ok_or(GptError::EosTokenNotInVocabulary(eos_token.to_string()))?;

        Ok(eos)
    }

    pub fn is_deepseek_qwen(&self) -> bool {
        match self {
            GeneratorModel::DeepseekR1Llama8bQ4KM
            | GeneratorModel::Qwen4bQ80 
            | GeneratorModel::Qwen8bQ80
            | GeneratorModel::Qwen14bQ80 
            | GeneratorModel::Qwen32bQ80 
            | GeneratorModel::Qwen4bQ4KXL 
            | GeneratorModel::Qwen8bQ4KXL 
            | GeneratorModel::Qwen14bQ40 
            | GeneratorModel::Qwen32bQ40 => false,
            GeneratorModel::DeepseekR1Qwen7bQ2KL 
            | GeneratorModel::DeepseekR1Qwen7bQ4KM 
            | GeneratorModel::DeepseekR1Qwen7bQ80
            | GeneratorModel::DeepseekR1Qwen14bQ2KL
            | GeneratorModel::DeepseekR1Qwen14bQ4KM
            | GeneratorModel::DeepseekR1Qwen14bQ80
            | GeneratorModel::DeepseekR1Qwen32bQ2KL
            | GeneratorModel::DeepseekR1Qwen32bQ4KM
            | GeneratorModel::DeepseekR1Qwen32bQ80 => true,
        }
    }
    pub fn is_deepseek_llama(&self) -> bool {
        match self {
            GeneratorModel::DeepseekR1Qwen7bQ2KL 
            | GeneratorModel::DeepseekR1Qwen7bQ4KM 
            | GeneratorModel::DeepseekR1Qwen7bQ80
            | GeneratorModel::DeepseekR1Qwen14bQ2KL
            | GeneratorModel::DeepseekR1Qwen14bQ4KM
            | GeneratorModel::DeepseekR1Qwen14bQ80
            | GeneratorModel::DeepseekR1Qwen32bQ2KL
            | GeneratorModel::DeepseekR1Qwen32bQ4KM
            | GeneratorModel::DeepseekR1Qwen32bQ80
            | GeneratorModel::Qwen4bQ80 
            | GeneratorModel::Qwen8bQ80
            | GeneratorModel::Qwen14bQ80 
            | GeneratorModel::Qwen32bQ80
            | GeneratorModel::Qwen4bQ4KXL 
            | GeneratorModel::Qwen8bQ4KXL 
            | GeneratorModel::Qwen14bQ40 
            | GeneratorModel::Qwen32bQ40 => false,
            GeneratorModel::DeepseekR1Llama8bQ4KM => true,
        }
    }
    pub fn is_qwen(&self) -> bool {
        match self {
            GeneratorModel::DeepseekR1Llama8bQ4KM
            | GeneratorModel::DeepseekR1Qwen7bQ2KL 
            | GeneratorModel::DeepseekR1Qwen7bQ4KM 
            | GeneratorModel::DeepseekR1Qwen7bQ80
            | GeneratorModel::DeepseekR1Qwen14bQ2KL
            | GeneratorModel::DeepseekR1Qwen14bQ4KM
            | GeneratorModel::DeepseekR1Qwen14bQ80
            | GeneratorModel::DeepseekR1Qwen32bQ2KL
            | GeneratorModel::DeepseekR1Qwen32bQ4KM
            | GeneratorModel::DeepseekR1Qwen32bQ80 => false,
            GeneratorModel::Qwen4bQ80 
            | GeneratorModel::Qwen8bQ80
            | GeneratorModel::Qwen14bQ80 
            | GeneratorModel::Qwen32bQ80
            | GeneratorModel::Qwen4bQ4KXL 
            | GeneratorModel::Qwen8bQ4KXL 
            | GeneratorModel::Qwen14bQ40 
            | GeneratorModel::Qwen32bQ40 => true,
        }
    }

    pub fn format_prompt(&self, prompt: &str, disable_thinking: bool) -> String {
        if self.is_deepseek_qwen() {
            if disable_thinking {
                format!("<｜User｜>{prompt}<｜Assistant｜>\n<think>\n\n</think>\n\n")
            } else {
                format!("<｜User｜>{prompt}<｜Assistant｜>")
            }
        } else if self.is_deepseek_llama() {
            format!("<｜user｜>{prompt}<｜assistant｜>")  // llama distillation only works with non-capitalized tags?
        } else if self.is_qwen() {
            if disable_thinking {
                format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n") 
            } else {
                format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n") 
            }
        } else {
            prompt.to_string()
        }
    }
}


fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{}B", size_in_bytes)
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}


#[derive(Args, Debug, Clone)]
pub struct TextGeneratorArgs {

    /// Text generation model.
    #[arg(long, short='m', default_value = "deepseekr1-qwen7b-q4-km")]
    pub model: GeneratorModel,

    /// Input user prompt.
    #[arg(long, short='p')]
    pub prompt: String,

    /// Model file and download directory.
    #[arg(long, short='d', default_value=".")]
    pub dir: PathBuf,

    /// Force download of model and tokenizer files if they do not exist in the model directory.
    #[arg(long, short='f')]
    pub force_download: bool,

    /// Force raw prompt instead of adding model tokens to input prompt.
    #[arg(long, short='r')]
    pub raw_prompt: bool,

    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 1000)]
    pub sample_len: usize,

    /// The temperature used to generate samples, use 0 for greedy sampling.
    #[arg(long, short='t', default_value_t = 0.8)]
    pub temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long, short='p')]
    pub top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long, short='k')]
    pub top_k: Option<usize>,

    /// GPU device index to run on.
    #[arg(long, short='g', default_value_t=0)]
    pub gpu: usize,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    pub seed: u64,

    /// Process prompt elements separately.
    #[arg(long)]
    pub split_prompt: bool,

    /// Clean output prompt e.g. by removing <think> tags for Deepseek.
    #[arg(long)]
    pub clean_output: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    pub repeat_last_n: usize,

    /// Log additional information.
    #[arg(long)]
    pub log_info: bool,

}


/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
/// 
/// https://github.com/huggingface/candle/blob/main/candle-examples/src/token_output_stream.rs
pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, GptError> {
        Ok(self.tokenizer.decode(tokens, true)?)
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>, GptError> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>, GptError> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_all(&self) -> Result<String, GptError> {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    pub model: GeneratorModel,
    pub model_dir: PathBuf, 
    pub force_download: bool,
    pub raw_prompt: bool,
    pub sample_len: usize,
    pub split_prompt: bool,
    pub clean_output: bool,
    pub temperature: f64,
    pub seed: u64,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub log_info: bool,
    pub gpu: usize,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        GeneratorConfig {
            model: GeneratorModel::DeepseekR1Qwen7bQ4KM,
            model_dir: PathBuf::from("."), 
            force_download: false,
            raw_prompt: false,
            sample_len: 10000,
            split_prompt: false,
            clean_output: false,
            temperature: 0.8,
            seed: 299792458,
            top_k: None,
            top_p: None,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            log_info: false,
            gpu: 0,
        }
    }
}

impl GeneratorConfig {
    pub fn with_default(model: GeneratorModel, model_dir: PathBuf, sample_len: usize, temperature: f64, gpu: usize) -> Self {
        GeneratorConfigBuilder::new()
            .model(model)
            .model_dir(model_dir)
            .sample_len(sample_len)
            .temperature(temperature)
            .gpu(gpu)
            .build()
    }
    pub fn from_args(args: &TextGeneratorArgs) -> Self {
        GeneratorConfig {
            model: args.model,
            model_dir: args.dir.clone(), 
            force_download: args.force_download,
            raw_prompt: args.raw_prompt,
            sample_len: args.sample_len,
            split_prompt: args.split_prompt,
            clean_output: args.clean_output,
            temperature: args.temperature,
            seed: args.seed,
            top_k: args.top_k.clone(),
            top_p: args.top_p.clone(),
            repeat_penalty: args.repeat_penalty,
            repeat_last_n: args.repeat_last_n,
            log_info: args.log_info,
            gpu: args.gpu,
        }
    }
}

pub struct GeneratorConfigBuilder {
    cfg: GeneratorConfig,
}

impl GeneratorConfigBuilder {
    /// Start a new builder with all defaults.
    pub fn new() -> Self {
        GeneratorConfigBuilder {
            cfg: GeneratorConfig::default(),
        }
    }
    pub fn model(mut self, model: GeneratorModel) -> Self {
        self.cfg.model = model;
        self
    }
    pub fn model_dir(mut self, model_dir: PathBuf) -> Self {
        self.cfg.model_dir = model_dir;
        self
    }
    pub fn force_download(mut self, force_download: bool) -> Self {
        self.cfg.force_download = force_download;
        self
    }
    pub fn raw_prompt(mut self, raw_prompt: bool) -> Self {
        self.cfg.raw_prompt = raw_prompt;
        self
    }
    pub fn sample_len(mut self, sample_len: usize) -> Self {
        self.cfg.sample_len = sample_len;
        self
    }
    pub fn split_prompt(mut self, split_prompt: bool) -> Self {
        self.cfg.split_prompt = split_prompt;
        self
    }
    pub fn clean_output(mut self, clean_output: bool) -> Self {
        self.cfg.clean_output = clean_output;
        self
    }
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.cfg.temperature = temperature;
        self
    }
    pub fn seed(mut self, seed: u64) -> Self {
        self.cfg.seed = seed;
        self
    }
    pub fn top_k(mut self, top_k: impl Into<Option<usize>>) -> Self {
        self.cfg.top_k = top_k.into();
        self
    }
    pub fn top_p(mut self, top_p: impl Into<Option<f64>>) -> Self {
        self.cfg.top_p = top_p.into();
        self
    }
    pub fn repeat_penalty(mut self, repeat_penalty: f32) -> Self {
        self.cfg.repeat_penalty = repeat_penalty;
        self
    }
    pub fn repeat_last_n(mut self, repeat_last_n: usize) -> Self {
        self.cfg.repeat_last_n = repeat_last_n;
        self
    }
    pub fn log_info(mut self, log_info: bool) -> Self {
        self.cfg.log_info = log_info;
        self
    }
    pub fn gpu(mut self, gpu: usize) -> Self {
        self.cfg.gpu = gpu;
        self
    }

    /// Finalize the builder and get your `GeneratorConfig`.
    pub fn build(self) -> GeneratorConfig {
        self.cfg
    }
}
