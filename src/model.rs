
use std::path::{Path, PathBuf};
use clap::ValueEnum;
use hf_hub::api::sync::Api;

use crate::{error::GptError, utils::TokenOutputStream};


#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ModelGroup {
    Qwen,
    NativeQwen,
    Deepseek,
    DeepseekQwen,
    DeepseekLlama,
}
impl ModelGroup {
    pub fn to_models(self) -> Vec<GeneratorModel> {
        match self {
            ModelGroup::Qwen => GeneratorModel::qwen(),
            ModelGroup::NativeQwen => GeneratorModel::native_qwen(),
            ModelGroup::Deepseek => GeneratorModel::deepseek(),
            ModelGroup::DeepseekQwen => GeneratorModel::deepseek_qwen(),
            ModelGroup::DeepseekLlama => GeneratorModel::deepseek_llama(),
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
pub enum GeneratorModel {

    #[value(name = "gemma-3-27b-it-q8-0")]
    Gemma327bQ80,
    #[value(name = "gemma-3-12b-it-q8-0")]
    Gemma312bQ80,
    #[value(name = "gemma-3-4b-it-q8-0")]
    Gemma34bQ80,

    #[value(name = "deepseekr1-llama8b-q4-km")]
    DeepseekR1Llama8bQ4KM,

    #[value(name = "deepseekr1-0528-qwen3-8b-bf16")]
    DeepseekR10528Qwen38bBf16,
    #[value(name = "deepseekr1-0528-qwen3-8b-q8-kxl")]
    DeepseekR10528Qwen38bQ8KXL,
    #[value(name = "deepseekr1-0528-qwen3-8b-q8-0")]
    DeepseekR10528Qwen38bQ80,

    #[value(name = "deepseekr1-qwen7b-q2-kl")]
    DeepseekR1Qwen7bQ2KL,
    #[value(name = "deepseekr1-qwen14b-q2-kl")]
    DeepseekR1Qwen14bQ2KL,
    #[value(name = "deepseekr1-qwen32b-q2-kl")]
    DeepseekR1Qwen32bQ2KL,

    #[value(name = "deepseekr1-qwen7b-q4-km")]
    DeepseekR1Qwen7bQ4KM,
    #[value(name = "deepseekr1-qwen14b-q4-km")]
    DeepseekR1Qwen14bQ4KM,
    #[value(name = "deepseekr1-qwen32b-q4-km")]
    DeepseekR1Qwen32bQ4KM,

    #[value(name = "deepseekr1-qwen7b-q8-0")]
    DeepseekR1Qwen7bQ80,
    #[value(name = "deepseekr1-qwen14b-q8-0")]
    DeepseekR1Qwen14bQ80,
    #[value(name = "deepseekr1-qwen32b-q8-0")]
    DeepseekR1Qwen32bQ80,


    #[value(name = "qwen3-4b-q2-kl")]
    Qwen4bQ2KL,
    #[value(name = "qwen3-8b-q2-kl")]
    Qwen8bQ2KL,
    #[value(name = "qwen3-14b-q2-kl")]
    Qwen14bQ2KL,
    #[value(name = "qwen3-32b-q2-kl")]
    Qwen32bQ2KL,

    #[value(name = "qwen3-4b-q4-km")]
    Qwen4bQ4KM,
    #[value(name = "qwen3-8b-q4-km")]
    Qwen8bQ4KM,
    #[value(name = "qwen3-14b-q4-km")]
    Qwen14bQ4KM,
    #[value(name = "qwen3-32b-q4-km")]
    Qwen32bQ4KM,

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
            GeneratorModel::Gemma327bQ80 => "google/gemma-3-27b-it",
            GeneratorModel::Gemma312bQ80 => "google/gemma-3-12b-it",
            GeneratorModel::Gemma34bQ80 => "google/gemma-3-4b-it",
            GeneratorModel::DeepseekR1Llama8bQ4KM 
                => "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            GeneratorModel::DeepseekR10528Qwen38bBf16  
                | GeneratorModel::DeepseekR10528Qwen38bQ8KXL  
                | GeneratorModel::DeepseekR10528Qwen38bQ80
                => "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
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
            GeneratorModel::Qwen4bQ80
                | GeneratorModel::Qwen4bQ4KM
                | GeneratorModel::Qwen4bQ2KL
                => "Qwen/Qwen3-4B",
            GeneratorModel::Qwen8bQ80
                | GeneratorModel::Qwen8bQ4KM
                | GeneratorModel::Qwen8bQ2KL
                => "Qwen/Qwen3-8B",
            GeneratorModel::Qwen14bQ80
                | GeneratorModel::Qwen14bQ4KM
                | GeneratorModel::Qwen14bQ2KL
                => "Qwen/Qwen3-14B",
            GeneratorModel::Qwen32bQ80
                | GeneratorModel::Qwen32bQ4KM
                | GeneratorModel::Qwen32bQ2KL
                => "Qwen/Qwen3-32B",
        }
    }
    pub fn model_repository(&self) -> &'static str {
        match self {
            GeneratorModel::Gemma327bQ80 => "unsloth/gemma-3-27b-it-GGUF",
            GeneratorModel::Gemma312bQ80 => "unsloth/gemma-3-12b-it-GGUF",
            GeneratorModel::Gemma34bQ80 => "unsloth/gemma-3-4b-it-GGUF",
            GeneratorModel::DeepseekR1Llama8bQ4KM 
                => "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
            GeneratorModel::DeepseekR10528Qwen38bBf16  
                | GeneratorModel::DeepseekR10528Qwen38bQ8KXL  
                | GeneratorModel::DeepseekR10528Qwen38bQ80
                => "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF",
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
            GeneratorModel::Qwen4bQ80
                | GeneratorModel::Qwen4bQ4KM
                | GeneratorModel::Qwen4bQ2KL
                => "unsloth/Qwen3-4B-GGUF",
            GeneratorModel::Qwen8bQ80
                | GeneratorModel::Qwen8bQ4KM
                | GeneratorModel::Qwen8bQ2KL
                => "unsloth/Qwen3-8B-GGUF",
            GeneratorModel::Qwen14bQ80
                | GeneratorModel::Qwen14bQ4KM
                | GeneratorModel::Qwen14bQ2KL
                => "unsloth/Qwen3-14B-GGUF",
            GeneratorModel::Qwen32bQ80
                | GeneratorModel::Qwen32bQ4KM
                | GeneratorModel::Qwen32bQ2KL
                => "unsloth/Qwen3-32B-GGUF",
        }
    }
    pub fn model_config(&self) -> &'static str {
        match self {
            GeneratorModel::Gemma327bQ80 => "gemma-3-27b-it-Q8_0.gguf",
            GeneratorModel::Gemma312bQ80 => "gemma-3-12b-it-Q8_0.gguf",
            GeneratorModel::Gemma34bQ80 => "gemma-3-4b-it-Q8_0.gguf",
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
            GeneratorModel::Qwen4bQ4KM => "Qwen3-4B-UD-Q4_K_XL.gguf",
            GeneratorModel::Qwen8bQ4KM => "Qwen3-8B-UD-Q4_K_XL.gguf",
            GeneratorModel::Qwen14bQ4KM => "Qwen3-14B-Q4_0.gguf",
            GeneratorModel::Qwen32bQ4KM => "Qwen3-32B-Q4_0.gguf",    
            GeneratorModel::Qwen4bQ2KL => "Qwen3-4B-Q2_K_L.gguf",
            GeneratorModel::Qwen8bQ2KL => "Qwen3-8B-Q2_K_L.gguf",
            GeneratorModel::Qwen14bQ2KL => "Qwen3-14B-Q2_K_L.gguf",
            GeneratorModel::Qwen32bQ2KL => "Qwen3-32B-Q2_K_L.gguf",
            GeneratorModel::DeepseekR10528Qwen38bBf16 => "DeepSeek-R1-0528-Qwen3-8B-BF16.gguf",
            GeneratorModel::DeepseekR10528Qwen38bQ8KXL  => "DeepSeek-R1-0528-Qwen3-8B-UD-Q8_K_XL.gguf",
            GeneratorModel::DeepseekR10528Qwen38bQ80 => "DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf",

        
        }
    }
    pub fn model_revision(&self) -> &'static str {
        "main"
    }
    // For writing files to disk
    pub fn model_name(&self) -> &'static str {
        match self {
            GeneratorModel::Gemma327bQ80 => "gemma-3-27b-q8-0",
            GeneratorModel::Gemma312bQ80 => "gemma-3-12b-q8-0",
            GeneratorModel::Gemma34bQ80 => "gemma-3-4b-q8-0",
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
            GeneratorModel::Qwen4bQ4KM => "qwen3-4b-q4-km",
            GeneratorModel::Qwen4bQ80 => "qwen3-4b-q8-0",
            GeneratorModel::Qwen8bQ4KM => "qwen3-8b-q4-km",
            GeneratorModel::Qwen8bQ80 => "qwen3-8b-q8-0",
            GeneratorModel::Qwen14bQ4KM => "qwen3-14b-q4-km",
            GeneratorModel::Qwen14bQ80 => "qwen3-14b-q8-0",
            GeneratorModel::Qwen32bQ4KM => "qwen3-32b-q4-km",
            GeneratorModel::Qwen32bQ80 => "qwen3-32b-q8-0",
            GeneratorModel::Qwen4bQ2KL => "qwen3-4b-q2-kl",
            GeneratorModel::Qwen8bQ2KL => "qwen3-8b-q2-kl",
            GeneratorModel::Qwen14bQ2KL => "qwen3-14b-q2-kl",
            GeneratorModel::Qwen32bQ2KL => "qwen3-32b-q2-kl",
            GeneratorModel::DeepseekR10528Qwen38bBf16 => "deepseekr1-0528-qwen3-8b-bf16",
            GeneratorModel::DeepseekR10528Qwen38bQ8KXL  => "deepseekr1-0528-qwen3-8b-q8-kxl",
            GeneratorModel::DeepseekR10528Qwen38bQ80 => "deepseekr1-0528-qwen3-8b-q8-0"
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
            GeneratorModel::Gemma327bQ80
                | GeneratorModel::Gemma312bQ80
                | GeneratorModel::Gemma34bQ80 => "<end_of_turn>",
            GeneratorModel::DeepseekR1Llama8bQ4KM 
                | GeneratorModel::DeepseekR10528Qwen38bBf16  
                | GeneratorModel::DeepseekR10528Qwen38bQ8KXL  
                | GeneratorModel::DeepseekR10528Qwen38bQ80
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
                | GeneratorModel::Qwen4bQ2KL
                | GeneratorModel::Qwen8bQ2KL
                | GeneratorModel::Qwen14bQ2KL
                | GeneratorModel::Qwen32bQ2KL
                | GeneratorModel::Qwen8bQ80
                | GeneratorModel::Qwen14bQ80 
                | GeneratorModel::Qwen32bQ80
                | GeneratorModel::Qwen4bQ4KM 
                | GeneratorModel::Qwen8bQ4KM 
                | GeneratorModel::Qwen14bQ4KM 
                | GeneratorModel::Qwen32bQ4KM => "<|im_end|>"
        };
        let eos = *tos.tokenizer()
            .get_vocab(true)
            .get(eos_token)
            .ok_or(GptError::EosTokenNotInVocabulary(eos_token.to_string()))?;

        Ok(eos)
    }

    pub fn is_deepseek_qwen(&self) -> bool {
        match self {
            GeneratorModel::Gemma327bQ80
            | GeneratorModel::Gemma312bQ80
            | GeneratorModel::Gemma34bQ80
            | GeneratorModel::DeepseekR1Llama8bQ4KM
            | GeneratorModel::Qwen4bQ80 
            | GeneratorModel::Qwen8bQ80
            | GeneratorModel::Qwen14bQ80 
            | GeneratorModel::Qwen32bQ80 
            | GeneratorModel::Qwen4bQ4KM 
            | GeneratorModel::Qwen8bQ4KM 
            | GeneratorModel::Qwen14bQ4KM 
            | GeneratorModel::Qwen32bQ4KM
            | GeneratorModel::Qwen4bQ2KL
            | GeneratorModel::Qwen8bQ2KL
            | GeneratorModel::Qwen14bQ2KL
            | GeneratorModel::Qwen32bQ2KL => false,
            GeneratorModel::DeepseekR10528Qwen38bBf16  
            | GeneratorModel::DeepseekR10528Qwen38bQ8KXL  
            | GeneratorModel::DeepseekR10528Qwen38bQ80
            | GeneratorModel::DeepseekR1Qwen7bQ2KL 
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
            GeneratorModel::DeepseekR10528Qwen38bBf16  
            | GeneratorModel::DeepseekR10528Qwen38bQ8KXL  
            | GeneratorModel::DeepseekR10528Qwen38bQ80
            | GeneratorModel::DeepseekR1Qwen7bQ2KL 
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
            | GeneratorModel::Qwen4bQ4KM 
            | GeneratorModel::Qwen8bQ4KM 
            | GeneratorModel::Qwen14bQ4KM 
            | GeneratorModel::Qwen32bQ4KM
            | GeneratorModel::Gemma327bQ80
            | GeneratorModel::Gemma312bQ80
            | GeneratorModel::Gemma34bQ80
            | GeneratorModel::Qwen4bQ2KL
            | GeneratorModel::Qwen8bQ2KL
            | GeneratorModel::Qwen14bQ2KL
            | GeneratorModel::Qwen32bQ2KL => false,
            GeneratorModel::DeepseekR1Llama8bQ4KM => true,
        }
    }
    pub fn is_qwen(&self) -> bool {
        match self {
            GeneratorModel::DeepseekR10528Qwen38bBf16  
            | GeneratorModel::DeepseekR10528Qwen38bQ8KXL  
            | GeneratorModel::DeepseekR10528Qwen38bQ80
            | GeneratorModel::DeepseekR1Llama8bQ4KM
            | GeneratorModel::DeepseekR1Qwen7bQ2KL 
            | GeneratorModel::DeepseekR1Qwen7bQ4KM 
            | GeneratorModel::DeepseekR1Qwen7bQ80
            | GeneratorModel::DeepseekR1Qwen14bQ2KL
            | GeneratorModel::DeepseekR1Qwen14bQ4KM
            | GeneratorModel::DeepseekR1Qwen14bQ80
            | GeneratorModel::DeepseekR1Qwen32bQ2KL
            | GeneratorModel::DeepseekR1Qwen32bQ4KM
            | GeneratorModel::DeepseekR1Qwen32bQ80
            | GeneratorModel::Gemma327bQ80
            | GeneratorModel::Gemma312bQ80
            | GeneratorModel::Gemma34bQ80 => false,
            GeneratorModel::Qwen4bQ80 
            | GeneratorModel::Qwen8bQ80
            | GeneratorModel::Qwen14bQ80 
            | GeneratorModel::Qwen32bQ80
            | GeneratorModel::Qwen4bQ4KM 
            | GeneratorModel::Qwen8bQ4KM 
            | GeneratorModel::Qwen14bQ4KM 
            | GeneratorModel::Qwen32bQ4KM
            | GeneratorModel::Qwen4bQ2KL
            | GeneratorModel::Qwen8bQ2KL
            | GeneratorModel::Qwen14bQ2KL
            | GeneratorModel::Qwen32bQ2KL => true,
        }
    }
    pub fn is_gemma(&self) -> bool {
        match self {
            GeneratorModel::DeepseekR10528Qwen38bBf16  
            | GeneratorModel::DeepseekR10528Qwen38bQ8KXL  
            | GeneratorModel::DeepseekR10528Qwen38bQ80
            | GeneratorModel::DeepseekR1Llama8bQ4KM
            | GeneratorModel::DeepseekR1Qwen7bQ2KL 
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
            | GeneratorModel::Qwen4bQ4KM 
            | GeneratorModel::Qwen8bQ4KM 
            | GeneratorModel::Qwen14bQ4KM 
            | GeneratorModel::Qwen32bQ4KM
            | GeneratorModel::Qwen4bQ2KL
            | GeneratorModel::Qwen8bQ2KL
            | GeneratorModel::Qwen14bQ2KL
            | GeneratorModel::Qwen32bQ2KL => false,
            | GeneratorModel::Gemma327bQ80
            | GeneratorModel::Gemma312bQ80
            | GeneratorModel::Gemma34bQ80 => true
        }
    }
    /// All Qwen-based models (including Deepseek-Qwen and regular Qwen3)
    pub fn qwen() -> Vec<GeneratorModel> {
        GeneratorModel::value_variants()
            .iter()
            .cloned()
            .filter(|m| m.is_qwen() || m.is_deepseek_qwen())
            .collect()
    }

    /// All Deepseek models (Qwen and Llama)
    pub fn deepseek() -> Vec<GeneratorModel> {
        GeneratorModel::value_variants()
            .iter()
            .cloned()
            .filter(|m| m.is_deepseek_qwen() || m.is_deepseek_llama())
            .collect()
    }

    /// Only native Qwen3 models (not Deepseek)
    pub fn native_qwen() -> Vec<GeneratorModel> {
        GeneratorModel::value_variants()
            .iter()
            .cloned()
            .filter(|m| m.is_qwen())
            .collect()
    }

    /// Only Deepseek-Qwen models
    pub fn deepseek_qwen() -> Vec<GeneratorModel> {
        GeneratorModel::value_variants()
            .iter()
            .cloned()
            .filter(|m| m.is_deepseek_qwen())
            .collect()
    }

    /// Only Deepseek-Llama models
    pub fn deepseek_llama() -> Vec<GeneratorModel> {
        GeneratorModel::value_variants()
            .iter()
            .cloned()
            .filter(|m| m.is_deepseek_llama())
            .collect()
    }

    /// Only Google-Gemma models
    pub fn gemma() -> Vec<GeneratorModel> {
        GeneratorModel::value_variants()
            .iter()
            .cloned()
            .filter(|m| m.is_gemma())
            .collect()
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

        } else if self.is_gemma() {
            format!("<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n")
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