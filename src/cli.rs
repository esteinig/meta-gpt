
use clap::Parser;
use meta_gpt::gpt::DiagnosticAgent;
use meta_gpt::gpt::ClinicalContext;
use meta_gpt::terminal::{App, Commands};
use meta_gpt::utils::{get_config, init_logger};
use cerebro_client::client::CerebroClient;
use meta_gpt::gpt::DiagnosticResult;
use nvml_wrapper::Nvml;

#[cfg(feature = "local")]
use meta_gpt::text::{TextGenerator, GeneratorConfig};

#[tokio::main]
async fn main() -> anyhow::Result<(), anyhow::Error> {
    
    init_logger();

    let cli = App::parse();

    match &cli.command {
        #[cfg(feature = "local")]
        Commands::Generate( args ) => {
            
            let nvml = Nvml::init()?;
            let nvml_device = nvml.device_by_index(args.gpu as u32)?;

            log::info!("Device name is: {}", nvml_device.name()?);
            
            let mut generator = TextGenerator::new(
                GeneratorConfig::from_args(&args)
            )?;

            generator.run(&args.prompt, false)?;

        },
        Commands::Download( args ) => {
            
            let mut selected = args.models.clone();

            if let Some(group) = args.group {
                selected.extend(group.to_models());
                selected.dedup(); // avoid duplicates
            }

            if selected.is_empty() {
                log::error!("No models or model groups were selected!")
            }

            for model in selected {
                model.save_model(&args.outdir)?;
                model.save_tokenizer(&args.outdir)?;
            }

        }
    }

    Ok(())

}

