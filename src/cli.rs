
use clap::Parser;
use meta_gpt::gpt::DiagnosticAgent;
use meta_gpt::terminal::{App, Commands};
use meta_gpt::utils::{get_config, init_logger};
use cerebro_client::client::CerebroClient;
use meta_gpt::gpt::DiagnosticResult;
use nvml_wrapper::Nvml;
use meta_gpt::error::GptError;

#[cfg(feature = "local")]
use meta_gpt::text::{TextGenerator, GeneratorConfig};

#[tokio::main]
async fn main() -> anyhow::Result<(), anyhow::Error> {
    
    init_logger();

    let cli = App::parse();

    match &cli.command {
        Commands::DiagnoseApi( args ) => {
            log::warn!("Not implemented yet")
        }
        Commands::Prefetch( args ) => {
           
            let api_client = CerebroClient::new(
                &cli.url,
                cli.token,
                false,
                cli.danger_invalid_certificate,
                cli.token_file,
                cli.team,
                cli.db,
                cli.project
            )?;

            log::info!("Checking status of Cerebro API at {}",  &api_client.url);
            api_client.ping_servers()?;

            for sample in &args.sample {
                let (gp_config, _) = get_config(
                    &args.json, 
                    Some(sample.clone()), 
                    &args.controls, 
                    &args.tags, 
                    &args.ignore_taxstr,
                    args.prevalence_outliers,
                    None,   // not relevant
                )?;
    
                log::info!("{:#?}", gp_config);
    
                DiagnosticAgent::new(Some(api_client.clone()))?
                    .prefetch(&args.output, &gp_config)?
            }
        }
        #[cfg(feature = "local")]
        Commands::DiagnoseLocal( args ) => {

            let api_client = match args.prefetch {
                Some(_) => None,
                None => {
                    let api_client = CerebroClient::new(
                        &cli.url,
                        cli.token,
                        false,
                        cli.danger_invalid_certificate,
                        cli.token_file,
                        cli.team,
                        cli.db,
                        cli.project
                    )?;
                    
                    log::info!("Checking status of Cerebro API at {}",  &api_client.url);
                    api_client.ping_servers()?;

                    Some(api_client)
                },
            };

                        
            let mut agent = DiagnosticAgent::new(api_client)?;


            let mut generator = TextGenerator::new(
                GeneratorConfig::with_default(
                    args.model.clone(),
                    args.model_dir.clone(),
                    args.sample_len,
                    args.temperature,
                    args.gpu
                )
            )?;

            let (gp_config, prefetch) = get_config(
                &args.json, 
                args.sample.clone(), 
                &args.controls, 
                &args.tags, 
                &args.ignore_taxstr,
                args.prevalence_outliers,
                args.prefetch.clone()
            )?;

            
            let post_filter = if args.post_filter.unwrap_or(false) {
                use cerebro_model::api::cerebro::schema::PostFilterConfig;
 
                let collapse_variants = match args.collapse_variants {
                    Some(value) => value,
                    None => false,
                };
                let species_domains = match &args.species_domains {
                    Some(domains) => domains.to_vec(),
                    None => vec!["Archaea".to_string(), "Bacteria".to_string(), "Eukaryota".to_string()]
                };
                let exclude_phage = match args.exclude_phage {
                    Some(value) => value,
                    None => false,
                };
                Some(PostFilterConfig::with_default(
                    collapse_variants,
                    args.min_species > 0,
                    args.min_species,
                    species_domains,
                    exclude_phage
                )) 
            } else { 
                None 
            };

           let result = match prefetch {
                Some(ref data) => {
                    // If all tiered filter categories are negative the agent pipeline returns a non-infectious result
                    // If we use prefetch data we anticipate this here. This circumvents loading the model to GPU for
                    // inference when we return a non-infectious result anyway.
                    if data.primary.is_empty() && data.secondary.is_empty() && data.target.is_empty() {

                        Some(DiagnosticResult::non_infectious())
                    } else {
                        None
                    }
                },
                None => None
            };

            let result = match result {
                Some(diagnostic_result) => diagnostic_result,
                None => {
                    agent.run_local(
                        &mut generator,
                        args.sample_context.clone(),
                        args.clinical_notes.clone(),
                        args.assay_context.clone(),
                        args.agent_primer.clone(),
                        &gp_config, 
                        prefetch,
                        post_filter,
                        args.disable_thinking
                    )?
                }
            };


            result.to_json(&args.diagnostic_log)?;

            log::info!("{:#?}", result);

            if let Some(state_log) = &args.state_log {
                agent.state.to_json(state_log)?;
            }
            

        },
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

