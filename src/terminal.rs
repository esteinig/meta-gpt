use std::path::PathBuf;
use clap::{ArgGroup, Args, Parser, Subcommand};

use crate::gpt::{AssayContext, GptModel, SampleContext};

#[cfg(feature = "local")]
use crate::text::{GeneratorModel, TextGeneratorArgs};

/// Cerebro: metagenomic generative practitioner (GPT)
#[derive(Debug, Parser)]
#[command(author, version, about)]
#[command(styles=get_styles())]
#[command(arg_required_else_help(true))]
#[clap(name = "meta-gp", version)]
pub struct App {
    /// API URL
    #[clap(
        long, 
        short = 'u', 
        default_value = "http://api.cerebro.localhost", 
        env = "CEREBRO_API_URL"
    )]
    pub url: String,
    /// API token - usually provided with CEREBRO_API_TOKEN
    #[clap(
        long, 
        short = 'e', 
        env = "CEREBRO_API_TOKEN",
        hide_env_values = true
    )]
    pub token: Option<String>,
    /// API token file - can be set from environment variable
    #[clap(
        long, 
        short = 'f', 
        env = "CEREBRO_API_TOKEN_FILE"
    )]
    pub token_file: Option<PathBuf>,
    /// User team name or identifier for requests that require team specification 
    #[clap(
        long, 
        short = 't', 
        env = "CEREBRO_USER_TEAM",
        hide_env_values = true
    )]
    pub team: Option<String>,
    /// Team database name or identifier for requests that require database access 
    #[clap(
        long, 
        short = 'd', 
        env = "CEREBRO_USER_DB",
        hide_env_values = true
    )]
    pub db: Option<String>,
    /// Team database project name or identifier for requests that require project access 
    #[clap(
        long, 
        short = 'p', 
        env = "CEREBRO_USER_PROJECT",
        hide_env_values = true
    )]
    pub project: Option<String>,
    /// SeaweedFS master node address
    #[clap(
        long, 
        short = 'a',
        default_value = "http://fs.cerebro.localhost", 
        env = "CEREBRO_FS_URL"
    )]
    pub fs_url: String,
    /// SeaweedFS master node port
    #[clap(
        long, 
        short = 'm',
        env = "CEREBRO_FS_PORT",
        default_value = "9333", 
    )]
    pub fs_port: String,
    /// SSL certificate verification is ignored [DANGER]
    #[clap(
        long, 
        env = "CEREBRO_DANGER_ACCEPT_INVALID_TLS_CERTIFICATE"
    )]
    pub danger_invalid_certificate: bool,
    
    #[clap(subcommand)]
    pub command: Commands,
}


#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Diagnose a sample given the evidence and decision tree using commercial models
    DiagnoseApi(DiagnoseApiArgs),


    /// Prefetch tiered data with filter settings for a sample and save to file
    PrefetchTiered(PrefetchTieredArgs),
    
    #[cfg(feature = "local")]
    /// Diagnose a sample given the evidence and decision tree using local models
    DiagnoseLocal(DiagnoseLocalArgs),

    #[cfg(feature = "local")]
    /// Run local text generation on GPU
    Generate(TextGeneratorArgs),
}

#[derive(Debug, Args)]
#[clap(group = ArgGroup::new("user")
    .required(false)
    .multiple(true)
    .args(&["controls", "tags"])
)]
#[clap(group = ArgGroup::new("file")
    .args(&["json"])
    .required(false)
    .multiple(true)
)]

pub struct DiagnoseApiArgs {
    /// Sample identifier for query
    #[clap(long, short = 's')]
    pub sample: String,
    /// Control sample identifiers associated with sample
    #[clap(long, short = 'c', num_args(0..), group = "user", help_heading = "Validation Query")]
    pub controls: Option<Vec<String>>,
    /// Tags for sample and model query
    #[clap(long, short = 't', num_args(0..), group = "user", help_heading = "Validation Query")]
    pub tags: Option<Vec<String>>,
    /// JSON file of request schema
    #[clap(long, short = 'j', group = "file", help_heading = "Schema Query")]
    pub json: Option<PathBuf>,
    /// OpenAI model for diagnostic queries
    #[clap(long, short = 'm', default_value="o3-mini")]
    pub model: GptModel,
    /// Dry run printing only decision tree and validating inputs
    #[clap(long, short = 'd')]
    pub dry_run: bool,
    /// State log at completion of diagnostic evidence synthesis
    #[clap(long, short = 'l')]
    pub state_log: Option<PathBuf>,
    /// Diagnostic log at completion of diagnostic evidence synthesis
    #[clap(long, short = 'o', default_value="diagnostic.log")]
    pub diagnostic_log: PathBuf,
    /// Clinical context model for diagnostic queries
    #[clap(long, default_value="csf")]
    pub clinical_context: SampleContext,
    /// Clinical notes to be added to clinical context
    #[clap(long)]
    pub clinical_notes: Option<String>,
    /// Check for contamination history outliers to be removed from prevalence filter
    #[clap(long)]
    pub contam_history: Option<bool>,
    /// Ignore these taxstr as filter for threshold values
    #[clap(long, num_args=1..)]
    pub ignore_taxstr: Option<Vec<String>>,
}


#[derive(Debug, Args)]
#[clap(group = ArgGroup::new("user")
    .multiple(true)
    .args(&["controls", "tags"])
)]
#[clap(group = ArgGroup::new("file")
    .args(&["json"])
)]
pub struct PrefetchTieredArgs {
    /// Sample identifier for query
    #[clap(long, short = 's', num_args(1..))]
    pub sample: Vec<String>,
    /// Output file for prefetched tiered filter data
    #[clap(long, short = 'o')]
    pub output: PathBuf,
    /// Control sample identifiers associated with sample
    #[clap(long, short = 'c', num_args(0..), group = "user", help_heading = "Validation Query")]
    pub controls: Option<Vec<String>>,
    /// Tags for sample and model query
    #[clap(long, short = 't', num_args(0..), group = "user", help_heading = "Validation Query")]
    pub tags: Option<Vec<String>>,
    /// Request schema (.json)
    #[clap(long, short = 'j', group = "file", help_heading = "Schema Query")]
    pub json: Option<PathBuf>,
    /// Override for prevalence contamination regresision outliers to be removed from prevalence filter and included in output (primary filter category, overrides specifications from JSON)
    #[clap(long)]
    pub prevalence_outliers: Option<bool>,
    /// Ignore these taxstr as filter for threshold values
    #[clap(long, num_args=1..)]
    pub ignore_taxstr: Option<Vec<String>>,
}

#[cfg(feature = "local")]
#[derive(Debug, Args)]
#[clap(group = ArgGroup::new("user")
    .multiple(true)
    .args(&["sample", "controls", "tags"])
)]
#[clap(group = ArgGroup::new("file")
    .args(&["sample", "json"])
)]
#[clap(group = ArgGroup::new("data")
    .args(&["prefetch"])
)]pub struct DiagnoseLocalArgs {
    /// Sample identifier for query
    #[clap(long, short = 's')]
    pub sample: Option<String>,
    /// Control sample identifiers associated with sample
    #[clap(long, short = 'c', num_args(0..), group = "user", help_heading = "User Query")]
    pub controls: Option<Vec<String>>,
    /// Tags for sample and model query
    #[clap(long, short = 't', num_args(0..), group = "user", help_heading = "User Query")]
    pub tags: Option<Vec<String>>,
    /// JSON file of request schema
    #[clap(long, short = 'j', group = "file", help_heading = "Schema Query")]
    pub json: Option<PathBuf>,
    /// Prefetched data for sample
    #[clap(long, short = 'p', group = "data", help_heading = "Prefetch Query")]
    pub prefetch: Option<PathBuf>,
    /// Local generator model for diagnostic queries
    #[clap(long, short = 'm', default_value="deepseekr1-qwen7b-q8-0")]
    pub model: GeneratorModel,
    /// Model directory for generator model(.gguf) and tokenizer file (.json)
    #[clap(long, default_value=".")]
    pub model_dir: PathBuf,
    /// The length of the sample to generate (in tokens)  - needs to be long enough to capture model thought process
    #[arg(short = 'n', long, default_value_t = 10000)]
    pub sample_len: usize,
    /// The temperature used to generate samples, use 0 for greedy sampling.
    #[arg(long, short='t', default_value_t = 0.8)]
    pub temperature: f64,
    /// GPU device index to run on.
    #[arg(long, short='g', default_value_t=0)]
    pub gpu: usize,
    /// State log at completion of diagnostic evidence synthesis
    #[clap(long, short = 'l')]
    pub state_log: Option<PathBuf>,
    /// Diagnostic log at completion of diagnostic evidence synthesis
    #[clap(long, short = 'o', default_value="diagnostic.log")]
    pub diagnostic_log: PathBuf,
    /// Clinical context model for diagnostic queries
    #[clap(long, default_value="none")]
    pub sample_context: SampleContext,
    /// Clinical context model for diagnostic queries
    #[clap(long, default_value="cerebro-filter")]
    pub assay_context: Option<AssayContext>,
    /// Clinical notes to be added to clinical context
    #[clap(long)]
    pub clinical_notes: Option<String>,
    /// Override for prevalence contamination regression outliers to be removed from prevalence filter and included in output (primary filter category, overrides specifications from JSON)
    #[clap(long)]
    pub prevalence_outliers: Option<bool>,
    /// Post process taxa after filtering and retrieval by collapsing species variants and selecting best species per genus (Archaea|Bacteria|Eukaryota)
    #[clap(long, default_value="true")]
    pub post_filter: Option<bool>,
    /// Minimum species per genus required to enable selecting best species for the genus during post filter processing (Archaea|Bacteria|Eukaryota)
    #[clap(long, default_value="3")]
    pub min_species: usize,
    /// Apply the species reduction filter to these domains (Archaea|Bacteria|Eukaryota)
    #[clap(long, num_args=1..)]
    pub species_domains: Option<Vec<String>>,
    /// Override deafault variant species collapse using pruned species name (GTDB, Archaea|Bacteria)
    #[clap(long, default_value="true")]
    pub collapse_variants: Option<bool>,
    /// Override default phage exclusion option for post processing taxa after filtering and retrieval
    #[clap(long, default_value="true")]
    pub exclude_phage: Option<bool>,
    /// Ignore these taxstr as filter for threshold values
    #[clap(long, num_args=1..)]
    pub ignore_taxstr: Option<Vec<String>>,
    /// Disable thinking in Qwen3
    #[clap(long)]
    pub disable_thinking: bool,
}

#[derive(Debug, Args)]
pub struct GlobalOptions {
    
}


pub fn get_styles() -> clap::builder::Styles {
	clap::builder::Styles::styled()
		.header(
			anstyle::Style::new()
				.bold()
				.underline()
				.fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::Yellow))),
		)
		.literal(
			anstyle::Style::new()
				.bold()
				.fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::Green))),
		)
}
