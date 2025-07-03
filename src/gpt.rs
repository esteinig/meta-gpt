// main.rs


use anyhow::Result;
use cerebro_client::client::CerebroClient;
use cerebro_model::api::cerebro::schema::{CerebroIdentifierSchema, MetaGpConfig, PostFilterConfig};
use cerebro_pipeline::taxa::filter::TaxonFilterConfig;
use cerebro_pipeline::taxa::taxon::{collapse_taxa, LineageOperations, Taxon};

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::Hash;
use std::io::{BufWriter, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};

use async_openai::{config::OpenAIConfig, types::*};
use colored::Colorize;
use anthropic_api::{messages::*, Credentials};


use petgraph::graph::Graph;
use petgraph::visit::EdgeRef;
use petgraph::graph::NodeIndex;
use petgraph::Direction;
use petgraph::visit::IntoNodeReferences;
use plotters::prelude::*;

use crate::error::GptError;
use crate::model::GeneratorModel;


#[cfg(feature = "local")]
use crate::text::TextGenerator;

//
// === Refined Question Types ===
//

#[derive(Debug, Clone, Deserialize, Serialize, Eq, Hash, PartialEq)]
#[serde(untagged)]
pub enum Question {
    Simple(String),
    Detailed {
        tasks: String,
        data: Option<String>,
        context: Option<String>,
        instructions: Option<String>,
    },
}
impl Question {
    /// [Context]
    /// ...
    /// 
    /// [Tasks]
    /// ...
    ///
    /// [Instructions]
    /// ...
    pub fn to_standard_prompt(&self) -> String {
        let (tasks, data, context, instructions) = match self {
            Question::Simple(p) => (p.clone(), None, None, None),
            Question::Detailed { tasks, data, context, instructions } => {
                (tasks.clone(), data.clone(), context.clone(), instructions.clone())
            }
        };

        let mut out = String::new();
        if let Some(context) = context {
            out.push_str("[Context]\n");
            out.push_str(&context);
            out.push_str("\n\n");
        }
        if let Some(data) = data {
            out.push_str("[Data]\n");
            out.push_str(&data);
            out.push_str("\n\n");
        }

        out.push_str("[Tasks]\n");
        out.push_str(&tasks);
        out.push_str("\n\n");

        if let Some(instructions) = instructions {
            out.push_str("[Instructions]\n");
            out.push_str(&instructions);
            out.push_str("\n");
        }
        out
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, clap::ValueEnum)]
pub enum GptStrategy {
    OneShotResponse,
    DecisionTreeChat,
    DecisionTreeResponse,
}

pub struct ThresholdCandidates {
    pub primary_threshold: Option<Vec<Taxon>>,
    pub secondary_threshold: Option<Vec<Taxon>>,
    pub target_threshold: Option<Vec<Taxon>>,
    pub integrate_threshold: Option<Vec<Taxon>>
}

impl ThresholdCandidates {
    pub fn from_primary_threshold(taxa: Vec<Taxon>) -> Self {
        Self { primary_threshold: Some(taxa), secondary_threshold: None, target_threshold: None, integrate_threshold: None }
    }
    pub fn from_secondary_threshold(taxa: Vec<Taxon>) -> Self {
        Self { primary_threshold: None, secondary_threshold: Some(taxa), target_threshold: None, integrate_threshold: None }
    }
    pub fn from_target_threshold(taxa: Vec<Taxon>) -> Self {
        Self { primary_threshold: None, secondary_threshold: None, target_threshold: Some(taxa), integrate_threshold: None }
    }
    pub fn from_integrate_threshold(taxa: Vec<Taxon>) -> Self {
        Self { primary_threshold: None, secondary_threshold: None, target_threshold: None, integrate_threshold: Some(taxa) }
    }
    pub fn to_str(&self, evidence: bool) -> String {
        let mut output = String::new();

        // Process above threshold candidates
        if let Some(ref above) = self.primary_threshold {
            if above.is_empty() {
                output.push_str("No taxa detected.");
            } else {
                output.push_str("Primary threshold taxa:\n\n");
                // Join the names of the taxa with commas.
                let taxa: Vec<String> = above.iter().map(|taxon| taxon.species_data(evidence)).collect();
                output.push_str(&taxa.join("\n\n"));
            }
            output.push('\n');
        }

        // Process below threshold candidates
        if let Some(ref below) = self.secondary_threshold {
            if below.is_empty() {
                output.push_str("No taxa detected.");
            } else {
                output.push_str("Secondary threshold taxa:\n\n");
                let taxa: Vec<String> = below.iter().map(|taxon| taxon.species_data(evidence)).collect();
                output.push_str(&taxa.join("\n\n"));
            }
        }

        // Process target list candidate taxa
        if let Some(ref target) = self.target_threshold {
            if target.is_empty() {
                output.push_str("No taxa detected.");
            } else {
                output.push_str("Target threshold taxa:\n\n");
                let taxa: Vec<String> = target.iter().map(|taxon| taxon.species_data(evidence)).collect();
                output.push_str(&taxa.join("\n\n"));
            }
        }

        // Process target list candidate taxa
        if let Some(ref integrate) = self.integrate_threshold {
            if integrate.is_empty() {
                output.push_str("No taxa detected.");
            } else {
                output.push_str("Secondary and target threshold taxa:\n\n");
                let taxa: Vec<String> = integrate.iter().map(|taxon| taxon.species_data(evidence)).collect();
                output.push_str(&taxa.join("\n\n"));
            }
        }

        output
    }
}


#[derive(Clone, Debug, Deserialize, Serialize, clap::ValueEnum)]
pub enum AssayContext {
    CerebroFilter,
    None
}
impl AssayContext {
    pub fn text(&self) -> String {
        match self {
            AssayContext::CerebroFilter => dedent(r"
                We conducted metagenomic sequencing for pathogen detection and diagnosis (Illumina PE, RNA and DNA libraries on NextSeq). Filtering the taxonomic profiling data from the bioinformatics pipeline produced three subsets of the same dataset: 
                primary threshold (specific but less sensitive for pathogen detection, moderate to high abundance organisms), secondary threshold (sensitive but less specific for pathogen detection, low to moderate abundance organisms), and a 
                target filter section containing high priority pathogens of interest (very sensitive but not specific for pathogen detection, very low to low abundance but may still be signficant). Our pipeline uses multiple profiling methods for pathogen detection - 
                read alignment (reads per million, RPM), k-mer classifiers (read per million, RPM) and metagenome assembly (contigs, bases). 
                
                Values for each species are the outputs from multiple methods or tools used for taxonomic profiling. Species names are taxonomic species name (genus name and species name). If you do not know a species, assume that the provided species name is correct - 
                do not interpret unknown species names as another species you know. You must make your considerations and determinations based on the species not the genus.
            "),
            AssayContext::None => String::new()
        }
    } 
}

#[derive(Clone, Debug, Deserialize, Serialize, clap::ValueEnum)]
pub enum SampleContext {
    Csf,
    Eye,
    None
}

impl SampleContext {
    pub fn text(&self) -> String {
        let sample_type = match self {
            SampleContext::Csf => String::from("Cerebrospinal fluid (CSF) sample."),
            SampleContext::Eye => String::from("Vitreous fluid sample."),
            SampleContext::None => String::new()
        };
        format!("Sample type: {}", sample_type)
    }
    pub fn with_clinical_notes(&self, notes: &str) -> String {
        format!("{}\nClinical notes: {}", self.text(), notes)
    }
}

//
// === Decision Tree Definitions ===
//

#[derive(Debug, Clone, Deserialize, Serialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticNode {
    AneuploidyQuery,
    AboveThresholdQuery,
    BelowThresholdQuery,
    TargetThresholdQuery,
    IntegrateThresholds,
    DiagnoseInfectious,
    DiagnoseNonInfectious
}

#[derive(Debug, Clone, Deserialize, Serialize, Eq, Hash, PartialEq)]
pub struct TreeNode {
    label: Option<String>,
    question: Option<Question>,
    check: Option<DiagnosticNode>,
    true_node: Option<String>,
    false_node: Option<String>,
    next: Option<String>,
    final_node: Option<bool>,
}


impl Default for TreeNode {
    fn default() -> Self {
        Self {
            label: None,
            question: None,
            check: None,
            true_node: None,
            false_node: None,
            next: None,
            final_node: None,
        }
    }
}

impl TreeNode {

    /// Set a simple prompt
    pub fn with_prompt<S: Into<String>>(mut self, prompt: S) -> Self {
        let p = prompt.into();
        self.question = Some(Question::Simple(p));
        self
    }

    /// Upgrade or set a Detailed question prompt
    pub fn with_tasks<S: Into<String>>(mut self, tasks: S) ->  Result<Self, GptError>  {
        let tasks = tasks.into();
        match self.question.take() {
            Some(Question::Detailed { data, context, instructions, .. }) => {
                self.question = Some(Question::Detailed { 
                    tasks,
                    context, 
                    data, 
                    instructions
                })
            }
            _ => self.question = Some(Question::Detailed {
                tasks,
                context: None,
                data: None,
                instructions: None,
            }),
        }
        Ok(self)
    }

    /// Add or replace the “Context:” block (use placeholders here)
    pub fn with_context<S: Into<String>>(mut self, context: S) -> Result<Self, GptError> {
        let context = context.into();
        match self.question.take() {
            Some(Question::Detailed { tasks, data, instructions, .. }) => {
                self.question = Some(Question::Detailed {
                    tasks,
                    context: Some(context),
                    data,
                    instructions,
                })
            }
            Some(Question::Simple(p)) => {
                self.question = Some(Question::Detailed {
                    tasks: p,
                    context: Some(context),
                    data: None,
                    instructions: None,
                })
            }
            None => {
                self.question = Some(Question::Detailed {
                    tasks: String::new(),
                    context: Some(context),
                    data: None,
                    instructions: None,
                })
            }
        }
        Ok(self)
    }

    /// Add or replace the “Instructions:” block
    pub fn with_instructions<S: Into<String>>(mut self, instructions: S) -> Result<Self, GptError> {
        let instructions = instructions.into();
        match self.question.take() {
            Some(Question::Detailed { tasks, context, data, .. }) => {
                self.question = Some(Question::Detailed {
                    tasks,
                    context,
                    data,
                    instructions: Some(instructions),
                })
            }
            Some(Question::Simple(p)) => {
                self.question = Some(Question::Detailed {
                    tasks: p,
                    context: None,
                    data: None,
                    instructions: Some(instructions),
                })
            }
            None => {
                self.question = Some(Question::Detailed {
                    tasks: String::new(),
                    context: None,
                    data: None,
                    instructions: Some(instructions),
                })
            }
        }
        Ok(self)
    }

    /// Add or replace the “Instructions:” block
    pub fn with_data<S: Into<String>>(mut self, data: S) -> Result<Self, GptError> {
        let data = data.into();
        match self.question.take() {
            Some(Question::Detailed { tasks, context, instructions, .. }) => {
                self.question = Some(Question::Detailed {
                    tasks,
                    context,
                    data: Some(data),
                    instructions,
                })
            }
            Some(Question::Simple(p)) => {
                self.question = Some(Question::Detailed {
                    tasks: p,
                    context: None,
                    data: Some(data),
                    instructions: None,
                })
            }
            None => return Err(GptError::TreeNodeQuestionMissing)
        }
        Ok(self)
    }

    pub fn with_check(mut self, c: DiagnosticNode) -> Self {
        self.check = Some(c);
        self
    }

    pub fn true_node<S: Into<String>>(mut self, tgt: S) -> Self {
        self.true_node = Some(tgt.into());
        self
    }

    pub fn false_node<S: Into<String>>(mut self, tgt: S) -> Self {
        self.false_node = Some(tgt.into());
        self
    }

    pub fn next<S: Into<String>>(mut self, nxt: S) -> Self {
        self.next = Some(nxt.into());
        self
    }

    pub fn final_node(mut self, is_final: bool) -> Self {
        self.final_node = Some(is_final);
        self
    }

    pub fn label<S: Into<String>>(mut self, lbl: S) -> Self {
        self.label = Some(lbl.into());
        self
    }
}


#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TreeEdge {
    action: TreeAction
}
impl TreeEdge {
    pub fn default_next() -> Self {
        Self {
            action: TreeAction::Next
        }
    }
    pub fn default_true() -> Self {
        Self {
            action: TreeAction::True
        }
    }
    pub fn default_false() -> Self {
        Self {
            action: TreeAction::False
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TreeAction {
    Next,
    True,
    False
}

//
// === Evaluation Details Structure ===
//

#[derive(Debug, Serialize, Deserialize)]
pub struct EvaluationDetails {
    pub taxa_contamination: Vec<String>,
    pub taxa_pathogens: Vec<String>,
    pub contamination_reason: String,
    pub pathogen_reason: String,
    pub positive_infection: Option<bool>,
}


//
// === Final Diagnostic Result Structure ===
//

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum Diagnosis {
    Infectious,
    InfectiousReview,
    NonInfectious,
    NonInfectiousReview,
    Tumor,
    Unknown
}

impl std::fmt::Display for Diagnosis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Diagnosis::Infectious => "infectious",
            Diagnosis::InfectiousReview => "infectious-review",
            Diagnosis::NonInfectious => "non-infectious",
            Diagnosis::NonInfectiousReview => "non-infectious-review",
            Diagnosis::Tumor => "tumor",
            Diagnosis::Unknown => "unknown",
        };
        write!(f, "{}", s)
    }
}


#[derive(Debug, Serialize, Deserialize)]
pub struct DiagnosticResult {
    pub diagnosis: Diagnosis,
    pub candidates: Vec<String>,
    pub pathogen: Option<String>,
}
impl DiagnosticResult {
    pub fn non_infectious() -> Self {
        Self {
            diagnosis: Diagnosis::NonInfectious,
            candidates: vec![],
            pathogen: None
        }
    }
    pub fn to_json(&self, path: &Path) -> Result<(), GptError> {
        let agent_state = serde_json::to_string_pretty(self).map_err(|err| GptError::SerdeJsonError(err))?;
        let mut writer = BufWriter::new(File::create(path)?);
        write!(writer, "{agent_state}")?;
        Ok(())
    }
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self, GptError> {
        let data = std::fs::read_to_string(path)?;
        let result = serde_json::from_str::<DiagnosticResult>(&data)?;
        Ok(result)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AgentBenchmark {
    pub seconds: f32
}
impl AgentBenchmark {
    pub fn to_json(&self, path: &Path) -> Result<(), GptError> {
        let writer = BufWriter::new(
            File::create(path)?
        );
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self, GptError> {
        let data: String = std::fs::read_to_string(path)?;
        let data = serde_json::from_str::<AgentBenchmark>(&data)?;
        Ok(data)
    }
}



#[derive(Serialize, Deserialize, Debug)]
pub struct GpuBenchmark {
    pub peak_vram: u64  // bytes
}
impl GpuBenchmark {
    pub fn to_json(&self, path: &Path) -> Result<(), GptError> {
        let writer = BufWriter::new(
            File::create(path)?
        );
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self, GptError> {
        let data: String = std::fs::read_to_string(path)?;
        let data = serde_json::from_str::<GpuBenchmark>(&data)?;
        Ok(data)
    }
}


//
// === Agent State (Memory) ===
//

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticMemory {
    pub node: DiagnosticNode,
    pub data: Vec<Taxon>,
    pub result: Option<bool>,
    pub prompt: Option<String>,
    pub thoughts: Option<String>,
    pub answer: Option<String>
}
impl DiagnosticMemory {
    pub fn new(node: DiagnosticNode, data: Vec<Taxon>, result: Option<bool>, prompt: Option<String>, thoughts: Option<String>, answer: Option<String>) -> Self {
        Self { node, data, result, prompt, thoughts, answer }
    }
    pub fn non_infectious(node: DiagnosticNode) -> Self {
        Self {
            node,
            data: vec![],
            result:  Some(false),
            prompt: None,
            thoughts: None,
            answer: None
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AgentState {
    pub memory: Vec<DiagnosticMemory>,
    pub post_filter_config: Option<PostFilterConfig>,
    pub repeat: HashMap<DiagnosticNode, usize>,
}

impl AgentState {
    fn new() -> Self {
        AgentState {
            memory: Vec::new(),
            post_filter_config: None,
            repeat: HashMap::new()
        }
    }

    pub fn memorize(&mut self, mem: DiagnosticMemory) {
        self.memory.push(mem);
    }

    pub fn retrieve(&self, node: DiagnosticNode) -> Option<&DiagnosticMemory> {
        self.memory.iter().find(|mem| mem.node == node)
    }
    pub fn to_json(&mut self, path: &Path) -> Result<(), GptError> {
        let agent_state = serde_json::to_string_pretty(self).map_err(|err| GptError::SerdeJsonError(err))?;
        let mut writer = BufWriter::new(File::create(path)?);
        write!(writer, "{agent_state}")?;
        Ok(())
    }
}

//
// === Diagnostic Agent ===
//

#[derive(Debug, Serialize, Deserialize)]
pub struct ThresholdTaxa {
    pub above: Vec<Taxon>,
    pub below: Vec<Taxon>,
    pub target: Vec<Taxon>
}
impl ThresholdTaxa {
    pub fn from_agent_memory(memory: &HashMap<String, String>) -> Result<Self, GptError> {

        let above: Vec<Taxon> = serde_json::from_str(
            memory.get("data__AboveThresholdQuery__CandidateTaxa")
                  .map_or("[]", |r| r.deref())
        )?;

        let below: Vec<Taxon> = serde_json::from_str(
            memory.get("data__BelowThresholdQuery__CandidateTaxa")
                  .map_or("[]", |r| r.deref())
        )?;


        let target: Vec<Taxon> = serde_json::from_str(
            memory.get("data__TargetThresholdQuery__CandidateTaxa")
                  .map_or("[]", |r| r.deref())
        )?;

        Ok(Self {
            above,
            below,
            target
        })
    }
    pub fn to_json(&self, path: &Path) -> Result<(), GptError> {
        let mut writer = BufWriter::new(File::create(path)?);
        write!(&mut writer, "{}", serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, clap::ValueEnum)]
pub enum GptModel {
    #[serde(rename="o4-mini")]
    O4Mini,
    #[serde(rename="o3-mini")]
    O3Mini,
    #[serde(rename="o1-mini")]
    O1Mini,
    #[serde(rename="gpt-4o-mini")]
    Gpt4oMini,
    #[serde(rename="gpt-4o")]
    Gpt4o,
    #[serde(rename="claude-3-7-sonnet-20250219")]
    Claude37Sonnet,
    #[serde(rename="claude-3-5-haiku-20241022")]
    Claude35Haiku,
    #[serde(rename="claude-3-5-sonnet-20241022")]
    Claude35Sonnet,
    #[serde(rename="claude-3-haiku-20240307")]
    Claude3Haiku,
    #[serde(rename="claude-3-sonnet-20240229")]
    Claude3Sonnet,
    #[serde(rename="claude-3-opus-20240229")]
    Claude3Opus
}
impl GptModel {
    pub fn is_openai(&self) -> bool {
        [GptModel::O3Mini, GptModel::O1Mini, GptModel::Gpt4o, GptModel::Gpt4oMini].contains(self)
    }
    pub fn is_anthropic(&self) -> bool {
        [GptModel::Claude37Sonnet, GptModel::Claude35Haiku, GptModel::Claude35Sonnet, GptModel::Claude3Haiku, GptModel::Claude3Opus, GptModel::Claude3Sonnet].contains(self)
    }
    pub fn has_system_message(&self) -> bool {
        [GptModel::Gpt4o, GptModel::Gpt4oMini].contains(self)
    }
    pub fn anthropic_credentials(&self) -> Credentials {
        Credentials::from_env()
    }

    pub fn anthropic_max_tokens(&self) -> u64 {
        match self {
            Self::Claude37Sonnet => 64000,
            Self::Claude35Sonnet => 8192,
            Self::Claude3Sonnet => 4096,
            Self::Claude35Haiku => 8192,
            Self::Claude3Haiku => 4096,
            Self::Claude3Opus => 4096,
            _ => 1024
        }
    }
}

impl From<&GptModel> for String {
    fn from(model: &GptModel) -> Self {
        serde_plain::to_string(model).expect("GptModel serialization failed")
    }
}

pub type TreeNodes = HashMap<String, TreeNode>;

pub trait TreeNodeReader {
    fn from_vec(v: Vec<TreeNode>) -> Result<TreeNodes, GptError>;
    fn from_str(s: &str) -> Result<TreeNodes, GptError>;
}

impl TreeNodeReader for TreeNodes {
    fn from_str(s: &str) -> Result<TreeNodes, GptError> {
        Ok(serde_json::from_str::<TreeNodes>(s)?)
    }
    fn from_vec(v: Vec<TreeNode>) -> Result<TreeNodes, GptError> {
        let mut label_nodes = Vec::new();
        for node in v {
            label_nodes.push((
                node.label.clone().ok_or(GptError::TreeNodeLabelMissing)?, 
                node
            ))
        }
        Ok(HashMap::from_iter(label_nodes))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecisionTree {
    pub name: String,
    pub version: String,
    pub description: String,
    pub max_repeats: usize,
    pub nodes: TreeNodes
}
impl DecisionTree {
    pub fn new(name: &str, version: &str, description: &str, nodes: &str) -> Result<Self, GptError> {
        Ok(
            Self {
                name: name.to_string(),
                version: version.to_string(),
                description: description.to_string(),
                max_repeats: 3,
                nodes: TreeNodes::from_str(nodes)?
            }
        )
    }
    pub fn tiered() -> Result<Self, GptError> {

        let check_above_threshold = TreeNode::default()
            .label("check_above_threshold")
            .true_node("diagnose_infectious")
            .false_node("check_below_threshold")
            .with_check(DiagnosticNode::AboveThresholdQuery)
            .with_tasks(
                dedent(r"  
                    1. Determine if the metagenomic taxonomic profiling data [Data] supports an infectious diagnosis or a non-infectious diagnosis. Infectious clinical symptoms do not necessarily indicate an infectious cause.
                    2. Consider making an infectious diagnosis if you are certain the species is a human pathogen, or the species occurs at very high abundance. Consider making an infectious diagnosis even if the pathogen is unusual or uncommon for the provided sample type or clinical context. Consider the potential for background contamination from reagents, sample site and the environment, but only if you are sure the species is not a typical human pathogen.
                    3. If a virus is detected, strongly consider an infectious diagnosis. 
                ")
            )?
            .with_instructions(
                dedent(r"
                    1.  Output your determination inside <result></result> tags (XML).
                    1a. Output 'yes' in <result></result> tags (<result>yes</result>) if the data supports an infectious diagnosis. 
                    1b. Output 'no' in <result></result> tags (<result>no</result>) if the data does not support an infectious diagnosis. 
                "))?;

        let check_below_threshold = TreeNode::default()
            .label("check_below_threshold")
            .next("check_target_threshold")
            .with_check(DiagnosticNode::BelowThresholdQuery)
            .with_tasks(
                dedent(r"  
                    1. Determine if the metagenomic taxonomic profiling data [Data] supports an infectious diagnosis or a non-infectious diagnosis. Infectious clinical symptoms do not necessarily indicate an infectious cause.
                    2. Consider the potential for background contamination from reagents, sample site and the environment. Consider making an infectious diagnosis if you are certain the species is a common human pathogen. Consider making a non-infectious diagnosis if the species is unusual or uncommon for the provided sample type or clinical context.
                    3. If a virus is detected, strongly consider an infectious diagnosis.
                ")
            )?
            .with_instructions(
                dedent(r"
                    1.  Output your determination inside <result></result> tags (XML).
                    1a. Output 'yes' in <result></result> tags (<result>yes</result>) if the data supports an infectious diagnosis. 
                    1b. Output 'no' in <result></result> tags (<result>no</result>) if the data does not support an infectious diagnosis. 
                ")
            )?;
        
            let check_target_threshold = TreeNode::default()
                .label("check_target_threshold")
                .next("integrate_thresholds")
                .with_check(DiagnosticNode::TargetThresholdQuery)
                .with_tasks(
                    dedent(r"  
                       1. Determine if the metagenomic taxonomic profiling data [Data] supports an infectious diagnosis or a non-infectious diagnosis. Infectious clinical symptoms do not necessarily indicate an infectious cause.
                       2. Consider the potential for background contamination from reagents, sample site and the environment. Consider making an infectious diagnosis if you are certain the species is a common human pathogen. Consider making a non-infectious diagnosis if the species is unusual or uncommon for the provided sample type or clinical context.
                       3. If a virus is detected, strongly consider an infectious diagnosis.
                    ")
                )?
                .with_instructions(
                    dedent(r"
                        1.  Output your determination inside <result></result> tags (XML).
                        1a. Output 'yes' in <result></result> tags (<result>yes</result>) if the data supports an infectious diagnosis. 
                        1b. Output 'no' in <result></result> tags (<result>no</result>) if the data does not support an infectious diagnosis. 
                    ")
                )?;

            let integrate_thresholds = TreeNode::default()
                .label("integrate_thresholds")
                .true_node("diagnose_infectious")
                .false_node("diagnose_non_infectious")
                .with_check(DiagnosticNode::IntegrateThresholds)
                .with_tasks(
                    dedent(r"  
                        1. Determine if the metagenomic taxonomic profiling data supports an infectious diagnosis or a non-infectious diagnosis. Infectious clinical symptoms do not necessarily indicate an infectious cause.
                        2. Consider the potential for background contamination from reagents, sample site and the environment. Consider making an infectious diagnosis if you are certain the species is a common human pathogen. Consider making a non-infectious diagnosis if the species is unusual or uncommon for the provided sample type or clinical context.
                        3. If a virus is detected, strongly consider an infectious diagnosis.
                    ")
                )?
                .with_instructions(
                    dedent(r"
                        1.  Output your determination inside <result></result> tags (XML).
                        1a. Output 'yes' in <result></result> tags (<result>yes</result>) if the data supports an infectious diagnosis. 
                        1b. Output 'no' in <result></result> tags (<result>no</result>) if the data does not support an infectious diagnosis. 
                    ")
                )?;
            
            let diagnose_infectious = TreeNode::default()
                .label("diagnose_infectious")
                .final_node(true)
                .with_check(DiagnosticNode::DiagnoseInfectious)
                .with_tasks(
                    dedent(r"  
                        You have made an infectious diagnosis for this sample. 

                        1. Determine the most likely pathogen from metagenomic taxonomic profiling data [Data] and the provided context [Context]. Infectious clinical symptoms do not necessarily indicate an infectious cause.
                        2. Consider the potential for background contamination from reagents, sample site and the environment. Consider making the determination if the species is a human pathogen.
                        3. If a virus is detected, strongly consider a selection as most likely pathogen.
                    ")
                )?
                .with_instructions(
                    dedent(r"
                        1.  Output the most likely pathogen inside <pathogen></pathogen> (XML).
                        1a. You must select only one of the species in [Data] - the most likely pathogen - and place it into <pathogen></pathogen> tags (XML)
                        1b. You are not allowed to put a value other than the pathogen species inside <pathogen></pathogen> tags (XML).
                        1c. You must place the full genus and species name from [Data] inside <pathogen></pathogen> tags (XML).
                        1d. You are not allowed to explain your selection.

                        Example: <pathogen>Rodorendens figura</pathogen>

                        Your output:
                    ")
                )?;
            
            let diagnose_non_infectious = TreeNode::default()
                .label("diagnose_non_infectious")
                .final_node(true)
                .with_check(DiagnosticNode::DiagnoseNonInfectious);

        let nodes = vec![
            check_above_threshold,
            check_below_threshold,
            check_target_threshold,
            integrate_thresholds,
            diagnose_infectious,
            diagnose_non_infectious
        ];
        
        Ok(
            Self {
                name: "tiered".to_string(),
                version: "0.2.0".to_string(),
                description: "Tiered decision making process using tiered filter sections of the metagenomic taxonomic profiling data as primary determination of infectious or non-infectious samples".to_string(),
                max_repeats: 3,
                nodes: TreeNodes::from_vec(nodes)?
            }
        )
    }

}

fn dedent(input: &str) -> String {

    let lines: Vec<&str> = input
        .lines()
        .skip_while(|l| l.trim().is_empty())
        .collect();
    let indent = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.chars().take_while(|c| c.is_whitespace()).count())
        .min()
        .unwrap_or(0);

    lines
        .iter()
        .map(|l| {
            if l.len() > indent {
                &l[indent..]
            } else {
                *l
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PrefetchData {
    pub primary: Vec<Taxon>,
    pub secondary: Vec<Taxon>,
    pub target: Vec<Taxon>,
    pub primary_filter: TaxonFilterConfig,
    pub secondary_filter: TaxonFilterConfig,
    pub target_filter: TaxonFilterConfig,
    pub config: MetaGpConfig
}
impl PrefetchData {
    pub fn new(
        primary: Vec<Taxon>,
        secondary: Vec<Taxon>,
        target: Vec<Taxon>,
        primary_filter: &TaxonFilterConfig,
        secondary_filter: &TaxonFilterConfig,
        target_filter: &TaxonFilterConfig,
        config: &MetaGpConfig
    ) -> Self {
        Self {
            primary,
            secondary,
            target,
            primary_filter: primary_filter.clone(),
            secondary_filter: secondary_filter.clone(),
            target_filter: target_filter.clone(),
            config: config.clone()
        }
    }
    /// Remove from `secondary` any Taxon whose `lineage` also appears in `primary`,
    /// then remove from `target` any Taxon whose `lineage` also appears in the (now-pruned) `secondary`.
    pub fn prune(&mut self) {

        log::info!("Pruning tiered filter categories...");
        
        // 1) collect all lineages present in primary
        let primary_lineages: HashSet<_> =
            self.primary.iter().map(|t| t.lineage.clone()).collect();

        // 2) drop from secondary anything already in primary
        self.secondary
            .retain(|t| !primary_lineages.contains(&t.lineage));

        // 3) now collect lineages in the (pruned) secondary
        let secondary_lineages: HashSet<_> =
            self.secondary.iter().map(|t| t.lineage.clone()).collect();

        // 4) drop from target anything already in secondary
        self.target
            .retain(|t| !secondary_lineages.contains(&t.lineage));
    }
    pub fn to_json(&self, path: &Path) -> Result<(), GptError> {
        let data = serde_json::to_string_pretty(self).map_err(|err| GptError::SerdeJsonError(err))?;
        let mut writer = BufWriter::new(File::create(path)?);
        write!(writer, "{data}")?;
        Ok(())
    }
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self, GptError> {
        let data = std::fs::read_to_string(path)?;
        let result = serde_json::from_str::<PrefetchData>(&data)?;
        Ok(result)
    }
}

impl PrefetchData {
    /// Draw a grouped horizontal bar chart of “# taxa per domain” in each tier,
    /// and save it as an SVG to `output_path`.
    pub fn plot_domain_counts_svg(&self, output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        // 1) Define domains and tiers
        let domains    = ["Bacteria", "Eukaryota", "Viruses"];
        let tier_names = ["primary", "secondary", "target"];
        let tier_data  = [&self.primary, &self.secondary, &self.target];

        // 2) Count per (tier, domain)
        let mut counts = Vec::with_capacity(tier_data.len());
        for taxa in &tier_data {
            let mut c = [0u32; 3];
            for (i, &domain) in domains.iter().enumerate() {
                c[i] = taxa
                    .iter()
                    .filter(|t| t.lineage.get_domain().as_deref() == Some(domain))
                    .count() as u32;
            }
            counts.push(c);
        }
        let max_count = counts
            .iter()
            .flat_map(|arr| arr.iter())
            .cloned()
            .max()
            .unwrap_or(0);

        // 3) Create SVG backend
        let root = SVGBackend::new(output_path, (800, 400)).into_drawing_area();
        root.fill(&WHITE)?;

        // 4) Build cartesian axes: X from 0..max_count, Y from 0.0..3.0
        let y_max = tier_data.len() as f64;
        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .caption("Taxa per Domain by Filter Tier", ("sans-serif", 20))
            .x_label_area_size(40)
            .y_label_area_size(80)
            .build_cartesian_2d(0u32..(max_count + 1), 0f64..y_max)?;

        // 5) Configure mesh, custom y‐axis labels
        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("Number of Taxa")
            .y_desc("Filter Tier")
            .y_labels(tier_data.len())
            .y_label_formatter(&|v| {
                let idx = *v as usize;
                if idx < tier_names.len() {
                    tier_names[idx].to_string()
                } else {
                    "".to_string()
                }
            })
            .draw()?;

        // 6) Bar geometry: three bars per tier
        let bar_height    = 0.2;
        let spacing       = 0.05;
        let group_height  = bar_height * domains.len() as f64 + spacing * (domains.len() - 1) as f64;
        let half_group    = group_height / 2.0;
        let colors        = [&BLUE, &GREEN, &RED];

        for (tier_idx, domain_counts) in counts.iter().enumerate() {
            let tier_y = tier_idx as f64;
            for (dom_idx, &val) in domain_counts.iter().enumerate() {
                // offset each bar within the group so they don't overlap
                let offset = (dom_idx as f64) * (bar_height + spacing) - half_group + (bar_height / 2.0);
                let y0 = tier_y + offset;
                let y1 = y0 + bar_height;

                chart.draw_series(std::iter::once(
                    Rectangle::new(
                        [(0u32, y0), (val, y1)],
                        colors[dom_idx].filled(),
                    ),
                ))?;
            }
        }

        Ok(())
    }
}

pub struct DiagnosticAgent {
    pub state: AgentState,
    pub tree: DecisionTree,
    pub client: Option<CerebroClient>,
    pub graph: Graph<TreeNode, TreeEdge>,
}

impl DiagnosticAgent {
    pub fn new(client: Option<CerebroClient>) -> Result<Self, GptError> {
        
        let tree = DecisionTree::tiered()?;
        
        Ok(DiagnosticAgent {
            tree: tree.clone(),
            client,
            state: AgentState::new(),
            graph: Self::graph(&tree)?
        })
    }

    pub fn prefetch(&self, output: &Path, config: &MetaGpConfig) -> Result<(), GptError> {
        
        log::info!("Fetching primary threshold data for sample '{}'", config.sample);

        if let Some(client) = &self.client {
            let primary_filter_config = &TaxonFilterConfig::gp_above_threshold(
                config.ignore_taxstr.clone()
            );
            let (primary, _) = client.get_taxa( 
                &CerebroIdentifierSchema::from_gp_config(config), 
                primary_filter_config, 
                &config.contamination,
                config.prevalence_outliers.primary
            )?;
    
            log::info!("Fetching secondary threshold data for sample '{}'", config.sample);
    
            let secondary_filter_config = &TaxonFilterConfig::gp_below_threshold(
                config.ignore_taxstr.clone()
            );
            let (secondary, _) = client.get_taxa( 
                &CerebroIdentifierSchema::from_gp_config(config), 
                secondary_filter_config, 
                &config.contamination,
                config.prevalence_outliers.secondary
            )?;
    
            log::info!("Fetching target threshold data for sample '{}'", config.sample);
            let target_filter_config = &TaxonFilterConfig::gp_target_threshold(
                config.ignore_taxstr.clone()
            );
            let (target, _) = client.get_taxa( 
                &CerebroIdentifierSchema::from_gp_config(config), 
                target_filter_config, 
                &config.contamination,
                config.prevalence_outliers.target
            )?;

            let mut prefetch_data = PrefetchData::new(
                primary, 
                secondary, 
                target, 
                primary_filter_config,
                secondary_filter_config,
                target_filter_config,
                config
            );

            prefetch_data.prune();
            prefetch_data.to_json(output)

        } else {
            Err(GptError::CerebroClientNotProvided)
        }
        

    }
    // Collapses GTDB species variants and sums the taxon evidence for
    // each combination of (id, tool, mode) returned from the taxon
    // retrieval request with the standard settings - the filter config
    // for the retrieval request itself can collapse variants; in this 
    // case we do this for the pathogen selection stage since the variants
    // are not informative for clinicians
    //
    // Note: the prevalence filter uses the variants in the retrieval request!
    pub fn collapse_variants(taxa: Vec<Taxon>) -> Result<Vec<Taxon>, GptError> {
        Ok(collapse_taxa(taxa)?)
    }
    // Selects the species with the most evidence support if multiple species
    // of the same genus are detected - used for bacteria and eukaryots
    pub fn select_best_species(taxa: Vec<Taxon>, post_filter: &PostFilterConfig) -> Result<Vec<Taxon>, GptError> {

        let base_weight = post_filter.best_species_base_weight.unwrap_or(1.0);

        let mut retained = Vec::new();

        let mut species_selection: HashMap<String, Vec<Taxon>> = HashMap::new();
        for taxon in taxa {
            match taxon.lineage.get_domain() {
                Some(domain_str) => {
                    if post_filter.best_species_domains.contains(&domain_str) {

                        // Aggregate only from the provided domains - we want to do this for Bacteria/Archaea/Eukaryota 
                        // but not necessarily for viruses as the genera are quite diverse and likely to co-occur whereas
                        // this is generally not the case for Eukaryots. Application of selection for Bacteria/Archaea may 
                        // need to be tested and optimized.

                        match taxon.lineage.get_genus() {
                            Some(genus_str) => {
                                species_selection
                                    .entry(genus_str.clone())
                                    .or_default()
                                    .push(taxon);
                            },
                            None => retained.push(taxon) // taxa with rank above genus are retained by default
                        }
                    } else {
                        retained.push(taxon) // taxon with domain rank not in targeted domains
                    }
                },
                None => retained.push(taxon) // taxon with missing domain rank in lineage
            }
        }

        // Pre-filter genera with too few species
        let mut filtered_selection = HashMap::new();
        for (genus, species_vec) in species_selection {
            if species_vec.len() < post_filter.best_species_min {
                retained.extend(species_vec);
            } else {
                filtered_selection.insert(genus, species_vec);
            }
        }

        let best_species: Vec<Taxon> = filtered_selection.into_iter()
            .filter_map(|(_, taxa)| {
                // Skip empty Vecs just in case
                let best = taxa.into_iter()
                    .max_by(|a, b| {
                        let sa = a.evidence.profile_score(base_weight);
                        let sb = b.evidence.profile_score(base_weight);
                        sa.partial_cmp(&sb).unwrap()
                    })?;
                Some(best)
            })
            .collect();
        
        for taxon in best_species {
            retained.push(taxon)
        }

        Ok(retained)
    }
    pub fn exclude_phage(taxa: Vec<Taxon>, post_filter: &PostFilterConfig) -> Vec<Taxon> {

        taxa.into_iter().filter(|taxon| {
            if let Some(domain) = taxon.lineage.get_domain() {
                if domain == "Viruses" {
                    !post_filter.exclude_phage_list.contains(&taxon.name) // do not return taxon if in phage list
                } else {
                    true // return taxon not virus
                }
            } else {
                true // return taxon no domain
            }
        }).collect()
    }
    pub fn apply_post_filter(taxa: Vec<Taxon>, post_filter: &PostFilterConfig) -> Result<Vec<Taxon>, GptError> {
        
        let taxa = if post_filter.collapse_variants {
            Self::collapse_variants(taxa)?
        } else {
            taxa
        };

        let taxa = if post_filter.best_species {
            Self::select_best_species(taxa, &post_filter)?
        } else {
            taxa
        };

        let taxa = if post_filter.exclude_phage {
            Self::exclude_phage(taxa, &post_filter)
        } else {
            taxa
        };

        Ok(taxa)
    }

    #[cfg(feature = "local")]
    pub fn run_local(
        &mut self, 
        text_generator: &mut TextGenerator, 
        sample_context: SampleContext, 
        clinical_notes: Option<String>, 
        assay_context: Option<AssayContext>, 
        config: &MetaGpConfig,
        prefetch: Option<PrefetchData>,
        post_filter: Option<PostFilterConfig>,
        disable_thinking: bool
    ) -> Result<DiagnosticResult, GptError> {

        self.state.post_filter_config = post_filter.clone();

        let mut node_label = "check_above_threshold".to_string();

        let sample_context = match clinical_notes {
            Some(notes) => sample_context.with_clinical_notes(&notes),
            None => sample_context.text()
        };
        
        let assay_context = match assay_context {
            Some(context) => context.text(),
            None => AssayContext::None.text()
        };

        let context = format!(
            "{}\n{}", 
            dedent(&assay_context), 
            dedent(&sample_context)
        );

        let mut result = DiagnosticResult {
            diagnosis: Diagnosis::Unknown,
            candidates: Vec::new(),
            pathogen: None
        };

        while let Some(node_ref) = self.tree.nodes.get(&node_label) {

            let current_node = node_ref.clone();

            log::info!("Processing node: {}", node_label);
            
            match current_node.check {
                Some(DiagnosticNode::AboveThresholdQuery) => {

                    // Taxon 
                    let primary_taxa = match prefetch {
                        Some(ref data) => data.primary.clone(),
                        None => {
                            let (primary, _) = match &self.client {
                                Some(client) => client.get_taxa( 
                                    &CerebroIdentifierSchema::from_gp_config(config), 
                                    &TaxonFilterConfig::gp_above_threshold(
                                        config.ignore_taxstr.clone()
                                    ), 
                                    &config.contamination,
                                    config.prevalence_outliers.primary
                                )?,
                                None => return Err(GptError::CerebroClientNotProvided)
                            };
                            primary
                        }
                    };

                    log::info!("Primary taxa: {}", primary_taxa.len());

                    let primary_taxa = if let Some(ref post_filter) = post_filter {
                        Self::apply_post_filter(primary_taxa, post_filter)?
                    } else {
                        primary_taxa
                    };


                    log::info!("Primary taxa post filter: {}", primary_taxa.len());
                    

                    let (result, confidence, prompt, thoughts, answer) = if !primary_taxa.is_empty() {

                        let candidates = ThresholdCandidates::from_primary_threshold(
                            primary_taxa.clone()
                        ).to_str(true);

                        let prompt = current_node
                            .clone()
                            .with_context(&context)?
                            .with_data(&candidates)?
                            .question
                            .unwrap()
                            .to_standard_prompt();
                        
                        log::info!("\n\n{prompt}");

                        let (thoughts, answer) = text_generator.run(&prompt, disable_thinking)?;
                        
                        log::info!("{thoughts}\n\n");
                        log::info!("{answer}");

                        (Self::extract_result(&answer, disable_thinking)?, None::<String>, Some(prompt), Some(thoughts), Some(answer))
                    } else {
                        log::info!("No data retrieved for this node");
                        (Some(false), None, None, None, None) // no taxa detected
                    };

                    self.state.memorize(
                        DiagnosticMemory::new(
                            DiagnosticNode::AboveThresholdQuery, 
                            primary_taxa, 
                            result, 
                            prompt, 
                            thoughts, 
                            answer
                        )
                    );
                    
                    match self.get_next_node_label(&current_node, result)? {
                        Some(label) => node_label = label,
                        None => break
                    }
                },

                Some(DiagnosticNode::BelowThresholdQuery) => {
                    
                    let secondary_taxa = match prefetch {
                        Some(ref data) => data.secondary.clone(),
                        None => {
                            let (secondary, _) = match &self.client { 
                                Some(client) => client.get_taxa( 
                                    &CerebroIdentifierSchema::from_gp_config(config), 
                                    &TaxonFilterConfig::gp_below_threshold(
                                        config.ignore_taxstr.clone()
                                    ), 
                                    &config.contamination,
                                    config.prevalence_outliers.secondary
                                )?,
                                None =>  return Err(GptError::CerebroClientNotProvided)
                            };
                            secondary
                        }
                    };

                    let secondary_taxa = if let Some(ref post_filter) = post_filter {
                        Self::apply_post_filter(secondary_taxa, post_filter)?
                    } else {
                        secondary_taxa
                    };

                    let (result, confidence, prompt, thoughts, answer) = if !secondary_taxa.is_empty() {

                        let candidates = ThresholdCandidates::from_secondary_threshold(
                            secondary_taxa.clone()
                        ).to_str(true);

                        let prompt = current_node
                            .clone()
                            .with_context(&context)?
                            .with_data(&candidates)?
                            .question
                            .unwrap()
                            .to_standard_prompt();
                        
                        log::info!("\n\n{prompt}");

                        let (thoughts, answer) = text_generator.run(&prompt, disable_thinking)?;
                        
                        log::info!("{thoughts}\n\n");
                        log::info!("{answer}");

                        (Self::extract_result(&answer,disable_thinking)?, None::<String>, Some(prompt), Some(thoughts), Some(answer))
                    } else {
                        log::info!("No data retrieved for this node");
                        (Some(false), None, None, None, None) // no taxa detected
                    };

                    self.state.memorize(
                        DiagnosticMemory::new(
                            DiagnosticNode::BelowThresholdQuery, 
                            secondary_taxa, 
                            result, 
                            prompt, 
                            thoughts, 
                            answer
                        )

                    );
                    
                    match self.get_next_node_label(&current_node, result)? {
                        Some(label) => node_label = label,
                        None => break
                    }
                    
                },
                Some(DiagnosticNode::TargetThresholdQuery) => {

                    let target_taxa = match prefetch {
                        Some(ref data) => data.target.clone(),
                        None => {
                            let (target, _) = match &self.client {
                                Some(client) => client.get_taxa( 
                                    &CerebroIdentifierSchema::from_gp_config(config), 
                                    &TaxonFilterConfig::gp_target_threshold(
                                        config.ignore_taxstr.clone()
                                    ), 
                                    &config.contamination,
                                    config.prevalence_outliers.target
                                )?,
                            None => return Err(GptError::CerebroClientNotProvided)
                            };
                            target
                        }
                    };

                    let target_taxa = if let Some(ref post_filter) = post_filter {
                        Self::apply_post_filter(target_taxa, post_filter)?
                    } else {
                        target_taxa
                    };

                    let (result, confidence, prompt, thoughts, answer) = if !target_taxa.is_empty() {

                        let candidates = ThresholdCandidates::from_target_threshold(
                            target_taxa.clone()
                        ).to_str(true);

                        let prompt = current_node
                            .clone()
                            .with_context(&context)?
                            .with_data(&candidates)?
                            .question
                            .unwrap()
                            .to_standard_prompt();
                        
                        log::info!("\n\n{prompt}");

                        let (thoughts, answer) = text_generator.run(&prompt, disable_thinking)?;
                        
                        log::info!("{thoughts}\n\n");
                        log::info!("{answer}");

                        (Self::extract_result(&answer, disable_thinking)?, None::<String>, Some(prompt), Some(thoughts), Some(answer))
                    } else {
                        log::info!("No data retrieved for this node");
                        (Some(false), None, None, None, None) // no taxa detected
                    };
                    
                    self.state.memorize(
                        DiagnosticMemory::new(
                            DiagnosticNode::TargetThresholdQuery, 
                            target_taxa, 
                            result, 
                            prompt, 
                            thoughts, 
                            answer
                        )
                    );

                    match self.get_next_node_label(&current_node, result)? {
                        Some(label) => node_label = label,
                        None => break
                    }
                    
                },
                Some(DiagnosticNode::IntegrateThresholds) => {
                    
                    // Retrieve the below and target threshold data and result memories
                    log::info!("IntegrateThresholds");
                    let secondary_memory = self.state.retrieve(DiagnosticNode::BelowThresholdQuery).cloned();
                    let target_memory = self.state.retrieve(DiagnosticNode::TargetThresholdQuery).cloned();

                    match (
                        secondary_memory, 
                        target_memory
                    ) {
                        (
                            Some(secondary_memory), 
                            Some(target_memory)
                        ) => {
                            match (
                                secondary_memory.data.is_empty(), 
                                target_memory.data.is_empty()
                            ) {
                                // Continue with diagnosis if data was available in one of the diagnostic nodes but not the other
                                // use the result from that stage to continue in the decision tree
                                (false, true) => {
                                    log::info!("Data only from secondary threshold node - continue to next node with result from secondary threshold node");

                                    let mut secondary_memory_integrated = secondary_memory.clone();
                                    secondary_memory_integrated.node = DiagnosticNode::IntegrateThresholds;

                                    self.state.memorize(secondary_memory_integrated);

                                    match self.get_next_node_label(&current_node, secondary_memory.result)? {
                                        Some(label) => node_label = label,
                                        None => break
                                    }
                                },
                                (true, false) => {
                                    let mut target_memory_integrated = target_memory.clone();
                                    target_memory_integrated.node = DiagnosticNode::IntegrateThresholds;

                                    self.state.memorize(target_memory_integrated);

                                    log::info!("Data only from target threshold node - continue to next node with result from target threshold node");
                                    match self.get_next_node_label(&current_node, target_memory.result)? {
                                        Some(label) => node_label = label,
                                        None => break
                                    }
                                },
                                (true, true) => {
                                    // No data from the sub-threshold nodes mean we didn't make a diagnosis in the primary threshold either 
                                    // so we assign the final result as non-infectious
                                    self.state.memorize(
                                        DiagnosticMemory::non_infectious(
                                            DiagnosticNode::IntegrateThresholds
                                        )
                                    );

                                    node_label = String::from("diagnose_non_infectious")
                                },
                                // Continue with integration node processing and decision making if bnoth nodes had data available -
                                // this will combine the data for the integration node prompt and override the decisions made previously
                                (false, false) => {

                                    let below_threshold_data = ThresholdCandidates::from_secondary_threshold(
                                        secondary_memory.data.clone()
                                    ).to_str(true);

                                    let target_threshold_data = ThresholdCandidates::from_target_threshold(
                                        target_memory.data.clone()
                                    ).to_str(true);
                                    
                                    let candidates = format!("{}\n\n{}", below_threshold_data, target_threshold_data);

                                    let prompt = current_node
                                        .clone()
                                        .with_context(&context)?
                                        .with_data(&candidates)?
                                        .question
                                        .unwrap()
                                        .to_standard_prompt();
                                    
                                    log::info!("\n\n{prompt}");
            
                                    let (thoughts, answer) = text_generator.run(&prompt, disable_thinking)?;
                                    
                                    log::info!("{thoughts}\n\n");
                                    log::info!("{answer}");

                                    let result = Self::extract_result(&answer,disable_thinking)?;

                                    let data = [
                                        secondary_memory.data.clone(), 
                                        target_memory.data.clone()
                                    ].concat();

                                    self.state.memorize(
                                        DiagnosticMemory::new(
                                            DiagnosticNode::IntegrateThresholds, 
                                            data, 
                                            result, 
                                            Some(prompt), 
                                            Some(thoughts), 
                                            Some(answer)
                                        )
                                    );

                                    match self.get_next_node_label(&current_node, result)? {
                                        Some(label) => node_label = label,
                                        None => break
                                    }

                                }
                            }
                        },
                        // If no memories were available for either of the low abundance diagnostic nodes
                        _ => {
                            log::warn!("No data available for the integration node - this should not happen!");
                            break
                        }
                    }

                },
                Some(DiagnosticNode::DiagnoseInfectious) => {
                    
                    let mut candidates = String::new();
                    let mut memory_candidates = Vec::new();

                    if let Some(memory) = self.state.retrieve(DiagnosticNode::AboveThresholdQuery) {
                        if let Some(result) = memory.result {
                            if result {
                                let primary_candidates = ThresholdCandidates::from_primary_threshold(
                                    memory.data.clone()
                                ).to_str(true);
                                
                                if !primary_candidates.is_empty() {
                                    candidates.push_str(&primary_candidates);
                                    candidates.push_str("\n\n");
                                }
                                memory_candidates.extend_from_slice(&memory.data);
                            }
                        };
                    };


                    if let Some(memory) = self.state.retrieve(DiagnosticNode::IntegrateThresholds) {
                        if let Some(result) = memory.result {
                            if result {
                                let integrate_candidates = ThresholdCandidates::from_integrate_threshold(
                                    memory.data.clone()
                                ).to_str(true);
                                
                                if !integrate_candidates.is_empty() {
                                    candidates.push_str(&integrate_candidates);
                                    candidates.push_str("\n\n");
                                }
                                memory_candidates.extend_from_slice(&memory.data);
                            }
                        }
                    };
    
                    let prompt = current_node
                        .clone()
                        .with_context(&context)? 
                        .with_data(&candidates)?
                        .question
                        .unwrap()
                        .to_standard_prompt();
                
                    log::info!("\n\n{prompt}");

                    let (thoughts, answer) = text_generator.run(&prompt, disable_thinking)?;
                    
                    log::info!("{thoughts}\n\n");
                    log::info!("{answer}");


                    let candidates = Self::extract_tags(&answer, "candidate", disable_thinking)?;
                    let pathogens = Self::extract_tags(&answer, "pathogen",disable_thinking)?;

                    result.diagnosis = Diagnosis::Infectious;
                    result.candidates = candidates;
                    result.pathogen = pathogens.first().cloned();


                    self.state.memorize(
                        DiagnosticMemory::new(
                            DiagnosticNode::DiagnoseInfectious, 
                            memory_candidates, 
                            None, 
                            Some(prompt), 
                            Some(thoughts), 
                            Some(answer)
                        )
                    );

                    break
                },
                Some(DiagnosticNode::DiagnoseNonInfectious) => {
                    result.diagnosis = Diagnosis::NonInfectious;
                    break
                },
                Some(_) => {
                    log::warn!("Node processing not implemented for node: {}", current_node.label.unwrap_or("no_label".to_string()));
                    break
                }
                _ => {
                    log::warn!("Node processing not implemented for node: {}", current_node.label.unwrap_or("no_label".to_string()));
                    break
                }
            }            
        }

        Ok(result)
    }
    fn get_next_node_label(&mut self, current_node: &TreeNode, result: Option<bool>) -> Result<Option<String>, GptError> {
        
        let node_label = if let Some(next_node_label) = &current_node.next {
            Some(next_node_label.clone())
        } else {
            // Failed to extract expected decision value -> repeat node question (self-loop)
            match result {
                None => {
                    log::info!("Failed to extract decision from answer - intitiate repeat check.");

                    // Check the current node repeats before returning
                    let check_type = current_node.check.as_ref()
                        .ok_or(GptError::NodeCheckTypeMissing)?;

                    let count = self
                        .state
                        .repeat
                        .entry(check_type.clone())
                        .and_modify(|c| *c += 1)   
                        .or_insert(1);

                    if *count > self.tree.max_repeats {
                        log::warn!("Maximum number of repeats exceeded, exit diagnostic process.");
                        None
                    } else { 
                        log::warn!("Initiate diagnostic node process ({count}).");
                        current_node.label.clone()
                    }
                },
                Some(decision) => {
                    if decision {
                        Some(current_node.true_node.clone().expect(&format!("True node expected")))
                    } else {
                        Some(current_node.false_node.clone().expect(&format!("False node expected")))
                    }
                }
            }
        };
        Ok(node_label)
    }
    fn extract_result(s: &str, relaxed: bool) -> Result<Option<bool>, GptError> {
        let results = Self::extract_tags(s, "result", relaxed)?;
        
        // Only consider the first result block
        let result = match results.first() {
            Some(value) => {
                if value.replace(" ", "").to_lowercase() == "yes" {
                    Some(true)
                } else if value.replace(" ", "").to_lowercase() == "no"  {
                    Some(false)
                } else {
                    None
                }
            },
            None => {
                // ALl the crazy ways the models can spit out identifiable yes or no answers... damn things
                if s.contains("Result: yes") {
                    Some(true)
                } else if s.contains("Result: no")   {
                    Some(false)
                } else if s.contains("```\nyes\n```") {
                    Some(true)
                } else if s.contains("=yes") {
                    Some(true)
                } else if s.contains("=no") {
                    Some(true)
                } else if s.contains("```\nno\n```") {
                    Some(false)
                } else if s.contains("```result\nyes\n```") {
                    Some(true)
                } else if s.contains("```result\nno\n```") {
                    Some(false)
                } else if s.contains("<result>yes</result>") {
                    Some(true)
                } else if s.contains("<result>no</result>") {
                    Some(false)
                } else {
                    log::warn!("Failed to find result in answer: {s}");
                    None
                }
            }
        };
        Ok(result)
    }

    /// Removes trailing variant tags (an underscore + uppercase letters) from each
    /// whitespace-separated token in the input.
    ///
    /// Example:
    /// "Staphylococcus aureus_A"     → "Staphylococcus aureus"
    /// "Escherichia coli_B"          → "Escherichia coli"
    /// "Bacillus subtilis"           → "Bacillus subtilis"  (unchanged)
    fn strip_variant_tags(input: &str) -> String {
        // match "_" followed by one or more uppercase A–Z at end of string
        let re = regex::Regex::new(r"_[A-Z]+$").expect("Invalid regex");
        input
            .split_whitespace()
            .map(|token| {
                // if the token ends in "_XXX", strip that off; otherwise leave it alone
                re.replace(token, "").into_owned()
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
    fn extract_tags(input: &str, tag: &str, relaxed: bool) -> Result<Vec<String>, GptError> {
        let tag = regex::escape(tag);
        
        let pat = if relaxed {
            format!(r">(?s)(.*?)</{0}>", tag)  // diabling thinking mode in Qwen3 results in partial tags for some answers such as: "ogen>Streptococcus mitis</pathogen>"
        } else {
            format!(r"<{0}>(?s)(.*?)</{0}>", tag)
        };

        let re = regex::Regex::new(&pat)?;

        let extracted = re.captures_iter(input)
            .filter_map(|cap| cap.get(1))
            .map(|m| Self::strip_variant_tags(m.as_str()))  // GTDB genus or species variant tags are not informative and too unpredictable for clinical evaluation
            .collect();
        
        Ok(extracted)
    }
    pub fn graph(tree: &DecisionTree) -> Result<petgraph::Graph<TreeNode, TreeEdge>, GptError> {

        // Build a simple knowledge graph from the decision tree.
        let mut graph = Graph::<TreeNode, TreeEdge>::new();
        let mut node_indices = HashMap::new();

        // Create a node in the graph for each decision tree node.
        for (key, node) in &tree.nodes {
            let mut labeled_node = node.clone();
            labeled_node.label = Some(key.to_string());
            let idx = graph.add_node(labeled_node.clone());
            node_indices.insert(key.clone(), idx);
        }
        // Add edges for transitions.
        for (key, node) in &tree.nodes  {
            if let Some(ref true_node) = node.true_node {
                if let (Some(&from), Some(&to)) =
                    (node_indices.get(key), node_indices.get(true_node))
                {
                    graph.add_edge(from, to, TreeEdge::default_true());
                }
            }
            if let Some(ref false_node) = node.false_node {
                if let (Some(&from), Some(&to)) =
                    (node_indices.get(key), node_indices.get(false_node))
                {
                    graph.add_edge(from, to, TreeEdge::default_false());
                }
            }
            if let Some(ref next) = node.next {
                if let (Some(&from), Some(&to)) =
                    (node_indices.get(key), node_indices.get(next))
                {
                    graph.add_edge(from, to, TreeEdge::default_next());
                }
            }
        }

        Ok(graph)
    }

    pub fn print_decision_tree(&self, tree: &DecisionTree, root: &str) {

        // Recursive helper function
        fn recurse(tree: &DecisionTree, node_key: &str, prefix: &str, is_last: bool) {
            let branch = if is_last { "└── " } else { "├── " };
            // Print the current node key in bold white.
            println!("{}{}{}", prefix, branch, node_key.bold().white());
            
            // Prepare the prefix for the children.
            let new_prefix = if is_last { format!("{}    ", prefix) } else { format!("{}│   ", prefix) };

            // Collect children from the node: we assume true_node, false_node, and next.
            let mut children: Vec<(&str, &String)> = Vec::new();
            if let Some(node) = tree.nodes.get(node_key) {
                if let Some(child) = &node.true_node {
                    children.push(("true", child));
                }
                if let Some(child) = &node.false_node {
                    children.push(("false", child));
                }
                if let Some(child) = &node.next {
                    children.push(("next", child));
                }
            }

            let count = children.len();
            for (i, (label, child_key)) in children.into_iter().enumerate() {
                
                // For extra style, we can color the label (e.g., yellow).
                let colored_label = format!("{}: ", label).yellow();
                let colored_child = child_key.bold().white();
                let line = format!("{}{}", colored_label, colored_child);

                // Instead of printing the label separately, we can include it in the recursion:
                // Print the child node line.
                let is_last_child = i == count - 1;
                println!("{}{}{}", new_prefix, if is_last_child { "└── " } else { "├── " }, line);

                // Then recurse on this child if it exists in the tree.
                if tree.nodes.contains_key(child_key) {
                    // Adjust prefix for further children.
                    let next_prefix = if is_last_child { format!("{}    ", new_prefix) } else { format!("{}│   ", new_prefix) };
                    recurse(tree, child_key, &next_prefix, true);
                }
            }
        }

        // Print header
        println!("{}", "Decision Tree:".bold().underline());

        // Start recursion from the root.
        recurse(tree, root, "", true);
    }
}


/// Draws a consensus decision tree (as a petgraph) using a hierarchical layout,
/// scaling node and edge opacities according to frequencies computed from a
/// collection of trees. If no additional trees are supplied, nodes and edges in the
/// consensus tree default to full opacity (alpha = 1).
///
/// # Arguments
/// - `consensus_tree`: A petgraph representing the consensus decision tree.
/// - `trees`: A slice of petgraph decision trees used to count node/edge frequencies.
/// - `filename`: The output image file (e.g. "consensus.png").
/// - `plot_width` and `plot_height`: Dimensions (in pixels) of the drawing area.
pub fn draw_consensus_tree(
    consensus_tree: &Graph<TreeNode, TreeEdge>,
    trees: &[Graph<TreeNode, TreeEdge>],
    filename: &str,
    plot_width: u32,
    plot_height: u32,
) -> Result<(), GptError>
{
    let mut node_freq: HashMap<TreeNode, u32> = HashMap::new();
    let mut edge_freq: HashMap<(TreeNode, TreeNode), u32> = HashMap::new();

    if trees.is_empty() {
        // Use the consensus tree's own nodes and edges with default frequency 1.
        for node_ref in consensus_tree.node_references() {
            node_freq.insert(node_ref.1.clone(), 1);
        }
        for edge in consensus_tree.edge_references() {
            let src = consensus_tree[edge.source()].clone();
            let tgt = consensus_tree[edge.target()].clone();
            edge_freq.insert((src, tgt), 1);
        }
    } else {
        // Count frequencies across all supplied trees.
        for tree in trees {
            for node_ref in tree.node_references() {
                *node_freq.entry(node_ref.1.clone()).or_insert(0) += 1;
            }
            for edge in tree.edge_references() {
                let src = tree[edge.source()].clone();
                let tgt = tree[edge.target()].clone();
                *edge_freq.entry((src, tgt)).or_insert(0) += 1;
            }
        }
    }
    let max_node_freq = node_freq.values().cloned().max().unwrap_or(1);
    let max_edge_freq = edge_freq.values().cloned().max().unwrap_or(1);

    let output = PathBuf::from(filename);
    let drawing_area = SVGBackend::new(
        &output, 
        (plot_width, plot_height)
    ).into_drawing_area();
    drawing_area.fill(&WHITE)?;


    fn layout_tree<N, E>(
        graph: &Graph<N, E>,
        node: NodeIndex,
        depth: u32,
        spacing_x: f64,
        spacing_y: f64,
        x_offset: &mut f64,
        layout: &mut HashMap<NodeIndex, (f64, f64)>,
    ) {
        // Gather children (nodes with outgoing edges).
        let children: Vec<NodeIndex> = graph
            .neighbors_directed(node, Direction::Outgoing)
            .collect();

        if children.is_empty() {
            // Leaf: assign current x position and increment the offset.
            layout.insert(node, (*x_offset, depth as f64 * spacing_y));
            *x_offset += spacing_x;
        } else {
            // Recursively determine positions for children.
            for child in &children {
                layout_tree(graph, *child, depth + 1, spacing_x, spacing_y, x_offset, layout);
            }
            // Position internal node at the horizontal midpoint of its children.
            let child_xs: Vec<f64> = children.iter().map(|c| layout[c].0).collect();
            let min_x = child_xs.first().cloned().unwrap_or(0.0);
            let max_x = child_xs.last().cloned().unwrap_or(0.0);
            let x = (min_x + max_x) / 2.0;
            layout.insert(node, (x, depth as f64 * spacing_y));
        }
    }

    let spacing_x = 80.0;
    let spacing_y = 100.0;
    let mut layout: HashMap<NodeIndex, (f64, f64)> = HashMap::new();
    let mut x_offset = 20.0; // Starting horizontal offset

    // Identify the root: a node with no incoming edges.
    let root = consensus_tree
        .node_indices()
        .find(|&n| consensus_tree.neighbors_directed(n, Direction::Incoming).next().is_none())
        .ok_or(GptError::TreeRootMissing)?;

    // Compute the tree layout recursively.
    layout_tree(consensus_tree, root, 0, spacing_x, spacing_y, &mut x_offset, &mut layout);

    // Center the tree horizontally within the plot.
    let (min_x, max_x, _) = layout.values().fold(
        (f64::INFINITY, f64::NEG_INFINITY, 0.0),
        |(min, max, _), &(x, y)| (min.min(x), max.max(x), y),
    );
    let tree_width = max_x - min_x;
    let shift_x = (plot_width as f64 - tree_width) / 2.0 - min_x;

    for pos in layout.values_mut() {
        pos.0 += shift_x;
    }

    for edge in consensus_tree.edge_references() {
        let src = edge.source();
        let tgt = edge.target();

        if let (Some(&(x1, y1)), Some(&(x2, y2))) = (layout.get(&src), layout.get(&tgt)) {
            let src_label = consensus_tree[src].clone();
            let tgt_label = consensus_tree[tgt].clone();
            let freq = edge_freq.get(&(src_label, tgt_label)).cloned().unwrap_or(1);
            let alpha = (freq as f64) / (max_edge_freq as f64);
            let edge_color = RGBAColor(50, 50, 200, alpha);
            drawing_area.draw(&PathElement::new(
                vec![(x1 as i32, y1 as i32), (x2 as i32, y2 as i32)],
                ShapeStyle {
                    color: edge_color,
                    filled: false,
                    stroke_width: 2,
                },
            ))?;
        }
    }

    for node in consensus_tree.node_indices() {
        if let Some(&(x, y)) = layout.get(&node) {
            let freq = node_freq.get(&consensus_tree[node]).cloned().unwrap_or(1);
            let alpha = (freq as f64) / (max_node_freq as f64);
            let node_color = RGBAColor(200, 50, 50, alpha);
            drawing_area.draw(&Circle::new((x as i32, y as i32), 10, ShapeStyle::from(&node_color).filled()))?;
            drawing_area.draw(&Text::new(
                format!("{}", match &consensus_tree[node].label { Some(label) => label.as_str(), None => "no_label" }),
                (x as i32 + 12, y as i32),
                ("sans-serif", 12).into_font().color(&BLACK),
            ))?;
        }
    }

    drawing_area.present()?;

    Ok(())
}