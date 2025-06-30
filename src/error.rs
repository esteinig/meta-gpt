use thiserror::Error;

/*
========================
Custom error definitions
========================
*/

#[derive(Error, Debug)]
pub enum GptError {
    #[error(transparent)]
    SerdeJsonError(#[from] serde_json::Error),
    #[cfg(feature = "local")]
    #[error(transparent)]
    HuggingfaceApiError(#[from] hf_hub::api::sync::ApiError),
    #[cfg(feature = "local")]
    #[error(transparent)]
    CandleCoreError(#[from] candle_core::Error),
    #[error(transparent)]
    NvmlError(#[from] nvml_wrapper::error::NvmlError),
    #[error(transparent)]
    IoError(#[from] std::io::Error),
    #[error(transparent)]
    CerebroModelError(#[from] cerebro_model::api::cerebro::model::ModelError),
    #[error(transparent)]
    CerebroClientError(#[from] cerebro_client::error::HttpClientError),
    #[error(transparent)]
    CerebroPipelineError(#[from] cerebro_pipeline::error::WorkflowError),
    #[error(transparent)]
    NifflerError(#[from] niffler::Error),
    #[error(transparent)]
    CsvError(#[from] csv::Error),
    #[error(transparent)]
    RegexError(#[from] regex::Error),
    #[error("plotters crate error: {0}")]
    PlottersError(#[from] Box<dyn std::error::Error + Send + Sync>), 
    #[error("decision tree root not found")]
    TreeRootMissing, 
    #[error("decision tree label not found")]
    TreeNodeLabelMissing, 
    #[error("decision tree question not found")]
    TreeNodeQuestionMissing, 
    #[error("end of sentence token not in vocabulary ({0})")]
    EosTokenNotInVocabulary(String), 
    #[error("sample identifier must be specified when not using prefetch data")]
    SampleIdentifierMissing, 
    #[error("node requires a check type")]
    NodeCheckTypeMissing, 
    #[error("Cerebro client not provided for diagnostic agent")]
    CerebroClientNotProvided, 
}

impl<T> From<plotters::drawing::DrawingAreaErrorKind<T>> for GptError
where
    T: std::error::Error + Send + Sync + 'static,
{
    fn from(err: plotters::drawing::DrawingAreaErrorKind<T>) -> Self {
        GptError::PlottersError(Box::new(err))
    }
}