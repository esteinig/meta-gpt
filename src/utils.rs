use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use cerebro_model::api::cerebro::schema::{MetaGpConfig, PostFilterConfig, PrevalenceOutliers};
use cerebro_pipeline::taxa::filter::PrevalenceContaminationConfig;
use csv::{Reader, ReaderBuilder, Writer, WriterBuilder};
use env_logger::Builder;
use env_logger::fmt::Color;
use log::{LevelFilter, Level};
use niffler::{get_reader, get_writer};
use serde::{Deserialize, Serialize};

use crate::error::GptError;
use crate::gpt::PrefetchData;

pub trait CompressionExt {
    fn from_path<S: AsRef<OsStr> + ?Sized>(p: &S) -> Self;
}

/// Attempts to infer the compression type from the file extension.
/// If the extension is not known, then Uncompressed is returned.
impl CompressionExt for niffler::compression::Format {
    fn from_path<S: AsRef<OsStr> + ?Sized>(p: &S) -> Self {
        let path = Path::new(p);
        match path.extension().map(|s| s.to_str()) {
            Some(Some("gz")) => Self::Gzip,
            Some(Some("bz") | Some("bz2")) => Self::Bzip,
            Some(Some("lzma")) => Self::Lzma,
            _ => Self::No,
        }
    }
}

pub trait StringUtils {
    fn substring(&self, start: usize, len: usize) -> Self;
}

impl StringUtils for String {
    fn substring(&self, start: usize, len: usize) -> Self {
        self.chars().skip(start).take(len).collect()
    }
}


pub trait UuidUtils {
    fn shorten(&self, len: usize) -> String;
}

impl UuidUtils for uuid::Uuid {
    fn shorten(&self, len: usize) -> String {
        self.to_string().substring(0, len)
    }
}

pub fn init_logger() {

    Builder::new()
        .format(|buf, record| {
            let timestamp = buf.timestamp();

            let mut red_style = buf.style();
            red_style.set_color(Color::Red).set_bold(true);
            let mut green_style = buf.style();
            green_style.set_color(Color::Green).set_bold(true);
            let mut white_style = buf.style();
            white_style.set_color(Color::White).set_bold(false);
            let mut orange_style = buf.style();
            orange_style.set_color(Color::Rgb(255, 102, 0)).set_bold(true);
            let mut apricot_style = buf.style();
            apricot_style.set_color(Color::Rgb(255, 195, 0)).set_bold(true);

            let msg = match record.level(){
                Level::Warn => (orange_style.value(record.level()), orange_style.value(record.args())),
                Level::Info => (green_style.value(record.level()), white_style.value(record.args())),
                Level::Debug => (apricot_style.value(record.level()), apricot_style.value(record.args())),
                Level::Error => (red_style.value(record.level()), red_style.value(record.args())),
                _ => (white_style.value(record.level()), white_style.value(record.args()))
            };

            writeln!(
                buf,
                "{} [{}] - {}",
                white_style.value(timestamp),
                msg.0,
                msg.1
            )
        })
        .filter(None, LevelFilter::Info)
        .init();
}


pub fn get_tsv_reader(file: &Path, flexible: bool, header: bool) -> Result<Reader<Box<dyn Read>>, GptError> {

    let buf_reader = BufReader::new(File::open(&file)?);
    let (reader, _format) = get_reader(Box::new(buf_reader))?;

    let tsv_reader = ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(header)
        .flexible(flexible) // Allows records with a different number of fields
        .from_reader(reader);

    Ok(tsv_reader)
}

pub fn get_tsv_writer(file: &Path, header: bool) -> Result<Writer<Box<dyn Write>>, GptError> {
    
    let buf_writer = BufWriter::new(File::create(&file)?);
    let writer = get_writer(Box::new(buf_writer), niffler::Format::from_path(file), niffler::compression::Level::Six)?;

    let csv_writer = WriterBuilder::new()
        .delimiter(b'\t')
        .has_headers(header)
        .from_writer(writer);

    Ok(csv_writer)
}

pub fn write_tsv<T: Serialize>(data: &Vec<T>, file: &Path, header: bool) -> Result<(), GptError> {

    let mut writer = get_tsv_writer(file, header)?;

    for value in data {
        // Serialize each value in the vector into the writer
        writer.serialize(&value)?;
    }

    // Flush and complete writing
    writer.flush()?;
    Ok(())
}

pub fn read_tsv<T: for<'de>Deserialize<'de>>(file: &Path, flexible: bool, header: bool) -> Result<Vec<T>, GptError> {

    let mut reader = get_tsv_reader(file, flexible, header)?;

    let mut records = Vec::new();
    for record in reader.deserialize() {
        records.push(record?)
    }

    Ok(records)
}


pub fn get_config(
    json: &Option<PathBuf>,
    sample: Option<String>,
    controls: &Option<Vec<String>>,
    tags: &Option<Vec<String>>,
    ignore_taxstr: &Option<Vec<String>>,
    prevalence_outliers: Option<bool>,
    prefetch: Option<PathBuf>,
) -> Result<(MetaGpConfig, Option<PrefetchData>), GptError> {

    // Parse generative practitioner configuration
    match &json {
        Some(path) => {
            // Config from JSON file:
            Ok((
                MetaGpConfig::with_json(
                    sample.ok_or(GptError::SampleIdentifierMissing)?, 
                    path,
                    ignore_taxstr.clone(),
                    match prevalence_outliers { 
                        Some(true) => Some(PrevalenceOutliers::default()),
                        Some(false) => Some(PrevalenceOutliers::disabled()),
                        None => None
                    },
                )?,
                None
            ))
        },
        None => {
            match &prefetch {
                Some(path) => {
                    let data = PrefetchData::from_json(path)?;
                    Ok((data.config.clone(), Some(data)))
                },
                None => {
                    Ok((
                        MetaGpConfig::new(
                            sample.ok_or(GptError::SampleIdentifierMissing)?, 
                            controls.clone(), 
                            tags.clone(),
                            ignore_taxstr.clone(),
                            PrevalenceContaminationConfig::gp_default(),
                            match prevalence_outliers { 
                                Some(true) => Some(PrevalenceOutliers::default()),
                                Some(false) => Some(PrevalenceOutliers::disabled()),
                                None => None
                            }
                        ),
                        None
                    ))
                }
            }
        }
    }
}