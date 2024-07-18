use std::{path::PathBuf, str::FromStr, time::Duration};

use cgmath::Vector2;
use derivative::Derivative;
use serde::{Deserialize, Serialize};

use crate::{raytracer::acceleration::Accel, scene_version::SceneVersion};

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum ImportanceSamplingMode {
    Visibility,
    VisibilityNEE,
    MeanEmitterSpectrum,
    MeanEmitterSpectrumAlbedo,
}

impl FromStr for ImportanceSamplingMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "visibility" => Ok(ImportanceSamplingMode::Visibility),
            "visibility-nee" => Ok(ImportanceSamplingMode::VisibilityNEE),
            "mean-spectrum" => Ok(ImportanceSamplingMode::MeanEmitterSpectrum),
            "mean-spectrum-albedo" => Ok(ImportanceSamplingMode::MeanEmitterSpectrumAlbedo),
            _ => Err(format!("Invalid importance sampling mode: {s}")),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Derivative)]
#[derivative(Debug)]
pub struct ImageSettings {
    pub size: Vector2<usize>,
    pub enable_dispersion: bool,
    pub max_depth: Option<usize>,
    pub always_sample_single_wavelength: bool,
    pub scene_version: Option<SceneVersion>,
}

impl ImageSettings {
    pub fn max_depth_reached(&self, depth: usize) -> bool {
        match self.max_depth {
            Some(max) => depth >= max,
            None => false,
        }
    }
}

#[derive(Clone)]
pub enum TerminationCondition {
    SampleCount(usize),
    Time(Duration),
}

#[derive(Clone)]
pub struct RenderSettings {
    pub termination_condition: TerminationCondition,
    pub thread_count: usize,
    pub accel_structure: Accel,
    pub intermediate_read_path: Option<PathBuf>,
    pub intermediate_write_path: Option<PathBuf>,
}
