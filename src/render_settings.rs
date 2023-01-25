use std::{path::PathBuf, time::Duration};

use cgmath::Vector2;
use serde::{Deserialize, Serialize};

use crate::{raytracer::acceleration::Accel, scene_version::SceneVersion};

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct VisibilitySettings {
    pub dump_debug_data: bool,
    pub spectral_importance_sampling: bool,
    pub nee_rejection: bool,
    pub nee_direct: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ImageSettings {
    pub size: Vector2<usize>,
    pub enable_dispersion: bool,
    pub max_depth: Option<usize>,
    pub always_sample_single_wavelength: bool,
    pub visibility: Option<VisibilitySettings>,
    pub scene_version: Option<SceneVersion>,
}

impl ImageSettings {
    pub fn use_visibility(&self) -> bool {
        self.visibility.is_some()
    }

    pub fn dump_visibility_data(&self) -> bool {
        if let Some(VisibilitySettings { dump_debug_data, .. }) = self.visibility {
            dump_debug_data
        } else {
            false
        }
    }

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
