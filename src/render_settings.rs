use std::path::PathBuf;

use cgmath::{vec2, Vector2};
use serde::{Deserialize, Serialize};

use crate::{raytracer::acceleration::Accel, scene_version::SceneVersion};

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct VisibilitySettings {
    pub dump_debug_data: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ImageSettings {
    pub width: usize,
    pub height: usize,
    pub enable_dispersion: bool,
    pub always_sample_single_wavelength: bool,
    pub visibility: Option<VisibilitySettings>,
    pub scene_version: Option<SceneVersion>,
}

impl ImageSettings {
    pub fn use_visibility(&self) -> bool {
        self.visibility.is_some()
    }

    pub fn dump_visibility_data(&self) -> bool {
        if let Some(VisibilitySettings { dump_debug_data }) = self.visibility {
            dump_debug_data
        } else {
            false
        }
    }

    pub fn size(&self) -> Vector2<usize> {
        vec2(self.width, self.height)
    }
}

pub struct RenderSettings {
    pub samples_per_pixel: usize,
    pub thread_count: usize,
    pub accel_structure: Accel,
    pub intermediate_read_path: Option<PathBuf>,
    pub intermediate_write_path: Option<PathBuf>,
}
