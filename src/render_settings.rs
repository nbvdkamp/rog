use cgmath::Vector2;

use crate::raytracer::acceleration::Accel;

pub struct RenderSettings {
    pub samples_per_pixel: usize,
    pub image_size: Vector2<usize>,
    pub thread_count: usize,
    pub accel_structure: Accel,
    pub enable_dispersion: bool,
    pub always_sample_single_wavelength: bool,
    pub use_visibility: bool,
    pub dump_visibility_debug_data: bool,
}
