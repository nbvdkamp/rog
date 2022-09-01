use cgmath::Vector2;

pub struct RenderSettings {
    pub samples_per_pixel: usize,
    pub image_size: Vector2<usize>,
    pub thread_count: usize,
    pub accel_structure_index: usize,
    pub enable_dispersion: bool,
}
