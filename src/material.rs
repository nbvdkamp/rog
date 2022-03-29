use super::color::RGBf32;

#[derive(Clone)]
pub struct Material {
    pub base_color: RGBf32,
    pub base_color_texture: Option<usize>,
    pub metallic: f32,
    pub roughness: f32,
    pub metallic_roughness_texture: Option<usize>,
    pub emissive: RGBf32,
    pub emissive_texture: Option<usize>,
}