use cgmath::{Vector3, Vector4};
use super::color::RGBf32;

#[derive(Clone)]
pub struct Material {
    pub base_color_factor: RGBf32,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_factor: Vector3<f32>,
}