use cgmath::Vector2;
use super::color::RGBf32;

use crate::texture::Texture;

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

pub struct MaterialSample {
    pub base_color: RGBf32,
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: RGBf32,
    pub specular: f32,
}

impl Material {
    pub fn sample(&self, texture_coordinates: Vector2<f32>, textures: &[Texture]) -> MaterialSample {
        let sample = |tex: Option<usize>| match tex {
            Some(index) => textures[index].sample(texture_coordinates.x, texture_coordinates.y),
            None => RGBf32::white()
        };

        // B channel is metallic, G is roughness
        let metallic_roughness = sample(self.metallic_roughness_texture);

        MaterialSample {
            base_color: self.base_color * sample(self.base_color_texture),
            metallic: self.metallic * metallic_roughness.b,
            roughness: (self.roughness * metallic_roughness.g).max(0.001), 
            emissive: self.emissive * sample(self.emissive_texture),
            specular: 0.5,
        }
    }
}