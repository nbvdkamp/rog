use cgmath::{Vector2, Vector3, vec3};
use super::color::RGBf32;

use crate::{
    texture::Texture,
    constants::GAMMA,
};

#[derive(Clone)]
pub struct Material {
    pub base_color: RGBf32,
    pub base_color_coefficients: [f32; 3],
    pub base_color_texture: Option<usize>,
    pub metallic: f32,
    pub roughness: f32,
    pub metallic_roughness_texture: Option<usize>,
    pub emissive: RGBf32,
    pub emissive_texture: Option<usize>,
    pub normal_texture: Option<usize>,
}

pub struct MaterialSample {
    pub base_color: RGBf32,
    pub base_color_coefficients: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: RGBf32,
    pub specular: f32,
    pub shading_normal: Option<Vector3<f32>>,
}

impl Material {
    pub fn sample(&self, texture_coordinates: Vector2<f32>, textures: &[Texture]) -> MaterialSample {
        let sample = |tex: Option<usize>| match tex {
            Some(index) => textures[index].sample(texture_coordinates.x, texture_coordinates.y),
            None => RGBf32::white()
        };

        // B channel is metallic, G is roughness
        let metallic_roughness = sample(self.metallic_roughness_texture);

        let shading_normal = self.normal_texture.map(|index| {
            let sample = textures[index].sample(texture_coordinates.x, texture_coordinates.y);
            let x = 2.0 * sample.r - 1.0;
            let y = 2.0 * sample.g - 1.0;
            let z = 2.0 * sample.b - 1.0;
            vec3(x, y, z)
        });

        MaterialSample {
            base_color: self.base_color * sample(self.base_color_texture).pow(GAMMA),
            //TODO: Figure out how to/whether to combine with texture sample
            base_color_coefficients: self.base_color_coefficients,
            metallic: self.metallic * metallic_roughness.b,
            roughness: (self.roughness * metallic_roughness.g).max(0.001), 
            emissive: self.emissive * sample(self.emissive_texture).pow(GAMMA),
            specular: 0.5,
            shading_normal,
        }
    }
}