use super::color::RGBf32;
use cgmath::{vec3, Vector2, Vector3};

use crate::{constants::GAMMA, spectrum::Spectrumf32, texture::Texture};

#[derive(Clone)]
pub struct Material {
    pub base_color: RGBf32,
    pub base_color_coefficients: [f32; 3],
    pub base_color_texture: Option<usize>,
    pub metallic: f32,
    pub roughness: f32,
    pub metallic_roughness_texture: Option<usize>,
    pub ior: f32,
    pub transmission: f32,
    pub emissive: RGBf32,
    pub emissive_texture: Option<usize>,
    pub normal_texture: Option<usize>,
}

pub struct MaterialSample {
    pub base_color: RGBf32,
    pub base_color_spectrum: Spectrumf32,
    pub metallic: f32,
    pub roughness: f32,
    pub ior: f32,
    pub transmission: f32,
    pub emissive: RGBf32,
    pub specular: f32,
    pub shading_normal: Option<Vector3<f32>>,
}

impl Material {
    pub fn sample(&self, texture_coordinates: Vector2<f32>, textures: &[Texture]) -> MaterialSample {
        let sample = |tex: Option<usize>| match tex {
            Some(index) => textures[index].sample(texture_coordinates.x, texture_coordinates.y),
            None => RGBf32::white(),
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

        let base_color_spectrum = Spectrumf32::from_coefficients(self.base_color_coefficients);

        let base_color_spectrum = match self.base_color_texture {
            Some(index) => {
                let coeffs_sample = textures[index]
                    .sample_coefficients(texture_coordinates.x, texture_coordinates.y)
                    .unwrap();
                base_color_spectrum * Spectrumf32::from_coefficients(coeffs_sample)
            }
            None => base_color_spectrum,
        };

        MaterialSample {
            base_color: self.base_color * sample(self.base_color_texture).pow(GAMMA),
            base_color_spectrum,
            metallic: self.metallic * metallic_roughness.b,
            roughness: (self.roughness * metallic_roughness.g).max(0.001),
            ior: self.ior,
            transmission: self.transmission,
            emissive: self.emissive * sample(self.emissive_texture).pow(GAMMA),
            specular: 0.5,
            shading_normal,
        }
    }
}
