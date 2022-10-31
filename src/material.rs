use super::color::{RGBAf32, RGBf32};
use cgmath::{vec3, InnerSpace, Point2, Vector3};

use crate::{raytracer::Textures, spectrum::Spectrumf32, texture::Texture};

#[derive(Copy, Clone)]
pub struct TextureRef {
    pub index: usize,
}

#[derive(Clone)]
pub struct Material {
    pub base_color: RGBf32,
    pub base_color_coefficients: [f32; 3],
    pub base_color_texture: Option<TextureRef>,
    pub metallic: f32,
    pub roughness: f32,
    pub metallic_roughness_texture: Option<TextureRef>,
    pub ior: f32,
    pub cauchy_coefficients: CauchyCoefficients,
    pub transmission_factor: f32,
    pub transmission_texture: Option<TextureRef>,
    pub emissive: RGBf32,
    pub emissive_texture: Option<TextureRef>,
    pub normal_texture: Option<TextureRef>,
}

pub struct MaterialSample {
    pub alpha: f32,
    pub base_color_spectrum: Spectrumf32,
    pub metallic: f32,
    pub roughness: f32,
    pub medium_ior: f32,
    pub ior: f32,
    pub cauchy_coefficients: CauchyCoefficients,
    pub transmission: f32,
    pub emissive: RGBf32,
    pub specular: f32,
    pub shading_normal: Option<Vector3<f32>>,
    pub sub_surface_scattering: f32,
}

impl Material {
    pub fn sample(&self, texture_coordinates: Point2<f32>, textures: &Textures) -> MaterialSample {
        let sample = |tex: Option<TextureRef>, textures: &Vec<Texture>| match tex {
            Some(tex) => textures[tex.index].sample(texture_coordinates.x, texture_coordinates.y),
            None => RGBAf32::white(),
        };

        // B channel is metallic, G is roughness
        let metallic_roughness = sample(self.metallic_roughness_texture, &textures.metallic_roughness);

        let shading_normal = self.normal_texture.map(|tex| {
            let RGBAf32 { r, g, b, .. } =
                textures.normal[tex.index].sample(texture_coordinates.x, texture_coordinates.y);
            (2.0 * vec3(r, g, b) - vec3(1.0, 1.0, 1.0)).normalize()
        });

        let base_color_spectrum = Spectrumf32::from_coefficients(self.base_color_coefficients);
        let mut alpha = 1.0;

        let base_color_spectrum = match self.base_color_texture {
            Some(tex) => {
                let coeffs_sample =
                    textures.base_color_coefficients[tex.index].sample(texture_coordinates.x, texture_coordinates.y);
                alpha = coeffs_sample.a;
                base_color_spectrum * Spectrumf32::from_coefficients(coeffs_sample.rgb().into())
            }
            None => base_color_spectrum,
        };

        MaterialSample {
            alpha,
            base_color_spectrum,
            metallic: self.metallic * metallic_roughness.b,
            roughness: (self.roughness * metallic_roughness.g).max(0.001),
            medium_ior: 1.0,
            ior: self.ior,
            cauchy_coefficients: self.cauchy_coefficients,
            transmission: self.transmission_factor * sample(self.transmission_texture, &textures.transimission).r,
            emissive: self.emissive * sample(self.emissive_texture, &textures.emissive).rgb().srgb_to_linear(),
            specular: 0.5,
            shading_normal,
            sub_surface_scattering: 0.0,
        }
    }

    pub fn sample_alpha(&self, texture_coordinates: Point2<f32>, textures: &Textures) -> f32 {
        match self.base_color_texture {
            Some(tex) => {
                textures.base_color_coefficients[tex.index].sample_alpha(texture_coordinates.x, texture_coordinates.y)
            }
            None => 1.0,
        }
    }
}

#[derive(Clone, Copy)]
pub struct CauchyCoefficients {
    pub a: f32,
    pub b: f32,
}

impl CauchyCoefficients {
    pub fn approx_from_ior(ior: f32) -> Self {
        // Using wavelengths in micrometers here
        const STANDARD_IOR_WAVELENGTH: f32 = 0.589;

        // Probably not the best approximation but it should work
        let b = (0.005 + 0.03 * (ior - 1.53)).max(0.003);
        let a = ior - b / (STANDARD_IOR_WAVELENGTH * STANDARD_IOR_WAVELENGTH);

        CauchyCoefficients { a, b }
    }

    pub fn ior_for_wavelength(&self, wavelength: f32) -> f32 {
        // To micrometers
        let l = wavelength / 1000.0;

        self.a + self.b / (l * l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let ior = 1.5;
        let c = CauchyCoefficients::approx_from_ior(ior);

        assert_eq!(ior, c.ior_for_wavelength(589.0));
        assert!(ior > c.ior_for_wavelength(700.0));
    }
}
