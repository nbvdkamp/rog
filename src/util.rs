use std::path::Path;

use cgmath::{Matrix4, Point3, Vector2, Vector3};
use luminance_front::shader::types::Mat44;

use rayon::prelude::*;

use crate::spectrum::Spectrumf32;

use super::color::{RGBf32, RGBu8};

pub trait ElementWiseMinMax {
    fn elementwise_min(self, other: Self) -> Self;
    fn elementwise_max(self, other: Self) -> Self;
}

impl ElementWiseMinMax for Vector3<f32> {
    fn elementwise_min(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    fn elementwise_max(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }
}

impl ElementWiseMinMax for Point3<f32> {
    fn elementwise_min(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    fn elementwise_max(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }
}

pub fn min_element(v: Vector3<f32>) -> f32 {
    f32::min(v.x, f32::min(v.y, v.z))
}

pub fn max_element(v: Vector3<f32>) -> f32 {
    f32::max(v.x, f32::max(v.y, v.z))
}

pub fn mat_to_shader_type<T>(m: Matrix4<T>) -> Mat44<T> {
    let x: [[T; 4]; 4] = m.into();
    x.into()
}

pub fn save_image<P>(buffer: &[RGBf32], image_size: Vector2<usize>, path: P)
where
    P: AsRef<Path>,
{
    let pixels: Vec<RGBu8> = buffer.iter().map(|c| c.normalized()).collect();

    let byte_buffer: &[u8] = unsafe {
        std::slice::from_raw_parts(
            pixels.as_ptr() as *const u8,
            pixels.len() * std::mem::size_of::<RGBu8>(),
        )
    };

    let save_result = image::save_buffer(
        path,
        byte_buffer,
        image_size.x as u32,
        image_size.y as u32,
        image::ColorType::Rgb8,
    );

    match save_result {
        Ok(_) => println!("File was saved succesfully"),
        Err(e) => println!("Couldn't save file: {}", e),
    }
}

pub fn convert_spectrum_buffer_to_rgb(buffer: Vec<Spectrumf32>) -> Vec<RGBf32> {
    buffer
        .into_par_iter()
        // .map(|spectrum| spectrum.to_srgb().linear_to_srgb())
        .map(|spectrum| {
            if spectrum.data[0] == 69.0 {
                return RGBf32::new(0.0, 1.0, 1.0);
            }

            if spectrum.data.iter().any(|v| v.is_nan()) {
                return RGBf32::new(1.0, 0.5, 0.0);
            }

            let c = spectrum.to_srgb().linear_to_srgb();
            if c.has_nan_component() {
                RGBf32::new(0.0, 1.0, 0.0)
            } else {
                c
            }
        })
        .collect()
}

pub mod bit_hacks {
    pub fn i32_as_f32(x: i32) -> f32 {
        unsafe { std::mem::transmute(x) }
    }

    pub fn f32_as_i32(x: f32) -> i32 {
        unsafe { std::mem::transmute(x) }
    }
}
