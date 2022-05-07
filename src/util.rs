use std::path::Path;

use cgmath::{Vector2, Vector3, Matrix4};
use luminance_front::shader::types::{Mat44};

use super::color::{RGBf32, RGBu8};

pub fn elementwise_min(a: Vector3<f32>, b: Vector3<f32>) -> Vector3<f32> {
    Vector3 {
        x: f32::min(a.x, b.x),
        y: f32::min(a.y, b.y),
        z: f32::min(a.z, b.z),
    }
}

pub fn elementwise_max(a: Vector3<f32>, b: Vector3<f32>) -> Vector3<f32> {
    Vector3 {
        x: f32::max(a.x, b.x),
        y: f32::max(a.y, b.y),
        z: f32::max(a.z, b.z),
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
    P: AsRef<Path>
{
    let pixels: Vec<RGBu8> = buffer.into_iter().map(|c| c.normalized()).collect();

    let byte_buffer: &[u8] = unsafe {
        std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * std::mem::size_of::<RGBu8>())
    };

    let save_result = image::save_buffer(path, &byte_buffer,
        image_size.x as u32, image_size.y as u32, image::ColorType::Rgb8);

    match save_result {
        Ok(_) => println!("File was saved succesfully"),
        Err(e) => println!("Couldn't save file: {}", e),
    }
}