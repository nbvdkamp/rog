use std::path::Path;

use cgmath::{Matrix, Matrix3, Matrix4, Point3, SquareMatrix, Vector2, Vector3};

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

pub fn normal_transform_from_mat4(transform: Matrix4<f32>) -> Matrix3<f32> {
    let m = Matrix3::from_cols(transform.x.truncate(), transform.y.truncate(), transform.z.truncate());
    m.invert().unwrap().transpose()
}

pub fn save_image<P>(buffer: &[RGBf32], image_size: Vector2<usize>, path: P)
where
    P: AsRef<Path>,
{
    let pixels: Vec<RGBu8> = buffer.iter().map(|c| c.normalized()).collect();
    save_u8_image(&pixels, image_size, path, true)
}

pub fn save_u8_image<P>(pixels: &[RGBu8], image_size: Vector2<usize>, path: P, success_message: bool)
where
    P: AsRef<Path>,
{
    let byte_buffer: &[u8] = unsafe {
        std::slice::from_raw_parts(
            pixels.as_ptr() as *const u8,
            pixels.len() * std::mem::size_of::<RGBu8>(),
        )
    };

    let save_result = image::save_buffer(
        &path,
        byte_buffer,
        image_size.x as u32,
        image_size.y as u32,
        image::ColorType::Rgb8,
    );

    match save_result {
        Ok(_) => {
            if success_message {
                println!("File was saved succesfully: {}", path.as_ref().display())
            }
        }
        Err(e) => eprintln!("Couldn't save file: {e}"),
    }
}

pub mod bit_hacks {
    pub fn i32_as_f32(x: i32) -> f32 {
        f32::from_bits(x as u32)
    }

    pub fn f32_as_i32(x: f32) -> i32 {
        x.to_bits() as i32
    }
}

pub fn align_to(x: usize, alignment: usize) -> usize {
    assert!(alignment.count_ones() == 1);

    let a = alignment - 1;
    (x + a) & !a
}
