use image::RgbaImage;

use crate::color::RGBf32;

pub struct Texture {
    image: RgbaImage,
}

impl Texture {
    pub fn new(image: RgbaImage) -> Self {
        Texture { image }
    }

    pub fn sample(&self, u: f32, v: f32) -> RGBf32 {
        let (width, height) = self.image.dimensions();
        // TODO: Bilinear interpolation
        let x = (u * width as f32).round() as u32;
        let y = (v * height as f32).round() as u32;
        let p = self.image.get_pixel(x, y);
        RGBf32::new(p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0)
    }
}