use lerp::Lerp;
use image::RgbaImage;

use crate::color::RGBf32;

#[derive(Clone)]
pub struct Texture {
    image: RgbaImage,
}

impl Texture {
    pub fn new(image: RgbaImage) -> Self {
        Texture { image }
    }

    pub fn sample(&self, u: f32, v: f32) -> RGBf32 {
        let (width, height) = self.image.dimensions();
        let x = u * (width - 1) as f32;
        let y = v * (height - 1) as f32;

        let x0 = x.trunc() as u32 % width;
        let y0 = y.trunc() as u32 % height;
        let x1 = (x0 + 1) % width;
        let y1 = (y0 + 1) % height;

        let to_RGBf32 = |p: &image::Rgba<u8>| 
            RGBf32::new(p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0);

        // Bilinear interpolation
        let p00 = to_RGBf32(self.image.get_pixel(x0, y0));
        let p01 = to_RGBf32(self.image.get_pixel(x0, y1));
        let p10 = to_RGBf32(self.image.get_pixel(x1, y0));
        let p11 = to_RGBf32(self.image.get_pixel(x1, y1));
        let p0 = p00.lerp(p01, y.fract());
        let p1 = p10.lerp(p11, y.fract());
        p0.lerp(p1, x.fract())
    }
}