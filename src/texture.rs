use cgmath::{vec2, Vector2};
use lerp::Lerp;
use rgb2spec::RGB2Spec;

use crate::{color::RGBf32, constants::GAMMA};

#[derive(Clone, PartialEq, Eq)]
pub enum Format {
    Rgb,
    Rgba,
}

impl Format {
    fn bytes(&self) -> u32 {
        match self {
            // We're only dealing with 8 bit textures for now
            Format::Rgb => 3,
            Format::Rgba => 4,
        }
    }
}

#[derive(Clone)]
pub struct Texture {
    pub image: Vec<u8>,
    coefficients_image: Option<Vec<f32>>,
    width: u32,
    height: u32,
    pub format: Format,
}

impl Texture {
    pub fn new(pixels: Vec<u8>, width: u32, height: u32, format: Format) -> Self {
        assert_eq!(pixels.len(), (width * height * format.bytes()) as usize);

        Texture {
            image: pixels,
            coefficients_image: None,
            width,
            height,
            format,
        }
    }

    pub fn sample_coefficients(&self, u: f32, v: f32) -> Option<[f32; 3]> {
        self.coefficients_image.as_ref().map(|image| {
            let x = u * self.width as f32;
            let y = v * self.height as f32;

            // Can't use % and u32 here because texture coordinates can be negative
            let x0 = (x.trunc() as i32).rem_euclid(self.width as i32) as u32;
            let y0 = (y.trunc() as i32).rem_euclid(self.height as i32) as u32;
            let x1 = (x0 + 1) % self.width;
            let y1 = (y0 + 1) % self.height;

            let to_rgbf32 = |x: u32, y: u32| {
                let pixel_index = y * self.width + x;
                let i = (pixel_index * self.format.bytes()) as usize;
                RGBf32::new(image[i], image[i + 1], image[i + 2])
            };

            // Bilinear interpolation
            let p00 = to_rgbf32(x0, y0);
            let p01 = to_rgbf32(x0, y1);
            let p10 = to_rgbf32(x1, y0);
            let p11 = to_rgbf32(x1, y1);
            let p0 = p00.lerp(p01, y.fract());
            let p1 = p10.lerp(p11, y.fract());
            p0.lerp(p1, x.fract()).into()
        })
    }

    pub fn sample(&self, u: f32, v: f32) -> RGBf32 {
        let x = u * self.width as f32;
        let y = v * self.height as f32;

        // Can't use % and u32 here because texture coordinates can be negative
        let x0 = (x.trunc() as i32).rem_euclid(self.width as i32) as u32;
        let y0 = (y.trunc() as i32).rem_euclid(self.height as i32) as u32;
        let x1 = (x0 + 1) % self.width;
        let y1 = (y0 + 1) % self.height;

        let to_rgbf32 = |x: u32, y: u32| {
            let pixel_index = y * self.width + x;
            let i = (pixel_index * self.format.bytes()) as usize;
            RGBf32::new(
                self.image[i] as f32 / 255.0,
                self.image[i + 1] as f32 / 255.0,
                self.image[i + 2] as f32 / 255.0,
            )
        };

        // Bilinear interpolation
        let p00 = to_rgbf32(x0, y0);
        let p01 = to_rgbf32(x0, y1);
        let p10 = to_rgbf32(x1, y0);
        let p11 = to_rgbf32(x1, y1);
        let p0 = p00.lerp(p01, y.fract());
        let p1 = p10.lerp(p11, y.fract());
        p0.lerp(p1, x.fract())
    }

    pub fn create_spectrum_coefficients(&mut self, rgb2spec: &RGB2Spec) {
        let size = (self.width * self.height * self.format.bytes()) as usize;
        let mut result = Vec::with_capacity(size);

        for y in 0..self.height {
            for x in 0..self.width {
                let pixel_index = y * self.width + x;
                let i = (pixel_index * self.format.bytes()) as usize;

                let coeffs = rgb2spec.fetch([
                    (self.image[i] as f32 / 255.0).powf(GAMMA),
                    (self.image[i + 1] as f32 / 255.0).powf(GAMMA),
                    (self.image[i + 2] as f32 / 255.0).powf(GAMMA),
                ]);

                result.extend(coeffs);

                if self.format == Format::Rgba {
                    result.push(self.image[i + 3] as f32 / 255.0);
                }
            }
        }

        assert_eq!(result.len(), size);
        self.coefficients_image = Some(result);
    }

    pub fn size(&self) -> Vector2<u32> {
        vec2(self.width, self.height)
    }
}
