use cgmath::{vec2, Vector2};
use lerp::Lerp;
use rayon::prelude::*;
use rgb2spec::RGB2Spec;

use crate::color::{RGBAf32, RGBf32};

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

    fn has_alpha(&self) -> bool {
        match self {
            Format::Rgb => false,
            Format::Rgba => true,
        }
    }
}

#[derive(Clone)]
struct CoefficientPixel {
    coeffs: [f32; 3],
    alpha: f32,
}

#[derive(Clone)]
pub struct Texture {
    pub image: Vec<u8>,
    width: u32,
    height: u32,
    pub format: Format,
}

pub struct CoefficientTexture {
    image: Vec<CoefficientPixel>,
    width: u32,
    height: u32,
    pub has_alpha: bool,
}

impl Texture {
    pub fn new(pixels: Vec<u8>, width: u32, height: u32, format: Format) -> Self {
        assert_eq!(pixels.len(), (width * height * format.bytes()) as usize);

        Texture {
            image: pixels,
            width,
            height,
            format,
        }
    }

    pub fn sample(&self, u: f32, v: f32) -> RGBAf32 {
        let get_pixel = |x: u32, y: u32| {
            let pixel_index = y * self.width + x;
            let i = (pixel_index * self.format.bytes()) as usize;

            let alpha = if self.format == Format::Rgba {
                self.image[i + 3] as f32 / 255.0
            } else {
                1.0
            };

            RGBAf32::new(
                self.image[i] as f32 / 255.0,
                self.image[i + 1] as f32 / 255.0,
                self.image[i + 2] as f32 / 255.0,
                alpha,
            )
        };

        wrapping_linear_interp(u, v, self.width, self.height, get_pixel)
    }

    pub fn create_spectrum_coefficients(&self, rgb2spec: &RGB2Spec) -> CoefficientTexture {
        let pixels = match self.format {
            Format::Rgb => self
                .image
                .par_chunks(3)
                .map(|slice| CoefficientPixel {
                    coeffs: rgb2spec.fetch(
                        (RGBf32::new(slice[0] as f32, slice[1] as f32, slice[2] as f32) / 255.0)
                            .srgb_to_linear()
                            .into(),
                    ),
                    alpha: 1.0,
                })
                .collect::<Vec<CoefficientPixel>>(),
            Format::Rgba => self
                .image
                .par_chunks(4)
                .map(|slice| CoefficientPixel {
                    coeffs: rgb2spec.fetch(
                        (RGBf32::new(slice[0] as f32, slice[1] as f32, slice[2] as f32) / 255.0)
                            .srgb_to_linear()
                            .into(),
                    ),
                    alpha: slice[3] as f32 / 255.0,
                })
                .collect::<Vec<CoefficientPixel>>(),
        };

        CoefficientTexture {
            image: pixels,
            width: self.width,
            height: self.height,
            has_alpha: self.format.has_alpha(),
        }
    }

    pub fn size(&self) -> Vector2<u32> {
        vec2(self.width, self.height)
    }
}

impl CoefficientTexture {
    /// Note that the data returned is rgb2spec coefficients not actually in any RGB color space
    pub fn sample(&self, u: f32, v: f32) -> RGBAf32 {
        let get_pixel = |x: u32, y: u32| {
            let pixel = &self.image[(y * self.width + x) as usize];
            let c = pixel.coeffs;
            RGBAf32::new(c[0], c[1], c[2], pixel.alpha)
        };

        wrapping_linear_interp(u, v, self.width, self.height, get_pixel)
    }

    pub fn sample_alpha(&self, u: f32, v: f32) -> f32 {
        if !self.has_alpha {
            return 1.0;
        }

        let get_pixel = |x: u32, y: u32| {
            let pixel_index = y * self.width + x;
            // let i = (pixel_index * if self.has_alpha { 3 } else { 4 }) as usize;
            // self.image[i + 3]

            self.image[pixel_index as usize].alpha
        };

        wrapping_linear_interp(u, v, self.width, self.height, get_pixel)
    }
}

fn wrapping_linear_interp<R>(u: f32, v: f32, width: u32, height: u32, get_pixel: impl Fn(u32, u32) -> R) -> R
where
    R: Lerp<f32>,
{
    let x = u * width as f32;
    let y = v * height as f32;

    // Can't use % and u32 here because texture coordinates can be negative
    let x0 = (x.floor() as i32).rem_euclid(width as i32) as u32;
    let y0 = (y.floor() as i32).rem_euclid(height as i32) as u32;
    let x1 = (x0 + 1) % width;
    let y1 = (y0 + 1) % height;

    // Bilinear interpolation
    let xfract = x - x.floor();
    let yfract = y - y.floor();

    let p00 = get_pixel(x0, y0);
    let p01 = get_pixel(x0, y1);
    let p10 = get_pixel(x1, y0);
    let p11 = get_pixel(x1, y1);
    let p0 = p00.lerp(p01, yfract);
    let p1 = p10.lerp(p11, yfract);
    p0.lerp(p1, xfract)
}
