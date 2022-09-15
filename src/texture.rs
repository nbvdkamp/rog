use cgmath::{vec2, Vector2};
use lerp::Lerp;
use rayon::prelude::*;
use rgb2spec::RGB2Spec;

use crate::color::{RGBAf32, RGBf32};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Format {
    Rgb,
    Rgba,
}

impl Format {
    fn bytes(&self) -> usize {
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
pub struct Texture {
    pub image: Vec<u8>,
    width: usize,
    height: usize,
    pub format: Format,
}

pub struct CoefficientTexture {
    image: Vec<f32>,
    width: usize,
    height: usize,
    pub has_alpha: bool,
}

impl Texture {
    pub fn new(pixels: Vec<u8>, width: usize, height: usize, format: Format) -> Self {
        assert_eq!(pixels.len(), (width * height * format.bytes()) as usize);

        Texture {
            image: pixels,
            width,
            height,
            format,
        }
    }

    pub fn sample(&self, u: f32, v: f32) -> RGBAf32 {
        let get_pixel = |x: usize, y: usize| {
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
            Format::Rgb => {
                let pixels = self
                    .image
                    .par_chunks(3)
                    .map(|slice: &[u8]| {
                        let coeffs = rgb2spec.fetch(
                            (RGBf32::new(slice[0] as f32, slice[1] as f32, slice[2] as f32) / 255.0)
                                .srgb_to_linear()
                                .into(),
                        );
                        CoefficientPixel { coeffs }
                    })
                    .collect::<Vec<CoefficientPixel>>();

                coefficient_pixel_buffer_to_f32_buffer(pixels)
            }
            Format::Rgba => {
                let pixels = self
                    .image
                    .par_chunks(4)
                    .map(|slice: &[u8]| {
                        let coeffs = rgb2spec.fetch(
                            (RGBf32::new(slice[0] as f32, slice[1] as f32, slice[2] as f32) / 255.0)
                                .srgb_to_linear()
                                .into(),
                        );
                        CoefficientPixelAlpha {
                            coeffs,
                            alpha: slice[3] as f32 / 255.0,
                        }
                    })
                    .collect::<Vec<CoefficientPixelAlpha>>();

                coefficient_pixel_alpha_buffer_to_f32_buffer(pixels)
            }
        };

        CoefficientTexture {
            image: pixels,
            width: self.width,
            height: self.height,
            has_alpha: self.format.has_alpha(),
        }
    }

    pub fn size(&self) -> Vector2<usize> {
        vec2(self.width, self.height)
    }
}

impl CoefficientTexture {
    /// Note that the data returned is rgb2spec coefficients not actually in any RGB color space
    pub fn sample(&self, u: f32, v: f32) -> RGBAf32 {
        let get_pixel = |x: usize, y: usize| {
            let pixel_index = y * self.width + x;
            let floats_per_pixel = if self.has_alpha { 4 } else { 3 };

            let i = pixel_index * floats_per_pixel;
            let alpha = if self.has_alpha { self.image[i + 3] } else { 1.0 };
            RGBAf32::new(self.image[i], self.image[i + 1], self.image[i + 2], alpha)
        };

        wrapping_linear_interp(u, v, self.width, self.height, get_pixel)
    }

    pub fn sample_alpha(&self, u: f32, v: f32) -> f32 {
        if !self.has_alpha {
            return 1.0;
        }

        let get_pixel = |x: usize, y: usize| {
            let pixel_index = y * self.width + x;
            let floats_per_pixel = if self.has_alpha { 4 } else { 3 };

            let i = pixel_index * floats_per_pixel;
            self.image[i + 3]
        };

        wrapping_linear_interp(u, v, self.width, self.height, get_pixel)
    }
}

fn wrapping_linear_interp<R>(u: f32, v: f32, width: usize, height: usize, get_pixel: impl Fn(usize, usize) -> R) -> R
where
    R: Lerp<f32>,
{
    let x = u * width as f32;
    let y = v * height as f32;

    // Can't use % and u32 here because texture coordinates can be negative
    let x0 = (x.floor() as i64).rem_euclid(width as i64) as usize;
    let y0 = (y.floor() as i64).rem_euclid(height as i64) as usize;
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

#[derive(Clone)]
#[repr(C, align(4))]
struct CoefficientPixel {
    coeffs: [f32; 3],
}

fn coefficient_pixel_buffer_to_f32_buffer(buffer: Vec<CoefficientPixel>) -> Vec<f32> {
    let (ptr, length, capacity, alloc) = buffer.into_raw_parts_with_alloc();
    unsafe { Vec::from_raw_parts_in(ptr as *mut f32, 3 * length, 3 * capacity, alloc) }
}

#[derive(Clone)]
#[repr(C, align(4))]
struct CoefficientPixelAlpha {
    coeffs: [f32; 3],
    alpha: f32,
}

fn coefficient_pixel_alpha_buffer_to_f32_buffer(buffer: Vec<CoefficientPixelAlpha>) -> Vec<f32> {
    let (ptr, length, capacity, alloc) = buffer.into_raw_parts_with_alloc();
    unsafe { Vec::from_raw_parts_in(ptr as *mut f32, 4 * length, 4 * capacity, alloc) }
}
