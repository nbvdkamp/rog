use lerp::Lerp;
use image::RgbaImage;
use rgb2spec::RGB2Spec;

use crate::color::RGBf32;

#[derive(Clone)]
pub struct Texture {
    image: RgbaImage,
    coefficients_image: Option<RgbaImage>,
}

impl Texture {
    pub fn new(image: RgbaImage, rgb2spec: Option<&RGB2Spec>) -> Self {
        let coefficients_image = rgb2spec.map(|rgb2spec| Texture::to_spectrum_coefficients(&image, rgb2spec));

        Texture {
            image,
            coefficients_image,
        }
    }

    pub fn sample(&self, u: f32, v: f32) -> RGBf32 {
        Texture::sample_image(&self.image, u, v)
    }

    pub fn sample_coefficients(&self, u: f32, v: f32) -> Option<RGBf32> {
       self.coefficients_image.as_ref().map(|i| Texture::sample_image(i, u, v))
    }

    fn sample_image(image: &RgbaImage, u: f32, v: f32) -> RGBf32 {
        let (width, height) = image.dimensions();
        let x = u * width as f32;
        let y = v * height as f32;

        // Can't use % and u32 here because texture coordinates can be negative
        let x0 = (x.trunc() as i32).rem_euclid(width as i32) as u32;
        let y0 = (y.trunc() as i32).rem_euclid(height as i32) as u32;
        let x1 = (x0 + 1) % width;
        let y1 = (y0 + 1) % height;

        let to_rgbf32 = |p: &image::Rgba<u8>| 
            RGBf32::new(p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0);

        // Bilinear interpolation
        let p00 = to_rgbf32(image.get_pixel(x0, y0));
        let p01 = to_rgbf32(image.get_pixel(x0, y1));
        let p10 = to_rgbf32(image.get_pixel(x1, y0));
        let p11 = to_rgbf32(image.get_pixel(x1, y1));
        let p0 = p00.lerp(p01, y.fract());
        let p1 = p10.lerp(p11, y.fract());
        p0.lerp(p1, x.fract())
    }

    pub fn has_coefficients_image(&self) -> bool {
        self.coefficients_image.is_some()
    }

    fn to_spectrum_coefficients(source: &RgbaImage, rgb2spec: &RGB2Spec) -> RgbaImage {
        let (width, height) = source.dimensions();
        let mut image = RgbaImage::new(width, height);

        for i in 0..width {
            for j in 0..height {
                let p = source.get_pixel(i, j);
                let coeffs = rgb2spec.fetch([p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0]);
                let new_p = image.get_pixel_mut(i, j);

                for k in 0..3 {
                    new_p[k] = (coeffs[k] * 255.0) as u8;
                }

                new_p[3] = p[3];
            }
        }

        image
    }
}
