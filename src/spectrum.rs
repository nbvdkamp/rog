#![allow(clippy::needless_range_loop)]
use std::ops::{self, AddAssign, DivAssign, MulAssign};

use serde::{Deserialize, Serialize};

use crate::{
    cie_data as CIE,
    color::{RGBf32, XYZf32},
};

pub struct Spectrumf32WithAlpha {
    pub spectrum: Spectrumf32,
    pub alpha: f32,
}

impl_op_ex!(+ |a: &Spectrumf32WithAlpha, b: &Spectrumf32WithAlpha| -> Spectrumf32WithAlpha {
    Spectrumf32WithAlpha {
        spectrum: a.spectrum + b.spectrum,
        alpha: a.alpha + b.alpha
    }
});

impl_op_ex!(*|a: Spectrumf32WithAlpha, s: f32| -> Spectrumf32WithAlpha {
    Spectrumf32WithAlpha {
        spectrum: a.spectrum * s,
        alpha: a.alpha * s,
    }
});

pub type Spectrumf32 = ArrSpectrumf32;

const RESOLUTION: usize = 60;
const STEP_SIZE: f32 = CIE::LAMBDA_RANGE / (RESOLUTION - 1) as f32;

const fn make_lambda() -> [f32; RESOLUTION] {
    let mut l = [0.0; RESOLUTION];
    let mut i = 0;

    while i < RESOLUTION {
        l[i] = CIE::LAMBDA_MIN + i as f32 * STEP_SIZE;
        i += 1;
    }

    l
}

const LAMBDA: [f32; RESOLUTION] = make_lambda();

impl Spectrumf32 {
    /// Converts to CIE 1931 XYZ color space
    pub fn to_xyz(self) -> XYZf32 {
        let mut xyz = XYZf32::from_grayscale(0.0);

        for i in 0..RESOLUTION {
            let wavelength = CIE::LAMBDA_MIN + i as f32 * STEP_SIZE;
            let whitepoint_sample = CIE::illuminant_d65_interp(wavelength);

            xyz += self.data[i] * CIE::observer_1931_interp(wavelength) * whitepoint_sample * STEP_SIZE;
        }

        xyz
    }

    pub fn to_srgb(self) -> RGBf32 {
        self.to_xyz().to_srgb()
    }

    /// Constructs discretized spectrum from rgb2spec coefficients
    pub fn from_coefficients(coeffs: [f32; 3]) -> Self {
        let mut data = [0.0; RESOLUTION];

        for i in 0..RESOLUTION {
            let wavelength = LAMBDA[i];
            data[i] = rgb2spec::eval_precise(coeffs, wavelength);
        }

        Self { data }
    }

    pub fn max_value(&self) -> f32 {
        self.data.iter().fold(f32::MIN, |old, v| old.max(*v))
    }

    pub fn mean_square_error(&self, other: &Self) -> f32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(s, o)| {
                let e = s - o;
                e * e
            })
            .sum::<f32>()
            / RESOLUTION as f32
    }

    pub fn relative_mean_square_error(&self, reference: &Self) -> f32 {
        let grayscale = reference.data.iter().sum::<f32>() / RESOLUTION as f32;

        self.data
            .iter()
            .zip(reference.data.iter())
            .map(|(s, r)| {
                let e = s - r;
                e * e / (grayscale * grayscale + 1.0e-5)
            })
            .sum::<f32>()
            / RESOLUTION as f32
    }

    pub fn mean(&self) -> f32 {
        self.data.iter().sum::<f32>() / RESOLUTION as f32
    }

    pub fn add_at_wavelength_lerp(&mut self, value: f32, wavelength: f32) {
        let x = (wavelength - CIE::LAMBDA_MIN) / CIE::LAMBDA_RANGE * (RESOLUTION - 2) as f32;
        let i = x.floor() as usize;
        let factor = x - x.floor();
        self.data[i] += (1.0 - factor) * value;
        self.data[i + 1] += factor * value;
    }

    pub fn at_wavelength_lerp(&self, wavelength: f32) -> f32 {
        let x = (wavelength - CIE::LAMBDA_MIN) / CIE::LAMBDA_RANGE * (RESOLUTION - 2) as f32;
        let i = x.floor() as usize;
        let factor = x - x.floor();
        self.data[i] * (1.0 - factor) + self.data[i + 1] * factor
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct ArrSpectrumf32 {
    #[serde(with = "serde_arrays")]
    pub data: [f32; RESOLUTION],
}

impl ArrSpectrumf32 {
    pub const RESOLUTION: usize = RESOLUTION;

    pub fn new(data: [f32; RESOLUTION]) -> Self {
        ArrSpectrumf32 { data }
    }

    pub fn constant(v: f32) -> Self {
        ArrSpectrumf32 { data: [v; RESOLUTION] }
    }

    pub fn sqrt(&self) -> Self {
        let mut data = [0.0; RESOLUTION];

        for i in 0..RESOLUTION {
            data[i] = self.data[i].sqrt();
        }

        Self { data }
    }

    pub fn lerp_with_scalar(&self, s: f32, t: f32) -> Self {
        let mut data = [0.0; RESOLUTION];

        for i in 0..RESOLUTION {
            data[i] = (1.0 - t) * self.data[i] + t * s;
        }

        Self { data }
    }
}

impl_op_ex!(+ |a: &ArrSpectrumf32, b: &ArrSpectrumf32| -> ArrSpectrumf32 {
    let mut data = [0.0; RESOLUTION];

    for i in 0..RESOLUTION {
        data[i] = a.data[i] + b.data[i];
    }

    ArrSpectrumf32 { data }
});

impl_op_ex!(-|a: &ArrSpectrumf32, b: &ArrSpectrumf32| -> ArrSpectrumf32 {
    let mut data = [0.0; RESOLUTION];

    for i in 0..RESOLUTION {
        data[i] = a.data[i] - b.data[i];
    }

    ArrSpectrumf32 { data }
});

impl_op_ex!(*|a: &ArrSpectrumf32, b: &ArrSpectrumf32| -> ArrSpectrumf32 {
    let mut data = [0.0; RESOLUTION];

    for i in 0..RESOLUTION {
        data[i] = a.data[i] * b.data[i];
    }

    ArrSpectrumf32 { data }
});

impl_op_ex!(/ |a: &ArrSpectrumf32, b: &ArrSpectrumf32| -> ArrSpectrumf32 {
    let mut data = [0.0; RESOLUTION];

    for i in 0..RESOLUTION {
        data[i] = a.data[i] / b.data[i];
    }

    ArrSpectrumf32 { data }
});

impl_op_ex_commutative!(*|a: &ArrSpectrumf32, b: f32| -> ArrSpectrumf32 {
    let mut data = [0.0; RESOLUTION];

    for i in 0..RESOLUTION {
        data[i] = a.data[i] * b;
    }

    ArrSpectrumf32 { data }
});

impl_op_ex_commutative!(/ |a: &ArrSpectrumf32, b: f32| -> ArrSpectrumf32 {
    let mut data = [0.0; RESOLUTION];

    for i in 0..RESOLUTION {
        data[i] = a.data[i] / b;
    }

    ArrSpectrumf32 { data }
});

impl AddAssign for ArrSpectrumf32 {
    fn add_assign(&mut self, other: Self) {
        self.data.iter_mut().zip(other.data.iter()).for_each(|(a, b)| *a += b);
    }
}

impl MulAssign<Self> for ArrSpectrumf32 {
    fn mul_assign(&mut self, other: Self) {
        self.data.iter_mut().zip(other.data.iter()).for_each(|(a, b)| *a *= b);
    }
}

impl MulAssign<f32> for ArrSpectrumf32 {
    fn mul_assign(&mut self, s: f32) {
        self.data.iter_mut().for_each(|a| *a *= s);
    }
}

impl DivAssign<f32> for ArrSpectrumf32 {
    fn div_assign(&mut self, s: f32) {
        self.data.iter_mut().for_each(|a| *a /= s);
    }
}

#[cfg(test)]
mod bench {
    extern crate test;
    use test::Bencher;

    use super::*;

    #[bench]
    fn add(b: &mut Bencher) {
        b.iter(|| {
            let p = ArrSpectrumf32::new([1.0; RESOLUTION]);
            let q = ArrSpectrumf32::new([10.0; RESOLUTION]);
            p + q
        });
    }

    #[bench]
    fn add_ref(b: &mut Bencher) {
        b.iter(|| {
            let p = ArrSpectrumf32::new([1.0; RESOLUTION]);
            let q = ArrSpectrumf32::new([10.0; RESOLUTION]);
            &p + &q
        });
    }

    #[bench]
    fn mul_then_div(b: &mut Bencher) {
        let p = ArrSpectrumf32::new([1.0; RESOLUTION]);
        b.iter(|| p * 10.0 / 3.0);
    }

    #[bench]
    fn div_then_mul(b: &mut Bencher) {
        let p = ArrSpectrumf32::new([1.0; RESOLUTION]);
        b.iter(|| p * (10.0 / 3.0));
    }

    #[bench]
    fn lerp_with_constant_spectrum(b: &mut Bencher) {
        let p = ArrSpectrumf32::new([1.0; RESOLUTION]);
        let q = ArrSpectrumf32::constant(7.0);
        use lerp::Lerp;

        b.iter(|| p.lerp(q, 0.5));
    }

    #[bench]
    fn lerp_with_scalar(b: &mut Bencher) {
        let p = ArrSpectrumf32::new([1.0; RESOLUTION]);
        let q = 7.0;

        b.iter(|| p.lerp_with_scalar(q, 0.5));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let a = Spectrumf32::constant(1.0);
        let b = Spectrumf32::constant(2.0);
        let c = a + b;

        assert_eq!(c.data, [3.0; RESOLUTION]);
    }

    #[test]
    fn mul() {
        let a = Spectrumf32::constant(2.0);
        let b = Spectrumf32::constant(3.0);
        let c = a * b;

        assert_eq!(c.data, [6.0; RESOLUTION]);
    }

    #[test]
    fn mul_scalar() {
        let a = Spectrumf32::constant(1.0) * 2.0;

        assert_eq!(a.data, [2.0; RESOLUTION]);
    }

    #[test]
    fn add_assign() {
        let mut a = Spectrumf32::constant(1.0);
        let b = Spectrumf32::constant(2.0);
        a += b;

        assert_eq!(a.data, [3.0; RESOLUTION]);
    }

    #[test]
    fn roundtrip() {
        let rgb2spec = match rgb2spec::RGB2Spec::load("res/out.spec") {
            Ok(rgb2spec) => rgb2spec,
            Err(e) => panic!("Can't load rgb2spec file: {}", e),
        };

        let res = 20;
        let mut max_error = RGBf32::from_grayscale(0.0);

        for i_r in 0..res {
            let r = i_r as f32 / res as f32;

            for i_g in 0..res {
                let g = i_g as f32 / res as f32;

                for i_b in 0..res {
                    let b = i_b as f32 / res as f32;

                    let coeffs = rgb2spec.fetch([r, g, b]);
                    let spectrum = Spectrumf32::from_coefficients(coeffs);
                    let error = RGBf32::new(r, g, b) - spectrum.to_srgb();

                    max_error.r = max_error.r.max(error.r.abs());
                    max_error.g = max_error.g.max(error.g.abs());
                    max_error.b = max_error.b.max(error.b.abs());
                }
            }
        }

        assert!(max_error.max_component() < 0.01);
    }
}
