use std::ops::{self, AddAssign, MulAssign, DivAssign};

use crate::{
    color::{RGBf32, XYZf32},
    cie_data as CIE,
};

type Spectrumf32 = ArrSpectrumf32;

const SPECTRUM_RES: usize = 100;

impl Spectrumf32 {
    pub fn to_rgb(&self) -> RGBf32 {
        let mut xyz = XYZf32::from_grayscale(0.0);

        let step_size = CIE::LAMBDA_RANGE / (SPECTRUM_RES - 1) as f32;

        for i in 0..SPECTRUM_RES {
            let wavelength = CIE::LAMBDA_MIN + i as f32 * step_size;
            let whitepoint_sample = CIE::illuminant_d65_interp(wavelength);

            xyz += self.data[i] * CIE::observer_1931_interp(wavelength) * whitepoint_sample * step_size ;
        }

        xyz_to_rgb(xyz)
    }

    pub fn from_coefficients(coeffs: [f32; 3]) -> Self {
        let mut spectrum = Self::zero();
        let step_size = CIE::LAMBDA_RANGE / (SPECTRUM_RES - 1) as f32;

        for i in 0..SPECTRUM_RES {
            let wavelength = CIE::LAMBDA_MIN + i as f32 * step_size;
            spectrum.data[i] = rgb2spec::eval_precise(coeffs, wavelength);
        }

        spectrum
    }
}

pub fn xyz_to_rgb(XYZf32 { x, y, z }: XYZf32) -> RGBf32 {
    let r =  3.240479 * x - 1.537150 * y - 0.498535 * z;
    let g = -0.969256 * x + 1.875991 * y + 0.041556 * z;
    let b =  0.055648 * x - 0.204043 * y + 1.057311 * z;

    RGBf32::new(r, g, b)
}

struct VecSpectrumf32 {
    data: Vec<f32>
}

impl VecSpectrumf32 {
    pub fn new(data: Vec<f32>) -> Self {
        VecSpectrumf32 { data }
    }

    pub fn constant(v: f32) -> Self {
        //TODO: const size for now
        VecSpectrumf32 { data: vec![v; SPECTRUM_RES] }
    }

    pub fn zero() -> Self {
        Self::constant(0.0)
    }
}

impl_op_ex!(+ |a: &VecSpectrumf32, b: &VecSpectrumf32| -> VecSpectrumf32 {
    let data = a.data.iter().zip(b.data.iter()).map(|(a, b)| a + b);

    VecSpectrumf32 { data: data.collect() }
});

impl_op_ex!(- |a: &VecSpectrumf32, b: &VecSpectrumf32| -> VecSpectrumf32 {
    let data = a.data.iter().zip(b.data.iter()).map(|(a, b)| a - b);

    VecSpectrumf32 { data: data.collect() }
});

impl_op_ex!(* |a: &VecSpectrumf32, b: &VecSpectrumf32| -> VecSpectrumf32 {
    let data = a.data.iter().zip(b.data.iter()).map(|(a, b)| a * b);

    VecSpectrumf32 { data: data.collect() }
});

impl_op_ex!(/ |a: &VecSpectrumf32, b: &VecSpectrumf32| -> VecSpectrumf32 {
    let data = a.data.iter().zip(b.data.iter()).map(|(a, b)| a / b);

    VecSpectrumf32 { data: data.collect() }
});

impl_op_commutative!(* |a: VecSpectrumf32, b: f32| -> VecSpectrumf32 {
    let data = a.data.iter().map(|a| a * b);

    VecSpectrumf32 { data: data.collect() }
});

impl_op_commutative!(/ |a: VecSpectrumf32, b: f32| -> VecSpectrumf32 {
    let data = a.data.iter().map(|a| a / b);

    VecSpectrumf32 { data: data.collect() }
});

impl AddAssign for VecSpectrumf32 {
    fn add_assign(&mut self, other: Self) {
        self.data.iter_mut().zip(other.data.iter()).for_each(|(a, b)| *a += b);
    }
}

impl MulAssign<Self> for VecSpectrumf32 {
    fn mul_assign(&mut self, other: Self) {
        self.data.iter_mut().zip(other.data.iter()).for_each(|(a, b)| *a *= b);
    }
}

impl MulAssign<f32> for VecSpectrumf32 {
    fn mul_assign(&mut self, s: f32) {
        self.data.iter_mut().for_each(|a| *a *= s);
    }
}

impl DivAssign<f32> for VecSpectrumf32 {
    fn div_assign(&mut self, s: f32) {
        self.data.iter_mut().for_each(|a| *a /= s);
    }
}


struct ArrSpectrumf32 {
    data: [f32; SPECTRUM_RES],
}

impl ArrSpectrumf32 {
    pub fn new(data: [f32; SPECTRUM_RES]) -> Self {
        ArrSpectrumf32 { data }
    }

    pub fn constant(v: f32) -> Self {
        ArrSpectrumf32 { data: [v; SPECTRUM_RES] }
    }

    pub fn zero() -> Self {
        Self::constant(0.0)
    }
}

impl_op_ex!(+ |a: &ArrSpectrumf32, b: &ArrSpectrumf32| -> ArrSpectrumf32 {
    let mut data = [0.0; SPECTRUM_RES];

    for i in 0..SPECTRUM_RES {
        data[i] = a.data[i] + b.data[i];
    }

    ArrSpectrumf32 { data }
});

impl_op_ex!(- |a: &ArrSpectrumf32, b: &ArrSpectrumf32| -> ArrSpectrumf32 {
    let mut data = [0.0; SPECTRUM_RES];

    for i in 0..SPECTRUM_RES {
        data[i] = a.data[i] - b.data[i];
    }

    ArrSpectrumf32 { data }
});

impl_op_ex!(* |a: &ArrSpectrumf32, b: &ArrSpectrumf32| -> ArrSpectrumf32 {
    let mut data = [0.0; SPECTRUM_RES];

    for i in 0..SPECTRUM_RES {
        data[i] = a.data[i] * b.data[i];
    }

    ArrSpectrumf32 { data }
});

impl_op_ex!(/ |a: &ArrSpectrumf32, b: &ArrSpectrumf32| -> ArrSpectrumf32 {
    let mut data = [0.0; SPECTRUM_RES];

    for i in 0..SPECTRUM_RES {
        data[i] = a.data[i] / b.data[i];
    }

    ArrSpectrumf32 { data }
});

impl_op_commutative!(* |a: ArrSpectrumf32, b: f32| -> ArrSpectrumf32 {
    let mut data = [0.0; SPECTRUM_RES];

    for i in 0..SPECTRUM_RES {
        data[i] = a.data[i] * b;
    }

    ArrSpectrumf32 { data }
});

impl_op_commutative!(/ |a: ArrSpectrumf32, b: f32| -> ArrSpectrumf32 {
    let mut data = [0.0; SPECTRUM_RES];

    for i in 0..SPECTRUM_RES {
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
mod tests {
    use super::*;

    #[test]
    fn add() {
        let a = VecSpectrumf32::new(vec![1.0, 2.0]);
        let b = VecSpectrumf32::new(vec![10.0, 20.0]);
        let c = a + b;

        assert_eq!(c.data, vec![11.0, 22.0]);
    }

    #[test]
    fn mul() {
        let a = VecSpectrumf32::new(vec![1.0, 2.0]);
        let b = VecSpectrumf32::new(vec![10.0, 20.0]);
        let c = a * b;

        assert_eq!(c.data, vec![10.0, 40.0]);
    }

    #[test]
    fn mul_scalar() {
        let a = VecSpectrumf32::new(vec![1.0, 2.0]) * 2.0;

        assert_eq!(a.data, vec![2.0, 4.0]);
    }

    #[test]
    fn add_assign() {
        let mut a = VecSpectrumf32::new(vec![1.0, 2.0]);
        let b = VecSpectrumf32::new(vec![10.0, 20.0]);
        a += b;

        assert_eq!(a.data, vec![11.0, 22.0]);
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
                    let error = RGBf32::new(r, g, b) - spectrum.to_rgb();

                    max_error.r = max_error.r.max(error.r.abs());
                    max_error.g = max_error.g.max(error.g.abs());
                    max_error.b = max_error.b.max(error.b.abs());
                }
            }
        }

        assert!(max_error.max_component() < 0.01);
    }

    #[test]
    fn asdf() {
        let mut spectrum = Spectrumf32::zero();
        spectrum.data[40] = 1.0;

        let new_rgb = spectrum.to_rgb();
        println!("{new_rgb:?}");
    }
}

#[cfg(test)]
mod bench {
    extern crate test;
    use test::Bencher;

    use super::*;

    #[bench]
    fn add(b: &mut Bencher) {
        let p = VecSpectrumf32::new(vec![1.0; SPECTRUM_RES]);
        let q = VecSpectrumf32::new(vec![10.0; SPECTRUM_RES]);

        b.iter(|| {
            &p + &q
        });
    }

    #[bench]
    fn add2(b: &mut Bencher) {
        let p = ArrSpectrumf32::new([1.0; SPECTRUM_RES]);
        let q = ArrSpectrumf32::new([10.0; SPECTRUM_RES]);

        b.iter(|| {
            &p + &q
        });
    }
}