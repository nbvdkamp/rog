use crate::{color::RGBf32, spectrum::Spectrumf32};
use cgmath::Vector3;

#[derive(Clone)]
pub struct Environment {
    pub color: RGBf32,
    pub spectrum: Spectrumf32,
}

impl Environment {
    pub fn sample(&self, _dir: Vector3<f32>) -> Spectrumf32 {
        self.spectrum
    }
}
