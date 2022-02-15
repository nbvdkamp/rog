use cgmath::Vector3;
use crate::color::RGBf32;

#[derive(Clone)]
pub struct Environment {
    pub color: RGBf32,
}

impl Environment {
    pub fn sample(&self, _dir: Vector3<f32>) -> RGBf32 {
        self.color
    }
}