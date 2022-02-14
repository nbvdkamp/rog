use cgmath::Point3;
use crate::color::RGBf32;

#[derive(Clone)]
pub struct Light {
    pub pos: Point3<f32>,
    pub color: RGBf32,
    pub intensity: f32,
    pub range: f32,
    pub kind: Kind
}

#[derive(PartialEq, Clone, Debug)]
pub enum Kind {
    Point,
    Directional,
    Spot,
}