use cgmath::Point3;
use crate::color::Color;

#[derive(Clone)]
pub struct Light {
    pub pos: Point3<f32>,
    pub color: Color,
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