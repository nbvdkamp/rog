use crate::{color::RGBf32, spectrum::Spectrumf32};
use cgmath::{InnerSpace, Point3, Vector3};

#[derive(Clone)]
pub struct Light {
    pub pos: Point3<f32>,
    pub color: RGBf32,
    pub spectrum: Spectrumf32,
    pub intensity: f32,
    pub range: f32,
    pub kind: Kind,
}

#[derive(PartialEq, Clone, Debug)]
pub enum Kind {
    Point,
    Directional {
        direction: Vector3<f32>,
    },
    Spot {
        inner_cone_angle: f32,
        outer_cone_angle: f32,
    },
}

pub struct LightSample {
    pub direction: Vector3<f32>,
    pub distance: f32,
    pub intensity: f32,
    pub pdf: f32,
}

impl Light {
    pub fn sample(&self, p: Point3<f32>) -> LightSample {
        match &self.kind {
            Kind::Point => {
                let v = self.pos - p;
                let distance = v.magnitude();
                let direction = v / distance;
                let falloff = distance * distance;
                let intensity = self.intensity / (falloff * 4.0 * std::f32::consts::PI);

                LightSample {
                    direction,
                    distance,
                    intensity,
                    pdf: 1.0,
                }
            }
            Kind::Directional { direction } => LightSample {
                direction: *direction,
                distance: f32::INFINITY,
                intensity: self.intensity * std::f32::consts::PI,
                pdf: 1.0,
            },
            Kind::Spot { .. } => {
                // TODO: Implement spot lights instead of just another point light
                let v = self.pos - p;
                let distance = v.magnitude();
                let direction = v / distance;
                let falloff = distance * distance;
                let intensity = self.intensity / (falloff * 4.0 * std::f32::consts::PI);

                LightSample {
                    direction,
                    distance,
                    intensity,
                    pdf: 1.0,
                }
            }
        }
    }
}
