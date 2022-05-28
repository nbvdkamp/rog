use crate::{color::RGBf32, raytracer::geometry::orthogonal_vector, spectrum::Spectrumf32};
use cgmath::{InnerSpace, Point3, Vector3};
use rand::Rng;

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
        radius: f32,
        area: f32,
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
    pub use_mis: bool,
}

impl Light {
    pub fn sample(&self, p: Point3<f32>) -> LightSample {
        match self.kind {
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
                    use_mis: false,
                }
            }
            Kind::Directional {
                direction,
                radius,
                area,
            } => {
                let sample_dir;
                let pdf;
                let use_mis;

                if area > 0.0 {
                    sample_dir = (direction + radius * orthogonal_disk_sample(direction)).normalize();
                    use_mis = true;

                    let cos_theta = sample_dir.dot(direction);
                    pdf = 1.0 / (area * cos_theta * cos_theta * cos_theta);
                } else {
                    sample_dir = direction;
                    use_mis = false;
                    pdf = 1.0;
                }

                LightSample {
                    direction: sample_dir,
                    distance: f32::INFINITY,
                    intensity: pdf * self.intensity * std::f32::consts::PI,
                    pdf,
                    use_mis,
                }
            }
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
                    use_mis: false,
                }
            }
        }
    }
}

fn orthogonal_disk_sample(direction: Vector3<f32>) -> Vector3<f32> {
    let tangent = orthogonal_vector(direction).normalize();
    let bitangent = direction.cross(tangent);

    let mut rng = rand::thread_rng();
    let theta: f32 = 2.0 * std::f32::consts::PI * rng.gen::<f32>();
    let r = rng.gen::<f32>().sqrt();

    r * (theta.cos() * tangent + theta.sin() * bitangent)
}
