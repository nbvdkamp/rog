use cgmath::{InnerSpace, Point3, Vector3};

pub struct Ray {
    pub origin: Point3<f32>,
    pub direction: Vector3<f32>,
}

#[derive(PartialEq, Debug)]
pub enum IntersectionResult {
    Miss,
    Hit { t: f32, u: f32, v: f32 },
}

impl Ray {
    // Möller–Trumbore intersection algorithm, without culling backfacing triangles
    pub fn intersect_triangle(&self, p0: Point3<f32>, p1: Point3<f32>, p2: Point3<f32>) -> IntersectionResult {
        let edge1 = p1 - p0;
        let edge2 = p2 - p0;

        let p = self.direction.cross(edge2);
        let determinant = edge1.dot(p);
        let inverse_determinant = 1.0 / determinant;
        let t = self.origin - p0;
        let u = inverse_determinant * t.dot(p);

        if u < 0.0 || u > 1.0 {
            return IntersectionResult::Miss;
        }

        let q = t.cross(edge1);
        let v = inverse_determinant * self.direction.dot(q);

        if v < 0.0 || u + v > 1.0 {
            return IntersectionResult::Miss;
        }

        let t = inverse_determinant * edge2.dot(q);

        if t > 0.0 {
            IntersectionResult::Hit { t, u, v }
        } else {
            IntersectionResult::Miss
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_triangle() {
        let ray = Ray {
            origin: Point3::new(0., 0., -1.),
            direction: Vector3::unit_z(),
        };
        assert_eq!(
            IntersectionResult::Hit { t: 1., u: 0.25, v: 0.5 },
            ray.intersect_triangle(
                Point3::new(-1.0, -1.0, 0.0),
                Point3::new(1.0, -1.0, 0.0),
                Point3::new(0.0, 1.0, 0.0)
            )
        );
    }

    #[test]
    fn barely_hit_triangle() {
        let ray = Ray {
            origin: Point3::new(0.0, 0.0, -1.0),
            direction: Vector3::unit_z(),
        };
        assert_eq!(
            IntersectionResult::Hit { t: 1., u: 0., v: 0.5 },
            ray.intersect_triangle(
                Point3::new(0.0, -1.0, 0.0),
                Point3::new(1.0, -1.0, 0.0),
                Point3::new(0.0, 1.0, 0.0)
            )
        );
    }

    #[test]
    fn miss_triangle() {
        let ray = Ray {
            origin: Point3::new(-0.01, 0.0, -1.0),
            direction: Vector3::unit_z(),
        };
        assert_eq!(
            IntersectionResult::Miss,
            ray.intersect_triangle(
                Point3::new(0.0, -1.0, 0.0),
                Point3::new(1.0, -1.0, 0.0),
                Point3::new(0.0, 1.0, 0.0)
            )
        );
    }

    #[test]
    fn miss_triangle_behind() {
        let ray = Ray {
            origin: Point3::new(0.0, 0.0, 1.0),
            direction: Vector3::unit_z(),
        };
        assert_eq!(
            IntersectionResult::Miss,
            ray.intersect_triangle(
                Point3::new(-1.0, -1.0, 0.0),
                Point3::new(1.0, -1.0, 0.0),
                Point3::new(0.0, 1.0, 0.0)
            )
        );
    }

    #[test]
    fn barely_miss_triangle_right() {
        let ray = Ray {
            origin: Point3::new(0.001, 0.0, -1.0),
            direction: Vector3::unit_z(),
        };
        assert_eq!(
            IntersectionResult::Miss,
            ray.intersect_triangle(
                Point3::new(-1.0, -1.0, 0.0),
                Point3::new(0.0, -1.0, 0.0),
                Point3::new(0.0, 1.0, 0.0)
            )
        );
    }
}
