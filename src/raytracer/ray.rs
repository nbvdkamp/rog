use cgmath::{Vector3, Point3, InnerSpace};

pub struct Ray {
    pub origin: Point3<f32>,
    pub direction: Vector3<f32>,
}

#[derive(PartialEq, Debug)]
pub enum IntersectionResult {
    Miss,
    Hit {
        t: f32,
        u: f32,
        v: f32,
    },
}

impl Ray {
    // Möller–Trumbore intersection algorithm:
    pub fn intersect_triangle(&self, p1: Point3<f32>, p2: Point3<f32>, p3: Point3<f32>) -> IntersectionResult {
        const EPSILON: f32 = 1.0e-7;
        let edge1 = p2 - p1;
        let edge2 = p3 - p1;

        let h = self.direction.cross(edge2);
        let a = edge1.dot(h);

        if  a > -EPSILON && a < EPSILON {
            return IntersectionResult::Miss;
        }
        
        let f = 1.0 / a;
        let s = self.origin - p1;
        let u = f * s.dot(h);

        const BIG_EPSILON: f32 = 1.0e-5;

        // Wider range to prevent rays from passing through seams between triangles
        if u < -BIG_EPSILON || u > 1.0 + BIG_EPSILON {
            return IntersectionResult::Miss;
        }

        let q = s.cross(edge1);
        let v = f * self.direction.dot(q);

        if v < -BIG_EPSILON || u + v > 1.0 + BIG_EPSILON {
            return IntersectionResult::Miss;
        }

        let t = f * edge2.dot(q);

        if t > EPSILON {
            IntersectionResult::Hit{ t, u, v }
        }
        else {
            IntersectionResult::Miss
        }
    }

    pub fn traverse(&self, t: f32) -> Point3<f32> {
        self.origin + t * self.direction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_triangle() {
        let ray = Ray { origin: Point3::new(0., 0., -1.), direction: Vector3::unit_z() };
        assert_eq!(IntersectionResult::Hit{ t: 1., u: 0.25, v: 0.5 }, ray.intersect_triangle(
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0)
        ));
    }

    #[test]
    fn barely_hit_triangle() {
        let ray = Ray { origin: Point3::new(0.0, 0.0, -1.0), direction: Vector3::unit_z() };
        assert_eq!(IntersectionResult::Hit{ t: 1., u: 0., v: 0.5 }, ray.intersect_triangle(
            Point3::new(0.0, -1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0)
        ));
    }

    

    #[test]
    fn miss_triangle() {
        let ray = Ray { origin: Point3::new(-0.01, 0.0, -1.0), direction: Vector3::unit_z() };
        assert_eq!(IntersectionResult::Miss, ray.intersect_triangle(
            Point3::new(0.0, -1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0)
        ));
    }

    #[test]
    fn miss_triangle_behind() {
        let ray = Ray { origin: Point3::new(0.0, 0.0, 1.0), direction: Vector3::unit_z() };
        assert_eq!(IntersectionResult::Miss, ray.intersect_triangle(
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0)
        ));
    }

    #[test]
    fn barely_miss_triangle_right() {
        let ray = Ray { origin: Point3::new(0.001, 0.0, -1.0), direction: Vector3::unit_z() };
        assert_eq!(IntersectionResult::Miss, ray.intersect_triangle(
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(0.0, -1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0)
        ));
    }
}