use cgmath::{Vector3, Point3, InnerSpace};

struct Ray {
    pub origin: Point3<f32>,
    pub direction: Vector3<f32>,
}

#[derive(PartialEq, Debug)]
enum IntersectionResult {
    Miss,
    Hit { position: Point3<f32> },
}

impl Ray {
    // Möller–Trumbore intersection algorithm:
    pub fn intersect_triangle(&self, p1: Point3<f32>, p2: Point3<f32>, p3: Point3<f32>) -> IntersectionResult {
        let epsilon = 1.0e-6;
        let edge1 = p2 - p1;
        let edge2 = p3 - p1;

        let h = self.direction.cross(edge2);
        let a = edge1.dot(h);

        if  a > -epsilon && a < epsilon {
            return IntersectionResult::Miss;
        }
        
        let f = 1.0 / a;
        let s = self.origin - p1;
        let u = f * s.dot(h);

        if u < 0.0 || u > 1.0 {
            return IntersectionResult::Miss;
        }

        let q = s.cross(edge1);
        let v = f * self.direction.dot(q);

        if v < 0.0 || v > 1.0 {
            return IntersectionResult::Miss;
        }

        let t = f * edge2.dot(q);

        if t > epsilon {
            IntersectionResult::Hit { position: self.origin + self.direction * t }
        }
        else {
            IntersectionResult::Miss
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_triangle() {
        let ray = Ray { origin: Point3::new(0., 0., -1.), direction: Vector3::unit_z() };
        assert_eq!(IntersectionResult::Hit { position: Point3::new(0., 0., 0.) }, ray.intersect_triangle(
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0)
        ));
    }

    #[test]
    fn miss_triangle() {
        let ray = Ray { origin: Point3::new(0.0, 0.0, 1.0), direction: Vector3::unit_z() };
        assert_eq!(IntersectionResult::Miss, ray.intersect_triangle(
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0)
        ));
    }
}