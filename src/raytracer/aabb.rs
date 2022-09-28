use super::{axis::Axis, Ray};
use crate::util::*;
use cgmath::{ElementWise, EuclideanSpace, Point3, Vector3};

#[derive(Debug, Copy, Clone)]
pub struct BoundingBox {
    pub min: Point3<f32>,
    pub max: Point3<f32>,
}

impl BoundingBox {
    pub fn new() -> Self {
        BoundingBox {
            min: Point3::new(f32::MAX, f32::MAX, f32::MAX),
            max: Point3::new(f32::MIN, f32::MIN, f32::MIN),
        }
    }

    pub fn add(&mut self, p: Point3<f32>) {
        self.min = self.min.elementwise_min(p);
        self.max = self.max.elementwise_max(p);
    }

    pub fn union(&mut self, other: Self) -> Self {
        Self {
            min: self.min.elementwise_min(other.min),
            max: self.max.elementwise_max(other.max),
        }
    }

    pub fn set_min(&mut self, axis: Axis, value: f32) {
        match axis {
            Axis::X => self.min.x = value,
            Axis::Y => self.min.y = value,
            Axis::Z => self.min.z = value,
        }
    }

    pub fn set_max(&mut self, axis: Axis, value: f32) {
        match axis {
            Axis::X => self.max.x = value,
            Axis::Y => self.max.y = value,
            Axis::Z => self.max.z = value,
        }
    }

    pub fn center(&self) -> Point3<f32> {
        self.min.midpoint(self.max)
    }

    pub fn surface_area(&self) -> f32 {
        if self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z {
            return 0.0;
        }

        let e = self.max - self.min;
        2.0 * (e.x * e.y + e.x * e.z + e.y * e.z)
    }

    pub fn intersects_ray(&self, ray: &Ray, inv_dir: &Vector3<f32>) -> bool {
        let t0 = (self.min - ray.origin).mul_element_wise(*inv_dir);
        let t1 = (self.max - ray.origin).mul_element_wise(*inv_dir);
        let tmin = t0.elementwise_min(t1);
        let tmax = t0.elementwise_max(t1);

        max_element(tmin) <= min_element(tmax) && min_element(tmax) >= 0.0 // Ensure the object isn't behind the ray
    }

    // Assumes there is an intersection
    pub fn t_distance_from_ray(&self, ray: &Ray, inv_dir: &Vector3<f32>) -> f32 {
        let t0 = (self.min - ray.origin).mul_element_wise(*inv_dir);
        let t1 = (self.max - ray.origin).mul_element_wise(*inv_dir);
        let tmin = t0.elementwise_min(t1);

        max_element(tmin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center() {
        let mut bb = BoundingBox::new();
        bb.add(Point3::new(0.0, 0.0, 0.0));
        bb.add(Point3::new(1.0, 1.0, 1.0));

        assert_eq!(bb.center(), Point3::new(0.5, 0.5, 0.5));
    }

    #[test]
    fn area() {
        let mut bb = BoundingBox::new();
        bb.add(Point3::new(0.0, -1.0, 0.0));
        bb.add(Point3::new(1.0, 1.0, 1.0));

        assert_eq!(bb.surface_area(), 10.0);
    }

    #[test]
    fn intersect_ray_hit() {
        let mut bb = BoundingBox::new();
        bb.add(Point3::new(-1.0, -1.0, -1.0));
        bb.add(Point3::new(1.0, 1.0, 1.0));

        let ray = Ray {
            origin: Point3::new(-2.0, 0.0, 0.0),
            direction: Vector3::new(1.0, 0.0, 0.0),
        };
        let inv_dir = 1.0 / ray.direction;

        assert_eq!(bb.intersects_ray(&ray, &inv_dir), true);
    }

    #[test]
    fn intersect_ray_miss() {
        let mut bb = BoundingBox::new();
        bb.add(Point3::new(-1.0, -1.0, -1.0));
        bb.add(Point3::new(1.0, 1.0, 1.0));

        let ray = Ray {
            origin: Point3::new(-2.0, 2.0, 0.0),
            direction: Vector3::new(1.0, 0.0, 0.0),
        };
        let inv_dir = 1.0 / ray.direction;

        assert_eq!(bb.intersects_ray(&ray, &inv_dir), false);
    }

    #[test]
    fn intersect_ray_miss_behind_ray() {
        let mut bb = BoundingBox::new();
        bb.add(Point3::new(-1.0, -1.0, -1.0));
        bb.add(Point3::new(1.0, 1.0, 1.0));

        let ray = Ray {
            origin: Point3::new(2.0, 0.0, 0.0),
            direction: Vector3::new(1.0, 0.0, 0.0),
        };
        let inv_dir = 1.0 / ray.direction;

        assert_eq!(bb.intersects_ray(&ray, &inv_dir), false);
    }

    #[test]
    fn distance_from_ray() {
        let mut bb = BoundingBox::new();
        bb.add(Point3::new(-1.0, -1.0, -1.0));
        bb.add(Point3::new(1.0, 1.0, 1.0));

        let direction = Vector3::new(1., 0., 0.);
        let inv_dir = 1. / direction;
        let ray = Ray {
            origin: Point3::new(-2., 0., 0.),
            direction,
        };

        assert_eq!(bb.t_distance_from_ray(&ray, &inv_dir), 1.0);
    }

    #[test]
    fn distance_from_ray_backwards() {
        let mut bb = BoundingBox::new();
        bb.add(Point3::new(-1.0, -1.0, -1.0));
        bb.add(Point3::new(1.0, 1.0, 1.0));

        let direction = Vector3::new(-1., 0., 0.);
        let inv_dir = 1. / direction;
        let ray = Ray {
            origin: Point3::new(2., 0., 0.),
            direction,
        };

        assert_eq!(bb.t_distance_from_ray(&ray, &inv_dir), 1.0);
    }
}
