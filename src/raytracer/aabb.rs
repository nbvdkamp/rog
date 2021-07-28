use cgmath::{ElementWise, Point3, Vector3};
use super::axis::Axis;
use super::Ray;
use crate::util::*;

#[derive(Debug, Copy, Clone)]
pub struct BoundingBox {
    min: Point3<f32>,
    max: Point3<f32>,
}

impl BoundingBox {
    pub fn new() -> Self {
        BoundingBox {
            min: Point3::new(f32::MAX, f32::MAX, f32::MAX),
            max: Point3::new(f32::MIN, f32::MIN, f32::MIN),
        }
    }

    pub fn add(&mut self, p: &Point3<f32>) {
        self.min.x = f32::min(self.min.x, p.x);
        self.min.y = f32::min(self.min.y, p.y);
        self.min.z = f32::min(self.min.z, p.z);

        self.max.x = f32::max(self.max.x, p.x);
        self.max.y = f32::max(self.max.y, p.y);
        self.max.z = f32::max(self.max.z, p.z);
    }

    pub fn find_split_plane(&self) -> (Axis, f32) {
        let x_size = self.max.x - self.min.x;
        let y_size = self.max.y - self.min.y;
        let z_size = self.max.z - self.min.z;

        if x_size >= y_size && x_size >= z_size {
            (Axis::X, self.min.x + x_size / 2.0)
        } else if y_size >= z_size {
            (Axis::Y, self.min.y + y_size / 2.0)
        } else {
            (Axis::Z, self.min.z + z_size / 2.0)
        }
    }

    pub fn set_min(&mut self, axis: &Axis, value: f32) {
        match axis {
            Axis::X => self.min.x = value,
            Axis::Y => self.min.y = value,
            Axis::Z => self.min.z = value,
        }
    }

    pub fn set_max(&mut self, axis: &Axis, value: f32) {
        match axis {
            Axis::X => self.max.x = value,
            Axis::Y => self.max.y = value,
            Axis::Z => self.max.z = value,
        }
    }

    pub fn intersects(&self, ray: &Ray, inv_dir: &Vector3<f32>) -> bool {
        let t0 = (self.min - ray.origin).mul_element_wise(*inv_dir);
        let t1 = (self.max - ray.origin).mul_element_wise(*inv_dir);
        let tmin = elementwise_min(t0, t1);
        let tmax = elementwise_max(t0, t1);

        max_element(tmin) <= min_element(tmax)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersect_hit() {
        let mut bb = BoundingBox::new();
        bb.add(&Point3::new(-1.0, -1.0, -1.0));
        bb.add(&Point3::new(1.0, 1.0, 1.0));

        let ray = Ray { origin: Point3::new(-2.0, 0.0, 0.0), direction: Vector3::new(1.0, 0.0, 0.0) };
        let inv_dir = 1.0 / ray.direction;

        assert_eq!(bb.intersects(&ray, &inv_dir), true);
    }

    #[test]
    fn intersect_miss() {
        let mut bb = BoundingBox::new();
        bb.add(&Point3::new(-1.0, -1.0, -1.0));
        bb.add(&Point3::new(1.0, 1.0, 1.0));

        let ray = Ray { origin: Point3::new(-2.0, 2.0, 0.0), direction: Vector3::new(1.0, 0.0, 0.0) };
        let inv_dir = 1.0 / ray.direction;

        assert_eq!(bb.intersects(&ray, &inv_dir), false);
    }
}