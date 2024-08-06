use super::{
    axis::Axis,
    ray::{Ray, RayWithInverseDir},
    sampling::sample_uniform_in_unit_sphere,
};
use crate::util::*;
use cgmath::{ElementWise, EuclideanSpace, Point3, Vector3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min: Point3<f32>,
    pub max: Point3<f32>,
}

#[derive(Debug, PartialEq)]
pub enum Intersects {
    Yes { distance: f32 },
    No,
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

    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: self.min.elementwise_min(other.min),
            max: self.max.elementwise_max(other.max),
        }
    }

    pub fn contains(&self, point: Point3<f32>) -> bool {
        for i in 0..3 {
            if point[i] < self.min[i] || point[i] > self.max[i] {
                return false;
            }
        }

        true
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

    #[inline(always)]
    pub fn intersects(&self, ray: &RayWithInverseDir) -> Intersects {
        let RayWithInverseDir {
            ray: Ray { origin, .. },
            inverse_direction,
        } = *ray;
        let t0 = (self.min - origin).mul_element_wise(inverse_direction);
        let t1 = (self.max - origin).mul_element_wise(inverse_direction);
        let tmin = t0.elementwise_min(t1);
        let tmax = t0.elementwise_max(t1);

        let max_tmin = max_element(tmin);
        let min_tmax = min_element(tmax);
        let intersects = max_tmin <= min_tmax && min_tmax >= 0.0; // Ensure the object isn't behind the ray

        if intersects {
            Intersects::Yes { distance: max_tmin }
        } else {
            Intersects::No
        }
    }

    // From Graphics Gems: "Arvo, James, A Simple Method for Box-Sphere Intersection Testing"
    pub fn intersects_sphere(&self, center: Point3<f32>, radius: f32) -> bool {
        let mut dmin = 0.0;
        let square = |x| x * x;

        for i in 0..3 {
            if center[i] < self.min[i] {
                dmin += square(center[i] - self.min[i]);
            } else if center[i] > self.max[i] {
                dmin += square(center[i] - self.max[i]);
            }
        }

        dmin <= square(radius)
    }

    /// Returns volume relative to the total volume of the sphere,
    /// so 1.0 if the whole sphere is contained in the BoundingBox.
    pub fn estimate_volume_of_intersection_with_sphere(&self, center: Point3<f32>, radius: f32, samples: usize) -> f32 {
        let mut hits = 0;

        for _ in 0..samples {
            let sample = sample_uniform_in_unit_sphere(center, radius);

            if self.contains(sample) {
                hits += 1;
            }
        }

        hits as f32 / samples as f32
    }

    pub fn within_min_bound_with_epsilon(&self, point: Point3<f32>, epsilon: f32) -> (bool, Vector3<bool>) {
        let min = (point - self.min).map(|x| x > -epsilon);
        (min.x && min.y && min.z, min)
    }

    pub fn within_max_bound_with_epsilon(&self, point: Point3<f32>, epsilon: f32) -> (bool, Vector3<bool>) {
        let max = (point - self.max).map(|x| x < epsilon);
        (max.x && max.y && max.z, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raytracer::Ray;

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
        }
        .with_inverse_dir();

        assert_eq!(bb.intersects(&ray), Intersects::Yes { distance: 1.0 });
    }

    #[test]
    fn intersect_ray_backwards_hit() {
        let mut bb = BoundingBox::new();
        bb.add(Point3::new(-1.0, -1.0, -1.0));
        bb.add(Point3::new(1.0, 1.0, 1.0));

        let ray = Ray {
            origin: Point3::new(2., 0., 0.),
            direction: Vector3::new(-1., 0., 0.),
        }
        .with_inverse_dir();

        assert_eq!(bb.intersects(&ray), Intersects::Yes { distance: 1.0 });
    }

    #[test]
    fn intersect_ray_miss() {
        let mut bb = BoundingBox::new();
        bb.add(Point3::new(-1.0, -1.0, -1.0));
        bb.add(Point3::new(1.0, 1.0, 1.0));

        let ray = Ray {
            origin: Point3::new(-2.0, 2.0, 0.0),
            direction: Vector3::new(1.0, 0.0, 0.0),
        }
        .with_inverse_dir();

        assert_eq!(bb.intersects(&ray), Intersects::No);
    }

    #[test]
    fn intersect_ray_miss_behind_ray() {
        let mut bb = BoundingBox::new();
        bb.add(Point3::new(-1.0, -1.0, -1.0));
        bb.add(Point3::new(1.0, 1.0, 1.0));

        let ray = Ray {
            origin: Point3::new(2.0, 0.0, 0.0),
            direction: Vector3::new(1.0, 0.0, 0.0),
        }
        .with_inverse_dir();

        assert_eq!(bb.intersects(&ray), Intersects::No);
    }

    #[test]
    fn intersect_sphere() {
        let mut bb = BoundingBox::new();
        bb.add(Point3::new(-1.0, -1.0, -1.0));
        bb.add(Point3::new(1.0, 1.0, 1.0));
        assert_eq!(bb.intersects_sphere(Point3::origin(), 0.0), true);
    }

    #[test]
    fn miss_sphere() {
        let mut bb = BoundingBox::new();
        bb.add(Point3::new(0.5, 0.5, 0.5));
        bb.add(Point3::new(1.0, 1.0, 1.0));
        assert_eq!(bb.intersects_sphere(Point3::origin(), 0.0), false);
    }

    #[test]
    fn just_intersect_sphere() {
        let mut bb = BoundingBox::new();
        bb.add(Point3::new(0.5, 0.5, 0.5));
        bb.add(Point3::new(1.0, 1.0, 1.0));
        assert_eq!(bb.intersects_sphere(Point3::origin(), 3.0_f32.sqrt() / 2.0), true);
    }

    #[test]
    fn just_miss_sphere() {
        let mut bb = BoundingBox::new();
        bb.add(Point3::new(0.5, 0.5, 0.5));
        bb.add(Point3::new(1.0, 1.0, 1.0));
        assert_eq!(
            bb.intersects_sphere(Point3::origin(), 3.0_f32.sqrt() / 2.0 - 0.000001),
            false
        );
    }

    #[test]
    fn sphere_volume_contained() {
        let mut bb = BoundingBox::new();
        bb.add(Point3::new(-1.0, -1.0, -1.0));
        bb.add(Point3::new(1.0, 1.0, 1.0));

        assert_eq!(
            bb.estimate_volume_of_intersection_with_sphere(Point3::origin(), 1.0, 100),
            1.0
        );
    }

    #[test]
    fn sphere_volume_no_intersection() {
        let mut bb = BoundingBox::new();
        bb.add(Point3::new(2.0, 2.0, 2.0));
        bb.add(Point3::new(4.0, 4.0, 4.0));

        assert_eq!(
            bb.estimate_volume_of_intersection_with_sphere(Point3::origin(), 1.0, 100),
            0.0
        );
    }
}

#[cfg(test)]
mod bench {
    extern crate test;
    use test::Bencher;

    use super::*;
    use crate::raytracer::Ray;

    #[bench]
    fn intersect(b: &mut Bencher) {
        let mut bb = BoundingBox::new();
        bb.add(Point3::new(-1.0, -1.0, -1.0));
        bb.add(Point3::new(1.0, 1.0, 1.0));

        let ray = Ray {
            origin: Point3::new(-2.0, 0.0, 0.0),
            direction: Vector3::new(1.0, 0.0, 0.0),
        }
        .with_inverse_dir();

        b.iter(|| bb.intersects(&ray));
    }
}
