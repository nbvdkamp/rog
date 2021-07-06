use cgmath::Point3;
use super::axis::Axis;

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
}