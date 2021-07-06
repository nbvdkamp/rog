use cgmath::Point3;
use super::axis::Axis;

#[derive(Debug, Copy, Clone)]
pub struct BoundingBox {
    x_max: f32,
    x_min: f32,
    y_max: f32,
    y_min: f32,
    z_max: f32,
    z_min: f32,
}

impl BoundingBox {
    pub fn new() -> Self {
        BoundingBox {
            x_max: f32::MIN,
            x_min: f32::MAX,
            y_max: f32::MIN,
            y_min: f32::MAX,
            z_max: f32::MIN,
            z_min: f32::MAX,
        }
    }

    pub fn add(&mut self, p: &Point3<f32>) {
        self.x_max = f32::max(self.x_max, p.x);
        self.x_min = f32::min(self.x_min, p.x);
        self.y_max = f32::max(self.y_max, p.y);
        self.y_min = f32::min(self.y_min, p.y);
        self.z_max = f32::max(self.z_max, p.z);
        self.z_min = f32::min(self.z_min, p.z);
    }

    pub fn find_split_plane(&self) -> (Axis, f32) {
        let x_size = self.x_max - self.x_min;
        let y_size = self.y_max - self.y_min;
        let z_size = self.z_max - self.z_min;

        if x_size >= y_size && x_size >= z_size {
            (Axis::X, self.x_min + x_size / 2.0)
        } else if y_size >= z_size {
            (Axis::Y, self.y_min + y_size / 2.0)
        } else {
            (Axis::Z, self.z_min + z_size / 2.0)
        }
    }

    pub fn set_min(&mut self, axis: &Axis, value: f32) {
        match axis {
            Axis::X => self.x_min = value,
            Axis::Y => self.y_min = value,
            Axis::Z => self.z_min = value,
        }
    }

    pub fn set_max(&mut self, axis: &Axis, value: f32) {
        match axis {
            Axis::X => self.x_max = value,
            Axis::Y => self.y_max = value,
            Axis::Z => self.z_max = value,
        }
    }
}