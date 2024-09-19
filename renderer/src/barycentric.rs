use cgmath::{EuclideanSpace, Point2, Point3, Vector3};

pub struct Barycentric {
    u: f32,
    v: f32,
}

impl Barycentric {
    pub fn new(u: f32, v: f32) -> Self {
        Self { u, v }
    }

    pub fn interpolate_point(&self, values: [Point3<f32>; 3]) -> Point3<f32> {
        self.w() * values[0] + self.u * values[1].to_vec() + self.v * values[2].to_vec()
    }

    pub fn interpolate_point2(&self, values: [Point2<f32>; 3]) -> Point2<f32> {
        self.w() * values[0] + self.u * values[1].to_vec() + self.v * values[2].to_vec()
    }

    pub fn interpolate_vector(&self, values: [Vector3<f32>; 3]) -> Vector3<f32> {
        self.w() * values[0] + self.u * values[1] + self.v * values[2]
    }

    fn w(&self) -> f32 {
        1.0 - self.u - self.v
    }
}
