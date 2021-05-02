use cgmath::{perspective, Matrix4, Rad, Vector3, Point3, EuclideanSpace};

pub struct PerspectiveCamera {
    pub aspect_ratio: f32,
    pub y_fov: f32,
    pub z_far: f32,
    pub z_near: f32,
    pub view: Matrix4<f32>,
}

impl PerspectiveCamera {
    pub fn default() -> PerspectiveCamera {
        PerspectiveCamera {
            aspect_ratio: 16. / 9.,
            y_fov: 1.5,
            z_far: 100.,
            z_near: 0.01,
            view: Matrix4::<f32>::look_at_rh(Point3::new(2., 2., 2.), Point3::origin(), Vector3::unit_y()),
        }
    }

    pub fn projection(&self) -> Matrix4<f32> {
        perspective(Rad(self.y_fov), self.aspect_ratio, self.z_near, self.z_far)
    }
}