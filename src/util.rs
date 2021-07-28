use cgmath::Vector3;

pub fn elementwise_min(a: Vector3<f32>, b: Vector3<f32>) -> Vector3<f32> {
    Vector3 {
        x: f32::min(a.x, b.x),
        y: f32::min(a.y, b.y),
        z: f32::min(a.z, b.z),
    }
}

pub fn elementwise_max(a: Vector3<f32>, b: Vector3<f32>) -> Vector3<f32> {
    Vector3 {
        x: f32::min(a.x, b.x),
        y: f32::min(a.y, b.y),
        z: f32::min(a.z, b.z),
    }
}

pub fn min_element(v: Vector3<f32>) -> f32 {
    f32::min(v.x, f32::min(v.y, v.z))
}

pub fn max_element(v: Vector3<f32>) -> f32 {
    f32::max(v.x, f32::max(v.y, v.z))
}