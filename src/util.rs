use cgmath::{Vector3, Vector4, Matrix4};
use luminance_front::shader::types::{Mat44, Vec4};

pub fn elementwise_min(a: Vector3<f32>, b: Vector3<f32>) -> Vector3<f32> {
    Vector3 {
        x: f32::min(a.x, b.x),
        y: f32::min(a.y, b.y),
        z: f32::min(a.z, b.z),
    }
}

pub fn elementwise_max(a: Vector3<f32>, b: Vector3<f32>) -> Vector3<f32> {
    Vector3 {
        x: f32::max(a.x, b.x),
        y: f32::max(a.y, b.y),
        z: f32::max(a.z, b.z),
    }
}

pub fn min_element(v: Vector3<f32>) -> f32 {
    f32::min(v.x, f32::min(v.y, v.z))
}

pub fn max_element(v: Vector3<f32>) -> f32 {
    f32::max(v.x, f32::max(v.y, v.z))
}

pub fn mat_to_shader_type<T>(m: Matrix4<T>) -> Mat44<T> {
    let x: [[T; 4]; 4] = m.into();
    x.into()
}

pub fn vec_to_shader_type<T>(v: Vector4<T>) -> Vec4<T> {
    let x: [T; 4] = v.into();
    x.into()
}