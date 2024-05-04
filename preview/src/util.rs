use cgmath::{Matrix3, Matrix4};
use luminance::shader::types::{Mat33, Vec4};
use luminance_front::shader::types::Mat44;
use renderer::color::RGBf32;

pub fn mat_to_shader_type<T>(m: Matrix4<T>) -> Mat44<T> {
    let x: [[T; 4]; 4] = m.into();
    x.into()
}

pub fn mat3_to_shader_type<T>(m: Matrix3<T>) -> Mat33<T> {
    let x: [[T; 3]; 3] = m.into();
    x.into()
}

#[inline]
pub fn vec4_from_rgb(v: RGBf32) -> Vec4<f32> {
    Vec4::new(v.r, v.g, v.b, 1.0)
}
