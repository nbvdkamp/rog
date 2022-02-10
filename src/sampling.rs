use rand::Rng;
use cgmath::{Vector3, InnerSpace, Matrix3};

fn orthogonal_vector(v: Vector3<f32>) -> Vector3<f32> {
    if v.x == 0.0 {
        Vector3::new(0.0, -v.z, v.y)
    } else if v.y == 0.0 {
        Vector3::new(-v.z, 0.0, v.x)
    } else {
        Vector3::new(-v.y, v.x, 0.0)
    }
}

fn to_tangent_space(normal: Vector3<f32>, v: Vector3<f32> ) -> Vector3<f32> {
    let tangent = orthogonal_vector(normal).normalize();
    let bitangent = tangent.cross(normal);

    Matrix3::from_cols(tangent, bitangent, normal) * v
}

fn cos_weighted_sample_hemisphere_z_up() -> Vector3<f32> {
    let mut rng = rand::thread_rng();

    let rand: f32 = rng.gen();
    let radius = rand.sqrt();
    let z = (1.0 - rand).sqrt();

    let theta: f32 = 2.0 * std::f32::consts::PI * rng.gen::<f32>();

    Vector3::new(radius * theta.cos(), radius * theta.sin(), z)
}

pub fn cos_weighted_sample_hemisphere(normal: Vector3<f32>) -> Vector3<f32> {
    to_tangent_space(normal, cos_weighted_sample_hemisphere_z_up())
}