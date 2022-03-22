use rand::Rng;
use cgmath::Vector3;
use super::util::to_tangent_space;

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