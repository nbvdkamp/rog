use rand::Rng;
use cgmath::Vector3;

fn _cos_weighted_sample_hemisphere_z_up() -> Vector3<f32> {
    let mut rng = rand::thread_rng();

    let rand: f32 = rng.gen();
    let radius = rand.sqrt();
    let z = (1.0 - rand).sqrt();

    let theta: f32 = 2.0 * std::f32::consts::PI * rng.gen::<f32>();

    Vector3::new(radius * theta.cos(), radius * theta.sin(), z)
}