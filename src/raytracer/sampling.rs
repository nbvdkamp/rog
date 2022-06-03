use cgmath::Vector3;
use rand::{thread_rng, Rng};

pub fn cos_weighted_sample_hemisphere() -> Vector3<f32> {
    let mut rng = thread_rng();

    let rand: f32 = rng.gen();
    let radius = rand.sqrt();
    let z = (1.0 - rand).sqrt();

    let theta: f32 = 2.0 * std::f32::consts::PI * rng.gen::<f32>();

    Vector3::new(radius * theta.cos(), radius * theta.sin(), z)
}

pub fn tent_sample() -> f32 {
    let r = 2.0 * thread_rng().gen::<f32>();

    if r < 1.0 {
        r.sqrt() - 1.0
    } else {
        1.0 - (2.0 - r).sqrt()
    }
}
