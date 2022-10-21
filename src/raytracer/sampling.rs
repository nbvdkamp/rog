use cgmath::{point2, Point2, Vector3};
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

pub fn sample_coordinates_on_triangle() -> Point2<f32> {
    let mut rng = rand::thread_rng();
    let r0 = rng.gen();
    let r1 = rng.gen();

    if r0 + r1 > 1.0 {
        point2(1.0 - r0, 1.0 - r1)
    } else {
        point2(r0, r1)
    }
}
