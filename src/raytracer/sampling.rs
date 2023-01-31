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

pub fn cumulative_probabilities_from_weights(weights: &[f32]) -> Vec<f32> {
    let sum = weights.iter().sum::<f32>();

    let mut cumulative_probabilities = Vec::new();
    cumulative_probabilities.reserve(weights.len());

    let mut acc = 0.0;

    for value in weights {
        acc += value / sum;
        cumulative_probabilities.push(acc);
    }

    cumulative_probabilities
}

pub fn sample_item_from_cumulative_probabilities(cumulative_probabilities: &[f32]) -> Option<usize> {
    if cumulative_probabilities.is_empty() {
        return None;
    }

    let mut rng = rand::thread_rng();
    let sample = rng.gen();

    //TODO: This sampling is linear in the number of items, we can improve it if necessary
    for (i, &c) in cumulative_probabilities.iter().enumerate() {
        if c >= sample {
            return Some(i);
        }
    }

    Some(cumulative_probabilities.len() - 1)
}
