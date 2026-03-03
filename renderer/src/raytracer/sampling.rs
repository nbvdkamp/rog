use std::f32::consts::PI;

use cgmath::{EuclideanSpace, InnerSpace, Point3, Vector3, point3};
use rand::RngExt;

use super::geometry::orthogonal_vector;
use crate::small_thread_rng::thread_rng;

pub fn cos_weighted_sample_hemisphere() -> Vector3<f32> {
    let mut rng = thread_rng();

    let rand: f32 = rng.random();
    let radius = rand.sqrt();
    let z = (1.0 - rand).sqrt();

    let theta: f32 = 2.0 * PI * rng.random::<f32>();

    Vector3::new(radius * theta.cos(), radius * theta.sin(), z)
}

pub fn sample_orthogonal_disk(direction: Vector3<f32>) -> Vector3<f32> {
    let tangent = orthogonal_vector(direction).normalize();
    let bitangent = direction.cross(tangent);

    let mut rng = thread_rng();
    let theta: f32 = 2.0 * PI * rng.random::<f32>();
    let r = rng.random::<f32>().sqrt();

    r * (theta.cos() * tangent + theta.sin() * bitangent)
}

pub fn sample_uniform_in_unit_sphere(center: Point3<f32>, radius: f32) -> Point3<f32> {
    let mut rng = thread_rng();
    let theta: f32 = 2.0 * PI * rng.random::<f32>();
    let z = 2.0 * rng.random::<f32>() - 1.0;
    let p = (1.0 - z * z).sqrt();
    let r = radius * rng.random::<f32>().cbrt();

    r * point3(p * theta.cos(), p * theta.sin(), z) + center.to_vec()
}

pub fn tent_sample() -> f32 {
    let r = 2.0 * thread_rng().random::<f32>();

    if r < 1.0 {
        r.sqrt() - 1.0
    } else {
        1.0 - (2.0 - r).sqrt()
    }
}

/// When a set of probabilities is sampled only once this is faster than
/// [sample_item_from_cumulative_probabilities] because it doesn't require allocating
pub fn sample_item_from_probabilities(probabilities: &[f32]) -> Option<(usize, f32)> {
    if probabilities.is_empty() {
        return None;
    }

    let sample = thread_rng().random::<f32>();

    let mut sum = 0.0;

    for (i, &w) in probabilities.iter().enumerate() {
        sum += w;
        if sum >= sample {
            return Some((i, probabilities[i]));
        }
    }

    let last_i = probabilities.len() - 1;
    Some((last_i, probabilities[last_i]))
}

pub fn sample_value_from_slice_uniform<T>(slice: &[T]) -> (&T, f32) {
    let value = &slice[thread_rng().random_range(0..slice.len())];
    let pdf = 1.0 / slice.len() as f32;
    (value, pdf)
}
