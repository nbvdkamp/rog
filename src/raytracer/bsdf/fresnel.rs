use crate::{material::MaterialSample, spectrum::Spectrumf32};
use cgmath::{InnerSpace, Vector3};
use lerp::Lerp;

pub fn schlick_weight(cos_theta: f32) -> f32 {
    let one_min_cos_theta = (1.0 - cos_theta).clamp(0.0, 1.0);
    let one_min_cos_squared = one_min_cos_theta * one_min_cos_theta;
    one_min_cos_squared * one_min_cos_squared * one_min_cos_theta
}

pub fn schlick_approximation(cos_theta: f32, zero_angle_reflection: f32) -> f32 {
    zero_angle_reflection + (1.0 - zero_angle_reflection) * schlick_weight(cos_theta)
}

pub fn f_zero(medium_ior: f32, material_ior: f32) -> f32 {
    let x = (medium_ior - material_ior) / (medium_ior + material_ior);
    x * x
}

pub fn dielectric(cos_theta_i: f32, eta: f32) -> f32 {
    let cos_theta_i = cos_theta_i.min(1.0).max(-1.0);

    let sin_theta_t_squared = eta * eta * (1.0 - cos_theta_i * cos_theta_i);

    if sin_theta_t_squared > 1.0 {
        // Total internal reflection
        return 1.0;
    }

    let cos_theta_t = (1.0 - sin_theta_t_squared).sqrt();

    let s = (cos_theta_i - cos_theta_t) / (cos_theta_i + cos_theta_t);
    let p = (cos_theta_t - cos_theta_i) / (cos_theta_t + cos_theta_i);
    // Assuming unpolarized light
    return 0.5 * (s * s + p * p);
}

pub fn disney(mat: &MaterialSample, m_dot_v: f32, eta: f32) -> Spectrumf32 {
    // Simplifying by assuming the specular tint factor = 0
    let f0 = Spectrumf32::constant(f_zero(1.0, mat.ior)).lerp(mat.base_color_spectrum, mat.metallic);
    let dielectric = dielectric(m_dot_v.abs(), eta);
    // TODO: Schlick is inaccurate for TIR
    let metallic = f0 + (Spectrumf32::constant(1.0) - f0) * schlick_weight(m_dot_v);
    Spectrumf32::constant(dielectric).lerp(metallic, mat.metallic)
}

#[cfg(test)]
mod tests {
    use cgmath::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn f0_ior() {
        assert_abs_diff_eq!(f_zero(1.0, 1.5), 0.04);
    }
}
