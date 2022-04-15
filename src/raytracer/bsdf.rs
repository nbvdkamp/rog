use cgmath::{Vector3, vec3, InnerSpace};
use lerp::Lerp;
use crate::{
    util::reflect,
    material::MaterialSample,
    color::RGBf32,
};

pub fn mis2(pdf1: f32, pdf2: f32) -> f32 {
    // Power heuristic (Veach 95)
    let sqp1 = pdf1 * pdf1;
    let sqp2 = pdf2 * pdf2;
    sqp1 / (sqp1 + sqp2)
}

fn schlick_fresnel_approximation(cos_theta: f32, zero_angle_reflection: f32) -> f32 {
    let one_min_cos_theta =  (1.0 - cos_theta).clamp(0.0, 1.0);
    let one_min_cos_squared = one_min_cos_theta * one_min_cos_theta;
    let one_min_cos_pow_5 = one_min_cos_squared * one_min_cos_squared * one_min_cos_theta;
    zero_angle_reflection + (1.0 - zero_angle_reflection) * one_min_cos_pow_5
}

fn fresnel_f_zero(ior: f32) -> f32 {
    (ior - 1.0) * (ior - 1.0) / ((ior + 1.0) * (ior + 1.0))
}

mod ggx {
    use rand::Rng;
    use cgmath::{Vector3, vec3};

    pub fn smith_shadow_term(v_dot_n: f32, alpha_squared: f32) -> f32 {
        let cos2 = v_dot_n * v_dot_n;
        let tan2 = (1.0 - cos2) / cos2;
        2.0 / (1.0 + (1.0 + alpha_squared * tan2).sqrt())
    }

    /// Using f64 since single precision leads to nans for near perfect mirror surfaces
    pub fn normal_distribution(alpha_squared: f64, cos_theta: f64) -> f32 {
        let cos_theta_squared = cos_theta * cos_theta;
        let b = (alpha_squared - 1.0) * cos_theta_squared + 1.0;

        (alpha_squared / (std::f64::consts::PI * b * b)) as f32
    }

    pub fn sample_micronormal(alpha_squared: f32) -> Vector3<f32> {
        let mut rng = rand::thread_rng();
        let r1 = rng.gen::<f32>();
        let r2 = rng.gen::<f32>();

        let phi = 2.0 * std::f32::consts::PI * r2;
        let cos_theta_squared = (1.0 - r1) / (r1 * (alpha_squared - 1.0) + 1.0);
        let cos_theta = cos_theta_squared.sqrt();
        let sin_theta= (1.0 - cos_theta_squared).sqrt();

        vec3(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
    }
}

fn brdf(mat: &MaterialSample, i_dot_n: f32, o_dot_n: f32, m_dot_n: f32, i_dot_m: f32) -> (RGBf32, f32) {
    let alpha = mat.roughness * mat.roughness;
    let alpha_squared = alpha * alpha;

    let g_i = ggx::smith_shadow_term(i_dot_n, alpha_squared);
    let g_o = ggx::smith_shadow_term(o_dot_n, alpha_squared);
    let shadow_masking = g_o * g_i;
    let normal_distrib = ggx::normal_distribution(alpha_squared as f64, m_dot_n as f64);

    let fresnel_m = schlick_fresnel_approximation(i_dot_m, fresnel_f_zero(1.45));
    let specular_color = 0.2;
    let fresnel = specular_color.lerp(1.0, fresnel_m);

    let brdf = RGBf32::white() * fresnel * shadow_masking * normal_distrib / (4.0 * i_dot_n * o_dot_n);
    let pdf = g_o * normal_distrib / (4.0 * i_dot_n * o_dot_n);

    (brdf, pdf)
}

pub fn brdf_sample(mat: &MaterialSample, incident: Vector3<f32>) -> (Vector3<f32>, RGBf32, f32) {
    let alpha = mat.roughness * mat.roughness;
    let micronormal = ggx::sample_micronormal(alpha * alpha);
    let outgoing = reflect(incident, micronormal);
    let i_dot_n = incident.z;
    let o_dot_n = outgoing.z;
    let m_dot_n = micronormal.z;
    let i_dot_m = incident.dot(micronormal).max(0.0);

    if i_dot_n <= 0.0 || o_dot_n <= 0.0 {
        return (vec3(0.0, 0.0, 0.0), RGBf32::from_grayscale(0.0), 0.0);
    }

    let (brdf, pdf) = brdf(mat, i_dot_n, o_dot_n, m_dot_n, i_dot_m);
    (outgoing, brdf, pdf)
}

pub fn brdf_eval(mat: &MaterialSample, incident: Vector3<f32>, outgoing: Vector3<f32>) -> (RGBf32, f32) {
    let micronormal = (incident + outgoing).normalize();
    let outgoing = reflect(incident, micronormal);
    let i_dot_n = incident.z;
    let o_dot_n = outgoing.z;
    let m_dot_n = micronormal.z;
    let i_dot_m = incident.dot(micronormal).max(0.0);

    if i_dot_n <= 0.0 || o_dot_n <= 0.0 {
        return (RGBf32::from_grayscale(0.0), 0.0);
    }

    brdf(mat, i_dot_n, o_dot_n, m_dot_n, i_dot_m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f0_ior() {
        assert_eq!(fresnel_f_zero(1.5), 0.04);
    }
}