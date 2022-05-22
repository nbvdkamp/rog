use cgmath::{vec2, InnerSpace, Vector3};
use lerp::Lerp;

use crate::{material::MaterialSample, spectrum::Spectrumf32};

use super::geometry::reflect;

pub fn mis2(pdf1: f32, pdf2: f32) -> f32 {
    // Power heuristic (Veach 95)
    let sqp1 = pdf1 * pdf1;
    let sqp2 = pdf2 * pdf2;
    sqp1 / (sqp1 + sqp2)
}

fn schlick_fresnel_approximation(cos_theta: f32, zero_angle_reflection: f32) -> f32 {
    let one_min_cos_theta = (1.0 - cos_theta).clamp(0.0, 1.0);
    let one_min_cos_squared = one_min_cos_theta * one_min_cos_theta;
    let one_min_cos_pow_5 = one_min_cos_squared * one_min_cos_squared * one_min_cos_theta;
    zero_angle_reflection + (1.0 - zero_angle_reflection) * one_min_cos_pow_5
}

fn fresnel_f_zero(ior: f32) -> f32 {
    (ior - 1.0) * (ior - 1.0) / ((ior + 1.0) * (ior + 1.0))
}

mod ggx {
    use cgmath::{vec3, InnerSpace, Vector2, Vector3};
    use rand::Rng;

    pub fn smith_shadow_term(v_dot_n: f32, alpha_squared: f32) -> f32 {
        let cos2 = v_dot_n * v_dot_n;
        let tan2 = (1.0 - cos2) / cos2;
        2.0 / (1.0 + (1.0 + alpha_squared * tan2).sqrt())
    }

    pub fn normal_distribution(alpha_squared: f32, cos_theta: f32) -> f32 {
        // Using f64 since single precision leads to nans for near perfect mirror surfaces
        let alpha_squared = alpha_squared as f64;
        let cos_theta = cos_theta as f64;

        let cos_theta_squared = cos_theta * cos_theta;
        let b = (alpha_squared - 1.0) * cos_theta_squared + 1.0;

        (alpha_squared / (std::f64::consts::PI * b * b)) as f32
    }

    /// Samples distribution of visible normals (Heitz 2018)
    pub fn sample_micronormal(incident: Vector3<f32>, alpha: Vector2<f32>) -> Vector3<f32> {
        let incident_h = vec3(alpha.x * incident.x, alpha.y * incident.y, incident.z).normalize();

        let length_squared = incident_h.x * incident_h.x + incident_h.y * incident_h.y;
        let tangent = if length_squared > 0.0 {
            vec3(-incident_h.y, incident_h.x, 0.0) / length_squared.sqrt()
        } else {
            vec3(1.0, 0.0, 0.0)
        };
        let bitangent = incident_h.cross(tangent);

        let mut rng = rand::thread_rng();
        let r1 = rng.gen::<f32>();
        let r2 = rng.gen::<f32>();

        let radius = r1.sqrt();
        let phi = 2.0 * std::f32::consts::PI * r2;
        let t1 = radius * phi.cos();
        let t2 = radius * phi.sin();
        let s = 0.5 * (1.0 + incident_h.z);

        let t2prime = (1.0 - s) * (1.0 - t1 * t1).sqrt() + s * t2;

        let normal_h =
            t1 * tangent + t2prime * bitangent + (1.0 - t1 * t1 - t2prime * t2prime).max(0.0).sqrt() * incident_h;

        vec3(alpha.x * normal_h.x, alpha.y * normal_h.y, normal_h.z.max(0.0)).normalize()
    }
}

fn brdf(mat: &MaterialSample, i_dot_n: f32, o_dot_n: f32, m_dot_n: f32, i_dot_m: f32) -> (Spectrumf32, f32) {
    let alpha = mat.roughness * mat.roughness;
    let alpha_squared = alpha * alpha;

    let g_i = ggx::smith_shadow_term(i_dot_n, alpha_squared);
    let g_o = ggx::smith_shadow_term(o_dot_n, alpha_squared);
    let shadow_masking = g_o * g_i;
    let normal_distrib = ggx::normal_distribution(alpha_squared, m_dot_n);
    // VNDF eq. 3 (Heitz 2018)
    let visible_normal_distrib = g_i * i_dot_m.max(0.0) * normal_distrib / i_dot_n;

    let fresnel_m = schlick_fresnel_approximation(i_dot_m, fresnel_f_zero(mat.ior));
    let specular_color = 0.2;
    let fresnel = specular_color.lerp(1.0, fresnel_m);
    let fresnel = 1.0;

    // Leaving o_dot_n out of the divisor to multiply the result by cos theta
    let brdf = Spectrumf32::constant(1.0) * fresnel * shadow_masking * normal_distrib / (4.0 * i_dot_n);
    // VNDF eq. 17 (Heitz 2018)
    let pdf = visible_normal_distrib / (4.0 * i_dot_m);

    (brdf, pdf)
}

pub enum Sample {
    Sample {
        outgoing: Vector3<f32>,
        brdf: Spectrumf32,
        pdf: f32,
    },
    Null,
}

pub fn brdf_sample(mat: &MaterialSample, incident: Vector3<f32>) -> Sample {
    let alpha = mat.roughness * mat.roughness;
    let micronormal = ggx::sample_micronormal(incident, vec2(alpha, alpha));
    let outgoing = reflect(incident, micronormal);
    let i_dot_n = incident.z;
    let o_dot_n = outgoing.z;
    let m_dot_n = micronormal.z;
    let i_dot_m = incident.dot(micronormal).max(0.0);

    if i_dot_n > 0.0 && o_dot_n > 0.0 {
        let (brdf, pdf) = brdf(mat, i_dot_n, o_dot_n, m_dot_n, i_dot_m);
        Sample::Sample { outgoing, brdf, pdf }
    } else {
        Sample::Null
    }
}

pub enum Evaluation {
    Evaluation { brdf: Spectrumf32, pdf: f32 },
    Null,
}

pub fn brdf_eval(mat: &MaterialSample, incident: Vector3<f32>, outgoing: Vector3<f32>) -> Evaluation {
    let micronormal = (incident + outgoing).normalize();
    let i_dot_n = incident.z;
    let o_dot_n = outgoing.z;
    let m_dot_n = micronormal.z;
    let i_dot_m = incident.dot(micronormal).max(0.0);

    if i_dot_n > 0.0 && o_dot_n > 0.0 {
        let (brdf, pdf) = brdf(mat, i_dot_n, o_dot_n, m_dot_n, i_dot_m);
        Evaluation::Evaluation { brdf, pdf }
    } else {
        Evaluation::Null
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f0_ior() {
        assert_eq!(fresnel_f_zero(1.5), 0.04);
    }
}
