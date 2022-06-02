mod fresnel;

use cgmath::{vec2, InnerSpace, Vector2, Vector3};
use lerp::Lerp;
use rand::{thread_rng, Rng};

use crate::{material::MaterialSample, spectrum::Spectrumf32};

use super::geometry::reflect;

pub fn mis2(pdf1: f32, pdf2: f32) -> f32 {
    // Power heuristic (Veach 95)
    let sqp1 = pdf1 * pdf1;
    let sqp2 = pdf2 * pdf2;
    sqp1 / (sqp1 + sqp2)
}

mod ggx {
    use cgmath::{vec2, vec3, InnerSpace, Vector2, Vector3};
    use rand::Rng;

    use crate::{material::MaterialSample, spectrum::Spectrumf32};

    use super::{fresnel, Evaluation};

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
    fn sample_micronormal(incident: Vector3<f32>, alpha: Vector2<f32>) -> Vector3<f32> {
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

    fn brdf(mat: &MaterialSample, i_dot_n: f32, o_dot_n: f32, m_dot_n: f32, i_dot_m: f32) -> (Spectrumf32, f32) {
        let alpha = mat.roughness * mat.roughness;
        let alpha_squared = alpha * alpha;

        let g_i = smith_shadow_term(i_dot_n, alpha_squared);
        let g_o = smith_shadow_term(o_dot_n, alpha_squared);
        let shadow_masking = g_o * g_i;
        let normal_distrib = normal_distribution(alpha_squared, m_dot_n);
        // VNDF eq. 3 (Heitz 2018)
        let visible_normal_distrib = g_i * i_dot_m.max(0.0) * normal_distrib / i_dot_n;

        let fresnel = fresnel::disney(mat, i_dot_m);

        // Leaving o_dot_n out of the divisor to multiply the result by cos theta
        let brdf = fresnel * shadow_masking * normal_distrib / (4.0 * i_dot_n);
        // VNDF eq. 17 (Heitz 2018)
        let pdf = visible_normal_distrib / (4.0 * i_dot_m);

        (brdf, pdf)
    }

    pub fn sample_m(mat: &MaterialSample, incident: Vector3<f32>) -> Vector3<f32> {
        let alpha = mat.roughness * mat.roughness;
        sample_micronormal(incident, vec2(alpha, alpha))
    }

    // pub fn sample(mat: &MaterialSample, incident: Vector3<f32>, micronormal: Vector3<f32>) -> (Vector3<f32>, f32, f32) {
    //     let i_dot_n = incident.z;
    //     let o_dot_n = outgoing.z;
    //     let m_dot_n = micronormal.z;
    //     let i_dot_m = incident.dot(micronormal).max(0.0);

    //     if i_dot_n > 0.0 && o_dot_n > 0.0 {
    //         let (brdf, pdf) = brdf(mat, i_dot_n, o_dot_n, m_dot_n, i_dot_m);
    //         (outgoing, brdf, pdf)
    //     } else {
    //         (vec3(0.0, 0.0, 0.0), 0.0, 0.0)
    //     }
    // }

    pub fn eval(mat: &MaterialSample, incident: Vector3<f32>, outgoing: Vector3<f32>) -> Evaluation {
        let micronormal = (incident + outgoing).normalize();
        eval_m(mat, incident, outgoing, micronormal)
    }

    pub fn eval_m(
        mat: &MaterialSample,
        incident: Vector3<f32>,
        outgoing: Vector3<f32>,
        micronormal: Vector3<f32>,
    ) -> Evaluation {
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
}

pub fn cos_weighted_sample_hemisphere() -> Vector3<f32> {
    let mut rng = rand::thread_rng();

    let rand: f32 = rng.gen();
    let radius = rand.sqrt();
    let z = (1.0 - rand).sqrt();

    let theta: f32 = 2.0 * std::f32::consts::PI * rng.gen::<f32>();

    Vector3::new(radius * theta.cos(), radius * theta.sin(), z)
}

pub enum Sample {
    Sample {
        outgoing: Vector3<f32>,
        brdf: Spectrumf32,
        pdf: f32,
    },
    Null,
}

pub enum Evaluation {
    Evaluation { brdf: Spectrumf32, pdf: f32 },
    Null,
}

fn eval_disney_diffuse(mat: &MaterialSample, wo: Vector3<f32>, wi: Vector3<f32>, wm: Vector3<f32>, thin: bool) -> f32 {
    let n_dot_l = wi.z.abs();
    let n_dot_v = wo.z.abs();

    let fl = fresnel::schlick_weight(n_dot_l);
    let fv = fresnel::schlick_weight(n_dot_v);

    let flatness = 0.0; //TODO:

    let hanrahan_krueger = if thin && flatness > 0.0 {
        let m_dot_l = wm.dot(wi);
        let fss90 = m_dot_l * m_dot_l * mat.roughness * mat.roughness;
        let fss = 1.0.lerp(fss90, fl) * 1.0.lerp(fss90, fv);

        1.25 * (fss * (1.0 / (n_dot_l + n_dot_v) - 0.5) + 0.5)
    } else {
        0.0
    };

    let retro = {
        let m_dot_l = wm.dot(wi);
        let rr = 2.0 * mat.roughness * m_dot_l * m_dot_l;
        rr * (fl + fv + fl * fv * (rr - 1.0))
    };

    let lambert = n_dot_l;

    let weight = if thin { flatness } else { 0.0 };
    let subsurf_approximation = lambert.lerp(hanrahan_krueger, weight);

    (retro + subsurf_approximation * (1.0 - 0.5 * fl) * (1.0 - 0.5 * fv)) / std::f32::consts::PI
}

struct LobePdfs {
    specular_reflection: f32,
    specular_transmission: f32,
    diffuse: f32,
    // clearcoat: f32,
}

fn lobe_pdfs(mat: &MaterialSample) -> LobePdfs {
    let metallic_brdf = mat.metallic;
    let specular_bsdf = (1.0 - metallic_brdf) * mat.transmission;
    let dielectric_brdf = (1.0 - metallic_brdf) * (1.0 - mat.transmission);

    let specular_weight = metallic_brdf + dielectric_brdf;
    let transmission_weight = specular_bsdf;
    let diffuse_weight = dielectric_brdf;
    // let clearcoat_weight = mat.clearcoat.min(1.0).max(0.0);

    let norm = 1.0 / (specular_weight + transmission_weight + diffuse_weight); // + clearcoat_weight);

    LobePdfs {
        specular_reflection: specular_weight * norm,
        specular_transmission: transmission_weight * norm,
        diffuse: diffuse_weight * norm,
    }
}

fn calculate_alpha(roughness: f32) -> Vector2<f32> {
    // TODO: Get anisotropy parameter
    let anisotropic: f32 = 0.0;
    let aspect = (1.0 - 0.9 * anisotropic).sqrt();
    let alpha = roughness * roughness;
    vec2(alpha / aspect, alpha * aspect)
}

pub fn eval(mat: &MaterialSample, incident: Vector3<f32>, outgoing: Vector3<f32>) -> Evaluation {
    let thin = false;
    let wo = incident;
    let wi = outgoing;
    let wm = (wo + wi).normalize();

    let n_dot_v = wo.z;
    let n_dot_l = wi.z;

    let mut reflectance = Spectrumf32::constant(0.0);
    let mut forward_pdf = 0.0;
    let mut reverse_pdf = 0.0;

    let lobe_pdfs = lobe_pdfs(mat);

    let alpha = calculate_alpha(mat.roughness);

    let transmission_weight = (1.0 - mat.metallic) * mat.transmission;
    let diffuse_weight = (1.0 - mat.metallic) * (1.0 - mat.transmission);

    let upper_hemisphere = n_dot_l > 0.0 && n_dot_v > 0.0;

    // TODO: Clearcoat

    if diffuse_weight > 0.0 {
        let diffuse = eval_disney_diffuse(mat, wo, wm, wi, thin);
        //TODO: Add Sheen
        reflectance += diffuse_weight * (diffuse * mat.base_color_spectrum); //+ sheen);

        forward_pdf += lobe_pdfs.diffuse * wi.z.abs();
        reverse_pdf += lobe_pdfs.diffuse * wo.z.abs();
    }

    // if transmission_weight > 0.0 {
    //     let scaled_rougness = if thin {
    //         mat.roughness // TODO:
    //     } else {
    //         mat.roughness
    //     };

    //     let transmission_alpha = calculate_alpha(scaled_roughness);

    //     let transmission = eval_disney_specular_transmission(mat, wo, wi, wm, alpha, thin);
    //     reflectance += transmission_weight * transmission;

    //     // TODO: These are the same ??
    //     let l_dot_m = wm.dot(wi);
    //     let v_dot_m = wm.dot(wo);
    //     let eta = mat.ior; // TODO: Correct?
    //     let square = |x| x * x;

    //     forward_pdf += lobe_pdfs.specular_transmission * forward_transmission_pdf / square(l_dot_m + eta * v_dot_m);
    //     reverse_pdf += lobe_pdfs.specular_transmission * reverse_transmission_pdf / square(v_dot_m + eta * l_dot_m);
    // }

    if upper_hemisphere {
        if let Evaluation::Evaluation { brdf: specular, pdf } = ggx::eval_m(mat, wo, wi, wm) {
            reflectance += specular;
            forward_pdf += lobe_pdfs.specular_reflection * pdf / /*maybe already in eval_m?*/ (4.0 * wm.dot(wo).abs());
            // TODO: reverse pdf
        }
    }

    Evaluation::Evaluation {
        brdf: reflectance,
        pdf: forward_pdf,
    }
}

// pub fn bsdf_sample_specular_transmission(mat: &MaterialSample, incident: Vector3<f32>) -> Sample {
//     let v_dot_n = incident.z;

//     if v_dot_n == 0.0 {
//         return Sample::Null;
//     }

//     let alpha = calculate_alpha(mat.roughness);

//     Sample::Null
// }

pub fn bsdf_sample_diffuse_reflection(mat: &MaterialSample, incident: Vector3<f32>) -> Sample {
    let wo = incident;
    let sign = wo.z.signum();

    let wi = sign * cos_weighted_sample_hemisphere();
    let wm = (wi + wo).normalize();

    let n_dot_l = wi.z;
    let n_dot_v = wo.z;

    if n_dot_l == 0.0 {
        return Sample::Null;
    }

    //TODO: Diffuse transmission? I suppose the general transmission value is used for both?
    // but then is it still logical to use that value to choose between specular transmission or diffuse?
    let pdf = 1.0;

    let thin = false;

    let diffuse = eval_disney_diffuse(mat, wo, wi, wm, thin);

    Sample::Sample {
        outgoing: wi,
        brdf: mat.base_color_spectrum * diffuse / pdf, // + sheen
        pdf: pdf * n_dot_l.abs(),
    }
}

pub fn sample(mat: &MaterialSample, incident: Vector3<f32>) -> Sample {
    let pdfs = lobe_pdfs(mat);

    let mut r = thread_rng().gen::<f32>();
    // return Sample::Sample { outgoing: , brdf: (), pdf: () }

    if r < pdfs.specular_reflection {
        let micronormal = ggx::sample_m(mat, incident);
        let outgoing = reflect(incident, micronormal);
        return if let Evaluation::Evaluation { brdf, pdf } = ggx::eval(mat, incident, outgoing) {
            // forward_pdf += lobe_pdfs.specular_reflection TODO: * pdf / /*maybe already in eval_m?*/ (4.0 * wm.dot(wo).abs());
            Sample::Sample {
                outgoing,
                brdf,
                // pdf: pdf * pdfs.specular_reflection,
                pdf,
            }
        } else {
            Sample::Null
        };
    } else {
        r -= pdfs.specular_reflection;
    }

    // if r < pdfs.specular_transmission {
    //     return bsdf_sample_specular_transmission(mat, incident);
    // } else {
    //     return bsdf_sample_diffuse_reflection(mat, incident);
    // }
    return if let Sample::Sample { outgoing, brdf, pdf } = bsdf_sample_diffuse_reflection(mat, incident) {
        Sample::Sample {
            outgoing,
            brdf,
            pdf: pdf * pdfs.diffuse,
        }
    } else {
        Sample::Null
    };
    // TODO: Clearcoat sampling
}
