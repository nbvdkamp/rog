mod fresnel;

use std::f32::consts::FRAC_1_PI;

use cgmath::{vec2, InnerSpace, Vector2, Vector3};
use lerp::Lerp;
use rand::{thread_rng, Rng};

use crate::{material::MaterialSample, spectrum::Spectrumf32};

use self::fresnel::Reflectance;

use super::{
    geometry::{reflect, refract},
    sampling::cos_weighted_sample_hemisphere,
};

pub fn mis2(pdf1: f32, pdf2: f32) -> f32 {
    // Power heuristic (Veach 95)
    let sqp1 = pdf1 * pdf1;
    let sqp2 = pdf2 * pdf2;
    sqp1 / (sqp1 + sqp2)
}

mod ggx {
    use cgmath::{vec3, InnerSpace, Vector2, Vector3};
    use rand::Rng;

    pub fn smith_shadow_term(n_dot_v: f32, alpha_squared: f32) -> f32 {
        let cos2 = n_dot_v * n_dot_v;
        let tan2 = (1.0 - cos2) / cos2;
        2.0 / (1.0 + (1.0 + alpha_squared * tan2).sqrt())
    }

    pub fn normal_distribution(alpha_squared: f32, n_dot_m: f32) -> f32 {
        // Using f64 since single precision leads to nans for near perfect mirror surfaces
        let alpha_squared = alpha_squared as f64;
        let cos_theta = n_dot_m as f64;

        let cos_theta_squared = cos_theta * cos_theta;
        let b = (alpha_squared - 1.0) * cos_theta_squared + 1.0;

        (alpha_squared / (std::f64::consts::PI * b * b)) as f32
    }

    /// Samples distribution of visible normals (Heitz 2018)
    pub fn sample_micronormal(outgoing: Vector3<f32>, alpha: Vector2<f32>) -> Vector3<f32> {
        let outgoing_h = vec3(alpha.x * outgoing.x, alpha.y * outgoing.y, outgoing.z).normalize();

        let length_squared = outgoing_h.x * outgoing_h.x + outgoing_h.y * outgoing_h.y;
        let tangent = if length_squared > 0.0 {
            vec3(-outgoing_h.y, outgoing_h.x, 0.0) / length_squared.sqrt()
        } else {
            vec3(1.0, 0.0, 0.0)
        };
        let bitangent = outgoing_h.cross(tangent);

        let mut rng = rand::thread_rng();
        let r1 = rng.gen::<f32>();
        let r2 = rng.gen::<f32>();

        let radius = r1.sqrt();
        let phi = 2.0 * std::f32::consts::PI * r2;
        let t1 = radius * phi.cos();
        let t2 = radius * phi.sin();
        let s = 0.5 * (1.0 + outgoing_h.z);

        let t2prime = (1.0 - s) * (1.0 - t1 * t1).sqrt() + s * t2;

        let normal_h =
            t1 * tangent + t2prime * bitangent + (1.0 - t1 * t1 - t2prime * t2prime).max(0.0).sqrt() * outgoing_h;

        vec3(alpha.x * normal_h.x, alpha.y * normal_h.y, normal_h.z.max(0.0)).normalize()
    }
}

pub enum Sample {
    Sample {
        /// Scattered direction, named this way because we are tracing the paths backwards
        incident: Vector3<f32>,
        weight: Spectrumf32,
        pdf: f32,
    },
    Null,
}

impl Sample {
    fn multiply_pdf(self, other_pdf: f32) -> Self {
        match self {
            Sample::Sample { incident, weight, pdf } => Sample::Sample {
                incident,
                weight,
                pdf: pdf * other_pdf,
            },
            Sample::Null => Sample::Null,
        }
    }
}

pub enum Evaluation {
    Evaluation { weight: Spectrumf32, pdf: f32 },
    Null,
}

impl Evaluation {
    fn into_sample_with_incident(self, incident: Vector3<f32>) -> Sample {
        match self {
            Evaluation::Evaluation { weight, pdf } => Sample::Sample { incident, weight, pdf },
            Evaluation::Null => Sample::Null,
        }
    }
}

pub fn eval_specular_reflection(
    mat: &MaterialSample,
    outgoing: Vector3<f32>,
    incident: Vector3<f32>,
    micronormal: Vector3<f32>,
) -> Evaluation {
    let n_dot_i = incident.z;
    let n_dot_o = outgoing.z;

    if n_dot_i <= 0.0 || n_dot_o <= 0.0 {
        return Evaluation::Null;
    }

    let n_dot_m = micronormal.z;
    let m_dot_i = incident.dot(micronormal).max(0.0);

    let alpha = mat.roughness * mat.roughness;
    let alpha_squared = alpha * alpha;

    let g_o = ggx::smith_shadow_term(n_dot_o, alpha_squared);
    let g_i = ggx::smith_shadow_term(n_dot_i, alpha_squared);
    let shadow_masking = g_o * g_i;
    let normal_distrib = ggx::normal_distribution(alpha_squared, n_dot_m);
    // VNDF eq. 3 (Heitz 2018)
    let visible_normal_distrib = g_o * m_dot_i.max(0.0) * normal_distrib / n_dot_o;

    let fresnel = fresnel::disney(mat, m_dot_i);

    let brdf = fresnel * (shadow_masking * normal_distrib / (4.0 * n_dot_o * n_dot_i));
    // VNDF eq. 17 (Heitz 2018)
    let pdf = visible_normal_distrib / (4.0 * m_dot_i);

    Evaluation::Evaluation { weight: brdf, pdf }
}

fn eval_disney_diffuse(
    mat: &MaterialSample,
    outgoing: Vector3<f32>,
    incident: Vector3<f32>,
    micronormal: Vector3<f32>,
) -> f32 {
    let (n_dot_i, n_dot_o) = if incident.z <= 0.0 && outgoing.z <= 0.0 {
        (incident.z.abs(), outgoing.z.abs())
    } else {
        (incident.z.max(0.0), outgoing.z.max(0.0))
    };
    let m_dot_i = micronormal.dot(incident);

    let fl = fresnel::schlick_weight(n_dot_i);
    let fv = fresnel::schlick_weight(n_dot_o);

    let hanrahan_krueger = {
        let fss90 = m_dot_i * m_dot_i * mat.roughness * mat.roughness;
        let fss = 1.0.lerp(fss90, fl) * 1.0.lerp(fss90, fv);

        1.25 * (fss * (1.0 / (n_dot_i + n_dot_o) - 0.5) + 0.5)
    };

    let lambert = 1.0;

    let subsurf_approximation = lambert.lerp(hanrahan_krueger, mat.sub_surface_scattering);

    let retro_reflection = {
        let rr = 2.0 * mat.roughness * m_dot_i * m_dot_i;
        rr * (fl + fv + fl * fv * (rr - 1.0))
    };

    (retro_reflection + subsurf_approximation * (1.0 - 0.5 * fl) * (1.0 - 0.5 * fv)) * FRAC_1_PI
}

fn eval_disney_specular_transmission(
    mat: &MaterialSample,
    outgoing: Vector3<f32>,
    incident: Vector3<f32>,
    micronormal: Vector3<f32>,
    alpha: Vector2<f32>,
) -> Evaluation {
    let n_dot_i = incident.z;
    let n_dot_o = outgoing.z;
    let m_dot_i = micronormal.dot(incident);
    let m_dot_o = micronormal.dot(outgoing);

    // TODO: Anisotropy
    let alpha_squared = alpha.x * alpha.x;

    let g_o = ggx::smith_shadow_term(outgoing.z, alpha_squared);
    let g_i = ggx::smith_shadow_term(incident.z, alpha_squared);
    let shadow_masking = g_o * g_i;
    let normal_distrib = ggx::normal_distribution(alpha_squared, micronormal.z);
    // VNDF eq. 3 (Heitz 2018)
    let visible_normal_distrib = g_o * m_dot_o.max(0.0) * normal_distrib / n_dot_o;

    let fresnel = fresnel::dielectric(m_dot_o, mat.medium_ior, mat.ior).reflectance();

    if n_dot_i * n_dot_o > 0.0 {
        // Reflection
        let jacobian = 1.0 / (4.0 * m_dot_i.abs());

        Evaluation::Evaluation {
            weight: Spectrumf32::constant(fresnel * shadow_masking * normal_distrib / (4.0 * n_dot_o * n_dot_i)),
            pdf: fresnel * visible_normal_distrib * jacobian,
        }
    } else {
        //Refraction

        let square = |x| x * x;
        let c = m_dot_i.abs() * m_dot_o.abs() / (n_dot_i.abs() * n_dot_o.abs());
        let t = square(mat.medium_ior) / square(mat.ior * m_dot_i + mat.medium_ior * m_dot_o);
        // Walter et al. 2007 eq. 17
        let jacobian = t * m_dot_i.abs();

        if (1.0 - fresnel) == 0.0 {
            Evaluation::Null
        } else {
            Evaluation::Evaluation {
                weight: mat.base_color_spectrum.sqrt() * c * t * (1.0 - fresnel) * shadow_masking * normal_distrib,
                pdf: (1.0 - fresnel) * visible_normal_distrib * jacobian,
            }
        }
    }
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

pub fn eval(mat: &MaterialSample, outgoing: Vector3<f32>, incident: Vector3<f32>) -> Evaluation {
    let micronormal = (outgoing + incident).normalize();

    let n_dot_o = outgoing.z;
    let n_dot_i = incident.z;

    let mut reflectance = Spectrumf32::constant(0.0);
    let mut forward_pdf = 0.0;
    let mut _reverse_pdf = 0.0;

    let lobe_pdfs = lobe_pdfs(mat);

    let _alpha = calculate_alpha(mat.roughness);

    let transmission_weight = (1.0 - mat.metallic) * mat.transmission;
    let diffuse_weight = (1.0 - mat.metallic) * (1.0 - mat.transmission);

    let upper_hemisphere = n_dot_i > 0.0 && n_dot_o > 0.0;

    // TODO: Clearcoat

    if diffuse_weight > 0.0 {
        let diffuse = eval_disney_diffuse(mat, outgoing, incident, micronormal);
        //TODO: Add Sheen
        reflectance += diffuse_weight * (diffuse * mat.base_color_spectrum); //+ sheen);

        forward_pdf += lobe_pdfs.diffuse * incident.z.abs();
        _reverse_pdf += lobe_pdfs.diffuse * outgoing.z.abs();
    }

    if transmission_weight > 0.0 {
        let transmission_alpha = calculate_alpha(mat.roughness);

        if let Evaluation::Evaluation {
            weight: transmission,
            pdf,
        } = eval_disney_specular_transmission(mat, outgoing, incident, micronormal, transmission_alpha)
        {
            reflectance += transmission_weight * transmission;

            let m_dot_i = micronormal.dot(incident);
            let m_dot_o = micronormal.dot(outgoing);
            let eta = mat.medium_ior / mat.ior;
            let square = |x| x * x;

            forward_pdf += lobe_pdfs.specular_transmission * pdf / square(m_dot_i + eta * m_dot_o);
            // reverse_pdf += lobe_pdfs.specular_transmission * reverse_transmission_pdf / square(m_dot_o + eta * m_dot_i);
        }
    }

    if upper_hemisphere {
        if let Evaluation::Evaluation { weight: specular, pdf } =
            eval_specular_reflection(mat, outgoing, incident, micronormal)
        {
            reflectance += specular;
            forward_pdf += lobe_pdfs.specular_reflection * pdf;
            // TODO: reverse pdf
        }
    }

    Evaluation::Evaluation {
        weight: reflectance,
        pdf: forward_pdf,
    }
}

pub fn bsdf_sample_specular_transmission(mat: &MaterialSample, outgoing: Vector3<f32>) -> Sample {
    let n_dot_o = outgoing.z;

    if n_dot_o <= 0.0 {
        return Sample::Null;
    }

    let alpha = calculate_alpha(mat.roughness);
    let alpha_squared = alpha.x * alpha.x;

    let micronormal = ggx::sample_micronormal(outgoing, alpha);

    let relative_ior = mat.medium_ior / mat.ior;

    let refl = fresnel::dielectric(n_dot_o, mat.medium_ior, mat.ior);
    let fresnel = refl.reflectance();

    let g_o = ggx::smith_shadow_term(n_dot_o, alpha_squared);
    let normal_distrib = ggx::normal_distribution(alpha_squared, micronormal.z);

    // VNDF eq. 3 (Heitz 2018)
    let visible_normal_distrib = g_o * n_dot_o.max(0.0) * normal_distrib / n_dot_o;

    let pdf;
    let incident;
    let weight;

    if thread_rng().gen::<f32>() <= fresnel {
        incident = reflect(outgoing, micronormal);
        let g_i = ggx::smith_shadow_term(incident.z, alpha_squared);
        let n_dot_i = incident.z;

        if n_dot_i <= 0.0 {
            return Sample::Null;
        }

        weight = Spectrumf32::constant(fresnel * g_o * g_i * normal_distrib / (4.0 * n_dot_o * n_dot_i));
        let jacobian = 1.0 / (4.0 * n_dot_o.abs());
        pdf = fresnel * visible_normal_distrib * jacobian;
    } else {
        match refl {
            Reflectance::TotalInternalReflection => unreachable!(),
            Reflectance::Refract {
                cos_theta_transmission, ..
            } => {
                incident = refract(outgoing, micronormal, relative_ior, cos_theta_transmission).normalize();
                let n_dot_i = incident.z;
                let m_dot_i = micronormal.dot(incident);
                let m_dot_o = micronormal.dot(outgoing);
                let g_i = ggx::smith_shadow_term(n_dot_i, alpha_squared);

                let square = |x| x * x;

                let c = m_dot_i.abs() * m_dot_o.abs() / (n_dot_i.abs() * n_dot_o.abs());
                let t = square(mat.medium_ior) / square(mat.ior * m_dot_i + mat.medium_ior * m_dot_o);
                // Walter et al. 2007 eq. 17
                let jacobian = t * m_dot_i.abs();

                weight = (1.0 - fresnel) * c * t * g_i * g_o * normal_distrib * mat.base_color_spectrum.sqrt();
                pdf = (1.0 - fresnel) * visible_normal_distrib * jacobian;
            }
        }
    }

    if incident.z == 0.0 {
        Sample::Null
    } else {
        Sample::Sample { incident, weight, pdf }
    }
}

pub fn bsdf_sample_diffuse_reflection(mat: &MaterialSample, outgoing: Vector3<f32>) -> Sample {
    let sign = outgoing.z.signum();

    let incident = sign * cos_weighted_sample_hemisphere();
    let micronormal = (incident + outgoing).normalize();

    let n_dot_i = incident.z;

    if n_dot_i == 0.0 {
        return Sample::Null;
    }

    //TODO: Diffuse transmission? I suppose the general transmission value is used for both?
    // but then is it still logical to use that value to choose between specular transmission or diffuse?
    let pdf = 1.0;

    let diffuse = eval_disney_diffuse(mat, outgoing, incident, micronormal);

    Sample::Sample {
        incident,
        weight: mat.base_color_spectrum * diffuse / pdf, // + sheen
        pdf: pdf * n_dot_i.abs(),
    }
}

pub fn sample(mat: &MaterialSample, outgoing: Vector3<f32>) -> Sample {
    let pdfs = lobe_pdfs(mat);

    let mut r = thread_rng().gen::<f32>();

    if r < pdfs.specular_reflection {
        let alpha = calculate_alpha(mat.roughness);
        let micronormal = ggx::sample_micronormal(outgoing, alpha);
        let incident = reflect(outgoing, micronormal);
        return eval_specular_reflection(mat, outgoing, incident, micronormal)
            .into_sample_with_incident(incident)
            .multiply_pdf(pdfs.specular_reflection);
    } else {
        r -= pdfs.specular_reflection;
    }

    if r < pdfs.specular_transmission {
        bsdf_sample_specular_transmission(mat, outgoing).multiply_pdf(pdfs.specular_transmission)
    } else {
        bsdf_sample_diffuse_reflection(mat, outgoing).multiply_pdf(pdfs.diffuse)
    }
    // TODO: Clearcoat sampling
}
