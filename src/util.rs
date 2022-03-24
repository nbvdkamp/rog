use cgmath::{Vector3, vec3, Point3, point3, Vector4, Matrix3, Matrix4, InnerSpace};
use luminance_front::shader::types::{Mat44};

pub fn elementwise_min(a: Vector3<f32>, b: Vector3<f32>) -> Vector3<f32> {
    Vector3 {
        x: f32::min(a.x, b.x),
        y: f32::min(a.y, b.y),
        z: f32::min(a.z, b.z),
    }
}

pub fn elementwise_max(a: Vector3<f32>, b: Vector3<f32>) -> Vector3<f32> {
    Vector3 {
        x: f32::max(a.x, b.x),
        y: f32::max(a.y, b.y),
        z: f32::max(a.z, b.z),
    }
}

pub fn min_element(v: Vector3<f32>) -> f32 {
    f32::min(v.x, f32::min(v.y, v.z))
}

pub fn max_element(v: Vector3<f32>) -> f32 {
    f32::max(v.x, f32::max(v.y, v.z))
}

pub fn from_homogenous(v: Vector4<f32>) -> Point3<f32> {
    point3(v.x / v.w, v.y / v.w, v.z / v.w)
}

pub fn mat_to_shader_type<T>(m: Matrix4<T>) -> Mat44<T> {
    let x: [[T; 4]; 4] = m.into();
    x.into()
}

pub fn orthogonal_vector(v: Vector3<f32>) -> Vector3<f32> {
    if v.x == 0.0 {
        Vector3::new(0.0, -v.z, v.y)
    } else if v.y == 0.0 {
        Vector3::new(-v.z, 0.0, v.x)
    } else {
        Vector3::new(-v.y, v.x, 0.0)
    }
}

pub fn to_tangent_space(normal: Vector3<f32>, v: Vector3<f32> ) -> Vector3<f32> {
    let tangent = orthogonal_vector(normal).normalize();
    let bitangent = normal.cross(tangent);

    Matrix3::from_cols(tangent, bitangent, normal) * v
}

/// normal must be a unit vector
pub fn reflect(v: Vector3<f32>, normal: Vector3<f32>) -> Vector3<f32> {
    -v + 2.0 * normal.dot(v) * normal
}

pub fn spherical_to_cartesian(theta: f32, phi: f32) -> Vector3<f32> {
    let sin_theta = theta.sin();
    vec3(sin_theta * phi.cos(), sin_theta * phi.sin(), theta.cos())
}

#[cfg(test)]
mod tests {
    use cgmath::assert_abs_diff_eq;
    use super::*;

    #[test]
    fn basic_reflect() {
        let v = vec3(1.0, 1.0, 0.0);
        let n = vec3(0.0, 1.0, 0.0);
        assert_eq!(reflect(v, n), vec3(-1.0, 1.0, 0.0));
    }

    #[test]
    fn diag_reflect() {
        let v = vec3(1.0, 0.0, 0.0);
        let n = vec3(1.0, 1.0, 0.0).normalize();
        assert_abs_diff_eq!(reflect(v, n), vec3(0.0, 1.0, 0.0), epsilon=0.0001);
    }
}