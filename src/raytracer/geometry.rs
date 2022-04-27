use cgmath::{Vector3, InnerSpace};

pub fn orthogonal_vector(v: Vector3<f32>) -> Vector3<f32> {
    if v.x == 0.0 {
        Vector3::new(0.0, -v.z, v.y)
    } else if v.y == 0.0 {
        Vector3::new(-v.z, 0.0, v.x)
    } else {
        Vector3::new(-v.y, v.x, 0.0)
    }
}

/// normal must be a unit vector
pub fn reflect(v: Vector3<f32>, normal: Vector3<f32>) -> Vector3<f32> {
    2.0 * normal.dot(v) * normal - v
}

#[cfg(test)]
mod tests {
    use cgmath::{assert_abs_diff_eq, vec3};
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

    #[test]
    fn simple_orthogonal_vector() {
        let v = vec3(1.0, 2.0, 3.0);
        let t = orthogonal_vector(v);
        assert_abs_diff_eq!(v.dot(t), 0.0);
    }
}