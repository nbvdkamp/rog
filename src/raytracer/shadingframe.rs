use cgmath::{Vector3, vec3, Matrix3, InnerSpace};
use super::geometry::orthogonal_vector;

pub struct ShadingFrame {
    normal: Vector3<f32>,
    tangent: Vector3<f32>,
    bitangent: Vector3<f32>,
}

impl ShadingFrame {
    pub fn new(normal: Vector3<f32>) -> Self {
        let tangent = orthogonal_vector(normal).normalize();
        let bitangent = normal.cross(tangent);

        ShadingFrame { normal, tangent, bitangent }
    }

    pub fn new_with_tangent(normal: Vector3<f32>, tangent: Vector3<f32>) -> Self {
        // Reorthogonalize with Gram-Schmidt
        let tangent = (tangent - tangent.dot(normal) * normal).normalize();
        let bitangent = normal.cross(tangent);

        ShadingFrame { normal, tangent, bitangent }
    }

    pub fn to_local(&self, global: Vector3<f32>) -> Vector3<f32> {
        vec3(self.tangent.dot(global), self.bitangent.dot(global), self.normal.dot(global))
    }

    pub fn to_global(&self, local: Vector3<f32>) -> Vector3<f32> {
        Matrix3::from_cols(self.tangent, self.bitangent, self.normal) * local 
    }
}

#[cfg(test)]
mod tests {
    use cgmath::assert_abs_diff_eq;
    use super::*;

    #[test]
    fn dot() {
        let frame = ShadingFrame::new(vec3(0.0, -1.0, 1.0).normalize());
        let a = vec3(1.0, 0.0, 0.0);
        let b = vec3(0.0, 1.0, 0.0);
        let a_local = frame.to_local(a);
        let b_local = frame.to_local(b);
        assert_abs_diff_eq!(a.dot(b), a_local.dot(b_local));
    }

    #[test]
    fn normal() {
        let frame = ShadingFrame::new(vec3(0.0, 0.0, 1.0));
        let global = vec3(1.0, 0.0, 0.0);
        let local = frame.to_local(global);
        assert_abs_diff_eq!(global, frame.to_global(local));
    }

    macro_rules! frame_tests {
        ($($name:ident: $value:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    let frame = ShadingFrame::new(vec3(0.0, -1.0, 1.0).normalize());
                    let global = $value;
                    let local = frame.to_local(global);
                    assert_abs_diff_eq!(global.magnitude(), 1.0);
                    assert_abs_diff_eq!(local.magnitude(), 1.0);
                    assert_abs_diff_eq!(global, frame.to_global(local));
                }
            )*
        };
    }

    frame_tests! {
        frame_test0: vec3(1.0, 0.0, 0.0),
        frame_test1: vec3(0.0, 1.0, 0.0),
        frame_test2: vec3(0.0, 0.0, 1.0),
        frame_test4: vec3(1.0, 1.0, 1.0).normalize(),
    }
}