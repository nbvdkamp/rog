use std::ops::{self, AddAssign};

use luminance::shader::types::Vec4;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct RGBf32 {
    r: f32,
    g: f32,
    b: f32,
}

impl RGBf32 {
    #[inline]
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self {r, g, b}
    }
    
    pub fn r_normalized(&self) -> u8 {
        (self.r * 255.0) as u8
    }

    pub fn g_normalized(&self) -> u8 {
        (self.g * 255.0) as u8
    }

    pub fn b_normalized(&self) -> u8 {
        (self.b * 255.0) as u8
    }
}

impl_op_ex!(+ |a: &RGBf32, b: &RGBf32| -> RGBf32 { RGBf32::new(a.r + b.r, a.g + b.g, a.b + b.b)} );
impl_op_ex!(- |a: &RGBf32, b: &RGBf32| -> RGBf32 { RGBf32::new(a.r - b.r, a.g - b.g, a.b - b.b)} );
impl_op_ex!(* |a: &RGBf32, b: &RGBf32| -> RGBf32 { RGBf32::new(a.r * b.r, a.g * b.g, a.b * b.b)} );
impl_op_ex!(/ |a: &RGBf32, b: &RGBf32| -> RGBf32 { RGBf32::new(a.r / b.r, a.g / b.g, a.b / b.b)} );

impl_op_commutative!(* |a: RGBf32, b: f32| -> RGBf32 { RGBf32::new(a.r * b, a.g * b, a.b * b)} );
impl_op_commutative!(/ |a: RGBf32, b: f32| -> RGBf32 { RGBf32::new(a.r / b, a.g / b, a.b / b)} );

impl AddAssign for RGBf32 {
    fn add_assign(&mut self, other: Self) {
        self.r += other.r;
        self.g += other.g;
        self.b += other.b;
    }
}

impl From<RGBf32> for Vec4<f32> {
    #[inline]
    fn from(v: RGBf32) -> Self {
        Vec4::new(v.r, v.g, v.b, 1.0)
    }
}

impl From<[f32; 3]> for RGBf32 {
    #[inline]
    fn from(v: [f32; 3]) -> Self {
        RGBf32::new(v[0], v[1], v[2])
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let c = RGBf32::new(0.1, 0.2, 0.5);
        let c2 = RGBf32::new(0.1, 0.4, 0.5);
        assert_eq!(c + c2, RGBf32::new(0.2, 0.6, 1.0));
    }

    #[test]
    fn mul_scalar() {
        let c = RGBf32::new(0.1, 0.2, 0.5);
        assert_eq!(c * 2.0, RGBf32::new(0.2, 0.4, 1.0));
    }

    #[test]
    fn add_assign() {
        let mut c = RGBf32::new(0.1, 0.2, 0.5);
        c += RGBf32::new(0.1, 0.1, 0.1);
        assert_eq!(c * 2.0, RGBf32::new(0.2, 0.4, 1.0));
    }
}