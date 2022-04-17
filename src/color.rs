use std::ops::{self, AddAssign, MulAssign};

use luminance::shader::types::Vec4;

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
pub struct RGBf32 {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}


macro_rules! impl_f32_color_tuple {
    ($name:ident { $($field:ident),+ }, { $($field_normalized:ident),+ }, $n:expr) => {
        impl $name {
            #[inline]
            pub fn new($($field: f32),+) -> Self {
                Self { $($field),+ }
            }
            
            $(
            pub fn $field_normalized(&self) -> u8 {
                (self.$field * 255.0) as u8
            })+

            pub fn from_hex(hex: &str) -> Self {
                let h = hex.trim_start_matches("#");
                let mut _i = 0;
                $(let $field = i32::from_str_radix(&h[(_i * 2)..((_i + 1) * 2)], 16).unwrap(); _i += 1;)+
                Self::new($($field as f32 / 255.0),+)
            }

            pub fn from_grayscale(value: f32) -> Self {
                Self { $($field: value),+ }
            }

            pub fn white() -> Self {
                Self::from_grayscale(1.0)
            }

            pub fn has_nan_component(&self) -> bool {
                $(self.$field.is_nan())||+
            }

            pub fn pow(&self, exponent: f32) -> Self {
                Self::new($(self.$field.powf(exponent)),+)
            }
        }

        impl_op_ex!(+ |a: &$name, b: &$name| -> $name { $name::new($(a.$field + b.$field),+)} );
        impl_op_ex!(- |a: &$name, b: &$name| -> $name { $name::new($(a.$field - b.$field),+)} );
        impl_op_ex!(* |a: &$name, b: &$name| -> $name { $name::new($(a.$field * b.$field),+)} );
        impl_op_ex!(/ |a: &$name, b: &$name| -> $name { $name::new($(a.$field / b.$field),+)} );

        impl_op_commutative!(* |a: $name, b: f32| -> $name { $name::new($(a.$field * b),+)} );
        impl_op_commutative!(/ |a: $name, b: f32| -> $name { $name::new($(a.$field / b),+)} );

        impl AddAssign for $name {
            fn add_assign(&mut self, other: Self) {
                $(self.$field += other.$field);+
            }
        }
        
        impl MulAssign<f32> for $name {
            fn mul_assign(&mut self, s: f32) {
                $(self.$field *= s);+
            }
        }
    };
}

impl_f32_color_tuple!(RGBf32 { r, g, b }, { r_normalized, g_normalized, b_normalized }, 3);

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

pub fn _debug_color(x: f32) -> RGBf32 {
    if (0.0..=1.0).contains(&x) { // green
        RGBf32::new(0.0, x, 0.0)
    } else if x > 1.0 &&  x <= 10.0 { // blue
        RGBf32::new(0.0, 0.0, x / 10.0)
    } else if x > 10.0 &&  x <= 100.0 { // red
        RGBf32::new(x / 100.0, 0.0, 0.0)
    } else if x > 100.0 { // white
        RGBf32::new(1.0, 1.0, 1.0)
    } else if x < 0.0 { // magenta to yellow
        RGBf32::new(1.0, -x, 1.0 + x)
    } else {
        RGBf32::new(0.0, 0.0, 0.0)
    }
}

pub fn _vec_debug_color(x: cgmath::Vector3<f32>) -> RGBf32 {
    RGBf32::new(x.x, x.y, x.z)
}

#[cfg(test)]
mod tests {
    use cgmath::assert_abs_diff_eq;

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
        assert_eq!(c, RGBf32::new(0.2, 0.3, 0.6));
    }

    #[test]
    fn from_hex() {
        let c = RGBf32::from_hex("#4080C0");
        assert_abs_diff_eq!(c.r, 0.25, epsilon = 0.01);
        assert_abs_diff_eq!(c.g, 0.5, epsilon = 0.01);
        assert_abs_diff_eq!(c.b, 0.75, epsilon = 0.01);
    }

    #[test]
    fn has_nan_component() {
        let mut c = RGBf32::new(f32::NAN, 0.0, 0.0);
        assert_eq!(c.has_nan_component(), true);
        c.r = 0.0;
        assert_eq!(c.has_nan_component(), false);
    }
}