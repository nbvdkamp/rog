use std::ops::{self, AddAssign};

use luminance::shader::types::Vec4;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct RGBf32 {
    r: f32,
    g: f32,
    b: f32,
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