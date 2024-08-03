use std::ops::{self, AddAssign, DivAssign, MulAssign};

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
pub struct RGBf32 {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct RGBu8 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct RGBAf32 {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct XYZf32 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

macro_rules! count {
    () => (0usize);
    ( $x:tt $($xs:tt)* ) => (1usize + count!($($xs)*));
}

macro_rules! impl_f32_color_tuple {
    ($name:ident { $($field:ident),+ }) => {
        impl $name {
            #[inline]
            pub fn new($($field: f32),+) -> Self {
                Self { $($field),+ }
            }

            pub fn from_hex(hex: &str) -> Self {
                let h = hex.trim_start_matches("#");
                let mut _i = 0;
                $(let $field = i32::from_str_radix(&h[(_i * 2)..((_i + 1) * 2)], 16).unwrap(); _i += 1;)+
                Self::new($($field as f32 / 255.0),+)
            }

            pub const fn from_grayscale(value: f32) -> Self {
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

            pub fn max_component(&self) -> f32 {
                let mut r = f32::MIN;

                $(r = r.max(self.$field);)+
                r
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

        impl DivAssign<f32> for $name {
            fn div_assign(&mut self, s: f32) {
                $(self.$field /= s);+
            }
        }

        impl MulAssign<Self> for $name {
            fn mul_assign(&mut self, other: Self) {
                $(self.$field *= other.$field);+
            }
        }

        impl From<[f32; count!($($field)*)]> for $name {
            #[inline]
            fn from(v: [f32; count!($($field)*)]) -> Self {
                let mut _i = 0;
                $(let $field = v[_i]; _i += 1;)+
                Self { $($field),+ }
            }
        }

        impl From<$name> for [f32; count!($($field)*)] {
            #[inline]
            fn from(v: $name) -> Self {
                [$(v.$field),+]
            }
        }
    };
}

macro_rules! impl_f32_color_tuple_normalized {
    ($name:ident { $($field:ident),+ }, $normalized_type:ident) => {
        impl $name {
            pub fn normalized(&self) -> $normalized_type {
                $normalized_type {
                    $($field: (self.$field * 255.0) as u8),+
                }
            }
        }
    };
}

macro_rules! impl_u8_color_tuple {
    ($name:ident { $($field:ident),+ }) => {
        impl $name {
            #[inline]
            pub fn new($($field: u8),+) -> Self {
                Self { $($field),+ }
            }

            pub fn from_grayscale(value: u8) -> Self {
                Self { $($field: value),+ }
            }
        }

        impl_op_ex!(+ |a: &$name, b: &$name| -> $name { $name::new($(a.$field + b.$field),+)} );
        impl_op_ex!(- |a: &$name, b: &$name| -> $name { $name::new($(a.$field - b.$field),+)} );

        impl AddAssign for $name {
            fn add_assign(&mut self, other: Self) {
                $(self.$field += other.$field);+
            }
        }
    };
}

impl_f32_color_tuple!(RGBf32 { r, g, b });
impl_f32_color_tuple_normalized!(RGBf32 { r, g, b }, RGBu8);
impl_f32_color_tuple!(RGBAf32 { r, g, b, a });
impl_u8_color_tuple!(RGBu8 { r, g, b });

impl_f32_color_tuple!(XYZf32 { x, y, z });

#[rustfmt::skip]
pub const SRGB8_TO_LINEAR_TABLE: [f32; 256] = [
    0.0, 0.000303527, 0.000607054, 0.000910581, 0.001214108, 0.001517635, 0.001821162, 0.0021246888,
    0.002428216, 0.002731743, 0.00303527, 0.0033465356, 0.003676507, 0.004024717, 0.004391442, 0.0047769533,
    0.005181517, 0.0056053917, 0.0060488326, 0.006512091, 0.00699541, 0.0074990317, 0.008023192, 0.008568125,
    0.009134057, 0.009721218, 0.010329823, 0.010960094, 0.011612245, 0.012286487, 0.012983031, 0.013702081,
    0.014443844, 0.015208514, 0.015996292, 0.016807375, 0.017641952, 0.018500218, 0.019382361, 0.020288562,
    0.02121901, 0.022173883, 0.023153365, 0.02415763, 0.025186857, 0.026241222, 0.027320892, 0.028426038,
    0.029556833, 0.03071344, 0.03189603, 0.033104762, 0.034339808, 0.035601314, 0.036889445, 0.038204364,
    0.039546236, 0.0409152, 0.04231141, 0.043735027, 0.045186203, 0.046665084, 0.048171822, 0.049706563,
    0.051269468, 0.052860655, 0.05448028, 0.056128494, 0.057805434, 0.05951124, 0.06124607, 0.06301003,
    0.06480328, 0.06662595, 0.06847818, 0.07036011, 0.07227186, 0.07421358, 0.07618539, 0.07818743,
    0.08021983, 0.082282715, 0.084376216, 0.086500466, 0.088655606, 0.09084173, 0.09305898, 0.095307484,
    0.09758736, 0.09989874, 0.10224175, 0.10461649, 0.10702311, 0.10946172, 0.111932434, 0.11443538,
    0.11697067, 0.119538434, 0.1221388, 0.12477184, 0.1274377, 0.13013649, 0.13286833, 0.13563335,
    0.13843162, 0.1412633, 0.14412849, 0.14702728, 0.1499598, 0.15292616, 0.15592647, 0.15896086,
    0.1620294, 0.16513222, 0.1682694, 0.1714411, 0.17464739, 0.17788841, 0.18116423, 0.18447499,
    0.18782076, 0.19120167, 0.19461781, 0.1980693, 0.20155624, 0.2050787, 0.20863685, 0.21223073,
    0.21586053, 0.21952623, 0.22322798, 0.22696589, 0.23074007, 0.23455065, 0.23839766, 0.2422812,
    0.2462014, 0.25015837, 0.25415218, 0.2581829, 0.26225072, 0.26635566, 0.27049786, 0.27467737,
    0.27889434, 0.2831488, 0.2874409, 0.2917707, 0.29613832, 0.30054384, 0.30498737, 0.30946895,
    0.31398875, 0.31854683, 0.32314324, 0.32777813, 0.33245158, 0.33716366, 0.34191445, 0.3467041,
    0.3515327, 0.35640025, 0.36130688, 0.3662527, 0.37123778, 0.37626222, 0.3813261, 0.38642952,
    0.39157256, 0.3967553, 0.40197787, 0.4072403, 0.4125427, 0.41788515, 0.42326775, 0.42869055,
    0.4341537, 0.43965724, 0.44520125, 0.45078585, 0.45641106, 0.46207705, 0.46778384, 0.47353154,
    0.47932023, 0.48514998, 0.4910209, 0.49693304, 0.5028866, 0.50888145, 0.5149178, 0.5209957,
    0.5271152, 0.5332765, 0.5394796, 0.5457246, 0.5520115, 0.5583405, 0.56471163, 0.5711249,
    0.5775805, 0.5840785, 0.5906189, 0.5972019, 0.6038274, 0.6104956, 0.61720663, 0.62396044,
    0.6307572, 0.63759696, 0.64447975, 0.6514057, 0.65837485, 0.66538733, 0.6724432, 0.67954254,
    0.68668544, 0.6938719, 0.701102, 0.70837593, 0.71569365, 0.72305524, 0.7304609, 0.73791057,
    0.74540436, 0.7529423, 0.76052463, 0.7681513, 0.77582234, 0.7835379, 0.79129803, 0.79910284,
    0.80695236, 0.8148467, 0.82278585, 0.83076996, 0.8387991, 0.8468733, 0.8549927, 0.8631573,
    0.8713672, 0.87962234, 0.8879232, 0.8962694, 0.90466136, 0.9130987, 0.92158204, 0.9301109,
    0.9386859, 0.9473066, 0.9559735, 0.9646863, 0.9734455, 0.9822506, 0.9911022, 1.0
];

impl RGBf32 {
    pub fn srgb_linear_to_gamma_compressed(&self) -> Self {
        let m = |c: f32| {
            if c <= 0.0031308 {
                if c < 0.0 {
                    0.0
                } else {
                    c * 12.92
                }
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            }
        };

        RGBf32 {
            r: m(self.r),
            g: m(self.g),
            b: m(self.b),
        }
    }

    pub fn srgb_gamma_compressed_to_linear(&self) -> Self {
        let m = |c| {
            if c <= 0.04045 {
                if c < 0.0 {
                    0.0
                } else {
                    c / 12.92
                }
            } else {
                let x: f32 = (c + 0.055) / 1.055;
                x.powf(2.4)
            }
        };

        RGBf32 {
            r: m(self.r),
            g: m(self.g),
            b: m(self.b),
        }
    }
}

impl XYZf32 {
    pub fn to_srgb(self) -> RGBf32 {
        let XYZf32 { x, y, z } = self;
        let r = 3.240479 * x - 1.53715 * y - 0.498535 * z;
        let g = -0.969256 * x + 1.875991 * y + 0.041556 * z;
        let b = 0.055648 * x - 0.204043 * y + 1.057311 * z;

        RGBf32::new(r.max(0.0), g.max(0.0), b.max(0.0))
    }

    /// To avoid operator overloads not being able to be const
    pub const fn mul(&self, scalar: f32) -> Self {
        let XYZf32 { x, y, z } = *self;
        XYZf32 {
            x: x * scalar,
            y: y * scalar,
            z: z * scalar,
        }
    }

    /// To avoid operator overloads not being able to be const
    pub const fn add(&self, other: Self) -> Self {
        let XYZf32 { x, y, z } = *self;
        XYZf32 {
            x: x + other.x,
            y: y + other.y,
            z: z + other.z,
        }
    }
}

impl RGBAf32 {
    pub fn rgb(&self) -> RGBf32 {
        RGBf32::new(self.r, self.g, self.b)
    }
}

pub fn _debug_color(x: f32) -> RGBf32 {
    if (0.0..=1.0).contains(&x) {
        // green
        RGBf32::new(0.0, x, 0.0)
    } else if x > 1.0 && x <= 10.0 {
        // blue
        RGBf32::new(0.0, 0.0, x / 10.0)
    } else if x > 10.0 && x <= 100.0 {
        // red
        RGBf32::new(x / 100.0, 0.0, 0.0)
    } else if x > 100.0 {
        // white
        RGBf32::new(1.0, 1.0, 1.0)
    } else if x < 0.0 {
        // magenta to yellow
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
