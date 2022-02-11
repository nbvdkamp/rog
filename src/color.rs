use cgmath::Vector4;

pub type Color = Vector4<f32>;

pub trait ColorNormalizable {
    fn r_normalized(&self) -> u8;
    fn g_normalized(&self) -> u8;
    fn b_normalized(&self) -> u8;
    fn a_normalized(&self) -> u8;
}

impl ColorNormalizable for Color {
    fn r_normalized(&self) -> u8 {
        (self.x * 255.0) as u8
    }
    fn g_normalized(&self) -> u8 {
        (self.y * 255.0) as u8
    }
    fn b_normalized(&self) -> u8 {
        (self.z * 255.0) as u8
    }
    fn a_normalized(&self) -> u8 {
        (self.w * 255.0) as u8
    }
}