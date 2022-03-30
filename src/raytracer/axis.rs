#[derive(Clone, Copy)]
pub enum Axis {
    X, Y, Z
}

impl Axis {
    pub fn index(&self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }
}