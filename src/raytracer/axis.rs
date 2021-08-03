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

    pub fn from_index(index: usize) -> Self {
        match index {
            0 => Axis::X,
            1 => Axis::Y,
            2 => Axis::Z,
            _ => { panic!("Invalid axis index"); }
        }
    }
}