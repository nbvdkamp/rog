use std::{f32::consts::TAU, ops};

pub struct SingleChannelImage {
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

impl SingleChannelImage {
    pub fn mean(&self) -> f32 {
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }

    pub fn assert_size_matches(&self, other: &Self) {
        assert_eq!(self.width, other.width);
        assert_eq!(self.height, other.height);
    }

    pub fn extent(&self) -> (f32, f32) {
        let mut min = self.data[0];
        let mut max = self.data[0];

        for &x in &self.data {
            min = min.min(x);
            max = max.max(x);
        }

        (min, max)
    }

    pub fn gaussian_blurred(&self, kernel_radius: i64) -> Self {
        let gaussian_1d: Vec<f32> = (-kernel_radius..=kernel_radius)
            .map(|i| gaussian(i as f32, 1.5))
            .collect();

        let mut x_blurred = Vec::new();
        x_blurred.reserve(self.data.len());

        for y in 0..self.height {
            for x in 0..self.width {
                let mut v = 0.0;

                for i in 0..gaussian_1d.len() {
                    let x = (x as i64 + i as i64 - 5).clamp(0, self.width as i64 - 1) as usize;
                    v += gaussian_1d[i] * self.data[y * self.width + x];
                }

                x_blurred.push(v);
            }
        }

        let mut xy_blurred = Vec::new();
        xy_blurred.reserve(self.data.len());

        for y in 0..self.height {
            for x in 0..self.width {
                let mut v = 0.0;

                for i in 0..gaussian_1d.len() {
                    let y = (y as i64 + i as i64 - 5).clamp(0, self.height as i64 - 1) as usize;
                    v += gaussian_1d[i] * x_blurred[y * self.width + x];
                }

                xy_blurred.push(v);
            }
        }

        Self {
            data: xy_blurred,
            ..*self
        }
    }
}

impl_op_ex!(+ |a: &SingleChannelImage, b: &SingleChannelImage| -> SingleChannelImage {
    a.assert_size_matches(b);
    let data = a.data.iter().zip(b.data.iter()).map(|(a, b)| a + b).collect();
    SingleChannelImage { data, ..*a }
});

impl_op_ex!(
    -|a: &SingleChannelImage, b: &SingleChannelImage| -> SingleChannelImage {
        a.assert_size_matches(b);
        let data = a.data.iter().zip(b.data.iter()).map(|(a, b)| a - b).collect();
        SingleChannelImage { data, ..*a }
    }
);

impl_op_ex!(
    *|a: &SingleChannelImage, b: &SingleChannelImage| -> SingleChannelImage {
        a.assert_size_matches(b);
        let data = a.data.iter().zip(b.data.iter()).map(|(a, b)| a * b).collect();
        SingleChannelImage { data, ..*a }
    }
);

impl_op_ex!(
    /|a: &SingleChannelImage, b: &SingleChannelImage| -> SingleChannelImage {
        a.assert_size_matches(b);
        let data = a.data.iter().zip(b.data.iter()).map(|(a, b)| a / b).collect();
        SingleChannelImage { data, ..*a }
    }
);

impl_op_ex_commutative!(+ |a: &SingleChannelImage, s: f32| -> SingleChannelImage {
    let data = a.data.iter().map(|a| a + s).collect();
    SingleChannelImage { data, ..*a }
});

impl_op_ex_commutative!(*|a: &SingleChannelImage, s: f32| -> SingleChannelImage {
    let data = a.data.iter().map(|a| a * s).collect();
    SingleChannelImage { data, ..*a }
});

fn square(x: f32) -> f32 {
    x * x
}

fn gaussian(x: f32, standard_deviation: f32) -> f32 {
    1.0 / (standard_deviation * TAU.sqrt()) * (-square(x) / (2.0 * square(standard_deviation))).exp()
}
