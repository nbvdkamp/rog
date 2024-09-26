use rand::{rngs::SmallRng, RngCore, SeedableRng};
use std::{
    cell::UnsafeCell,
    hash::{DefaultHasher, Hash, Hasher},
    rc::Rc,
};

use crate::raytracer::PixelSample;

#[derive(Clone, Debug)]
pub struct SmallThreadRng {
    rng: Rc<UnsafeCell<SmallRng>>,
}

thread_local!(
    static THREAD_RNG_KEY: Rc<UnsafeCell<SmallRng>> = {
        let rng = SmallRng::seed_from_u64(0);
        Rc::new(UnsafeCell::new(rng))
    }
);

#[inline(always)]
pub fn thread_rng() -> SmallThreadRng {
    let rng = THREAD_RNG_KEY.with(|t| t.clone());
    SmallThreadRng { rng }
}

pub fn seed_thread_rng_for_path(PixelSample { pixel_pos, sample }: PixelSample) {
    let mut hasher = DefaultHasher::new();
    pixel_pos.hash(&mut hasher);
    hasher.write_usize(sample);
    seed_thread_rng(hasher.finish());
}

pub fn seed_thread_rng(seed: u64) {
    let rng = SmallRng::seed_from_u64(seed);
    THREAD_RNG_KEY.with(|t| {
        unsafe { *t.get() = rng };
    });
}

impl RngCore for SmallThreadRng {
    #[inline(always)]
    fn next_u32(&mut self) -> u32 {
        let rng = unsafe { &mut *self.rng.get() };
        rng.next_u32()
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let rng = unsafe { &mut *self.rng.get() };
        rng.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let rng = unsafe { &mut *self.rng.get() };
        rng.fill_bytes(dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        let rng = unsafe { &mut *self.rng.get() };
        rng.try_fill_bytes(dest)
    }
}
