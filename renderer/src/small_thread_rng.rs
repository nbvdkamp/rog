use core::convert::Infallible;
use rand::{Rng, SeedableRng, TryRng, rngs::SmallRng};
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

impl TryRng for SmallThreadRng {
    type Error = Infallible;

    #[inline(always)]
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        let rng = unsafe { &mut *self.rng.get() };
        Ok(rng.next_u32())
    }

    #[inline(always)]
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let rng = unsafe { &mut *self.rng.get() };
        Ok(rng.next_u64())
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        let rng = unsafe { &mut *self.rng.get() };
        Ok(rng.fill_bytes(dst))
    }
}
