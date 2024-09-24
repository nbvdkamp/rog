#![feature(test)]
#![feature(iter_partition_in_place)]
#![feature(allocator_api)]
#![feature(let_chains)]
#[macro_use]
extern crate impl_ops;
extern crate static_assertions;

pub mod args;
pub(crate) mod barycentric;
pub mod camera;
pub(crate) mod cie_data;
pub mod color;
pub(crate) mod environment;
pub(crate) mod light;
pub(crate) mod material;
pub mod mesh;
pub mod raytracer;
pub mod render_settings;
pub mod scene;
pub(crate) mod scene_version;
pub(crate) mod small_thread_rng;
pub(crate) mod spectrum;
pub mod texture;
pub mod util;
