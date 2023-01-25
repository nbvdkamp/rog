#![feature(test)]
#![feature(const_fn_floating_point_arithmetic)]
#![feature(iter_partition_in_place)]
#![feature(allocator_api)]
#![feature(let_chains)]
#[macro_use]
extern crate impl_ops;
extern crate static_assertions;

pub mod args;
pub mod camera;
pub mod cie_data;
pub mod color;
pub mod environment;
pub mod light;
pub mod material;
pub mod mesh;
pub mod preview;
pub mod raytracer;
pub mod render_settings;
pub mod scene;
pub mod scene_version;
pub mod spectrum;
pub mod texture;
pub mod util;
