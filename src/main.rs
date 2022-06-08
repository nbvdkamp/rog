#![feature(test)]
#![feature(const_fn_floating_point_arithmetic)]
#[macro_use]
extern crate impl_ops;

use std::{time::Duration, vec};

use cgmath::{vec2, Vector2};

mod args;
mod camera;
mod cie_data;
mod color;
mod constants;
mod environment;
mod light;
mod material;
mod mesh;
mod preview;
mod raytracer;
mod scene;
mod spectrum;
mod texture;
mod util;

use args::Args;
use preview::app::App;
use raytracer::Raytracer;
use scene::Scene;
use util::{convert_spectrum_buffer_to_rgb, save_image};

pub const ACCEL_INDEX: usize = 2;

fn main() {
    let args = Args::parse();
    // FIXME: Verify scene aspect ratio with given image size

    if args.benchmark {
        accel_benchmark();
    } else if args.headless {
        headless_render(args);
    } else {
        let mut app = App::new(args);
        app.run();
    }
}

fn accel_benchmark() {
    let test_scene_filenames = vec![
        "simple_raytracer_test",
        "sea_test",
        "sea_test_obscured",
        "cube",
        "simplest",
    ];

    #[cfg(not(feature = "stats"))]
    println!("Build with --features stats for traversal statistics");

    let resolution_factor = 15;
    let image_size = Vector2::new(16 * resolution_factor, 9 * resolution_factor);
    let samples = 1;

    for path in test_scene_filenames {
        let scene = Scene::load(format!("res/{}.glb", path)).unwrap();
        let raytracer = Raytracer::new(&scene);

        println!("\nFilename: {}, tris: {}", path, raytracer.get_num_tris());
        #[cfg(feature = "stats")]
        println!(
            "{: <25} | {: <10} | {: <10} | {: <10} | {: <10}",
            "Acceleration structure", "Time (s)", "Nodes/ray", "Tests/ray", "Hits/test"
        );
        #[cfg(not(feature = "stats"))]
        println!("{: <25} | {: <10}", "Acceleration structure", "Time (s)");

        for i in 0..raytracer.accel_structures.len() {
            let (_, time_elapsed) = raytracer.render(image_size, samples, i, None);

            #[cfg(feature = "stats")]
            {
                let stats = raytracer.accel_structures[i].get_statistics();
                let traversals_per_ray = stats.inner_node_traversals as f32 / stats.rays as f32;
                let tests_per_ray = stats.intersection_tests as f32 / stats.rays as f32;
                let hits_per_test = stats.intersection_hits as f32 / stats.intersection_tests as f32;
                println!(
                    "{: <25} | {: <10} | {: <10} | {: <10} | {: <10}",
                    raytracer.accel_structures[i].get_name(),
                    time_elapsed,
                    traversals_per_ray,
                    tests_per_ray,
                    hits_per_test
                );
            }
            #[cfg(not(feature = "stats"))]
            println!(
                "{: <25} | {: <10}",
                raytracer.accel_structures[i].get_name(),
                time_elapsed
            );
        }
    }
}

fn headless_render(args: Args) {
    let scene = match Scene::load(args.file) {
        Ok(scene) => scene,
        Err(message) => {
            eprintln!("{}", message);
            std::process::exit(-1);
        }
    };

    let raytracer = Raytracer::new(&scene);
    let image_size = vec2(args.width, args.height);

    let report_progress = |completed, total, seconds_per_tile| {
        let time_remaining = (total - completed) as f32 * seconds_per_tile;
        println!("Completed {completed}/{total} tiles. Approximately {time_remaining} seconds remaining");
    };

    let progress = Some(raytracer::RenderProgress {
        report_interval: Duration::from_secs(5),
        report: Box::new(report_progress),
    });

    let (buffer, time_elapsed) = raytracer.render(image_size, args.samples, ACCEL_INDEX, progress);
    println!("Finished rendering in {} seconds", time_elapsed);

    let buffer = convert_spectrum_buffer_to_rgb(buffer);
    save_image(&buffer, image_size, args.output_file);
}
