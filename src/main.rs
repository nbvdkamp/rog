#![feature(test)]
#![feature(const_fn_floating_point_arithmetic)]
#[macro_use]
extern crate impl_ops;

use std::{io::Write, time::Duration, vec};

use cgmath::Vector2;

mod args;
mod camera;
mod cie_data;
mod color;
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
    let thread_count = (num_cpus::get() - 2).max(1);

    for path in test_scene_filenames {
        let scene = Scene::load(format!("res/{path}.glb")).unwrap();
        let raytracer = Raytracer::new(&scene);

        println!("Filename: {}, tris: {}", path, raytracer.get_num_tris());
        #[cfg(feature = "stats")]
        println!(
            "{: <25} | {: <10} | {}",
            "Acceleration structure",
            "Time (s)",
            raytracer::acceleration::statistics::Statistics::format_header()
        );
        #[cfg(not(feature = "stats"))]
        println!("{: <25} | {: <10}", "Acceleration structure", "Time (s)");

        for i in 0..raytracer.accel_structures.len() {
            let (_, time_elapsed) = raytracer.render(image_size, samples, thread_count, i, None);
            let name = raytracer.accel_structures[i].get_name();

            let _stats = raytracer.accel_structures[i].get_statistics();
            #[cfg(feature = "stats")]
            println!("{name: <25} | {time_elapsed: <10} | {_stats}");
            #[cfg(not(feature = "stats"))]
            println!("{name: <25} | {time_elapsed: <10}");
        }
        println!();
    }
}

fn headless_render(args: Args) {
    let scene = match Scene::load(args.file) {
        Ok(scene) => scene,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(-1);
        }
    };

    let raytracer = Raytracer::new(&scene);

    let report_progress = |completed, total, seconds_per_tile| {
        let time_remaining = (total - completed) as f32 * seconds_per_tile;
        print!("\r\x1b[2K Completed {completed}/{total} tiles. Approximately {time_remaining:.2} seconds remaining");
        std::io::stdout().flush().unwrap();
    };

    let progress = Some(raytracer::RenderProgress {
        report_interval: Duration::from_secs(3),
        report: Box::new(report_progress),
    });

    let (buffer, time_elapsed) =
        raytracer.render(args.image_size, args.samples, args.thread_count, ACCEL_INDEX, progress);
    println!("\r\x1b[2KFinished rendering in {time_elapsed} seconds");

    let buffer = convert_spectrum_buffer_to_rgb(buffer);
    save_image(&buffer, args.image_size, args.output_file);
}
