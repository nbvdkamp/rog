use std::str::FromStr;

use crate::raytracer::acceleration::Accel;

use super::render_settings::RenderSettings;
use cgmath::vec2;
use clap::{arg, Command};
use image::ImageFormat;

pub struct Args {
    pub scene_file: String,
    pub output_file: String,
    pub render_settings: RenderSettings,
    pub headless: bool,
    pub benchmark: bool,
}

impl Args {
    pub fn parse() -> Self {
        let default_thread_count = (num_cpus::get() - 2).max(1);

        let matches = Command::new("RustRays")
            .version("0.1.0")
            .author("Nathan van der Kamp")
            .about("A pathtracer")
            .args(&[
                arg!(-f --file <FILE> "Path to .gltf or .glb file to render")
                    .default_value("res/simple_raytracer_test.glb")
                    .required(false)
                    .display_order(1),
                arg!(-h --headless "Run without a window").display_order(2),
                arg!(--samples --spp <NUM> "Number of samples per pixel")
                    .default_value("1")
                    .required(false)
                    .display_order(3),
                arg!(--width <NUM> "Image width")
                    .default_value("1920")
                    .required(false)
                    .display_order(4),
                arg!(--height <NUM> "Image height")
                    .default_value("1080")
                    .required(false)
                    .display_order(5),
                arg!(-o --output <FILE> "Path/filename where image should be saved")
                    .default_value("output/result.png")
                    .required(false)
                    .display_order(6),
                arg!(-t --threads <NUM> "Number of threads to use for rendering (default is based on available threads)")
                    .default_value(&format!("{}", default_thread_count))
                    .required(false)
                    .display_order(7),
                arg!(--nodispersion "Disable dispersion")
                    .display_order(8),
                arg!(--alwayssamplewavelength "Sample only one wavelength per path, even if it doesn't encounter any dispersive surfaces")
                    .display_order(9),
                arg!(-a --accel <NUM> "Name of acceleration structure to use (in snake_case)")
                    .required(false),
                arg!(-b --benchmark "Benchmark acceleration structures"),
            ])
            .get_matches();

        let read_usize = |name, default| match matches.value_of(name).unwrap().parse::<usize>() {
            Ok(v) => v,
            Err(_) => {
                println!(
                    "Unable to parse argument {} as an integer, using value {} instead",
                    name, default
                );
                default
            }
        };

        let scene_file: String = matches.value_of("file").unwrap().into();

        if scene_file.contains(' ') {
            println!("Warning: Spaces in filename might cause issues in opening the file.");
        }

        let output_file = matches.value_of("output").unwrap().into();

        if let Err(e) = ImageFormat::from_path(&output_file) {
            eprintln!("Invalid output file specified:\n\t{e}");
            std::process::exit(-1);
        }

        let accel_structure = match matches.value_of("accel") {
            Some(name) => match Accel::from_str(name) {
                Ok(accel) => accel,
                Err(_) => {
                    eprintln!("Wrong acceleration structure name provided, using KD tree instead");
                    Accel::KdTree
                }
            },
            None => Accel::KdTree,
        };

        let render_settings = RenderSettings {
            samples_per_pixel: read_usize("samples", 1),
            image_size: vec2(read_usize("width", 1920), read_usize("height", 1080)),
            thread_count: read_usize("threads", default_thread_count).clamp(1, 2048),
            accel_structure,
            enable_dispersion: !matches.is_present("nodispersion"),
            always_sample_single_wavelength: matches.is_present("alwayssamplewavelength"),
        };

        Args {
            scene_file,
            output_file,
            render_settings,
            headless: matches.is_present("headless"),
            benchmark: matches.is_present("benchmark"),
        }
    }
}
