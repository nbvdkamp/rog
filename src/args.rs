use std::{path::PathBuf, str::FromStr};

use crate::{raytracer::acceleration::Accel, scene_version::SceneVersion};

use super::render_settings::{ImageSettings, RenderSettings, VisibilitySettings};
use clap::{arg, value_parser, Command};
use image::ImageFormat;

pub struct Args {
    pub scene_file: String,
    pub output_file: String,
    pub render_settings: RenderSettings,
    pub image_settings: ImageSettings,
    pub headless: bool,
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
                    .default_value("res/scenes/simple_raytracer_test.glb")
                    .required(false),
                arg!(-H --headless "Run without a window"),
                arg!(--samples --spp <NUM> "Number of samples per pixel")
                    .value_parser(value_parser!(usize))
                    .default_value("1")
                    .required(false),
                arg!(--width <NUM> "Image width")
                    .value_parser(value_parser!(usize))
                    .default_value("1920")
                    .required(false),
                arg!(--height <NUM> "Image height")
                    .value_parser(value_parser!(usize))
                    .default_value("1080")
                    .required(false),
                arg!(-o --output <FILE> "Path/filename where image should be saved")
                    .default_value("output/result.png")
                    .required(false),
                arg!(-t --threads <NUM> "Number of threads to use for rendering (default is based on available threads)")
                    .value_parser(value_parser!(usize))
                    .default_value(format!("{}", default_thread_count))
                    .required(false),
                arg!(--"no-dispersion" "Disable dispersion"),
                arg!(--"always-sample-wavelength" "Sample only one wavelength per path, even if it doesn't encounter any dispersive surfaces"),
                arg!(-a --accel <NAME> "Name of acceleration structure to use (in snake_case)")
                    .required(false),
                arg!(-w --"write-intermediate" <FILE> "Write intermediate image to file to be able to continue rendering later")
                    .value_parser(value_parser!(PathBuf))
                    .required(false),
                arg!(-r --"read-intermediate" <FILE> "Read from intermediate image file to resume rendering")
                    .value_parser(value_parser!(PathBuf))
                    .required(false),
                arg!(-v --visibility "Sample visibility data for the scene and use it for importance sampling"),
                arg!(--"visibility-debug" "Write computed visibility related data to disk for debugging"),
            ])
            .get_matches();

        let read_usize = |name, default| {
            if let Some(&v) = matches.get_one::<usize>(name) {
                v
            } else {
                println!(
                    "Unable to parse argument {} as an integer, using value {} instead",
                    name, default
                );
                default
            }
        };

        let scene_file = matches.get_one::<String>("file").expect("defaulted").clone();
        let output_file = matches.get_one::<String>("output").expect("defaulted").clone();

        if let Err(e) = ImageFormat::from_path(&output_file) {
            eprintln!("Invalid output file specified:\n\t{e}");
            std::process::exit(-1);
        }

        let accel_structure = if let Some(name) = matches.get_one::<String>("accel") {
            if let Ok(accel) = Accel::from_str(name) {
                accel
            } else {
                eprintln!("Can't parse \"{name}\" as an acceleration structure name, using KD tree instead");
                Accel::KdTree
            }
        } else {
            Accel::KdTree
        };

        let use_visibility = matches.get_flag("visibility");

        let dump_debug_data = if matches.get_flag("visibility-debug") && !use_visibility {
            eprintln!("Can't dump visibility data if using visibility data is not enabled");
            std::process::exit(-1);
        } else {
            matches.get_flag("visibility-debug")
        };

        let render_settings = RenderSettings {
            samples_per_pixel: read_usize("samples", 1),
            thread_count: read_usize("threads", default_thread_count).clamp(1, 2048),
            accel_structure,
            intermediate_read_path: matches.get_one::<PathBuf>("read-intermediate").map(|p| p.clone()),
            intermediate_write_path: matches.get_one::<PathBuf>("write-intermediate").map(|p| p.clone()),
        };

        let scene_version = match SceneVersion::new(scene_file.clone()) {
            Ok(scene_version) => Some(scene_version),
            Err(e) => {
                eprintln!("Can't read scene file: {e}");
                std::process::exit(-1);
            }
        };

        let image_settings = ImageSettings {
            width: read_usize("width", 1920),
            height: read_usize("height", 1080),
            enable_dispersion: !matches.get_flag("no-dispersion"),
            always_sample_single_wavelength: matches.get_flag("always-sample-wavelength"),
            visibility: if use_visibility {
                Some(VisibilitySettings { dump_debug_data })
            } else {
                None
            },
            scene_version,
        };

        Args {
            scene_file,
            output_file,
            render_settings,
            image_settings,
            headless: matches.get_flag("headless"),
        }
    }
}
