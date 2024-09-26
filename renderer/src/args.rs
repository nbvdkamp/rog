use std::{fmt::Display, path::PathBuf, str::FromStr, time::Duration};

use cgmath::vec2;
use clap::{arg, value_parser, Command};
use image::ImageFormat;

use crate::{
    raytracer::{acceleration::Accel, PixelSample},
    render_settings::{ImageSettings, RenderSettings, TerminationCondition},
    scene_version::SceneVersion,
};

pub struct Args {
    pub scene_file: PathBuf,
    pub output_file: PathBuf,
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
                    .value_parser(value_parser!(PathBuf))
                    .required(false),
                arg!(-H --headless "Run without a window"),
                arg!(-s --samples <NUM> "Number of samples per pixel")
                    .value_parser(value_parser!(usize))
                    .required(false),
                arg!(--time <NUM> "Number of seconds to render for")
                    .value_parser(value_parser!(u64))
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
                    .value_parser(value_parser!(PathBuf))
                    .required(false),
                arg!(-t --threads <NUM> "Number of threads to use for rendering (default is based on available threads)")
                    .value_parser(value_parser!(usize))
                    .default_value(format!("{default_thread_count}"))
                    .required(false),
                arg!(--"max-bounces" <NUM> "Maximum bounces per path, no hard limit if omitted")
                    .value_parser(value_parser!(usize))
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
                arg!(--"debug-render-single-path" <VALUE> "Specify pixel coordinates and sample number as 3 comma separated numbers")
                    .required(false),
            ])
            .get_matches();

        let read_usize = |name, default| {
            if let Some(&v) = matches.get_one::<usize>(name) {
                v
            } else {
                println!("Unable to parse argument {name} as an integer, using value {default} instead");
                default
            }
        };

        let termination_condition = {
            let samples = matches.get_one::<usize>("samples");
            let time = matches.get_one::<u64>("time");

            match (samples, time) {
                (Some(_), Some(_)) => error("Termination conditions are mutually exlusive."),
                (Some(samples), None) => TerminationCondition::SampleCount(*samples),
                (None, Some(seconds)) => TerminationCondition::Time(Duration::from_secs(*seconds)),
                (None, None) => TerminationCondition::SampleCount(1),
            }
        };

        let scene_file = matches.get_one::<PathBuf>("file").expect("defaulted").clone();
        let output_file = matches.get_one::<PathBuf>("output").expect("defaulted").clone();

        if let Err(e) = ImageFormat::from_path(&output_file) {
            error(format!("Invalid output file specified:\n\t{e}"))
        }

        let accel_structure = if let Some(name) = matches.get_one::<String>("accel") {
            if let Ok(accel) = Accel::from_str(name) {
                accel
            } else {
                eprintln!("Can't parse \"{name}\" as an acceleration structure name, using recursive BVH instead");
                Accel::BvhRecursive
            }
        } else {
            Accel::BvhRecursive
        };

        let headless = matches.get_flag("headless");
        let intermediate_read_path = matches.get_one::<PathBuf>("read-intermediate").cloned();
        let intermediate_write_path = matches.get_one::<PathBuf>("write-intermediate").cloned();

        if let Some(path) = &intermediate_write_path {
            let parent = path.parent().unwrap();

            if !parent.exists() {
                error(format!("Directory {} doesn't exist", parent.display()))
            }
        }

        if !headless && intermediate_read_path.is_some() {
            error("Resuming rendering currently only works in --headless mode")
        }

        let scene_version = match SceneVersion::new(scene_file.clone()) {
            Ok(scene_version) => Some(scene_version),
            Err(e) => error(format!("Can't read scene file: {e}")),
        };

        let image_settings = ImageSettings {
            size: vec2(read_usize("width", 1920), read_usize("height", 1080)),
            enable_dispersion: !matches.get_flag("no-dispersion"),
            always_sample_single_wavelength: matches.get_flag("always-sample-wavelength"),
            scene_version,
            max_depth: matches.get_one::<usize>("max-bounces").copied(),
        };

        let debug_render_single_path = matches.get_one::<String>("debug-render-single-path").map(|s| {
            if !headless {
                error("debug-render-single-path is only available in headless mode")
            }

            let mut s = s.split(',');
            let c = s.clone().count();

            if c != 3 {
                error("debug-render-single-path requires three comma separated numbers: x,y,sample")
            }

            let p = |x: &str| match x.parse::<usize>() {
                Ok(v) => v,
                Err(e) => error(format!("Can't parse \"{x}\" as usize, {e}")),
            };

            let size = image_settings.size;
            let x = p(s.next().unwrap());
            if x >= size.x {
                error(format!(
                    "Pixel x coordinate {x} should be smaller than image width {}",
                    size.x
                ))
            }
            let y = p(s.next().unwrap());
            if y >= size.y {
                error(format!(
                    "Pixel y coordinate {y} should be smaller than image height {}",
                    size.y
                ))
            }
            let sample = p(s.next().unwrap());

            PixelSample {
                pixel_pos: vec2(x, y),
                sample,
            }
        });

        let render_settings = RenderSettings {
            termination_condition,
            thread_count: read_usize("threads", default_thread_count).clamp(1, 2048),
            accel_structure,
            intermediate_read_path,
            intermediate_write_path,
            debug_render_single_path,
        };

        Args {
            scene_file,
            output_file,
            render_settings,
            image_settings,
            headless,
        }
    }
}

fn error(msg: impl Display) -> ! {
    eprintln!("Argument error: {msg}");
    std::process::exit(-1)
}
