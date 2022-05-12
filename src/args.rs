use clap::{arg, Command};
use image::ImageFormat;

pub struct Args {
    pub file: String,
    pub output_file: String,
    pub headless: bool,
    pub samples: usize,
    pub width: usize,
    pub height: usize,
    pub benchmark: bool,
}

impl Args {
    pub fn parse() -> Self {
        let matches = Command::new("RustRays")
            .version("0.1.0")
            .author("Nathan van der Kamp")
            .about("A pathtracer")
            .args(&[
                arg!(-f --file <FILE> "Path to .gltf or .glb file to render")
                    .default_value("res/simple_raytracer_test.glb")
                    .required(false)
                    .display_order(1),
                arg!(-h --headless "Run without a window")
                    .display_order(2),
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
                arg!(-b --benchmark "Benchmark acceleration structures"),
            ]).get_matches();

        let read_usize = |name, default| {
            let opt = matches.value_of(name).unwrap().parse::<usize>();

            match opt {
                Ok(v) => v,
                Err(_) => {
                    println!("Unable to parse argument {} as an integer, using value {} instead", name, default);
                    default
                }
            }
        };

        let input_file: String = matches.value_of("file").unwrap().into();

        if input_file.contains(" ") {
            println!("Warning: Spaces in filename might cause issues in opening the file.");
        }

        let output_file = matches.value_of("output").unwrap().into();

        match ImageFormat::from_path(&output_file) {
            Err(e) => {
                eprintln!("Invalid output file specified:\n\t{e}");
                std::process::exit(-1);
            }
            Ok(_) => (),
        }

        Args {
            file: input_file,
            headless: matches.is_present("headless"),
            samples: read_usize("samples", 1),
            width: read_usize("width", 1920),
            height: read_usize("height", 1080),
            output_file,
            benchmark: matches.is_present("benchmark"),
        }
    }
}