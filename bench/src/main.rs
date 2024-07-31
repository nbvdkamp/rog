use std::path::PathBuf;

use cgmath::{vec2, Angle, EuclideanSpace, InnerSpace, Matrix4, Rad, Vector3};
use clap::{arg, value_parser, Command};

use csv::Writer;
use renderer::{
    raytracer::{aabb::BoundingBox, acceleration::Accel, working_image::WorkingImage, Raytracer},
    render_settings::{ImageSettings, RenderSettings, TerminationCondition},
    scene::Scene,
};

fn main() {
    let matches = Command::new("bench")
        .args(&[
            arg!(--file <FILE>).required(true).value_parser(value_parser!(PathBuf)),
            arg!(--output <FILE>)
                .required(true)
                .value_parser(value_parser!(PathBuf)),
            arg!(--image_output_dir <DIR> "a directory to store the generated images in")
                .value_parser(value_parser!(PathBuf)),
        ])
        .get_matches();

    let image_output_dir = matches.get_one::<PathBuf>("image_output_dir").map(|i| {
        if !i.is_dir() {
            println!("Path is not a directory: {i:?}");
            std::process::exit(-1);
        }
        i
    });

    let test_scene_filename = matches.get_one::<PathBuf>("file").unwrap();
    let (scene, _) = match Scene::load(test_scene_filename) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("Couldn't open scene {test_scene_filename:?}: {e}");
            std::process::exit(-1);
        }
    };

    let resolution_factor = 15;
    let width = 16 * resolution_factor;
    let height = 9 * resolution_factor;
    let thread_count = (num_cpus::get() - 2).max(1);
    let accel_structures_to_construct = vec![Accel::Bvh];

    let image_settings = ImageSettings {
        size: vec2(width, height),
        enable_dispersion: true,
        max_depth: None,
        always_sample_single_wavelength: false,
        scene_version: None,
    };

    let mut scene_bounds = BoundingBox::new();
    scene.instances.iter().for_each(|i| {
        scene_bounds = scene_bounds.union(&i.bounds);
    });

    let scene_center = scene_bounds.center();
    let scene_diameter = (scene_bounds.max - scene_center).magnitude() * 1.2;

    let mut raytracer = Raytracer::new(scene, &accel_structures_to_construct, image_settings.clone());

    let angle_divisions = 120;
    let angle_increment = Rad::full_turn() / angle_divisions as f32;

    let output_path = matches.get_one::<PathBuf>("output").unwrap();
    let mut wrt = match Writer::from_path(output_path) {
        Ok(wrt) => wrt,
        Err(e) => {
            println!("Cant write to path: {output_path:?}\n\terror: {e}");
            std::process::exit(-1);
        }
    };

    let settings = RenderSettings {
        termination_condition: TerminationCondition::SampleCount(1),
        thread_count,
        accel_structure: accel_structures_to_construct[0],
        intermediate_read_path: None,
        intermediate_write_path: None,
    };

    // Do a warmup run:
    let image = WorkingImage::new(image_settings.clone());
    let (_, _) = raytracer.render(&settings, None, None, None, image);

    for i in 0..angle_divisions {
        let angle = angle_increment * i as f32;

        raytracer.scene.camera.model = Matrix4::from_translation(scene_center.to_vec())
            * Matrix4::from_angle_y(angle)
            * Matrix4::from_translation(Vector3::new(0.0, 0.0, scene_diameter));

        let image = WorkingImage::new(image_settings.clone());
        let (result, time_elapsed) = raytracer.render(&settings, None, None, None, image);

        if let Some(dir) = image_output_dir {
            let mut buf = dir.clone();
            buf.push(format!("{angle:?}.png"));
            result.save_as_rgb(buf);
        }

        if let Err(e) = wrt.serialize(&[angle.0, time_elapsed]) {
            println!("Writing failed: {e}");
        }
    }

    if let Err(e) = wrt.flush() {
        println!("Flushing writer failed: {e}");
    }
}
