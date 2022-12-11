use cgmath::vec2;

#[cfg(feature = "stats")]
use renderer::raytracer::acceleration::statistics::Statistics;
use renderer::{
    raytracer::{acceleration::Accel, working_image::WorkingImage, Raytracer},
    render_settings::{ImageSettings, RenderSettings, TerminationCondition},
    scene::Scene,
};

fn main() {
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
    let width = 16 * resolution_factor;
    let height = 9 * resolution_factor;
    let thread_count = (num_cpus::get() - 2).max(1);
    let accel_structures_to_construct = vec![Accel::Bvh, Accel::BvhRecursive, Accel::KdTree];

    for scene_name in test_scene_filenames {
        let (scene, _) = match Scene::load(format!("res/scenes/{scene_name}.glb")) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("Couldn't open scene {scene_name}: {e}");
                continue;
            }
        };

        let raytracer = Raytracer::new(scene, &accel_structures_to_construct, false, None);

        println!("Filename: {scene_name}, tris: {}", raytracer.get_num_tris());
        #[cfg(feature = "stats")]
        println!(
            "{: <25} | {: <10} | {}",
            "Acceleration structure",
            "Time (s)",
            Statistics::format_header()
        );
        #[cfg(not(feature = "stats"))]
        println!("{: <25} | {: <10}", "Acceleration structure", "Time (s)");

        for &structure in &accel_structures_to_construct {
            let settings = RenderSettings {
                termination_condition: TerminationCondition::SampleCount(1),
                thread_count,
                accel_structure: structure,
                intermediate_read_path: None,
                intermediate_write_path: None,
            };

            let image_settings = ImageSettings {
                size: vec2(width, height),
                enable_dispersion: true,
                max_depth: None,
                always_sample_single_wavelength: false,
                visibility: None,
                scene_version: None,
            };

            let image = WorkingImage::new(image_settings);

            let (_, time_elapsed) = raytracer.render(&settings, None, None, None, image);
            let name = structure.name();

            #[cfg(feature = "stats")]
            {
                let mut stats = raytracer.accel_structures.get(structure).get_statistics();
                let top_level_stats = raytracer.accel_structures.get(structure).get_top_level_statistics();
                println!("{name: <25} | {time_elapsed: <10} | {stats}");
                println!("{:<25} | {:10} | {top_level_stats}", "Top Level:", "");
            }
            #[cfg(not(feature = "stats"))]
            println!("{name: <25} | {time_elapsed: <10}");
        }
        println!();
    }
}
