use renderer::{
    args::Args,
    raytracer::{render_and_save, working_image::WorkingImage, Raytracer},
    scene::Scene,
};

use preview::app::App;

fn main() {
    let args = Args::parse();
    // FIXME: Verify scene aspect ratio with given image size

    if args.headless {
        headless_render(args);
    } else {
        let mut app = App::new(args);
        app.run();
    }
}

fn headless_render(args: Args) {
    let image = if let Some(path) = &args.render_settings.intermediate_read_path {
        match WorkingImage::read_from_file(path, &args.image_settings.scene_version) {
            Ok(image) => image,
            Err(e) => {
                eprintln!("An error occured while opening intermediate image: {e}");
                std::process::exit(-1);
            }
        }
    } else {
        WorkingImage::new(args.image_settings.clone())
    };

    let (scene, _) = match Scene::load(args.scene_file) {
        Ok(scene) => scene,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(-1);
        }
    };

    let raytracer = Raytracer::new(
        scene,
        &[args.render_settings.accel_structure],
        args.image_settings.clone(),
    );

    render_and_save(&raytracer, &args.render_settings, image, args.output_file, None, None);
}
