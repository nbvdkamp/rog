use renderer::{
    args::Args,
    preview::app::App,
    raytracer::{render_and_save, Raytracer},
    scene::Scene,
};

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
    let (scene, textures) = match Scene::load(args.scene_file) {
        Ok(scene) => scene,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(-1);
        }
    };

    let raytracer = Raytracer::new(
        &scene,
        textures,
        &[args.render_settings.accel_structure],
        args.render_settings.use_visibility,
    );

    if args.render_settings.dump_visibility_debug_data {
        raytracer.dump_visibility_data();
    }

    render_and_save(&raytracer, &args.render_settings, args.output_file);
}
