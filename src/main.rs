use glfw::{Context as _, WindowEvent, Key, Action};

use luminance_glfw::GlfwSurface;
use luminance_windowing::{WindowDim, WindowOpt};
use luminance_derive::UniformInterface;
use luminance_front::pipeline::PipelineState;
use luminance_front::render_state::RenderState;
use luminance_front::context::GraphicsContext;
use luminance_front::tess::{Tess, Interleaved};
use luminance::shader::Uniform;

use std::process::exit;
use std::path::Path;
use std::env;

mod mesh;
mod scene;
mod camera;
mod material;
mod raytracer;
mod util;
use mesh::{LuminanceVertex, VertexIndex, VertexSemantics};
use scene::Scene;
use material::Material;
use raytracer::Raytracer;

fn main() {
    if env::args().any(|arg| arg == "--bench") {
        accel_benchmark();
    } else {
        let scene_filename = "res/simple_raytracer_test.glb";
        let app = App::new(scene_filename);
        app.run();
    }
}

#[derive(Debug, UniformInterface)]
struct ShaderInterface {
    #[uniform(unbound)]
    u_projection: Uniform<[[f32; 4]; 4]>,
    #[uniform(unbound)]
    u_view: Uniform<[[f32; 4]; 4]>,
    #[uniform(unbound)]
    u_base_color: Uniform<[f32; 4]>,
}


const VS_STR: &str = include_str!("passthrough.vs");
const FS_STR: &str = include_str!("color.fs");

struct App {
    raytracer: Raytracer,
    scene: Scene,
}

impl App {
    fn new<P>(scene_path: P) -> Self
        where P: AsRef<Path>,
    {
        let scene = Scene::load(scene_path).unwrap();
        let raytracer = Raytracer::new(&scene);

        App { raytracer, scene }
    }

    fn run(&self) {
        let dim = WindowDim::Windowed {
            width: 960,
            height: 540,
        };

        let surface_result = GlfwSurface::new_gl33("Window Title", WindowOpt::default().set_dim(dim));

        let surface = match surface_result {
            Ok(surface) => {
                eprintln!("Graphics surface created");
                surface
            }
            Err(e) => {
                eprintln!("Could not create graphics surface:\n{}", e);
                exit(1);
            }
        };

        let mut context = surface.context;
        let events = surface.events_rx;
        let back_buffer = context.back_buffer().expect("back buffer");

        let tesses = self.scene.meshes.as_slice().iter()
            .map(|mesh| (mesh.to_tess(&mut context).unwrap(), mesh.material.clone()))
            .collect::<Vec<(Tess<LuminanceVertex, VertexIndex, (), Interleaved>, Material)>>();

        let [width, height] = back_buffer.size();
        let projection = self.scene.camera.projection();
        let view = self.scene.camera.view;
        
        let mut program = context
            .new_shader_program::<VertexSemantics, (), ShaderInterface>()
            .from_strings(VS_STR, None, None, FS_STR)
            .unwrap()
            .ignore_warnings();

        'app: loop {
            context.window.glfw.poll_events();

            for (_, event) in glfw::flush_messages(&events) {
                match event {
                    WindowEvent::Close => break 'app,
                    WindowEvent::Key(key, _, action, _) => self.handle_key_event(key, action),
                    _ => ()
                }
            }

            let render = context
                .new_pipeline_gate()
                .pipeline(
                    &back_buffer,
                    &PipelineState::default().set_clear_color([0.1, 0.1, 0.1, 1.0]),
                    |_, mut shd_gate| {
                        for (tess, material) in &tesses {
                            shd_gate.shade(&mut program, |mut iface, unif, mut rdr_gate| {
                                iface.set(&unif.u_projection, projection.into());
                                iface.set(&unif.u_view, view.into());
                                iface.set(&unif.u_base_color , material.base_color_factor.into());

                                rdr_gate.render(&RenderState::default(), |mut tess_gate| {
                                    tess_gate.render(tess)
                                })
                            })?;
                        }

                        Ok(())
                    },
                )
                .assume();
            
            if !render.is_ok() {
                break 'app;
            }

            context.window.swap_buffers();
        }
    }

    fn handle_key_event(&self, key: Key, action: Action) {
        if key == Key::Enter && action == Action::Press {
            let width = 1920;
            let height = 1080;

            let (buffer, time_elapsed) = self.raytracer.render(width, height, 2);
            println!("Finished rendering in {} seconds", time_elapsed);

            let save_result = image::save_buffer("output/result.png", &buffer, width, height, image::ColorType::Rgb8);

            match save_result {
                Ok(_) => println!("File was saved succesfully"),
                Err(e) => println!("Couldn't save file: {}", e),
            }
        }
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
    
    let resolution_factor = 15;
    let image_width = 16 * resolution_factor;
    let image_height = 9 * resolution_factor;

    for path in test_scene_filenames {
        let scene = Scene::load(format!("res/{}.glb", path)).unwrap();
        let raytracer = Raytracer::new(&scene);

        println!("\nFilename: {}", path);
        println!("{: <25} | {: <10} | {: <10} | {: <10} | {: <10}", "Acceleration structure", "Time (s)", "Nodes/ray", "Tests/ray", "Hits/test", );

        for i in 0..raytracer.accel_structures.len() {
            let (_, time_elapsed) = raytracer.render(image_width, image_height, i);

            let stats = raytracer.accel_structures[i].get_statistics();
            let traversals_per_ray = stats.inner_node_traversals as f32 / stats.rays as f32;
            let tests_per_ray = stats.intersection_tests as f32 / stats.rays as f32;
            let hits_per_test = stats.intersection_hits as f32 / stats.intersection_tests as f32;

            println!("{: <25} | {: <10} | {: <10} | {: <10} | {: <10}", raytracer.accel_structures[i].get_name(), time_elapsed, traversals_per_ray, tests_per_ray, hits_per_test);
        }
    }
}