#[macro_use] extern crate impl_ops;

use glfw::{Context as _, WindowEvent, Key, Action, WindowMode, SwapInterval};
use cgmath::{Vector2, vec2};

use luminance_glfw::{GlfwSurface, GlfwSurfaceError};
use luminance_derive::UniformInterface;
use luminance_front::{
    pipeline::PipelineState,
    render_state::RenderState,
    context::GraphicsContext,
    shader::{
        Uniform, 
        types::{Mat44, Vec4}
    },
    tess::{Tess, Interleaved}
};

use std::vec;

mod args;
mod color;
mod mesh;
mod scene;
mod camera;
mod material;
mod texture;
mod light;
mod environment;
mod raytracer;
mod util;
mod sampling;

use args::Args;
use mesh::{LuminanceVertex, VertexIndex, VertexSemantics};
use scene::Scene;
use material::Material;
use raytracer::Raytracer;
use util::mat_to_shader_type;
use color::{RGBf32, RGBu8};

fn main() {
    let args = Args::parse();
    // FIXME: Verify scene aspect ratio with given image size

    if args.benchmark {
        accel_benchmark();
    } else if args.headless {
        headless_render(args);
    } else {
        let app = App::new(args);
        app.run();
    }
}

#[derive(Debug, UniformInterface)]
struct ShaderInterface {
    #[uniform(unbound)]
    u_projection: Uniform<Mat44<f32>>,
    #[uniform(unbound)]
    u_view: Uniform<Mat44<f32>>,
    #[uniform(unbound)]
    u_base_color: Uniform<Vec4<f32>>,
}


const VS_STR: &str = include_str!("passthrough.vs");
const FS_STR: &str = include_str!("color.fs");

const ACCEL_INDEX: usize = 1;

struct App {
    raytracer: Raytracer,
    scene: Scene,
    image_size: Vector2<usize>,
    samples_per_pixel: usize,
}

#[derive(Debug)]
pub enum PlatformError {
  CannotCreateWindow,
}

impl App {
    fn new(args: Args) -> Self
    {
        let scene = match Scene::load(args.file) {
            Ok(scene) => scene,
            Err(message) => {
                eprintln!("{}", message);
                std::process::exit(-1);
            }
        };

        let raytracer = Raytracer::new(&scene);
        let image_size = vec2(args.width, args.height);

        App {
            raytracer,
            scene,
            image_size,
            samples_per_pixel: args.samples
        }
    }

    fn run(&self) {
        let surface = GlfwSurface::new(|glfw| {
            let (mut window, events) = glfw
                .create_window(960, 540, "Rust renderer", WindowMode::Windowed)
                .ok_or(GlfwSurfaceError::UserError(PlatformError::CannotCreateWindow))?;
            
            window.make_current();
            window.set_all_polling(true);
            glfw.set_swap_interval(SwapInterval::Sync(1));

            Ok((window, events))
        })
        .expect("Failed to create GLFW surface");

        let mut context = surface.context;
        let events = surface.events_rx;
        let back_buffer = context.back_buffer().expect("back buffer");

        let tesses = self.scene.meshes.as_slice().iter()
            .map(|mesh| (mesh.to_tess(&mut context).unwrap(), mesh.material.clone()))
            .collect::<Vec<(Tess<LuminanceVertex, VertexIndex, (), Interleaved>, Material)>>();

        let [_width, _height] = back_buffer.size();
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
                                iface.set(&unif.u_projection, mat_to_shader_type(projection));
                                iface.set(&unif.u_view, mat_to_shader_type(view));
                                iface.set(&unif.u_base_color , material.base_color.into());

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
            let (buffer, time_elapsed) = self.raytracer.render(self.image_size, self.samples_per_pixel, ACCEL_INDEX);
            println!("Finished rendering in {} seconds", time_elapsed);

            save_image(&buffer, self.image_size);
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

    #[cfg(not(feature = "stats"))]
    println!("Build with --features stats for traversal statistics");
    
    let resolution_factor = 15;
    let image_size = Vector2::new(16 * resolution_factor, 9 * resolution_factor);
    let samples = 1;

    for path in test_scene_filenames {
        let scene = Scene::load(format!("res/{}.glb", path)).unwrap();
        let raytracer = Raytracer::new(&scene);

        println!("\nFilename: {}, tris: {}", path, raytracer.get_num_tris());
        #[cfg(feature = "stats")]
        println!("{: <25} | {: <10} | {: <10} | {: <10} | {: <10}", "Acceleration structure", "Time (s)", "Nodes/ray", "Tests/ray", "Hits/test");
        #[cfg(not(feature = "stats"))]
        println!("{: <25} | {: <10}", "Acceleration structure", "Time (s)");

        for i in 0..raytracer.accel_structures.len() {
            let (_, time_elapsed) = raytracer.render(image_size, samples, i);

            #[cfg(feature = "stats")]
            {
                let stats = raytracer.accel_structures[i].get_statistics();
                let traversals_per_ray = stats.inner_node_traversals as f32 / stats.rays as f32;
                let tests_per_ray = stats.intersection_tests as f32 / stats.rays as f32;
                let hits_per_test = stats.intersection_hits as f32 / stats.intersection_tests as f32;
                println!("{: <25} | {: <10} | {: <10} | {: <10} | {: <10}", raytracer.accel_structures[i].get_name(), time_elapsed, traversals_per_ray, tests_per_ray, hits_per_test);
            }
            #[cfg(not(feature = "stats"))]
            println!("{: <25} | {: <10}", raytracer.accel_structures[i].get_name(), time_elapsed);
        }
    }
}

fn headless_render(args: Args)
{
    let scene = match Scene::load(args.file) {
        Ok(scene) => scene,
        Err(message) => {
            eprintln!("{}", message);
            std::process::exit(-1);
        }
    };

    let raytracer = Raytracer::new(&scene);
    let image_size = vec2(args.width, args.height);

    let (buffer, time_elapsed) = raytracer.render(image_size, args.samples, ACCEL_INDEX);
    println!("Finished rendering in {} seconds", time_elapsed);

    save_image(&buffer, image_size);
}

fn save_image(buffer: &[RGBf32], image_size: Vector2<usize>) {
    let pixels: Vec<RGBu8> = buffer.into_iter().map(|c| c.normalized()).collect();

    let byte_buffer: &[u8] = unsafe { 
        std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * std::mem::size_of::<RGBu8>())
    };

    let save_result = image::save_buffer("output/result.png", &byte_buffer,
        image_size.x as u32, image_size.y as u32, image::ColorType::Rgb8);

    match save_result {
        Ok(_) => println!("File was saved succesfully"),
        Err(e) => println!("Couldn't save file: {}", e),
    }
}