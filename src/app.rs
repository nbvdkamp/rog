
use cgmath::{Vector2, vec2};

use glfw::{Context as _, WindowEvent, Key, Action, WindowMode, SwapInterval};

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

use crate::{
    args::Args,
    material::Material,
    mesh::{LuminanceVertex, VertexIndex, VertexSemantics},
    raytracer::Raytracer,
    scene::Scene,
    util::{mat_to_shader_type, save_image},
};

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

pub struct App {
    raytracer: Raytracer,
    scene: Scene,
    image_size: Vector2<usize>,
    samples_per_pixel: usize,
    output_file: String,
}

#[derive(Debug)]
pub enum PlatformError {
  CannotCreateWindow,
}

impl App {
    pub fn new(args: Args) -> Self
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
            samples_per_pixel: args.samples,
            output_file: args.output_file,
        }
    }

    pub fn run(&self) {
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
            let (buffer, time_elapsed) = self.raytracer.render(self.image_size, self.samples_per_pixel, super::ACCEL_INDEX);
            println!("Finished rendering in {} seconds", time_elapsed);

            save_image(&buffer, self.image_size, &self.output_file);
        }
    }
}