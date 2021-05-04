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

mod mesh;
mod scene;
mod camera;
mod material;
mod raytracer;
use mesh::{Vertex, VertexIndex, VertexSemantics};
use scene::Scene;
use material::Material;

fn main() {
    let dim = WindowDim::Windowed {
        width: 960,
        height: 540,
    };

    let surface = GlfwSurface::new_gl33("Window Title", WindowOpt::default().set_dim(dim));

    match surface {
        Ok(surface) => {
            eprintln!("Graphics surface created");
            main_loop(surface);
        }
        Err(e) => {
            eprintln!("Could not create graphics surface:\n{}", e);
            exit(1);
        }
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

fn main_loop(surface: GlfwSurface) {
    let mut context = surface.context;
    let events = surface.events_rx;
    let back_buffer = context.back_buffer().expect("back buffer");

    let scene = Scene::load(Path::new("res/simple_raytracer_test.glb")).unwrap();
    let tesses = scene.meshes.as_slice().into_iter()
        .map(|mesh| (mesh.to_tess(&mut context).unwrap(), mesh.material.clone()))
        .collect::<Vec<(Tess<Vertex, VertexIndex, (), Interleaved>, Material)>>();

    let [width, height] = back_buffer.size();
    let projection = scene.camera.projection();
    let view = scene.camera.view;
    
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
                WindowEvent::Key(key, _, action, _) => handle_key_event(key, action),
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

fn handle_key_event(key: Key, action: Action) {
    if (key == Key::Enter && action == Action::Press) {
        println!("Do a thing");
    }
}