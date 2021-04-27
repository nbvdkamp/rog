use glfw::{Context as _, WindowEvent};

use luminance_glfw::GlfwSurface;
use luminance_windowing::{WindowDim, WindowOpt};
use luminance_derive::{Semantics, Vertex};
use luminance::pipeline::PipelineState;
use luminance::tess::Mode;
use luminance::render_state::RenderState;
use luminance_front::context::GraphicsContext;
use luminance_front::tess::{Tess, TessError, Interleaved};
use luminance_front::Backend;

//use gltf::*;

use cgmath::Vector4;

use std::process::exit;
use std::time::Instant;
use std::path::Path;

fn main() {
    let dim = WindowDim::Windowed {
        width: 960,
        height: 540,
    };

    let surface = GlfwSurface::new_gl33("Window Title", WindowOpt::default().set_dim(dim));
    Mesh::load(Path::new("res/cube.glb"));

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

#[derive(Copy, Clone, Debug, Semantics)]
pub enum VertexSemantics {
    #[sem(name = "position", repr = "[f32; 3]", wrapper = "VertexPosition")]
    Position,   
}

#[derive(Copy, Clone, Vertex)]
#[vertex(sem = "VertexSemantics")]
pub struct Vertex {
    #[allow(dead_code)]
    position: VertexPosition,
}

type VertexIndex = u32;
pub struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<VertexIndex>,
}

impl Mesh {
    fn to_tess<C>(self, context: &mut C) -> Result<Tess<Vertex, VertexIndex, (), Interleaved>, TessError>
    where
        C: GraphicsContext<Backend = Backend>,
    {
        context
            .new_tess()
            .set_mode(Mode::Triangle)
            .set_vertices(self.vertices)
            .set_indices(self.indices)
            .build()
    }

    fn load<P>(path: P) -> Result<Self, String>
    where
        P: AsRef<Path>,
    {
        /*let gltf = Gltf::open(path)?;
        for scene in gltf.scenes() {
            for node in scene.nodes() {
                println!(
                    "Node #{} has {} children",
                    node.index(),
                    node.children().count(),
                );
            }
        }*/
        Ok(Mesh { vertices: Vec::new(), indices: Vec::new() })
    }
}


const VS_STR: &str = include_str!("passthrough.vs");
const FS_STR: &str = include_str!("color.fs");

const VERTICES: [Vertex; 3] = [
  Vertex::new(
    VertexPosition::new([-0.5, -0.5, 0.0]),
  ),
  Vertex::new(
    VertexPosition::new([0.5, -0.5, 0.0]),
  ),
  Vertex::new(
    VertexPosition::new([0.0, 0.5, 0.0]),
  ),
];

fn main_loop(surface: GlfwSurface) {
    let mut context = surface.context;
    let events = surface.events_rx;
    let back_buffer = context.back_buffer().expect("back buffer");

    let start_t = Instant::now();

    let triangle = context
        .new_tess()
        .set_vertices(&VERTICES[..])
        .set_mode(Mode::Triangle)
        .build()
        .unwrap();
    
    let mut program = context
        .new_shader_program::<VertexSemantics, (), ()>()
        .from_strings(VS_STR, None, None, FS_STR)
        .unwrap()
        .ignore_warnings();

    'app: loop {
        context.window.glfw.poll_events();

        for (_, event) in glfw::flush_messages(&events) {
            match event {
                WindowEvent::Close => break 'app,
                _ => ()
            }
        }

        let t = start_t.elapsed().as_millis() as f32 * 1e-3;
        let color = Vector4::new(t.cos(), t.sin(), 0.5, 1.);

        let render = context
            .new_pipeline_gate()
            .pipeline(
                &back_buffer,
                &PipelineState::default().set_clear_color(color.into()),
                |_, mut shd_gate| {
                    shd_gate.shade(&mut program, |_, _, mut rdr_gate| {
                        rdr_gate.render(&RenderState::default(), |mut tess_gate| {
                            tess_gate.render(&triangle)
                        })
                    })
                },
            )
            .assume();
        
        if render.is_ok() {
            context.window.swap_buffers();
        } else {
            break 'app;
        }
    }
}