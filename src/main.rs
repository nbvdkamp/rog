use glfw::{Context as _, WindowEvent};

use luminance_glfw::GlfwSurface;
use luminance_windowing::{WindowDim, WindowOpt};
use luminance_derive::{Semantics, Vertex};
use luminance::context::GraphicsContext as _;
use luminance::pipeline::PipelineState;
use luminance::tess::Mode;
use luminance::shader::Program;
use luminance::render_state::RenderState;

use cgmath::Vector2;
use cgmath::Vector3;
use cgmath::Vector4;
use cgmath::prelude::*;

use std::process::exit;
use std::time::Instant;

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

#[derive(Copy, Clone, Debug, Semantics)]
pub enum VertexSemantics {
    #[sem(name = "position", repr = "[f32; 2]", wrapper = "VertexPosition")]
    Position,   
    #[sem(name = "color", repr = "[u8; 3]", wrapper = "VertexRGB")]
    Color,
}

#[derive(Copy, Clone, Vertex)]
#[vertex(sem = "VertexSemantics")]
pub struct Vertex {
    #[allow(dead_code)]
    position: VertexPosition,
    #[allow(dead_code)]
    #[vertex(normalized = "true")]
    color: VertexRGB
}

const VS_STR: &str = include_str!("passthrough.vs");
const FS_STR: &str = include_str!("color.fs");

const VERTICES: [Vertex; 3] = [
  Vertex::new(
    VertexPosition::new([-0.5, -0.5]),
    VertexRGB::new([255, 0, 0]),
  ),
  Vertex::new(
    VertexPosition::new([0.5, -0.5]),
    VertexRGB::new([0, 255, 0]),
  ),
  Vertex::new(
    VertexPosition::new([0., 0.5]),
    VertexRGB::new([0, 0, 255])
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