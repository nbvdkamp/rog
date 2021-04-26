use glfw::{Context as _, WindowEvent};

use luminance_glfw::GlfwSurface;
use luminance_windowing::{WindowDim, WindowOpt};
use luminance::context::GraphicsContext as _;
use luminance::pipeline::PipelineState;

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

fn main_loop(surface: GlfwSurface) {
    let mut context = surface.context;
    let events = surface.events_rx;
    let back_buffer = context.back_buffer().expect("back buffer");

    let start_t = Instant::now();

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
                |_, _| Ok(()),
            )
            .assume();
        
        if render.is_ok() {
            context.window.swap_buffers();
        } else {
            break 'app;
        }
    }
}