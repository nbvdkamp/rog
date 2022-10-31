use std::{io::Write, time::Duration};

use cgmath::{vec2, vec3, Matrix4, Rad, SquareMatrix, Vector2, Vector3};

use glfw::{Action, Context as _, Key, MouseButton, SwapInterval, WindowEvent, WindowMode};

use luminance_derive::UniformInterface;
use luminance_front::{
    blending::{Blending, Equation, Factor},
    context::GraphicsContext,
    pipeline::{BoundTexture, PipelineState, TextureBinding},
    pixel::{NormRGB8UI, NormRGBA8UI, NormUnsigned},
    render_state::RenderState,
    shader::{
        types::{Mat44, Vec3, Vec4},
        Uniform,
    },
    tess::{Interleaved, Tess},
    texture::{Dim2, MagFilter, MinFilter, Sampler, TexelUpload, Texture, Wrap},
};
use luminance_glfw::{GlfwSurface, GlfwSurfaceError};

use crate::{
    args::Args,
    material::Material,
    mesh::{LuminanceVertex, VertexIndex, VertexSemantics},
    raytracer::{Raytracer, RenderProgress},
    render_settings::RenderSettings,
    scene::Scene,
    texture::Format,
    util::{convert_spectrum_buffer_to_rgb, mat_to_shader_type, save_image},
};

#[derive(Debug, UniformInterface)]
struct ShaderInterface {
    u_projection: Uniform<Mat44<f32>>,
    u_view: Uniform<Mat44<f32>>,
    u_base_color: Uniform<Vec4<f32>>,
    u_base_color_texture: Uniform<TextureBinding<Dim2, NormUnsigned>>,
    u_use_texture: Uniform<bool>,
    u_light_position: Uniform<Vec3<f32>>,
}

const VS_STR: &str = include_str!("vertex.vs");
const FS_STR: &str = include_str!("fragment.fs");

pub struct App {
    raytracer: Raytracer,
    scene: Scene,
    render_settings: RenderSettings,
    output_file: String,
    movement: Movement,
}

#[derive(Debug)]
pub enum PlatformError {
    CannotCreateWindow,
}

impl App {
    pub fn new(args: Args) -> Self {
        let (scene, textures) = match Scene::load(args.scene_file) {
            Ok(scene) => scene,
            Err(message) => {
                eprintln!("{}", message);
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

        App {
            raytracer,
            scene,
            render_settings: args.render_settings,
            output_file: args.output_file,
            movement: Movement::new(),
        }
    }

    pub fn run(&mut self) {
        let surface = GlfwSurface::new(|glfw| {
            let mut width = 960;
            let mut height = 540;

            glfw.with_primary_monitor(|_, monitor| {
                if let Some(monitor) = monitor {
                    let (_, _, w, h) = monitor.get_workarea();
                    let max_width = 2 * w as u32 / 3;
                    let max_height = 4 * h as u32 / 5;
                    let ratio = self.scene.camera.aspect_ratio;

                    if ratio * max_height as f32 <= max_width as f32 {
                        width = (ratio * max_height as f32) as u32;
                        height = max_height;
                    } else {
                        width = max_width;
                        height = (max_width as f32 / ratio) as u32;
                    }
                }
            });

            let (mut window, events) = glfw
                .create_window(width, height, "Rust renderer", WindowMode::Windowed)
                .ok_or(GlfwSurfaceError::UserError(PlatformError::CannotCreateWindow))?;

            window.make_current();
            window.set_all_polling(true);
            glfw.set_swap_interval(SwapInterval::Sync(1));

            Ok((window, events))
        })
        .expect("Failed to create GLFW surface");

        let background_color = {
            let c = self.scene.environment.color.linear_to_srgb();
            [c.r, c.g, c.b, 1.0]
        };

        let light_position = self.scene.lights.first().map_or(Vec3::new(1.0, 1.0, 1.0), |p| {
            let p = p.pos;
            Vec3::new(p.x, p.y, p.z)
        });

        let mut context = surface.context;
        let events = surface.events_rx;
        let mut back_buffer = context.back_buffer().expect("back buffer");

        let tesses = self
            .scene
            .meshes
            .iter()
            .map(|mesh| (mesh.to_tess(&mut context).unwrap(), mesh.material.clone()))
            .collect::<Vec<(Tess<LuminanceVertex, VertexIndex, (), Interleaved>, Material)>>();

        let sampler = Sampler {
            wrap_r: Wrap::Repeat,
            wrap_s: Wrap::Repeat,
            wrap_t: Wrap::Repeat,
            min_filter: MinFilter::LinearMipmapLinear,
            mag_filter: MagFilter::Linear,
            depth_comparison: None,
        };

        enum Tex {
            Rgb(Texture<Dim2, NormRGB8UI>),
            Rgba(Texture<Dim2, NormRGBA8UI>),
        }

        enum BoundTex<'a> {
            Rgb(BoundTexture<'a, Dim2, NormRGB8UI>),
            Rgba(BoundTexture<'a, Dim2, NormRGBA8UI>),
        }

        let mut textures: Vec<Option<Tex>> = self
            .scene
            .textures
            .iter()
            .map(|texture| {
                let size = texture.size();
                let size = [size.x as u32, size.y as u32];
                let upload = TexelUpload::base_level(texture.image.as_slice(), 2);

                match texture.format {
                    Format::Rgb => match context.new_texture_raw(size, sampler, upload) {
                        Ok(texture) => Some(Tex::Rgb(texture)),
                        Err(e) => {
                            println!("An error occured while uploading textures: {e}");
                            None
                        }
                    },
                    Format::Rgba => match context.new_texture_raw(size, sampler, upload) {
                        Ok(texture) => Some(Tex::Rgba(texture)),
                        Err(e) => {
                            println!("An error occured while uploading textures: {e}");
                            None
                        }
                    },
                }
            })
            .collect();

        // The textures are now in GPU memory so no longer needed
        self.scene.textures.clear();

        let blending = Blending {
            equation: Equation::Additive,
            src: Factor::SrcAlpha,
            dst: Factor::SrcAlphaComplement,
        };

        let mut projection = self.scene.camera.projection();

        let mut program = context
            .new_shader_program::<VertexSemantics, (), ShaderInterface>()
            .from_strings(VS_STR, None, None, FS_STR)
            .unwrap()
            .ignore_warnings();

        let mut frame_start = context.window.glfw.get_time();

        'app: loop {
            context.window.glfw.poll_events();

            for (_, event) in glfw::flush_messages(&events) {
                match event {
                    WindowEvent::Close => break 'app,
                    WindowEvent::Size(width, height) => {
                        self.scene.camera.aspect_ratio = width as f32 / height as f32;
                        projection = self.scene.camera.projection();
                        back_buffer = context.back_buffer().expect("Unable to create new back buffer");
                    }
                    e => self.handle_event(e),
                }
            }

            let view = self.scene.camera.view;

            let render = context
                .new_pipeline_gate()
                .pipeline(
                    &back_buffer,
                    &PipelineState::default().set_clear_color(background_color),
                    |pipeline, mut shd_gate| {
                        for (tess, material) in &tesses {
                            let tex = material.base_color_texture.and_then(|tex| textures[tex.index].as_mut());

                            let bound_tex = match tex {
                                Some(Tex::Rgb(rgb)) => Some(BoundTex::Rgb(pipeline.bind_texture(rgb)?)),
                                Some(Tex::Rgba(rgba)) => Some(BoundTex::Rgba(pipeline.bind_texture(rgba)?)),
                                None => None,
                            };

                            shd_gate.shade(&mut program, |mut iface, unif, mut rdr_gate| {
                                iface.set(&unif.u_projection, mat_to_shader_type(projection));
                                iface.set(&unif.u_view, mat_to_shader_type(view));
                                iface.set(&unif.u_base_color, material.base_color.into());
                                iface.set(&unif.u_light_position, light_position);

                                bound_tex.iter().for_each(|tex| match tex {
                                    BoundTex::Rgb(t) => iface.set(&unif.u_base_color_texture, t.binding()),
                                    BoundTex::Rgba(t) => iface.set(&unif.u_base_color_texture, t.binding()),
                                });

                                iface.set(&unif.u_use_texture, bound_tex.is_some());

                                let render_state = RenderState::default().set_blending(blending);

                                rdr_gate.render(&render_state, |mut tess_gate| tess_gate.render(tess))
                            })?;
                        }

                        Ok(())
                    },
                )
                .assume();

            if !render.is_ok() {
                break 'app;
            }

            let frame_end = context.window.glfw.get_time();
            let frame_time = (frame_end - frame_start) as f32;
            frame_start = frame_end;

            self.do_movement(frame_time);

            context.window.swap_buffers();
        }
    }

    fn render(&mut self) {
        self.raytracer.camera = self.scene.camera;

        let report_progress = |completed, total, seconds_per_tile| {
            let time_remaining = (total - completed) as f32 * seconds_per_tile;
            print!(
                "\r\x1b[2K Completed {completed}/{total} tiles. Approximately {time_remaining:.2} seconds remaining"
            );
            std::io::stdout().flush().unwrap();
        };

        let progress = Some(RenderProgress {
            report_interval: Duration::from_secs(3),
            report: Box::new(report_progress),
        });

        let (buffer, time_elapsed) = self.raytracer.render(&self.render_settings, progress);
        println!("\r\x1b[2KFinished rendering in {time_elapsed} seconds");

        let buffer = convert_spectrum_buffer_to_rgb(buffer);
        save_image(&buffer, self.render_settings.image_size, &self.output_file);
    }

    fn do_movement(&mut self, delta_time: f32) -> bool {
        let moving = self.movement.moving();

        if moving {
            let rotation = if self.movement.turning {
                Matrix4::from_angle_x(Rad(-self.movement.mouse_delta.y as f32) * delta_time)
                    * Matrix4::from_angle_y(Rad(-self.movement.mouse_delta.x as f32) * delta_time)
                    * Matrix4::from_angle_z(Rad(self.movement.roll.value() as f32) * delta_time)
            } else {
                Matrix4::from_angle_z(Rad(self.movement.roll.value() as f32) * delta_time)
            };

            let translation = Matrix4::from_translation(delta_time * self.movement.translation());
            self.scene.camera.model = self.scene.camera.model * rotation * translation;
            self.scene.camera.view = self.scene.camera.model.invert().unwrap();
        }

        self.movement.mouse_delta = vec2(0.0, 0.0);
        moving
    }

    fn handle_event(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::Key(key, _, action, _) => self.handle_key_event(key, action),
            WindowEvent::MouseButton(button, action, _) => self.handle_mousebutton_event(button, action),
            WindowEvent::CursorPos(x, y) => {
                let pos = vec2(x, y);
                self.movement.mouse_delta = pos - self.movement.mouse_position;
                self.movement.mouse_position = pos;
            }
            WindowEvent::Scroll(_, y_offset) => {
                self.movement.speed *= if y_offset > 0.0 { 1.2 } else { 0.8 };
            }
            _ => (),
        }
    }

    fn handle_key_event(&mut self, key: Key, action: Action) {
        match action {
            Action::Press => match key {
                Key::Enter => self.render(),
                Key::W => self.movement.forward_backward.set(1),
                Key::S => self.movement.forward_backward.set(-1),
                Key::A => self.movement.left_right.set(1),
                Key::D => self.movement.left_right.set(-1),
                Key::Space => self.movement.up_down.set(1),
                Key::LeftShift => self.movement.up_down.set(-1),
                Key::Q => self.movement.roll.set(1),
                Key::E => self.movement.roll.set(-1),
                _ => (),
            },
            Action::Release => match key {
                Key::W => self.movement.forward_backward.reset(1),
                Key::S => self.movement.forward_backward.reset(-1),
                Key::A => self.movement.left_right.reset(1),
                Key::D => self.movement.left_right.reset(-1),
                Key::Space => self.movement.up_down.reset(1),
                Key::LeftShift => self.movement.up_down.reset(-1),
                Key::Q => self.movement.roll.reset(1),
                Key::E => self.movement.roll.reset(-1),
                _ => (),
            },
            Action::Repeat => (),
        }
    }

    fn handle_mousebutton_event(&mut self, button: MouseButton, action: Action) {
        match button {
            MouseButton::Button1 => match action {
                Action::Press => self.movement.turning = true,
                Action::Release => self.movement.turning = false,
                Action::Repeat => (),
            },
            _ => (),
        }
    }
}

struct MovementDirection {
    stack: Vec<i8>,
}

impl MovementDirection {
    pub fn new() -> Self {
        MovementDirection { stack: Vec::new() }
    }

    pub fn set(&mut self, value: i8) {
        self.stack.push(value);
    }

    pub fn reset(&mut self, released_value: i8) {
        self.stack.retain(|x| *x != released_value);
    }

    pub fn value(&self) -> i8 {
        *self.stack.last().unwrap_or(&0)
    }
}

struct Movement {
    pub left_right: MovementDirection,
    pub forward_backward: MovementDirection,
    pub up_down: MovementDirection,
    pub roll: MovementDirection,
    pub speed: f32,
    pub turning: bool,
    pub mouse_position: Vector2<f64>,
    pub mouse_delta: Vector2<f64>,
}

impl Movement {
    pub fn new() -> Self {
        Movement {
            forward_backward: MovementDirection::new(),
            left_right: MovementDirection::new(),
            up_down: MovementDirection::new(),
            roll: MovementDirection::new(),
            speed: 2.0,
            turning: false,
            mouse_position: vec2(0.0, 0.0),
            mouse_delta: vec2(0.0, 0.0),
        }
    }

    pub fn translation(&self) -> Vector3<f32> {
        self.speed
            * (self.forward_backward.value() as f32 * vec3(0.0, 0.0, -1.0)
                + self.left_right.value() as f32 * vec3(-1.0, 0.0, 0.0)
                + self.up_down.value() as f32 * vec3(0.0, 1.0, 0.0))
    }

    pub fn moving(&self) -> bool {
        self.turning && self.mouse_delta != vec2(0.0, 0.0)
            || self.forward_backward.value() != 0
            || self.left_right.value() != 0
            || self.up_down.value() != 0
            || self.roll.value() != 0
    }
}
