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
    raytracer::Raytracer,
    render_settings::RenderSettings,
    scene::Scene,
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
        let scene = match Scene::load(args.scene_file) {
            Ok(scene) => scene,
            Err(message) => {
                eprintln!("{}", message);
                std::process::exit(-1);
            }
        };

        App {
            raytracer: Raytracer::new(&scene),
            scene,
            render_settings: args.render_settings,
            output_file: args.output_file,
            movement: Movement::new(),
        }
    }

    pub fn run(&mut self) {
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

        let background_color = {
            let c = self.scene.environment.color.srgb_to_linear();
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
            None,
            Rgb(Texture<Dim2, NormRGB8UI>),
            Rgba(Texture<Dim2, NormRGBA8UI>),
        }

        enum BoundTex<'a> {
            None,
            Rgb(BoundTexture<'a, Dim2, NormRGB8UI>),
            Rgba(BoundTexture<'a, Dim2, NormRGBA8UI>),
        }

        let mut textures: Vec<Tex> = self
            .scene
            .textures
            .iter()
            .map(|texture| {
                let size = texture.size();
                let upload = TexelUpload::base_level(texture.image.as_slice(), 2);

                match texture.format {
                    crate::texture::Format::Rgb => match context.new_texture_raw([size.x, size.y], sampler, upload) {
                        Ok(texture) => Tex::Rgb(texture),
                        Err(e) => {
                            println!("An error occured while uploading textures: {e}");
                            Tex::None
                        }
                    },
                    crate::texture::Format::Rgba => match context.new_texture_raw([size.x, size.y], sampler, upload) {
                        Ok(texture) => Tex::Rgba(texture),
                        Err(e) => {
                            println!("An error occured while uploading textures: {e}");
                            Tex::None
                        }
                    },
                }
            })
            .collect();

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
                            let mut none = Tex::None;

                            let tex = if let Some(i) = material.base_color_texture {
                                &mut textures[i]
                            } else {
                                &mut none
                            };

                            let bound_tex = match tex {
                                Tex::None => BoundTex::None,
                                Tex::Rgb(rgb) => BoundTex::Rgb(pipeline.bind_texture(rgb)?),
                                Tex::Rgba(rgba) => BoundTex::Rgba(pipeline.bind_texture(rgba)?),
                            };

                            shd_gate.shade(&mut program, |mut iface, unif, mut rdr_gate| {
                                iface.set(&unif.u_projection, mat_to_shader_type(projection));
                                iface.set(&unif.u_view, mat_to_shader_type(view));
                                iface.set(&unif.u_base_color, material.base_color.into());
                                iface.set(&unif.u_light_position, light_position);

                                match bound_tex {
                                    BoundTex::Rgb(rgb) => {
                                        iface.set(&unif.u_base_color_texture, rgb.binding());
                                        iface.set(&unif.u_use_texture, true);
                                    }
                                    BoundTex::Rgba(rgba) => {
                                        iface.set(&unif.u_base_color_texture, rgba.binding());
                                        iface.set(&unif.u_use_texture, true);
                                    }
                                    BoundTex::None => {
                                        iface.set(&unif.u_use_texture, false);
                                    }
                                }

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

    fn do_movement(&mut self, delta_time: f32) -> bool {
        let moving = self.movement.moving();

        if moving {
            let rotation = if self.movement.turning {
                Matrix4::from_angle_x(Rad(-self.movement.mouse_delta.y as f32) * delta_time)
                    * Matrix4::from_angle_y(Rad(-self.movement.mouse_delta.x as f32) * delta_time)
                    * Matrix4::from_angle_z(Rad(self.movement.roll.value as f32) * delta_time)
            } else {
                Matrix4::from_angle_z(Rad(self.movement.roll.value as f32) * delta_time)
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
                Key::Enter => {
                    self.raytracer.camera = self.scene.camera;
                    let (buffer, time_elapsed) = self.raytracer.render(&self.render_settings, None);
                    println!("Finished rendering in {time_elapsed} seconds");

                    let buffer = convert_spectrum_buffer_to_rgb(buffer);
                    save_image(&buffer, self.render_settings.image_size, &self.output_file);
                }
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
    pub value: i8,
}

impl MovementDirection {
    pub fn set(&mut self, value: i8) {
        self.value = value;
    }

    pub fn reset(&mut self, released_value: i8) {
        if self.value == released_value {
            self.value = 0;
        }
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
            forward_backward: MovementDirection { value: 0 },
            left_right: MovementDirection { value: 0 },
            up_down: MovementDirection { value: 0 },
            roll: MovementDirection { value: 0 },
            speed: 2.0,
            turning: false,
            mouse_position: vec2(0.0, 0.0),
            mouse_delta: vec2(0.0, 0.0),
        }
    }

    pub fn translation(&self) -> Vector3<f32> {
        self.speed
            * (self.forward_backward.value as f32 * vec3(0.0, 0.0, -1.0)
                + self.left_right.value as f32 * vec3(-1.0, 0.0, 0.0)
                + self.up_down.value as f32 * vec3(0.0, 1.0, 0.0))
    }

    pub fn moving(&self) -> bool {
        self.turning && self.mouse_delta != vec2(0.0, 0.0)
            || self.forward_backward.value != 0
            || self.left_right.value != 0
            || self.up_down.value != 0
            || self.roll.value != 0
    }
}
