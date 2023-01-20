use std::{
    path::PathBuf,
    sync::{
        mpsc::{channel, Sender, TryRecvError},
        Arc,
        Mutex,
    },
    thread,
    time::Duration,
};

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
        types::{Mat33, Mat44, Vec3, Vec4},
        Uniform,
    },
    tess::{Interleaved, Tess},
    texture::{Dim2, MagFilter, MinFilter, Sampler, TexelUpload, Texture as LuminanceTexture, Wrap},
};
use luminance_glfw::{GlfwSurface, GlfwSurfaceError};

use crate::{
    args::Args,
    camera::PerspectiveCamera,
    color::RGBu8,
    material::Material,
    mesh::{LuminanceVertex, VertexIndex, VertexSemantics},
    preview::preview_quad::PreviewQuad,
    raytracer::{render_and_save, working_image::WorkingImage, ImageUpdateReporting, Raytracer, RenderMessage},
    render_settings::{ImageSettings, RenderSettings},
    scene::Scene,
    texture::{Format, Texture},
    util::{mat3_to_shader_type, mat_to_shader_type, normal_transform_from_mat4},
};

#[derive(Debug, UniformInterface)]
struct ShaderInterface {
    u_projection: Uniform<Mat44<f32>>,
    u_view: Uniform<Mat44<f32>>,
    u_model: Uniform<Mat44<f32>>,
    u_normal_transform: Uniform<Mat33<f32>>,
    u_base_color: Uniform<Vec4<f32>>,
    u_base_color_texture: Uniform<TextureBinding<Dim2, NormUnsigned>>,
    u_use_texture: Uniform<bool>,
    u_light_position: Uniform<Vec3<f32>>,
}

const VS_STR: &str = include_str!("vertex.vs");
const FS_STR: &str = include_str!("fragment.fs");

pub struct App {
    raytracer: Arc<Mutex<Raytracer>>,
    textures: Vec<Texture>,
    camera: PerspectiveCamera,
    render_settings: RenderSettings,
    image_settings: ImageSettings,
    output_file: PathBuf,
    movement: Movement,
    rendering: Arc<Mutex<bool>>,
}

#[derive(Debug)]
pub enum PlatformError {
    CannotCreateWindow,
}

struct RGBImage {
    pixels: Vec<RGBu8>,
    size: Vector2<usize>,
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
            scene,
            &[args.render_settings.accel_structure],
            args.image_settings.use_visibility(),
            args.image_settings.scene_version.clone(),
        );

        if args.image_settings.dump_visibility_data() {
            raytracer.dump_visibility_data();
        }

        App {
            camera: raytracer.scene.camera,
            raytracer: Arc::new(Mutex::new(raytracer)),
            textures,
            render_settings: args.render_settings,
            image_settings: args.image_settings,
            output_file: args.output_file,
            movement: Movement::new(),
            rendering: Arc::new(Mutex::new(false)),
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
                    let ratio = self.camera.aspect_ratio;

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

        let raytracer = self.raytracer.lock().unwrap();

        let background_color = {
            let c = raytracer.scene.environment.color.linear_to_srgb();
            [c.r, c.g, c.b, 1.0]
        };

        let light_position = raytracer.scene.lights.first().map_or(Vec3::new(1.0, 1.0, 1.0), |p| {
            let p = p.pos;
            Vec3::new(p.x, p.y, p.z)
        });

        let mut context = surface.context;
        let events = surface.events_rx;
        let mut back_buffer = context.back_buffer().expect("back buffer");

        let tesses = raytracer
            .scene
            .meshes
            .iter()
            .map(|mesh| (mesh.to_tess(&mut context).unwrap(), mesh.material.clone()))
            .collect::<Vec<(Tess<LuminanceVertex, VertexIndex, (), Interleaved>, Material)>>();

        let instances = raytracer.scene.instances.clone();

        // Unlock it again
        drop(raytracer);

        let (width, height) = context.window.get_size();
        let mut progress_display_quad = PreviewQuad::create(&mut context, width as f32 / height as f32).unwrap();

        let sampler = Sampler {
            wrap_r: Wrap::Repeat,
            wrap_s: Wrap::Repeat,
            wrap_t: Wrap::Repeat,
            min_filter: MinFilter::LinearMipmapLinear,
            mag_filter: MagFilter::Linear,
            depth_comparison: None,
        };

        enum Tex {
            Rgb(LuminanceTexture<Dim2, NormRGB8UI>),
            Rgba(LuminanceTexture<Dim2, NormRGBA8UI>),
        }

        enum BoundTex<'a> {
            Rgb(BoundTexture<'a, Dim2, NormRGB8UI>),
            Rgba(BoundTexture<'a, Dim2, NormRGBA8UI>),
        }

        let mut textures: Vec<Option<Tex>> = self
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
        self.textures.clear();

        let blending = Blending {
            equation: Equation::Additive,
            src: Factor::SrcAlpha,
            dst: Factor::SrcAlphaComplement,
        };

        let mut program = context
            .new_shader_program::<VertexSemantics, (), ShaderInterface>()
            .from_strings(VS_STR, None, None, FS_STR)
            .unwrap()
            .ignore_warnings();

        let mut frame_start = context.window.glfw.get_time();
        let (progress_sender, progress_receiver) = channel();
        let mut cancel_sender: Option<Sender<RenderMessage>> = None;

        thread::scope(|scope| 'app: loop {
            context.window.glfw.poll_events();

            for (_, event) in glfw::flush_messages(&events) {
                match event {
                    WindowEvent::Close => {
                        if let Some(sender) = cancel_sender {
                            let _ = sender.send(RenderMessage::Cancel);
                        }

                        break 'app;
                    }
                    WindowEvent::Size(width, height) => {
                        let ratio = width as f32 / height as f32;
                        self.camera.aspect_ratio = ratio;
                        progress_display_quad.window_aspect_ratio = ratio;
                        back_buffer = context.back_buffer().expect("Unable to create new back buffer");
                    }
                    WindowEvent::Key(Key::Enter, _, Action::Press, _) => {
                        let mut rendering = self.rendering.lock().unwrap();
                        if *rendering {
                            continue;
                        }
                        *rendering = true;

                        self.movement.reset();
                        progress_display_quad.reset();

                        let (sender, cancel_receiver) = channel();
                        cancel_sender = Some(sender);

                        let mut raytracer = self.raytracer.lock().unwrap();
                        raytracer.scene.camera = self.camera;

                        let image_settings = self.image_settings.clone();
                        let render_settings = self.render_settings.clone();
                        let output_file = self.output_file.clone();
                        let raytracer = self.raytracer.clone();
                        let rendering = self.rendering.clone();
                        let progress_sender = progress_sender.clone();

                        if let Err(e) = thread::Builder::new()
                            .name("Main render thread".to_string())
                            .spawn_scoped(scope, move || {
                                let raytracer = raytracer.lock().unwrap();

                                let image_update = ImageUpdateReporting {
                                    update: Box::new(move |image, _| {
                                        let _ = progress_sender.send(RGBImage {
                                            pixels: image
                                                .to_rgb_buffer()
                                                .iter()
                                                .map(|c| c.normalized())
                                                .collect::<Vec<_>>(),
                                            size: image.settings.size,
                                        });
                                    }),
                                    update_interval: Duration::from_secs(10),
                                };

                                let image = WorkingImage::new(image_settings);

                                render_and_save(
                                    &raytracer,
                                    &render_settings,
                                    image,
                                    output_file,
                                    Some(image_update),
                                    Some(cancel_receiver),
                                );
                                *rendering.lock().unwrap() = false;
                            })
                        {
                            eprintln!("Unable to spawn main render thread: {e}");
                            std::process::exit(-1);
                        };
                    }
                    e => {
                        if *self.rendering.lock().unwrap() {
                            progress_display_quad.handle_event(e);
                        } else {
                            self.movement.handle_event(e);
                        }
                    }
                }
            }

            match progress_receiver.try_recv() {
                Ok(image) => {
                    let size = [image.size.x as u32, image.size.y as u32];
                    progress_display_quad.update_texture(&mut context, size, &image.pixels);
                }
                Err(TryRecvError::Empty) => (),
                Err(TryRecvError::Disconnected) => {
                    panic!("Receiver disconnected");
                }
            }

            let render = context
                .new_pipeline_gate()
                .pipeline(
                    &back_buffer,
                    &PipelineState::default().set_clear_color(background_color),
                    |pipeline, mut shd_gate| {
                        if !*self.rendering.lock().unwrap() {
                            for instance in &instances {
                                let (tess, material) = &tesses[instance.mesh_index as usize];
                                let tex = material.base_color_texture.and_then(|tex| textures[tex.index].as_mut());

                                let bound_tex = match tex {
                                    Some(Tex::Rgb(rgb)) => Some(BoundTex::Rgb(pipeline.bind_texture(rgb)?)),
                                    Some(Tex::Rgba(rgba)) => Some(BoundTex::Rgba(pipeline.bind_texture(rgba)?)),
                                    None => None,
                                };

                                shd_gate.shade(&mut program, |mut iface, unif, mut rdr_gate| {
                                    iface.set(&unif.u_projection, mat_to_shader_type(self.camera.projection()));
                                    iface.set(&unif.u_view, mat_to_shader_type(self.camera.view));
                                    iface.set(&unif.u_model, mat_to_shader_type(instance.transform));
                                    let normal_transform = normal_transform_from_mat4(instance.transform);
                                    iface.set(&unif.u_normal_transform, mat3_to_shader_type(normal_transform));
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
                        } else {
                            progress_display_quad.render(&mut shd_gate, pipeline)?;
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
        });
    }

    fn do_movement(&mut self, delta_time: f32) -> bool {
        let moving = !*self.rendering.lock().unwrap() && self.movement.moving();

        if moving {
            let rotation = if self.movement.turning {
                Matrix4::from_angle_x(Rad(-self.movement.mouse_delta.y as f32) * delta_time)
                    * Matrix4::from_angle_y(Rad(-self.movement.mouse_delta.x as f32) * delta_time)
                    * Matrix4::from_angle_z(Rad(self.movement.roll.value() as f32) * delta_time)
            } else {
                Matrix4::from_angle_z(Rad(self.movement.roll.value() as f32) * delta_time)
            };

            let translation = Matrix4::from_translation(delta_time * self.movement.translation());
            self.camera.model = self.camera.model * rotation * translation;
            self.camera.view = self.camera.model.invert().unwrap();
        }

        self.movement.mouse_delta = vec2(0.0, 0.0);
        moving
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

    pub fn reset(&mut self) {
        self.forward_backward = MovementDirection::new();
        self.left_right = MovementDirection::new();
        self.up_down = MovementDirection::new();
        self.roll = MovementDirection::new();
        self.turning = false;
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

    fn handle_event(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::Key(key, _, action, _) => self.handle_key_event(key, action),
            WindowEvent::MouseButton(button, action, _) => self.handle_mousebutton_event(button, action),
            WindowEvent::CursorPos(x, y) => {
                let pos = vec2(x, y);
                self.mouse_delta = pos - self.mouse_position;
                self.mouse_position = pos;
            }
            WindowEvent::Scroll(_, y_offset) => {
                self.speed *= if y_offset > 0.0 { 1.2 } else { 0.8 };
            }
            _ => (),
        }
    }

    fn handle_key_event(&mut self, key: Key, action: Action) {
        match action {
            Action::Press => match key {
                Key::W => self.forward_backward.set(1),
                Key::S => self.forward_backward.set(-1),
                Key::A => self.left_right.set(1),
                Key::D => self.left_right.set(-1),
                Key::Space => self.up_down.set(1),
                Key::LeftShift => self.up_down.set(-1),
                Key::Q => self.roll.set(1),
                Key::E => self.roll.set(-1),
                _ => (),
            },
            Action::Release => match key {
                Key::W => self.forward_backward.reset(1),
                Key::S => self.forward_backward.reset(-1),
                Key::A => self.left_right.reset(1),
                Key::D => self.left_right.reset(-1),
                Key::Space => self.up_down.reset(1),
                Key::LeftShift => self.up_down.reset(-1),
                Key::Q => self.roll.reset(1),
                Key::E => self.roll.reset(-1),
                _ => (),
            },
            Action::Repeat => (),
        }
    }

    fn handle_mousebutton_event(&mut self, button: MouseButton, action: Action) {
        match button {
            MouseButton::Button1 => match action {
                Action::Press => self.turning = true,
                Action::Release => self.turning = false,
                Action::Repeat => (),
            },
            _ => (),
        }
    }
}
