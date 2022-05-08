
use cgmath::{Vector2, vec2, Vector3, vec3, Matrix4, SquareMatrix, Rad};

use glfw::{Context as _, WindowEvent, Key, Action, WindowMode, SwapInterval, MouseButton};

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
    movement: Movement,
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
            let c = self.scene.environment.color.pow(1.0 / crate::constants::GAMMA);
            [c.r, c.g, c.b, 1.0]
        };

        let mut context = surface.context;
        let events = surface.events_rx;
        let mut back_buffer = context.back_buffer().expect("back buffer");

        let tesses = self.scene.meshes.as_slice().iter()
            .map(|mesh| (mesh.to_tess(&mut context).unwrap(), mesh.material.clone()))
            .collect::<Vec<(Tess<LuminanceVertex, VertexIndex, (), Interleaved>, Material)>>();

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
                    WindowEvent::Size(width, height) => {
                        self.scene.camera.aspect_ratio = width as f32 / height as f32;
                        projection = self.scene.camera.projection();
                        back_buffer = context.back_buffer().expect("Unable to create new back buffer");
                    }
                    _ => ()
                }
            }

            let view = self.scene.camera.view;

            let render = context
                .new_pipeline_gate()
                .pipeline(
                    &back_buffer,
                    &PipelineState::default().set_clear_color(background_color),
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

            let frame_end= context.window.glfw.get_time();
            let frame_time = (frame_end - frame_start) as f32;
            frame_start = frame_end;

            if self.movement.moving() {
                let rotation = if self.movement.turning {
                    Matrix4::from_angle_x(Rad(-self.movement.mouse_delta.y as f32) * frame_time) *
                    Matrix4::from_angle_y(Rad(-self.movement.mouse_delta.x as f32) * frame_time) *
                    Matrix4::from_angle_z(Rad(self.movement.roll.value as f32) * frame_time)
                } else {
                    Matrix4::from_angle_z(Rad(self.movement.roll.value as f32) * frame_time)
                };

                let translation = Matrix4::from_translation(frame_time * self.movement.translation());
                self.scene.camera.model =  self.scene.camera.model * rotation *  translation;
                self.scene.camera.view = self.scene.camera.model.invert().unwrap();
            }

            self.movement.mouse_delta =  vec2(0.0, 0.0);

            context.window.swap_buffers();
        }
    }

    fn handle_key_event(&mut self, key: Key, action: Action) {
        match action {
            Action::Press => {
                match key {
                    Key::Enter => {
                        self.raytracer.camera = self.scene.camera;
                        let (buffer, time_elapsed) = self.raytracer.render(self.image_size, self.samples_per_pixel, super::ACCEL_INDEX);
                        println!("Finished rendering in {} seconds", time_elapsed);

                        save_image(&buffer, self.image_size, &self.output_file);
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
                }
            },
            Action::Release => {
                match key {
                    Key::W => self.movement.forward_backward.reset(1),
                    Key::S => self.movement.forward_backward.reset(-1),
                    Key::A => self.movement.left_right.reset(1),
                    Key::D => self.movement.left_right.reset(-1),
                    Key::Space => self.movement.up_down.reset(1),
                    Key::LeftShift => self.movement.up_down.reset(-1),
                    Key::Q => self.movement.roll.reset(1),
                    Key::E => self.movement.roll.reset(-1),
                    _ => (),
                }
            },
            Action::Repeat => (),
        }
    }

    fn handle_mousebutton_event(&mut self, button: MouseButton, action: Action) {
        match button {
            MouseButton::Button1 => {
                match action {
                    Action::Press => self.movement.turning = true,
                    Action::Release => self.movement.turning = false,
                    Action::Repeat => (),
                }
            }
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
        self.speed * (
            self.forward_backward.value as f32 * vec3(0.0, 0.0, -1.0) +
            self.left_right.value as f32 * vec3(-1.0, 0.0, 0.0) +
            self.up_down.value as f32 * vec3(0.0, 1.0, 0.0)
        )
    }

    pub fn moving(&self) -> bool {
        self.turning && self.mouse_delta != vec2(0.0, 0.0) ||
        self.forward_backward.value != 0 ||
        self.left_right.value != 0 ||
        self.up_down.value != 0 ||
        self.roll.value != 0
    }
}