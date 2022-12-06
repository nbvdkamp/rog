use cgmath::{vec2, Vector2};
use glfw::{Action, Key, MouseButton, WindowEvent};
use luminance::{pipeline::PipelineError, shader::types::Vec2};
use luminance_derive::{Semantics, UniformInterface, Vertex};
use luminance_front::{
    blending::{Blending, Equation, Factor},
    context::GraphicsContext,
    pipeline::{Pipeline, TextureBinding},
    pixel::{NormRGB8UI, NormUnsigned},
    render_state::RenderState,
    shader::{Program, Uniform},
    shading_gate::ShadingGate,
    tess::{Interleaved, Mode, Tess, TessError},
    texture::{Dim2, MagFilter, MinFilter, Sampler, TexelUpload, Texture as LuminanceTexture, Wrap},
    Backend,
};

use crate::color::RGBu8;

#[derive(Copy, Clone, Debug, Semantics)]
pub enum VertexSemantics {
    #[sem(name = "position", repr = "[f32; 2]", wrapper = "VertexPosition")]
    Position,
    #[sem(name = "uv", repr = "[f32; 2]", wrapper = "VertexUV")]
    TextureCoords,
}

#[derive(Copy, Clone, Vertex)]
#[vertex(sem = "VertexSemantics")]
pub struct Vertex {
    pub position: VertexPosition,
    pub uv: VertexUV,
}

type VertexIndex = u32;

const VS_STR: &str = include_str!("full_screen_quad/vertex.vs");
const FS_STR: &str = include_str!("full_screen_quad/fragment.fs");

#[derive(Debug, UniformInterface)]
struct ShaderInterface {
    u_texture: Uniform<TextureBinding<Dim2, NormUnsigned>>,
    u_scale: Uniform<Vec2<f32>>,
    u_translation: Uniform<Vec2<f32>>,
}

pub struct PreviewQuad {
    tess: Tess<Vertex, VertexIndex, (), Interleaved>,
    shader: Program<VertexSemantics, (), ShaderInterface>,
    blending: Blending,
    // Having two copies of the texture is not an efficient way of changing
    // the magnification filter but Luminance doesn't seem to have a better way.
    texture: LuminanceTexture<Dim2, NormRGB8UI>,
    zoomed_texture: LuminanceTexture<Dim2, NormRGB8UI>,
    scale: f32,
    translation: Vector2<f32>,
    mouse_delta: Vector2<f32>,
    mouse_position: Vector2<f32>,
    dragging: bool,
}

impl PreviewQuad {
    pub fn create<C>(context: &mut C) -> Result<Self, TessError>
    where
        C: GraphicsContext<Backend = Backend>,
    {
        let indices = vec![0, 1, 2, 3];
        let vertices = vec![
            Vertex {
                position: [-1.0, -1.0].into(),
                uv: [0.0, 1.0].into(),
            },
            Vertex {
                position: [1.0, -1.0].into(),
                uv: [1.0, 1.0].into(),
            },
            Vertex {
                position: [-1.0, 1.0].into(),
                uv: [0.0, 0.0].into(),
            },
            Vertex {
                position: [1.0, 1.0].into(),
                uv: [1.0, 0.0].into(),
            },
        ];

        let tess = context
            .new_tess()
            .set_mode(Mode::TriangleStrip)
            .set_vertices(vertices)
            .set_indices(indices)
            .build()?;

        let shader = context
            .new_shader_program::<VertexSemantics, (), ShaderInterface>()
            .from_strings(VS_STR, None, None, FS_STR)
            .unwrap()
            .ignore_warnings();

        let texture = make_texture(context, [1, 1], &[0, 0, 0], MagFilter::Linear);
        let zoomed_texture = make_texture(context, [1, 1], &[0, 0, 0], MagFilter::Nearest);

        let blending = Blending {
            equation: Equation::Additive,
            src: Factor::SrcAlpha,
            dst: Factor::SrcAlphaComplement,
        };

        Ok(Self {
            tess,
            shader,
            blending,
            texture,
            zoomed_texture,
            scale: 1.0,
            translation: vec2(0.0, 0.0),
            mouse_position: vec2(0.0, 0.0),
            mouse_delta: vec2(0.0, 0.0),
            dragging: false,
        })
    }

    pub fn render(&mut self, shd_gate: &mut ShadingGate, pipeline: Pipeline) -> Result<(), PipelineError> {
        let tex = if self.scale < 4.0 {
            pipeline.bind_texture(&mut self.texture)?
        } else {
            pipeline.bind_texture(&mut self.zoomed_texture)?
        };

        shd_gate.shade(&mut self.shader, |mut iface, unif, mut rdr_gate| {
            iface.set(&unif.u_texture, tex.binding());
            iface.set(&unif.u_scale, [self.scale, self.scale].into());
            let translation: [f32; 2] = self.translation.into();
            iface.set(&unif.u_translation, translation.into());
            let render_state = RenderState::default().set_blending(self.blending);

            rdr_gate.render(&render_state, |mut tess_gate| tess_gate.render(&self.tess))
        })
    }

    pub fn update_texture<C>(&mut self, context: &mut C, size: [u32; 2], image_buffer: &[RGBu8])
    where
        C: GraphicsContext<Backend = Backend>,
    {
        let data = image_buffer.as_ptr() as *const u8;
        let len = image_buffer.len() * std::mem::size_of::<RGBu8>();

        let image_buffer = unsafe { std::slice::from_raw_parts(data, len) };
        self.texture = make_texture(context, size, image_buffer, MagFilter::Linear);
        self.zoomed_texture = make_texture(context, size, image_buffer, MagFilter::Nearest);
    }

    pub fn handle_event(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::MouseButton(MouseButton::Button1, action, _) => match action {
                Action::Press => self.dragging = true,
                Action::Release => self.dragging = false,
                Action::Repeat => (),
            },
            WindowEvent::CursorPos(x, y) => {
                let pos = vec2(x as f32, y as f32);
                self.mouse_delta = pos - self.mouse_position;
                self.mouse_position = pos;

                if self.dragging {
                    self.translation += 0.002 * vec2(self.mouse_delta.x, -self.mouse_delta.y);
                }
            }
            WindowEvent::Scroll(_, y_offset) => {
                self.scale *= if y_offset > 0.0 { 1.2 } else { 0.8 };
            }
            WindowEvent::Key(Key::Space, _, Action::Press, _) => {
                self.scale = 1.0;
                self.translation = vec2(0.0, 0.0);
            }
            _ => (),
        }
    }
}

fn make_texture<C>(
    context: &mut C,
    size: [u32; 2],
    image_buffer: &[u8],
    mag_filter: MagFilter,
) -> LuminanceTexture<Dim2, NormRGB8UI>
where
    C: GraphicsContext<Backend = Backend>,
{
    let sampler = Sampler {
        wrap_r: Wrap::ClampToEdge,
        wrap_s: Wrap::ClampToEdge,
        wrap_t: Wrap::ClampToEdge,
        min_filter: MinFilter::LinearMipmapLinear,
        mag_filter,
        depth_comparison: None,
    };

    let upload = TexelUpload::base_level(image_buffer, 0);

    context
        .new_texture_raw(size, sampler, upload)
        .expect("unable to upload preview texture")
}
