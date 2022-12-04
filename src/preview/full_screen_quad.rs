use luminance::pipeline::PipelineError;
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
}

pub struct FullScreenQuad {
    tess: Tess<Vertex, VertexIndex, (), Interleaved>,
    shader: Program<VertexSemantics, (), ShaderInterface>,
    blending: Blending,
    texture: LuminanceTexture<Dim2, NormRGB8UI>,
}

impl FullScreenQuad {
    pub fn create<C>(context: &mut C) -> Result<Self, TessError>
    where
        C: GraphicsContext<Backend = Backend>,
    {
        let indices = vec![0, 1, 2, 3];
        let vertices = vec![
            Vertex {
                position: [-1.0, -1.0].into(),
                uv: [0.0, 0.0].into(),
            },
            Vertex {
                position: [1.0, -1.0].into(),
                uv: [1.0, 0.0].into(),
            },
            Vertex {
                position: [-1.0, 1.0].into(),
                uv: [0.0, 1.0].into(),
            },
            Vertex {
                position: [1.0, 1.0].into(),
                uv: [1.0, 1.0].into(),
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

        let texture = make_texture(context, [1, 1], &[0, 0, 0]);

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
        })
    }

    pub fn render(&mut self, shd_gate: &mut ShadingGate, pipeline: Pipeline) -> Result<(), PipelineError> {
        let tex = pipeline.bind_texture(&mut self.texture)?;

        shd_gate.shade(&mut self.shader, |mut iface, unif, mut rdr_gate| {
            iface.set(&unif.u_texture, tex.binding());
            let render_state = RenderState::default().set_blending(self.blending);

            rdr_gate.render(&render_state, |mut tess_gate| tess_gate.render(&self.tess))
        })
    }
}

fn make_texture<C>(context: &mut C, size: [u32; 2], image_buffer: &[u8]) -> LuminanceTexture<Dim2, NormRGB8UI>
where
    C: GraphicsContext<Backend = Backend>,
{
    let sampler = Sampler {
        wrap_r: Wrap::ClampToEdge,
        wrap_s: Wrap::ClampToEdge,
        wrap_t: Wrap::ClampToEdge,
        min_filter: MinFilter::LinearMipmapLinear,
        mag_filter: MagFilter::Linear,
        depth_comparison: None,
    };

    let upload = TexelUpload::base_level(image_buffer, 0);

    context
        .new_texture_raw(size, sampler, upload)
        .expect("unable to upload preview texture")
}
