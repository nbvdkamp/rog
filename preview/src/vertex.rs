use cgmath::Point2;
use luminance_derive::{Semantics, Vertex};
use luminance_front::{
    context::GraphicsContext,
    tess::{Interleaved, Mode, Tess, TessError},
    Backend,
};
use renderer::mesh::{Mesh, VertexIndex};

#[derive(Copy, Clone, Debug, Semantics)]
pub enum VertexSemantics {
    #[sem(name = "position", repr = "[f32; 3]", wrapper = "VertexPosition")]
    Position,
    #[sem(name = "normal", repr = "[f32; 3]", wrapper = "VertexNormal")]
    Normal,
    // Not all vertices actually have UVs but for simplicity just default to 0,0
    #[sem(name = "uv", repr = "[f32; 2]", wrapper = "VertexUV")]
    TextureCoords,
}

#[repr(C)]
#[derive(Copy, Clone, Vertex)]
#[vertex(sem = "VertexSemantics")]
pub struct LuminanceVertex {
    pub position: VertexPosition,
    pub normal: VertexNormal,
    pub uv: VertexUV,
}

pub fn mesh_to_tess<C>(
    mesh: &Mesh,
    context: &mut C,
) -> Result<Tess<LuminanceVertex, VertexIndex, (), Interleaved>, TessError>
where
    C: GraphicsContext<Backend = Backend>,
{
    let indices: Vec<_> = mesh
        .triangles
        .iter()
        .flat_map(|tri| [tri.indices[0], tri.indices[1], tri.indices[2]])
        .collect();

    let luminance_vertices: Vec<_> = (0..mesh.vertices.len())
        .map(|i| {
            let pos: [f32; 3] = mesh.vertices.positions[i].into();
            let norm: [f32; 3] = mesh.vertices.normals[i].into();
            let uv = if let Some(tex_coords) = mesh.vertices.tex_coords.first() {
                let Point2 { x: u, y: v } = tex_coords[i];
                [u, v]
            } else {
                [0.0, 0.0]
            };
            LuminanceVertex {
                position: pos.into(),
                normal: norm.into(),
                uv: uv.into(),
            }
        })
        .collect();

    context
        .new_tess()
        .set_mode(Mode::Triangle)
        .set_vertices(luminance_vertices)
        .set_indices(indices)
        .build()
}
