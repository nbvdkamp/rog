use luminance_derive::{Semantics, Vertex};
use luminance_front::context::GraphicsContext;
use luminance_front::tess::{Mode, Tess, TessError, Interleaved};
use luminance_front::Backend;

use crate::material::Material;

#[derive(Copy, Clone, Debug, Semantics)]
pub enum VertexSemantics {
    #[sem(name = "position", repr = "[f32; 3]", wrapper = "VertexPosition")]
    Position,   
    #[sem(name = "normal", repr = "[f32; 3]", wrapper = "VertexNormal")]
    Normal,   
}

#[derive(Copy, Clone, Vertex)]
#[vertex(sem = "VertexSemantics")]
pub struct Vertex {
    #[allow(dead_code)]
    pub position: VertexPosition,
    #[allow(dead_code)]
    pub normal: VertexNormal,
}

pub type VertexIndex = u32;

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<VertexIndex>,
    pub material: Material,
}

impl Mesh {
    pub fn to_tess<C>(&self, context: &mut C) -> Result<Tess<Vertex, VertexIndex, (), Interleaved>, TessError>
    where
        C: GraphicsContext<Backend = Backend>,
    {
        context
            .new_tess()
            .set_mode(Mode::Triangle)
            .set_vertices(self.vertices.clone())
            .set_indices(self.indices.clone())
            .build()
    }
}