use luminance_derive::{Semantics, Vertex};
use luminance_front::{
    context::GraphicsContext,
    tess::{Interleaved, Mode, Tess, TessError},
    Backend,
};

use cgmath::{Point2, Point3, Vector3};

use crate::material::Material;

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

#[derive(Copy, Clone, Vertex)]
#[vertex(sem = "VertexSemantics")]
pub struct LuminanceVertex {
    pub position: VertexPosition,
    pub normal: VertexNormal,
    pub uv: VertexUV,
}

#[derive(Clone)]
pub struct Vertex {
    pub position: Point3<f32>,
    pub normal: Vector3<f32>,
    pub tangent: Vector3<f32>,
    pub tex_coord: Option<Point2<f32>>,
}

pub type VertexIndex = u32;

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<VertexIndex>,
    pub material: Material,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<VertexIndex>, material: Material) -> Self {
        Mesh {
            vertices,
            indices,
            material,
        }
    }

    pub fn to_tess<C>(&self, context: &mut C) -> Result<Tess<LuminanceVertex, VertexIndex, (), Interleaved>, TessError>
    where
        C: GraphicsContext<Backend = Backend>,
    {
        let luminance_vertices = self
            .vertices
            .iter()
            .map(|v| {
                let pos: [f32; 3] = v.position.into();
                let norm: [f32; 3] = v.normal.into();
                LuminanceVertex {
                    position: pos.into(),
                    normal: norm.into(),
                    uv: match (v.tex_coord, self.material.base_color_texture) {
                        (Some(uv), Some(tex_ref)) => {
                            let Point2 { x: u, y: v } = tex_ref.transform_texture_coordinates(uv);
                            [u, v].into()
                        }
                        _ => [0.0, 0.0].into(),
                    },
                }
            })
            .collect::<Vec<_>>();

        context
            .new_tess()
            .set_mode(Mode::Triangle)
            .set_vertices(luminance_vertices)
            .set_indices(self.indices.clone())
            .build()
    }
}
