use luminance_derive::{Semantics, Vertex};
use luminance_front::{
    context::GraphicsContext,
    tess::{Interleaved, Mode, Tess, TessError},
    Backend,
};

use cgmath::{Matrix4, Point2, Point3, Vector3};

use crate::{
    material::Material,
    raytracer::{aabb::BoundingBox, triangle::Triangle},
};

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

pub struct Vertices {
    pub positions: Vec<Point3<f32>>,
    pub normals: Vec<Vector3<f32>>,
    pub tangents: Vec<Vector3<f32>>,
    pub tex_coords: Vec<Vec<Point2<f32>>>,
}

impl Vertices {
    pub fn len(&self) -> usize {
        self.positions.len()
    }
}

pub type VertexIndex = u32;

pub struct Mesh {
    pub vertices: Vertices,
    pub triangles: Vec<Triangle>,
}

impl Mesh {
    pub fn new(vertices: Vertices, triangles: Vec<Triangle>) -> Self {
        Mesh { vertices, triangles }
    }

    pub fn to_tess<C>(&self, context: &mut C) -> Result<Tess<LuminanceVertex, VertexIndex, (), Interleaved>, TessError>
    where
        C: GraphicsContext<Backend = Backend>,
    {
        let indices: Vec<_> = self
            .triangles
            .iter()
            .flat_map(|tri| [tri.indices[0], tri.indices[1], tri.indices[2]])
            .collect();

        let luminance_vertices: Vec<_> = (0..self.vertices.len())
            .map(|i| {
                let pos: [f32; 3] = self.vertices.positions[i].into();
                let norm: [f32; 3] = self.vertices.normals[i].into();
                let uv = if let Some(tex_coords) = self.vertices.tex_coords.first() {
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
}

#[derive(Clone)]
pub struct Instance {
    pub mesh_index: u32,
    pub transform: Matrix4<f32>,
    pub inverse_transform: Matrix4<f32>,
    pub bounds: BoundingBox,
    pub material: Material,
}

impl Instance {
    pub fn new(
        mesh_index: usize,
        mesh: &Mesh,
        transform: Matrix4<f32>,
        inverse_transform: Matrix4<f32>,
        material: Material,
    ) -> Self {
        let mut bounds = BoundingBox::new();

        for &position in &mesh.vertices.positions {
            let transformed = transform * position.to_homogeneous();
            bounds.add(Point3::from_homogeneous(transformed));
        }

        Instance {
            mesh_index: mesh_index as u32,
            transform,
            inverse_transform,
            bounds,
            material,
        }
    }
}
