use arrayvec::ArrayVec;
use cgmath::{Matrix4, Point2, Point3, Vector3};

use crate::{
    material::Material,
    raytracer::{aabb::BoundingBox, triangle::Triangle},
};

pub const MAX_TEX_COORD_SETS: usize = 4;
pub type TextureCoordinates = ArrayVec<Point2<f32>, MAX_TEX_COORD_SETS>;

pub struct Vertices {
    pub positions: Vec<Point3<f32>>,
    pub normals: Vec<Vector3<f32>>,
    pub tangents: Vec<Vector3<f32>>,
    pub tex_coords: Vec<Vec<Point2<f32>>>,
}

pub struct TriangleVertices {
    pub positions: [Point3<f32>; 3],
    pub normals: [Vector3<f32>; 3],
    pub tangents: [Vector3<f32>; 3],
    pub tex_coords: ArrayVec<[Point2<f32>; 3], MAX_TEX_COORD_SETS>,
}

impl Vertices {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn get(&self, indices: [usize; 3]) -> TriangleVertices {
        TriangleVertices {
            positions: indices.map(|i| self.positions[i]),
            normals: indices.map(|i| self.normals[i]),
            tangents: indices.map(|i| self.tangents[i]),
            tex_coords: self.tex_coords.iter().map(|t| indices.map(|i| t[i])).collect(),
        }
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
