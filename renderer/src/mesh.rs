use cgmath::{Matrix4, Point2, Point3, Vector3};

use crate::{
    material::Material,
    raytracer::{aabb::BoundingBox, triangle::Triangle},
};

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
