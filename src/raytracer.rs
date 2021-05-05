use std::convert::TryFrom;
use cgmath::{Point3, Vector3};

mod ray;
mod triangle;
use triangle::Triangle;
use crate::{material::Material, mesh::{Mesh, Vertex}};

pub struct Raytracer {
    verts: Vec<Vertex>,
    triangles: Vec<Triangle>,
    materials: Vec<Material>,
}

impl Raytracer {
    pub fn new(meshes: Vec<Mesh>) -> Self {
        let mut result = Raytracer { verts: Vec::new(), triangles: Vec::new(), materials: Vec::new() };

        for mesh in meshes {
            let start_index = u32::try_from(result.verts.len()).unwrap();
            let material_index = u32::try_from(result.materials.len()).unwrap();

            for v in mesh.vertices {
                result.verts.push(v);
            }

            result.materials.push(mesh.material);

            for i in (0..mesh.indices.len()).step_by(3) {
                result.triangles.push(Triangle {
                    index1: mesh.indices[i] + start_index,
                    index2: mesh.indices[i + 1] + start_index,
                    index3: mesh.indices[i + 2] + start_index,
                    material_index 
                });
            }
        };

        result
    }
}