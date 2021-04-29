use crate::mesh::{Vertex, VertexIndex, Mesh};
use std::path::Path;

pub struct Scene {
    pub meshes: Vec<Mesh>,
}

impl Scene {
    pub fn load<P>(path: P) -> Result<Self, String>
    where
        P: AsRef<Path>,
    {
        if let Ok((document, buffers, _)) = gltf::import(path) {
            let mut meshes = Vec::<Mesh>::new();
            
            for scene in document.scenes() {
                for node in scene.nodes() {
                    if let Some(mesh) = node.mesh() {
                        for primitive in mesh.primitives() {
                            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                            let positions = {
                                let iter = reader
                                    .read_positions()
                                    .unwrap_or_else(||
                                        panic!("Primitive does not have POSITION attribute (mesh: {}, primitive: {})", mesh.index(), primitive.index())
                                    );
                                iter.collect::<Vec<_>>()
                            };

                            let vertices: Vec<Vertex> = positions
                                .into_iter()
                                .map(|position| {
                                    Vertex {
                                        position: position.into()
                                    }
                                }).collect();
                            
                            let indices: Vec<VertexIndex> = reader
                                .read_indices()
                                .map(|read_indices| {
                                    read_indices.into_u32().collect::<Vec<_>>()
                                })
                                .unwrap_or_else(||
                                    panic!("Primitive has no indices (mesh: {}, primitive: {})", mesh.index(), primitive.index())
                                );
                            
                            //TODO get normals and stuff.
                            meshes.push(Mesh { vertices, indices });
                        }
                    }
                }
            }

            Ok(Scene { meshes })
        }
        else {
            Err("Couldn't open glTF file.".into())
        }
    }
}