use crate::mesh::{Vertex, VertexIndex, Mesh};
use std::path::Path;
use cgmath::{Matrix4, Vector4, Quaternion};
use gltf::scene::Transform;

pub struct Scene {
    pub meshes: Vec<Mesh>,
}

fn transform_to_mat(t: Transform) -> Matrix4<f32> {
    match t {
        Transform::Matrix { matrix } => { matrix.into() }
        Transform::Decomposed { translation, rotation, scale } => { 
            let r: Matrix4<f32> = Quaternion::from(rotation).into();
            let t = Matrix4::from_translation(translation.into());
            let s = Matrix4::from_nonuniform_scale(scale[0], scale[1], scale[2]);
            t * r * s
        }
    }
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
                    let transform = transform_to_mat(node.transform());

                    if let Some(mesh) = node.mesh() {
                        for primitive in mesh.primitives() {
                            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                            let positions = {
                                let iter = reader
                                    .read_positions()
                                    .unwrap_or_else(||
                                        panic!("Primitive does not have POSITION attribute (mesh: {}, primitive: {})", mesh.index(), primitive.index())
                                    )
                                    .map(|pos| {
                                        let v = transform * Vector4::new(pos[0], pos[1], pos[2], 1.0);
                                        [v[0] / v[3], v[1] / v[3], v[2] / v[3]]
                                    });
                                
                                iter.collect::<Vec<_>>()
                            };
                            
                            let normals= {
                                let iter = reader
                                    .read_normals()
                                    .unwrap_or_else(||
                                        panic!("Primitive does not have NORMAL attribute (mesh: {}, primitive: {})", mesh.index(), primitive.index())
                                    );
                                iter.collect::<Vec<_>>()
                            };

                            let vertices: Vec<Vertex> = positions
                                .into_iter()
                                .zip(normals.into_iter())
                                .map(|(position, normal)| {
                                    Vertex {
                                        position: position.into(),
                                        normal: normal.into(),
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