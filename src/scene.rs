use std::path::Path;

use cgmath::{Matrix4, Quaternion, Point3, Vector4, SquareMatrix, vec4};
use gltf::scene::Transform;
use gltf::camera::Projection;

use crate::{
    mesh::{Vertex, VertexIndex, Mesh},
    camera::PerspectiveCamera,
    material::Material,
    light::Light, 
    color::RGBf32,
    environment::Environment,
};

pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub lights: Vec<Light>,
    pub camera: PerspectiveCamera,
    pub environment: Environment,
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
        match  gltf::import(path) {
            Ok((document, buffers, _)) => {
                let gamma = 2.2;
                let environment = Environment {
                    color: RGBf32::from_hex("#404040").pow(gamma),
                };

                let mut result = Scene {
                    meshes: Vec::new(),
                    lights: Vec::new(),
                    camera: PerspectiveCamera::default(),
                    environment,
                };
                
                for scene in document.scenes() {
                    result.parse_nodes(scene.nodes().collect(), &buffers, Matrix4::identity());
                }


                Ok(result)
            }
            Err(e) => {
                Err(format!("An error occured while opening the glTF file:\n\t{}", e))
            }
        }
    }

    fn parse_nodes(&mut self,
        nodes: Vec<gltf::Node>,
        buffers: &[gltf::buffer::Data], 
        base_transform: Matrix4<f32>) {

        for node in nodes {
            let transform = base_transform * transform_to_mat(node.transform());

            if let Some(mesh) = node.mesh() {
                self.add_meshes_from_gltf_mesh(mesh, buffers, transform);
            } else if let Some(cam) = node.camera() {
                if let Projection::Perspective(perspective) = cam.projection() {
                    self.camera = PerspectiveCamera {
                        aspect_ratio: perspective.aspect_ratio().unwrap(),
                        y_fov: perspective.yfov(),
                        z_far: perspective.zfar().unwrap(),
                        z_near: perspective.znear(),
                        view: transform.invert().unwrap(),
                        model: transform
                    }
                } else {
                    println!("Non-perspective cameras are not supported");
                }
            } else if let Some(light) = node.light() {
                if let Some(light) = parse_light(light, transform) {
                    self.lights.push(light);
                }
            }

            self.parse_nodes(node.children().collect(), buffers, transform);
        }
    }

    fn add_meshes_from_gltf_mesh(&mut self, mesh: gltf::Mesh, buffers: &[gltf::buffer::Data], transform: Matrix4<f32>) {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let mat = primitive.material();
            let pbr = mat.pbr_metallic_roughness();
            let base = pbr.base_color_factor();
            
            let material = Material {
                base_color: RGBf32::new(base[0], base[1], base[2]),
                roughness: pbr.roughness_factor(),
                metallic: pbr.metallic_factor(),
                emissive: mat.emissive_factor().into(),
            };

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
            
            //TODO get more optional vertex data.
            self.meshes.push(Mesh::new(vertices, indices, material));
        }
    }
}

fn parse_light(light: gltf::khr_lights_punctual::Light, transform: Matrix4<f32>) -> Option<Light> {
    let kind = match light.kind() {
        gltf::khr_lights_punctual::Kind::Point => crate::light::Kind::Point,
        gltf::khr_lights_punctual::Kind::Directional => crate::light::Kind::Directional,
        gltf::khr_lights_punctual::Kind::Spot { .. }=> crate::light::Kind::Spot,
    };

    // FIXME: remove this once all light types are supported.
    if kind != crate::light::Kind::Point {
        println!("Lights of type {:?} are not yet implemented.", kind);
        return None;
    }

    let range;

    if let Some(r) = light.range() {
        range = r;
    } else {
        range = f32::INFINITY;
    }

    Some(Light {
        pos: Point3::from_homogeneous(transform * vec4(0.0, 0.0, 0.0, 1.0)),
        intensity: light.intensity(),
        range,
        color: light.color().into(),
        kind
    })
}