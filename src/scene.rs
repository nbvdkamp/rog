use std::path::Path;
use std::time::Instant;

use gltf::scene::Transform;
use cgmath::{
    Quaternion,
    Point3,
    Vector2, Vector3, Vector4,
    vec3, vec4,
    InnerSpace,
    Matrix, SquareMatrix,
    Matrix3, Matrix4,
};
use gltf::camera::Projection;

use crate::{
    mesh::{Vertex, Mesh},
    camera::PerspectiveCamera,
    constants::GAMMA,
    material::Material,
    texture::{Texture, Format},
    light::Light, 
    color::RGBf32,
    environment::Environment,
};

use rgb2spec::RGB2Spec;

pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub lights: Vec<Light>,
    pub textures: Vec<Texture>,
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
        let rgb2spec = match RGB2Spec::load("res/out.spec") {
            Ok(rgb2spec) => rgb2spec,
            Err(e) => panic!("Can't load rgb2spec file: {}", e),
        };

        let start = Instant::now();

        match  gltf::import(path) {
            Ok((document, buffers, images)) => {
                let lib_time = start.elapsed().as_secs_f32();

                let textures = images.into_iter().map(|i| {
                    let format = match i.format {
                        gltf::image::Format::R8G8B8 => Format::RGB,
                        gltf::image::Format::R8G8B8A8 => Format::RGBA,
                        other => panic!("Texture format {:?} is not implemented", other)
                    };

                    Texture::new(i.pixels, i.width, i.height, format)
                }).collect();

                let environment = Environment {
                    color: RGBf32::from_hex("#404040").pow(GAMMA),
                };

                let mut result = Scene {
                    meshes: Vec::new(),
                    lights: Vec::new(),
                    textures,
                    camera: PerspectiveCamera::default(),
                    environment,
                };
                
                for scene in document.scenes() {
                    result.parse_nodes(scene.nodes().collect(), &buffers, Matrix4::identity(), &rgb2spec);
                }

                let total_time = start.elapsed().as_secs_f32();
                println!("Parsed scene in {} seconds ({} in library, {} in own code)", total_time, lib_time, total_time - lib_time);

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
        base_transform: Matrix4<f32>,
        rgb2spec: &RGB2Spec) {

        for node in nodes {
            let transform = base_transform * transform_to_mat(node.transform());

            if let Some(mesh) = node.mesh() {
                self.add_meshes_from_gltf_mesh(mesh, buffers, transform, rgb2spec);
            } else if let Some(cam) = node.camera() {
                if let Projection::Perspective(perspective) = cam.projection() {
                    self.camera = PerspectiveCamera {
                        aspect_ratio: perspective.aspect_ratio().unwrap_or(16.0 / 9.0),
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

            self.parse_nodes(node.children().collect(), buffers, transform, rgb2spec);
        }
    }

    fn add_meshes_from_gltf_mesh(&mut self, mesh: gltf::Mesh, buffers: &[gltf::buffer::Data], transform: Matrix4<f32>, rgb2spec: &RGB2Spec) {
        let m = Matrix3::from_cols(transform.x.truncate(), transform.y.truncate(), transform.z.truncate());
        let normal_transform = m.invert().unwrap().transpose();

        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let mat = primitive.material();
            let pbr = mat.pbr_metallic_roughness();
            let base = pbr.base_color_factor();

            let base_color = RGBf32::new(base[0], base[1], base[2]);
            let base_color_coefficients = rgb2spec.fetch([base_color.r, base_color.g, base_color.b]);

            let get_index = |t: gltf::texture::Info| t.texture().source().index();
            let base_color_texture= pbr.base_color_texture().map(get_index);

            base_color_texture.map(|i| self.textures[i].create_spectrum_coefficients(rgb2spec));

            let material = Material {
                base_color,
                base_color_coefficients,
                base_color_texture,
                roughness: pbr.roughness_factor(),
                metallic: pbr.metallic_factor(),
                metallic_roughness_texture: pbr.metallic_roughness_texture().map(get_index),
                emissive: mat.emissive_factor().into(),
                emissive_texture: mat.emissive_texture().map(get_index),
                normal_texture: mat.normal_texture().map(|t| t.texture().source().index()),
            };

            let positions = reader
                    .read_positions()
                    .unwrap_or_else(||
                        panic!("Primitive does not have POSITION attribute (mesh: {}, primitive: {})", mesh.index(), primitive.index())
                    )
                    .map(|pos| Point3::from_homogeneous(transform * Vector4::new(pos[0], pos[1], pos[2], 1.0)));
            
            let normals = reader
                    .read_normals()
                    .unwrap_or_else(||
                        panic!("Primitive does not have NORMAL attribute (mesh: {}, primitive: {})", mesh.index(), primitive.index())
                    ).map(|normal| {
                        normal_transform * Vector3::from(normal)
                    });

            let tangents = reader.read_tangents();

            if material.normal_texture.is_some() && !tangents.is_some() {
                println!("Primitive has normal map but no TANGENT attribute! (mesh: {}, primitive: {})",
                    mesh.name().unwrap_or(&mesh.index().to_string()), primitive.index());
            }

            let tex_coords = reader.read_tex_coords(0)
                    .map(|read_tex_coords| 
                        read_tex_coords.into_f32()
                    ).map(|iter| iter.map(|uv| Vector2::<f32>::new(uv[0], uv[1])));

            // FIXME: Find a way to remove the duplication
            let vertices = if let Some(tex_coords) = tex_coords {
                if let Some(tangents) = tangents {
                    let tangents = tangents
                            .map(|tangent| {
                                normal_transform * (tangent[3] * vec3(tangent[0], tangent[1], tangent[2]))
                            });

                    positions.zip(normals).zip(tex_coords.zip(tangents))
                    .map(|((position, normal), (tex_coord, tangent))| {
                        Vertex {
                            position,
                            normal,
                            tangent: Some(tangent),
                            tex_coord: Some(tex_coord),
                        }
                    }).collect()
                } else {
                    positions.zip(normals).zip(tex_coords)
                    .map(|((position, normal), tex_coord)| {
                        Vertex {
                            position,
                            normal,
                            tangent: None,
                            tex_coord: Some(tex_coord),
                        }
                    }).collect()
                }
            } else {
                positions.zip(normals)
                .map(|(position, normal)| {
                    Vertex {
                        position,
                        normal,
                        tangent: None,
                        tex_coord: None,
                    }
                }).collect()
            };
            
            let indices = reader
                .read_indices()
                .map(|read_indices| {
                    read_indices.into_u32().collect::<Vec<_>>()
                })
                .unwrap_or_else(||
                    panic!("Primitive has no indices (mesh: {}, primitive: {})", mesh.index(), primitive.index())
                );
            
            self.meshes.push(Mesh::new(vertices, indices, material));
        }
    }
}

fn parse_light(light: gltf::khr_lights_punctual::Light, transform: Matrix4<f32>) -> Option<Light> {
    let kind = match light.kind() {
        gltf::khr_lights_punctual::Kind::Point => crate::light::Kind::Point,
        gltf::khr_lights_punctual::Kind::Directional => {
            let m = Matrix3::from_cols(transform.x.truncate(), transform.y.truncate(), transform.z.truncate());
            let normal_transform = m.invert().unwrap().transpose();
            let direction = -(normal_transform * vec3(0.0, 0.0, -1.0)).normalize();
            crate::light::Kind::Directional { direction }
        },
        gltf::khr_lights_punctual::Kind::Spot { inner_cone_angle, outer_cone_angle} => {
            crate::light::Kind::Spot { inner_cone_angle, outer_cone_angle }
        },
    };

    Some(Light {
        pos: Point3::from_homogeneous(transform * vec4(0.0, 0.0, 0.0, 1.0)),
        intensity: light.intensity(),
        range: light.range().unwrap_or(f32::INFINITY),
        color: light.color().into(),
        kind
    })
}