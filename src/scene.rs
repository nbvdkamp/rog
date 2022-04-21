use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use cgmath::{Matrix4, Quaternion, Point3, Vector4, SquareMatrix, vec4, Vector2};
use gltf::{scene::Transform};
use gltf::camera::Projection;

use crate::{
    mesh::{Vertex, Mesh},
    camera::PerspectiveCamera,
    material::Material,
    texture::Texture,
    light::Light, 
    color::RGBf32,
    environment::Environment,
};

pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub lights: Vec<Light>,
    pub textures: Vec<Texture>,
    pub camera: PerspectiveCamera,
    pub environment: Environment,
    texture_indices: HashMap<TextureView, usize>,
}

/// Used to keep track of which Textures were loaded from the glTF
#[derive(PartialEq, Eq, Hash)]
struct TextureView {
    buffer_index: usize,
    offset: usize,
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
        let start = Instant::now();

        match  gltf::import(path) {
            Ok((document, buffers, _)) => {
                let lib_time = start.elapsed().as_secs_f32();

                let gamma = 2.2;
                let environment = Environment {
                    color: RGBf32::from_hex("#404040").pow(gamma),
                };

                let mut result = Scene {
                    meshes: Vec::new(),
                    lights: Vec::new(),
                    textures: Vec::new(),
                    camera: PerspectiveCamera::default(),
                    environment,
                    texture_indices: HashMap::new(),
                };
                
                for scene in document.scenes() {
                    result.parse_nodes(scene.nodes().collect(), &buffers, Matrix4::identity());
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
        base_transform: Matrix4<f32>) {

        for node in nodes {
            let transform = base_transform * transform_to_mat(node.transform());

            if let Some(mesh) = node.mesh() {
                self.add_meshes_from_gltf_mesh(mesh, buffers, transform);
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

            self.parse_nodes(node.children().collect(), buffers, transform);
        }
    }

    fn add_meshes_from_gltf_mesh(&mut self, mesh: gltf::Mesh, buffers: &[gltf::buffer::Data], transform: Matrix4<f32>) {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let mat = primitive.material();
            let pbr = mat.pbr_metallic_roughness();
            let base = pbr.base_color_factor();

            let mut add_texture = |texture_info: Option<gltf::texture::Info>|
                texture_info.map(|t| {
                    self.add_texture(t.texture(), buffers)
                });

            let material = Material {
                base_color: RGBf32::new(base[0], base[1], base[2]),
                base_color_texture: add_texture(pbr.base_color_texture()),
                roughness: pbr.roughness_factor(),
                metallic: pbr.metallic_factor(),
                metallic_roughness_texture: add_texture(pbr.metallic_roughness_texture()),
                emissive: mat.emissive_factor().into(),
                emissive_texture: add_texture(mat.emissive_texture()),
                normal_texture: mat.normal_texture().map(|t| self.add_texture(t.texture(), buffers)),
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
                    );

            let tex_coords = reader.read_tex_coords(0)
                    .map(|read_tex_coords| 
                        read_tex_coords.into_f32()
                    ).map(|iter| iter.map(|uv| Vector2::<f32>::new(uv[0], uv[1])));

            // FIXME: Find a way to remove the duplication
            let vertices = if let Some(tex_coords) = tex_coords {
                positions.zip(normals).zip(tex_coords)
                .map(|((position, normal), tex_coord)| {
                    Vertex {
                        position,
                        normal: normal.into(),
                        tex_coord: Some(tex_coord),
                    }
                }).collect()
            } else {
                positions.zip(normals)
                .map(|(position, normal)| {
                    Vertex {
                        position,
                        normal: normal.into(),
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
            
            //TODO get more optional vertex data.
            self.meshes.push(Mesh::new(vertices, indices, material));
        }
    }

    /// Adds the texture if it hasn't been added yet, returns its index 
    fn add_texture(&mut self, texture: gltf::Texture, buffers: &[gltf::buffer::Data]) -> usize {
        let texture = texture.source().source();

        match texture {
            gltf::image::Source::View { view, .. } => {
                let buffer_index = view.buffer().index();
                let start = view.offset();
                let end = start + view.length();
                let texture_view = TextureView { buffer_index, offset: start };

                if let Some(index) = self.texture_indices.get(&texture_view) {
                    return *index;
                }

                let cursor = std::io::Cursor::new(&buffers[buffer_index][start..end]);

                let reader = match image::io::Reader::new(cursor).with_guessed_format() {
                    Ok(reader) => reader,
                    Err(e) => {
                        panic!("Inferring image type failed: {}", e);
                    }
                };

                let image = match reader.decode() {
                    Ok(image) => image.to_rgba8(),
                    Err(e) => {
                        panic!("Decoding image failed: {}", e);
                    }
                };

                self.textures.push(Texture::new(image));
                let index = self.textures.len() - 1;
                self.texture_indices.insert(texture_view, index);

                index
            },
            gltf::image::Source::Uri { .. } => {
                panic!("Loading images from URIs is not yet implemented!");
            },
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

    Some(Light {
        pos: Point3::from_homogeneous(transform * vec4(0.0, 0.0, 0.0, 1.0)),
        intensity: light.intensity(),
        range: light.range().unwrap_or(f32::INFINITY),
        color: light.color().into(),
        kind
    })
}