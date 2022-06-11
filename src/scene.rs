use std::{path::Path, time::Instant};

use cgmath::{
    vec3,
    vec4,
    InnerSpace,
    Matrix,
    Matrix3,
    Matrix4,
    Point3,
    Quaternion,
    SquareMatrix,
    Vector2,
    Vector3,
    Vector4,
    Zero,
};
use gltf::{camera::Projection, scene::Transform};
use serde::Deserialize;

use crate::{
    camera::PerspectiveCamera,
    color::RGBf32,
    constants::GAMMA,
    environment::Environment,
    light::Light,
    material::Material,
    mesh::{Mesh, Vertex},
    spectrum::Spectrumf32,
    texture::{Format, Texture},
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
        Transform::Matrix { matrix } => matrix.into(),
        Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => {
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

        match gltf::import(path) {
            Ok((document, buffers, images)) => {
                let lib_time = start.elapsed().as_secs_f32();

                let textures = images
                    .into_iter()
                    .map(|i| {
                        let (format, pixels) = match i.format {
                            gltf::image::Format::R8G8B8 => (Format::Rgb, i.pixels),
                            gltf::image::Format::R8G8B8A8 => (Format::Rgba, i.pixels),
                            gltf::image::Format::R16G16B16 => (Format::Rgb, drop_every_other_byte(i.pixels)),
                            gltf::image::Format::R16G16B16A16 => (Format::Rgba, drop_every_other_byte(i.pixels)),
                            gltf::image::Format::R8 => (Format::Rgb, repeat_every_byte_thrice(i.pixels)),
                            gltf::image::Format::R8G8 => (Format::Rgb, insert_zero_byte_every_two(i.pixels)),
                            other => panic!("Texture format {:?} is not implemented", other),
                        };

                        Texture::new(pixels, i.width, i.height, format)
                    })
                    .collect();

                let environment = {
                    let color = RGBf32::from_hex("#404040").pow(GAMMA);
                    let coeffs = rgb2spec.fetch(color.into());

                    Environment {
                        color,
                        spectrum: Spectrumf32::from_coefficients(coeffs),
                    }
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
                println!(
                    "Parsed scene in {} seconds ({} in library, {} in own code)",
                    total_time,
                    lib_time,
                    total_time - lib_time
                );

                Ok(result)
            }
            Err(e) => Err(format!("An error occured while opening the glTF file:\n\t{}", e)),
        }
    }

    fn parse_nodes(
        &mut self,
        nodes: Vec<gltf::Node>,
        buffers: &[gltf::buffer::Data],
        base_transform: Matrix4<f32>,
        rgb2spec: &RGB2Spec,
    ) {
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
                        model: transform,
                    }
                } else {
                    println!("Non-perspective cameras are not supported");
                }
            } else if let Some(light) = node.light() {
                if let Some(light) = parse_light(light, transform, rgb2spec) {
                    self.lights.push(light);
                }
            }

            self.parse_nodes(node.children().collect(), buffers, transform, rgb2spec);
        }
    }

    fn add_meshes_from_gltf_mesh(
        &mut self,
        mesh: gltf::Mesh,
        buffers: &[gltf::buffer::Data],
        transform: Matrix4<f32>,
        rgb2spec: &RGB2Spec,
    ) {
        let m = Matrix3::from_cols(transform.x.truncate(), transform.y.truncate(), transform.z.truncate());
        let normal_transform = m.invert().unwrap().transpose();

        'primitive: for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let mat = primitive.material();
            let pbr = mat.pbr_metallic_roughness();
            let base = pbr.base_color_factor();

            let base_color = RGBf32::new(base[0], base[1], base[2]);
            let base_color_coefficients = rgb2spec.fetch([base_color.r, base_color.g, base_color.b]);

            let get_index = |t: gltf::texture::Info| t.texture().source().index();
            let base_color_texture = pbr.base_color_texture().map(get_index);

            if let Some(i) = base_color_texture {
                self.textures[i].create_spectrum_coefficients(rgb2spec)
            }

            let (transmission_factor, transmission_texture) = if let Some(transmission) = mat.transmission() {
                let texture = transmission.transmission_texture().map(get_index);
                (transmission.transmission_factor(), texture)
            } else {
                (0.0, None)
            };

            let material = Material {
                base_color,
                base_color_coefficients,
                base_color_texture,
                roughness: pbr.roughness_factor(),
                metallic: pbr.metallic_factor(),
                metallic_roughness_texture: pbr.metallic_roughness_texture().map(get_index),
                ior: mat.ior().unwrap_or(1.5),
                transmission_factor,
                transmission_texture,
                emissive: mat.emissive_factor().into(),
                emissive_texture: mat.emissive_texture().map(get_index),
                normal_texture: mat.normal_texture().map(|t| t.texture().source().index()),
            };

            let positions: Vec<Point3<f32>> = reader
                .read_positions()
                .unwrap_or_else(|| {
                    panic!(
                        "Primitive does not have POSITION attribute (mesh: {}, primitive: {})",
                        mesh.index(),
                        primitive.index()
                    )
                })
                .map(|pos| Point3::from_homogeneous(transform * Vector4::new(pos[0], pos[1], pos[2], 1.0)))
                .collect();

            let indices = reader
                .read_indices()
                .map(|read_indices| read_indices.into_u32().collect::<Vec<_>>())
                .unwrap_or_else(|| {
                    let count = positions.len();
                    let mut v = Vec::with_capacity(count);

                    for i in 0..count {
                        v.push(i as u32);
                    }

                    v
                });

            let normals: Vec<Vector3<f32>> = match reader.read_normals() {
                Some(normals) => normals
                    .map(|normal| -> Vector3<f32> { normal_transform * Vector3::from(normal) })
                    .collect(),
                None => {
                    let mut tri_normals = Vec::with_capacity(indices.len() / 3);

                    indices.chunks_exact(3).for_each(|i| {
                        let p0 = positions[i[0] as usize];
                        let p1 = positions[i[1] as usize];
                        let p2 = positions[i[2] as usize];

                        let edge1 = p0 - p1;
                        let edge2 = p0 - p2;
                        tri_normals.push(edge1.cross(edge2).normalize());
                    });

                    let mut vert_normals = vec![Vector3::zero(); positions.len()];

                    for i in 0..tri_normals.len() {
                        vert_normals[indices[3 * i] as usize] += tri_normals[i];
                        vert_normals[indices[3 * i + 1] as usize] += tri_normals[i];
                        vert_normals[indices[3 * i + 2] as usize] += tri_normals[i];
                    }

                    vert_normals.into_iter().map(|v| v.normalize()).collect()
                }
            };

            let has_tex_coords = reader.read_tex_coords(0).is_some();

            let tex_coords = match reader.read_tex_coords(0) {
                Some(reader) => reader
                    .into_f32()
                    .map(|uv| Some(Vector2::<f32>::new(uv[0], uv[1])))
                    .collect(),
                None => vec![None; positions.len()],
            };

            let tangents: Vec<Vector3<f32>> = match reader.read_tangents() {
                Some(reader) => reader
                    .map(|tangent| normal_transform * (tangent[3] * vec3(tangent[0], tangent[1], tangent[2])))
                    .collect(),
                None => {
                    let mut tri_tangents = Vec::with_capacity(indices.len() / 3);

                    indices.chunks_exact(3).for_each(|i| {
                        let p0 = positions[i[0] as usize];
                        let p1 = positions[i[1] as usize];
                        let p2 = positions[i[2] as usize];

                        let edge1 = p1 - p0;
                        let edge2 = p2 - p0;

                        if has_tex_coords {
                            let uv0 = tex_coords[i[0] as usize].unwrap();
                            let uv1 = tex_coords[i[1] as usize].unwrap();
                            let uv2 = tex_coords[i[2] as usize].unwrap();
                            let uv_edge1 = uv1 - uv0;
                            let uv_edge2 = uv2 - uv0;

                            let r = 1.0 / (uv_edge1.x * uv_edge2.y - uv_edge1.y * uv_edge2.x);
                            let tangent = (edge1 * uv_edge2.y - edge2 * uv_edge1.y) * r;

                            tri_tangents.push(tangent);
                        } else {
                            tri_tangents.push(edge1);
                        }
                    });

                    let mut vert_tangents = vec![Vector3::zero(); positions.len()];

                    for i in 0..tri_tangents.len() {
                        vert_tangents[indices[3 * i] as usize] += tri_tangents[i];
                        vert_tangents[indices[3 * i + 1] as usize] += tri_tangents[i];
                        vert_tangents[indices[3 * i + 2] as usize] += tri_tangents[i];
                    }

                    for i in 0..vert_tangents.len() {
                        let t = vert_tangents[i];

                        if t == Vector3::zero() {
                            println!(
                                "Encountered an error in computing tangents for primitive {} of mesh {}, skipping the primitive.",
                                primitive.index(),
                                mesh.name().unwrap_or(&mesh.index().to_string())
                            );
                            continue 'primitive;
                        }

                        let n = normals[i];
                        vert_tangents[i] = (t - n * t.dot(n)).normalize();
                    }

                    vert_tangents
                }
            };

            let vertices = positions
                .into_iter()
                .zip(normals)
                .zip(tex_coords.into_iter().zip(tangents.into_iter()))
                .map(|((position, normal), (tex_coord, tangent))| Vertex {
                    position,
                    normal,
                    tangent,
                    tex_coord,
                })
                .collect();

            let mesh = Mesh::new(vertices, indices, material);

            // Put meshes with RGBA textures at the end of the list so alpha blending works correctly
            if mesh
                .material
                .base_color_texture
                .map(|i| self.textures[i].format == Format::Rgba)
                .unwrap_or(false)
            {
                self.meshes.push(mesh);
            } else {
                self.meshes.insert(0, mesh);
            }
        }
    }
}

#[derive(Deserialize)]
struct LightExtras {
    radius: Option<f32>,
    angular_diameter: Option<f32>,
}

fn parse_light(light: gltf::khr_lights_punctual::Light, transform: Matrix4<f32>, rgb2spec: &RGB2Spec) -> Option<Light> {
    let extras: Option<LightExtras> = light
        .extras()
        .as_ref()
        .and_then(|extras| serde_json::from_str(&extras.as_ref().get()).ok());

    let kind = match light.kind() {
        gltf::khr_lights_punctual::Kind::Point => {
            let radius = (|| extras?.radius)().unwrap_or(0.0);
            crate::light::Kind::Point { radius }
        }
        gltf::khr_lights_punctual::Kind::Directional => {
            let m = Matrix3::from_cols(transform.x.truncate(), transform.y.truncate(), transform.z.truncate());
            let normal_transform = m.invert().unwrap().transpose();
            let direction = -(normal_transform * vec3(0.0, 0.0, -1.0)).normalize();

            let angular_diameter = (|| extras?.angular_diameter)().map_or(0.0, |degrees| degrees.to_radians());
            let angle = angular_diameter / 2.0;
            let radius = angle.tan();
            let area = std::f32::consts::PI * radius * radius;

            crate::light::Kind::Directional {
                direction,
                radius,
                area,
            }
        }
        gltf::khr_lights_punctual::Kind::Spot {
            inner_cone_angle,
            outer_cone_angle,
        } => crate::light::Kind::Spot {
            inner_cone_angle,
            outer_cone_angle,
        },
    };

    let color: RGBf32 = light.color().into();

    // TODO: rgb2spec is intended for reflectances, not emission,
    // so using it like this is not very physically accurate
    // though it seems to be a reasonable placeholder / fallback method.
    let spectrum = Spectrumf32::from_coefficients(rgb2spec.fetch(color.into()));

    Some(Light {
        pos: Point3::from_homogeneous(transform * vec4(0.0, 0.0, 0.0, 1.0)),
        intensity: light.intensity(),
        range: light.range().unwrap_or(f32::INFINITY),
        color,
        spectrum,
        kind,
    })
}

pub fn drop_every_other_byte(v: Vec<u8>) -> Vec<u8> {
    v.chunks_exact(2).map(|s| s[1]).collect()
}

pub fn repeat_every_byte_thrice(v: Vec<u8>) -> Vec<u8> {
    v.into_iter().flat_map(|s| [s, s, s]).collect()
}

pub fn insert_zero_byte_every_two(v: Vec<u8>) -> Vec<u8> {
    v.chunks_exact(2).flat_map(|s| [s[0], s[1], 0]).collect()
}
