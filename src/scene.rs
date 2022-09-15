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
    environment::Environment,
    light::Light,
    material::{CauchyCoefficients, Material},
    mesh::{Mesh, Vertex},
    raytracer::Textures,
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
    pub fn load<P>(path: P) -> Result<(Self, Textures), String>
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
                let start = Instant::now();

                let textures = images
                    .into_iter()
                    .map(|i| {
                        use gltf::image::Format as ImageFormat;

                        let (format, pixels) = match i.format {
                            ImageFormat::R8G8B8 => (Format::Rgb, i.pixels),
                            ImageFormat::R8G8B8A8 => (Format::Rgba, i.pixels),
                            ImageFormat::R16G16B16 => (Format::Rgb, drop_every_other_byte(i.pixels)),
                            ImageFormat::R16G16B16A16 => (Format::Rgba, drop_every_other_byte(i.pixels)),
                            ImageFormat::R8 => (Format::Rgb, repeat_every_byte_thrice(i.pixels)),
                            ImageFormat::R8G8 => (Format::Rgb, insert_zero_byte_every_two(i.pixels)),
                            other => panic!("Texture format {:?} is not implemented", other),
                        };

                        Texture::new(pixels, i.width as usize, i.height as usize, format)
                    })
                    .collect::<Vec<Texture>>();

                let environment = {
                    let color = RGBf32::from_hex("#404040").srgb_to_linear();
                    let coeffs = rgb2spec.fetch(color.into());

                    Environment {
                        color,
                        spectrum: Spectrumf32::from_coefficients(coeffs),
                    }
                };

                let mut result_scene = Scene {
                    meshes: Vec::new(),
                    lights: Vec::new(),
                    textures: Vec::new(),
                    camera: PerspectiveCamera::default(),
                    environment,
                };

                for scene in document.scenes() {
                    result_scene.parse_nodes(scene.nodes().collect(), &buffers, Matrix4::identity(), &rgb2spec);
                }

                let parse_time = start.elapsed().as_secs_f32();
                let start = Instant::now();

                let mut texture_types = Vec::new();
                texture_types.resize_with(textures.len(), || Vec::new());

                #[derive(Debug, PartialEq, Clone, Copy)]
                enum TextureType {
                    BaseColor,
                    MetallicRoughness,
                    Transmission,
                    Emissive,
                    Normal,
                }

                for mesh in &mut result_scene.meshes {
                    macro_rules! set_type {
                        ( $tex:expr, $tex_type:expr ) => {
                            match $tex {
                                Some(index) => {
                                    texture_types[index].push((&mut $tex, $tex_type));
                                }
                                None => (),
                            }
                        };
                    }

                    set_type!(mesh.material.base_color_texture, TextureType::BaseColor);
                    set_type!(mesh.material.metallic_roughness_texture, TextureType::MetallicRoughness);
                    set_type!(mesh.material.transmission_texture, TextureType::Transmission);
                    set_type!(mesh.material.emissive_texture, TextureType::Emissive);
                    set_type!(mesh.material.normal_texture, TextureType::Normal);
                }

                let mut sorted_textures = Textures {
                    base_color_coefficients: Vec::new(),
                    metallic_roughness: Vec::new(),
                    transimission: Vec::new(),
                    emissive: Vec::new(),
                    normal: Vec::new(),
                };

                textures
                    .into_iter()
                    .zip(texture_types.into_iter())
                    .for_each(|(texture, types)| {
                        if !types.is_empty() {
                            let texture_type = types[0].1.clone();

                            for (_, other_type) in &types {
                                if other_type != &texture_type {
                                    panic!("Texture is being used for multiple purposes: {texture_type:?} and {other_type:?}");
                                }
                            }

                            match texture_type {
                                TextureType::BaseColor => {
                                    sorted_textures
                                        .base_color_coefficients
                                        .push(texture.create_spectrum_coefficients(&rgb2spec));

                                    result_scene.textures.push(texture);
                                }
                                TextureType::MetallicRoughness => sorted_textures.metallic_roughness.push(texture),
                                TextureType::Transmission => sorted_textures.emissive.push(texture),
                                TextureType::Emissive => sorted_textures.emissive.push(texture),
                                TextureType::Normal => sorted_textures.normal.push(texture),
                            };

                            for (tex_opt, _) in types {
                                match tex_opt {
                                    Some(index) => {
                                        *index = match texture_type {
                                            // The index for the result scene texture is the same as for the coefficient texture because they are inserted together
                                            TextureType::BaseColor => &result_scene.textures,
                                            TextureType::MetallicRoughness => &sorted_textures.metallic_roughness,
                                            TextureType::Transmission => &sorted_textures.emissive,
                                            TextureType::Emissive => &sorted_textures.emissive,
                                            TextureType::Normal => &sorted_textures.normal,
                                        }.len() - 1;
                                    }
                                    None => unreachable!(),
                                }
                            }
                        }
                    });

                // To draw meshes with alpha textures last
                result_scene.meshes.iter_mut().partition_in_place(|mesh| {
                    mesh.material
                        .base_color_texture
                        .map_or(true, |index| result_scene.textures[index].format == Format::Rgb)
                });

                let textures_time = start.elapsed().as_secs_f32();
                let total_time = lib_time + parse_time + textures_time;
                println!(
                    "Loaded scene in {} seconds ({} in library, {} converting scene, {} converting textures)",
                    total_time, lib_time, parse_time, textures_time,
                );

                Ok((result_scene, sorted_textures))
            }
            Err(e) => {
                let hint = if let gltf::Error::Io(_) = e {
                    // TODO: Remove this when the gltf crate updates to include the fix
                    "\n\tIf the path to the file is correct this issue may be caused by URL encoded characters in resources used by the file."
                } else {
                    ""
                };

                Err(format!(
                    "An error occured while opening the glTF file:\n\t{}{}",
                    e, hint
                ))
            }
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

        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let mat = primitive.material();
            let pbr = mat.pbr_metallic_roughness();
            let base = pbr.base_color_factor();

            #[derive(Deserialize)]
            struct MaterialExtras {
                cauchy_a: Option<f32>,
                cauchy_b: Option<f32>,
            }

            let extras: Option<MaterialExtras> = mat
                .extras()
                .as_ref()
                .and_then(|extras| serde_json::from_str(&extras.as_ref().get()).ok());

            let base_color = RGBf32::new(base[0], base[1], base[2]);
            let base_color_coefficients = rgb2spec.fetch([base_color.r, base_color.g, base_color.b]);

            let get_index = |t: gltf::texture::Info| t.texture().source().index();
            let base_color_texture = pbr.base_color_texture().map(get_index);

            let (transmission_factor, transmission_texture) = if let Some(transmission) = mat.transmission() {
                let texture = transmission.transmission_texture().map(get_index);
                (transmission.transmission_factor(), texture)
            } else {
                (0.0, None)
            };

            let ior = mat.ior().unwrap_or(1.5);

            let cauchy_coefficients = if let Some(extras) = extras {
                if extras.cauchy_a.is_some() && extras.cauchy_b.is_some() {
                    CauchyCoefficients {
                        a: extras.cauchy_a.unwrap(),
                        b: extras.cauchy_b.unwrap(),
                    }
                } else {
                    CauchyCoefficients::approx_from_ior(ior)
                }
            } else {
                CauchyCoefficients::approx_from_ior(ior)
            };

            let material = Material {
                base_color,
                base_color_coefficients,
                base_color_texture,
                roughness: pbr.roughness_factor(),
                metallic: pbr.metallic_factor(),
                metallic_roughness_texture: pbr.metallic_roughness_texture().map(get_index),
                ior,
                cauchy_coefficients,
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

                    let mut notified_of_error = false;

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

                            // Bad UVs can cause infinite or NaN values so fall back to an edge
                            if !tangent.x.is_finite() || !tangent.y.is_finite() || !tangent.z.is_finite() {
                                tri_tangents.push(edge1);
                            } else {
                                tri_tangents.push(tangent);
                            }
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

                        if t == Vector3::zero() && !notified_of_error {
                            println!(
                                "Encountered an error in computing tangents for primitive {} of mesh {}.",
                                primitive.index(),
                                mesh.name().unwrap_or(&mesh.index().to_string())
                            );
                            notified_of_error = true;
                        } else {
                            let n = normals[i];
                            vert_tangents[i] = (t - n * t.dot(n)).normalize();
                        }
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

            self.meshes.push(Mesh::new(vertices, indices, material));
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
