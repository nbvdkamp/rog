use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    io::Cursor,
    path::Path,
    time::Instant,
};

use cgmath::{
    point2,
    vec3,
    vec4,
    InnerSpace,
    Matrix4,
    Point2,
    Point3,
    Quaternion,
    Rad,
    Rotation2,
    SquareMatrix,
    Vector3,
    Zero,
};
use gltf::{camera::Projection, mesh::Mode, scene::Transform};
use serde::Deserialize;

use crate::{
    camera::PerspectiveCamera,
    color::RGBf32,
    environment::Environment,
    light::{Kind, Light},
    material::{CauchyCoefficients, Material, TextureRef, TextureTransform},
    mesh::{Instance, Mesh, Vertices},
    raytracer::{aabb::BoundingBox, triangle::Triangle, Textures},
    spectrum::Spectrumf32,
    texture::{Format, Texture},
    util::normal_transform_from_mat4,
};

use rgb2spec::RGB2Spec;

const RGB2SPEC_BYTES: &[u8; 9437448] = include_bytes!("../res/out.spec");

pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub instances: Vec<Instance>,
    pub lights: Vec<Light>,
    pub textures: Textures,
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
    pub fn load<P>(path: P) -> Result<(Self, Vec<Texture>), String>
    where
        P: AsRef<Path>,
    {
        let rgb2spec = match RGB2Spec::from_reader(&mut Cursor::new(RGB2SPEC_BYTES)) {
            Ok(rgb2spec) => rgb2spec,
            Err(e) => panic!("Can't load rgb2spec file: {e}"),
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
                            other => panic!("Texture format {other:?} is not implemented"),
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

                let mut scene = Scene {
                    meshes: Vec::new(),
                    instances: Vec::new(),
                    lights: Vec::new(),
                    textures: Textures {
                        base_color_coefficients: Vec::new(),
                        metallic_roughness: Vec::new(),
                        transimission: Vec::new(),
                        emissive: Vec::new(),
                        normal: Vec::new(),
                    },
                    camera: PerspectiveCamera::default(),
                    environment,
                };

                let gltf_scene = document.default_scene().unwrap_or(document.scenes().next().unwrap());
                scene.parse_nodes(gltf_scene.nodes().collect(), &buffers, Matrix4::identity(), &rgb2spec);

                let sorted_textures = &mut scene.textures;

                let parse_time = start.elapsed().as_secs_f32();
                let start = Instant::now();

                let mut texture_types = Vec::new();
                texture_types.resize_with(textures.len(), HashMap::new);

                #[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
                enum TextureType {
                    BaseColor,
                    MetallicRoughness,
                    Transmission,
                    Emissive,
                    Normal,
                }

                for instance in &mut scene.instances {
                    macro_rules! set_type {
                        ( $tex:expr, $tex_type:expr ) => {
                            match $tex {
                                Some(tex) => match texture_types[tex.index].entry($tex_type) {
                                    Entry::Vacant(e) => {
                                        e.insert(vec![&mut $tex]);
                                    }
                                    Entry::Occupied(mut l) => {
                                        l.get_mut().push(&mut $tex);
                                    }
                                },
                                None => (),
                            }
                        };
                    }

                    set_type!(instance.material.base_color_texture, TextureType::BaseColor);
                    set_type!(
                        instance.material.metallic_roughness_texture,
                        TextureType::MetallicRoughness
                    );
                    set_type!(instance.material.transmission_texture, TextureType::Transmission);
                    set_type!(instance.material.emissive_texture, TextureType::Emissive);
                    set_type!(instance.material.normal_texture, TextureType::Normal);
                }

                let mut preview_textures = Vec::new();

                textures
                    .into_iter()
                    .zip(texture_types.into_iter())
                    .for_each(|(texture, types_map)| {
                        let mut types = types_map.into_iter();

                        let num_types = types.len();

                        if num_types == 0 {
                            return;
                        }

                        let mut insert = |texture: Texture, texture_type, opts: Vec<&mut Option<TextureRef>>| {
                            match texture_type {
                                TextureType::BaseColor => {
                                    sorted_textures
                                        .base_color_coefficients
                                        .push(texture.create_spectrum_coefficients(&rgb2spec));

                                    preview_textures.push(texture);
                                }
                                TextureType::MetallicRoughness => sorted_textures.metallic_roughness.push(texture),
                                TextureType::Transmission => sorted_textures.transimission.push(texture),
                                TextureType::Emissive => sorted_textures.emissive.push(texture),
                                TextureType::Normal => sorted_textures.normal.push(texture),
                            };

                            let last_index = match texture_type {
                                // The index for the result scene texture is the same as for the coefficient texture because they are inserted together
                                TextureType::BaseColor => &preview_textures,
                                TextureType::MetallicRoughness => &sorted_textures.metallic_roughness,
                                TextureType::Transmission => &sorted_textures.transimission,
                                TextureType::Emissive => &sorted_textures.emissive,
                                TextureType::Normal => &sorted_textures.normal,
                            }
                            .len()
                                - 1;

                            for tex_opt in opts {
                                match tex_opt {
                                    Some(tex) => tex.index = last_index,
                                    None => unreachable!(),
                                }
                            }
                        };

                        for _ in 0..num_types - 1 {
                            let (texture_type, opts) = types.next().unwrap();
                            insert(texture.clone(), texture_type, opts);
                        }

                        let (texture_type, opts) = types.last().unwrap();
                        insert(texture, texture_type, opts);
                    });

                // To draw meshes with alpha textures last
                scene.instances.iter_mut().partition_in_place(|instance| {
                    instance
                        .material
                        .base_color_texture
                        .map_or(true, |tex| preview_textures[tex.index].format == Format::Rgb)
                });

                let textures_time = start.elapsed().as_secs_f32();
                let total_time = lib_time + parse_time + textures_time;
                println!(
                    "Loaded scene in {total_time} seconds ({lib_time} in library, {parse_time} converting scene, {textures_time} converting textures)"
                );

                Ok((scene, preview_textures))
            }
            Err(e) => {
                let hint = if let gltf::Error::Io(_) = e {
                    // TODO: Remove this when the gltf crate updates to include the fix
                    "\n\tIf the path to the file is correct this issue may be caused by URL encoded characters in resources used by the file."
                } else {
                    ""
                };

                Err(format!("An error occured while opening the glTF file:\n\t{e}{hint}"))
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
                self.lights.push(parse_light(light, transform, rgb2spec));
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
        let Some(inverse_transform) = transform.invert() else {
            eprintln!("Singular transform matrix for mesh: {}, skipping it", mesh.name().unwrap_or(&mesh.index().to_string()));
            return;
        };

        for primitive in mesh.primitives() {
            if primitive.mode() != Mode::Triangles {
                eprintln!(
                    "Primitives with render mode {:?} are currently not supported",
                    primitive.mode()
                );
                continue;
            }

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
                .and_then(|extras| serde_json::from_str(extras.as_ref().get()).ok());

            let base_color = RGBf32::new(base[0], base[1], base[2]);
            let base_color_coefficients = rgb2spec.fetch([base_color.r, base_color.g, base_color.b]);

            let read_texture_transform = |transform: gltf::texture::TextureTransform| TextureTransform {
                offset: transform.offset().into(),
                rotation: Rotation2::from_angle(Rad(transform.rotation())),
                scale: transform.scale().into(),
            };

            let get_tex_ref = |t: gltf::texture::Info| TextureRef {
                index: t.texture().source().index(),
                texture_coordinate_set: t.tex_coord() as usize,
                transform: t.texture_transform().map(read_texture_transform),
            };
            let base_color_texture = pbr.base_color_texture().map(get_tex_ref);

            let (transmission_factor, transmission_texture) = if let Some(transmission) = mat.transmission() {
                let texture = transmission.transmission_texture().map(get_tex_ref);
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
                metallic_roughness_texture: pbr.metallic_roughness_texture().map(get_tex_ref),
                ior,
                cauchy_coefficients,
                transmission_factor,
                transmission_texture,
                emissive: mat.emissive_factor().into(),
                emissive_texture: mat.emissive_texture().map(get_tex_ref),
                normal_texture: mat.normal_texture().map(|t| TextureRef {
                    index: t.texture().source().index(),
                    texture_coordinate_set: t.tex_coord() as usize,
                    transform: t.texture_transform().map(read_texture_transform),
                }),
            };

            let positions: Vec<Point3<f32>> = if let Some(iter) = reader.read_positions() {
                iter.map(|pos| pos.into()).collect()
            } else {
                eprintln!(
                    "Skipping primitive with missing POSITION attribute (mesh: {}, primitive: {})",
                    mesh.index(),
                    primitive.index()
                );
                continue;
            };

            let indices = reader
                .read_indices()
                .map(|read_indices| read_indices.into_u32().collect::<Vec<_>>())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());

            // Triangle meshes should always have an index count of a multiple of 3
            assert_eq!(indices.len() % 3, 0);

            let normals: Vec<Vector3<f32>> = match reader.read_normals() {
                Some(normals) => normals.map(Vector3::from).collect(),
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

            let mut tex_coords: Vec<Vec<Point2<f32>>> = Vec::new();
            let mut i = 0;

            while let Some(reader) = reader.read_tex_coords(i) {
                tex_coords.push(reader.into_f32().map(|uv| point2(uv[0], uv[1])).collect());
                i += 1;
            }

            let tangents: Vec<Vector3<f32>> = match reader.read_tangents() {
                Some(reader) => reader
                    .map(|tangent| tangent[3] * vec3(tangent[0], tangent[1], tangent[2]))
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

                        if let Some(tex_coords) = tex_coords.first() {
                            let uv0 = tex_coords[i[0] as usize];
                            let uv1 = tex_coords[i[1] as usize];
                            let uv2 = tex_coords[i[2] as usize];
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

                        if t == Vector3::zero() {
                            if !notified_of_error {
                                println!(
                                    "Encountered an error in computing tangents for primitive {} of mesh {}.",
                                    primitive.index(),
                                    mesh.name().unwrap_or(&mesh.index().to_string())
                                );
                                notified_of_error = true;
                            }
                        } else {
                            let n = normals[i];
                            vert_tangents[i] = (t - n * t.dot(n)).normalize();
                        }
                    }

                    vert_tangents
                }
            };

            let triangles: Vec<Triangle> = indices
                .chunks(3)
                .map(|v| Triangle {
                    indices: [v[0], v[1], v[2]],
                })
                .collect();

            let vertices = Vertices {
                positions,
                normals,
                tangents,
                tex_coords,
            };

            // Split topologically and spatially separated parts of the mesh into separate meshes
            // to improve acceleration structure performance.
            let mut assigned = vec![false; triangles.len()];

            while let Some(first_unassigned) = assigned.iter().position(|&v| !v) {
                let mut visited = HashSet::<u32>::new();
                let mut new_triangles = Vec::new();
                let mut new_tri_added = true;
                let mut new_bounds = BoundingBox::new();

                let tri = &triangles[first_unassigned];
                visited.extend(tri.indices.iter());
                new_triangles.push(triangles[first_unassigned]);
                new_bounds.add(vertices.positions[tri.indices[0] as usize]);
                new_bounds.add(vertices.positions[tri.indices[1] as usize]);
                new_bounds.add(vertices.positions[tri.indices[2] as usize]);
                assigned[first_unassigned] = true;

                while new_tri_added {
                    new_tri_added = false;
                    for tri_index in 0..triangles.len() {
                        let tri = &triangles[tri_index];
                        if !assigned[tri_index]
                            && (visited.contains(&tri.indices[0])
                                || visited.contains(&tri.indices[1])
                                || visited.contains(&tri.indices[2])
                                || new_bounds.contains(vertices.positions[tri.indices[0] as usize])
                                || new_bounds.contains(vertices.positions[tri.indices[1] as usize])
                                || new_bounds.contains(vertices.positions[tri.indices[2] as usize]))
                        {
                            visited.extend(triangles[tri_index].indices.iter());
                            new_triangles.push(triangles[tri_index]);
                            new_bounds.add(vertices.positions[tri.indices[0] as usize]);
                            new_bounds.add(vertices.positions[tri.indices[1] as usize]);
                            new_bounds.add(vertices.positions[tri.indices[2] as usize]);
                            assigned[tri_index] = true;
                            new_tri_added = true;
                        }
                    }
                }

                // Fast path if the whole mesh is one piece
                if visited.len() == vertices.positions.len() {
                    self.meshes.push(Mesh::new(vertices, triangles));
                    let mesh_index = self.meshes.len() - 1;
                    self.instances.push(Instance::new(
                        mesh_index,
                        &self.meshes[mesh_index],
                        transform,
                        inverse_transform,
                        material,
                    ));
                    break;
                }

                let indices: Vec<u32> = visited.into_iter().collect();
                let new_triangles = new_triangles
                    .into_iter()
                    .map(|tri| Triangle {
                        indices: [
                            indices.iter().position(|&i| i == tri.indices[0]).unwrap() as u32,
                            indices.iter().position(|&i| i == tri.indices[1]).unwrap() as u32,
                            indices.iter().position(|&i| i == tri.indices[2]).unwrap() as u32,
                        ],
                    })
                    .collect();

                let vertices = Vertices {
                    positions: indices.iter().map(|&i| vertices.positions[i as usize]).collect(),
                    normals: indices.iter().map(|&i| vertices.normals[i as usize]).collect(),
                    tangents: indices.iter().map(|&i| vertices.tangents[i as usize]).collect(),
                    tex_coords: vertices
                        .tex_coords
                        .iter()
                        .map(|t| indices.iter().map(|&i| t[i as usize]).collect())
                        .collect(),
                };

                self.meshes.push(Mesh::new(vertices, new_triangles));
                let mesh_index = self.meshes.len() - 1;
                self.instances.push(Instance::new(
                    mesh_index,
                    &self.meshes[mesh_index],
                    transform,
                    inverse_transform,
                    material.clone(),
                ));
            }
        }
    }
}

#[derive(Deserialize)]
struct LightExtras {
    radius: Option<f32>,
    angular_diameter: Option<f32>,
}

fn parse_light(light: gltf::khr_lights_punctual::Light, transform: Matrix4<f32>, rgb2spec: &RGB2Spec) -> Light {
    let extras: Option<LightExtras> = light
        .extras()
        .as_ref()
        .and_then(|extras| serde_json::from_str(extras.as_ref().get()).ok());

    use gltf::khr_lights_punctual::Kind as GltfKind;

    let kind = match light.kind() {
        GltfKind::Point => {
            let radius = (|| extras?.radius)().unwrap_or(0.0);
            Kind::Point { radius }
        }
        GltfKind::Directional => {
            let normal_transform = normal_transform_from_mat4(transform);
            let direction = -(normal_transform * vec3(0.0, 0.0, -1.0)).normalize();

            let angular_diameter = (|| extras?.angular_diameter)().map_or(0.0, |degrees| degrees.to_radians());
            let angle = angular_diameter / 2.0;
            let radius = angle.tan();
            let area = std::f32::consts::PI * radius * radius;

            Kind::Directional {
                direction,
                radius,
                area,
            }
        }
        GltfKind::Spot {
            inner_cone_angle,
            outer_cone_angle,
        } => Kind::Spot {
            inner_cone_angle,
            outer_cone_angle,
        },
    };

    let color: RGBf32 = light.color().into();

    // TODO: rgb2spec is intended for reflectances, not emission,
    // so using it like this is not very physically accurate
    // though it seems to be a reasonable placeholder / fallback method.
    let spectrum = Spectrumf32::from_coefficients(rgb2spec.fetch(color.into()));

    Light {
        pos: Point3::from_homogeneous(transform * vec4(0.0, 0.0, 0.0, 1.0)),
        intensity: light.intensity(),
        range: light.range().unwrap_or(f32::INFINITY),
        color,
        spectrum,
        kind,
    }
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
