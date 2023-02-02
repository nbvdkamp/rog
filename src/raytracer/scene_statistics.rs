use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use arrayvec::ArrayVec;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rayon::iter::ParallelIterator;
use serde::{Deserialize, Serialize};
use static_assertions::const_assert;

use cgmath::{point2, point3, vec2, vec3, ElementWise, Point2, Point3, Vector3};
use rand::Rng;
use rayon::prelude::IntoParallelIterator;

use crate::{
    color::RGBf32,
    raytracer::{file_formatting::SectionHeader, geometry::triangle_area, sampling::sample_coordinates_on_triangle},
    scene::Scene,
    scene_version::SceneVersion,
    spectrum::Spectrumf32,
    util::{align_to, save_image},
};

use super::{
    aabb::BoundingBox,
    acceleration::Accel,
    axis::Axis,
    file_formatting::Error,
    geometry::{interpolate_point_on_triangle, line_axis_plane_intersect},
    ray::Ray,
    sampling::{cumulative_probabilities_from_weights, sample_item_from_cumulative_probabilities},
    Raytracer,
};

const VISIBILITY_SAMPLES: usize = 16;
const MATERIAL_SAMPLES: usize = 128;

// File format
const FORMAT_VERSION: u32 = 1;
const TAG: &str = "SCENE_STATISTICS";
const JSON_TAG: &str = "JSON";
const VISIBILITY_TAG: &str = "VIS ";
const MATERIALS_TAG: &str = "MATS";
static_assertions::const_assert_eq!(TAG.len() % 4, 0);
static_assertions::const_assert_eq!(JSON_TAG.len() % 4, 0);
static_assertions::const_assert_eq!(VISIBILITY_TAG.len() % 4, 0);

type Visibility = u8;
const_assert!(Visibility::MAX as usize >= VISIBILITY_SAMPLES);

pub struct Distribution {
    pub probabilities: Spectrumf32,
    pub cumulative_probabilities: Spectrumf32,
}

pub struct VoxelLights {
    pub voxel_index: usize,
    pub light_indices: Vec<usize>,
}

#[derive(Serialize, Deserialize)]
pub struct SceneStatistics {
    resolution: usize,
    voxel_count: usize,
    table_size: usize,
    scene_bounds: BoundingBox,
    scene_extent: Vector3<f32>,
    voxel_extent: Vector3<f32>,
    scene_version: SceneVersion,
    #[serde(skip)]
    pub voxels_with_lights: Vec<VoxelLights>,
    #[serde(skip)]
    pub positionless_lights: Vec<usize>,
    #[serde(skip)]
    visibility: Vec<Visibility>,
    #[serde(skip)]
    materials: Vec<Option<Spectrumf32>>,
    #[serde(skip)]
    pub spectral_distributions: Vec<Option<Distribution>>,
}

impl SceneStatistics {
    pub fn new(scene_bounds: BoundingBox, scene_version: SceneVersion, resolution: usize) -> Self {
        let scene_extent = scene_bounds.max - scene_bounds.min;
        let voxel_count = resolution * resolution * resolution;

        SceneStatistics {
            resolution,
            voxel_count,
            table_size: (voxel_count * (voxel_count + 1)) / 2,
            scene_bounds,
            scene_extent,
            voxel_extent: scene_extent / resolution as f32,
            scene_version,
            voxels_with_lights: Vec::new(),
            positionless_lights: Vec::new(),
            visibility: Vec::new(),
            materials: Vec::new(),
            spectral_distributions: Vec::new(),
        }
    }

    pub fn get_estimated_visibility(&self, first_voxel_index: usize, voxel_b_index: usize) -> f32 {
        let i = self.get_table_index(first_voxel_index, voxel_b_index);
        let vis = self.visibility[i] as f32 / VISIBILITY_SAMPLES as f32;
        // Ensure estimate is never 0 to prevent bias
        (0.99 * vis + 0.01).clamp(0.01, 1.0)
    }

    pub fn sample_visibility(&mut self, raytracer: &Raytracer, accel: Accel) {
        self.visibility = (0..self.voxel_count)
            .into_par_iter()
            .flat_map(|b| {
                (b..self.voxel_count)
                    .map(|a| self.measure_visibility_between_voxels(a, b, raytracer, accel))
                    .collect::<Vec<_>>()
            })
            .collect();
    }

    fn measure_visibility_between_voxels(&self, a: usize, b: usize, raytracer: &Raytracer, accel: Accel) -> Visibility {
        (0..VISIBILITY_SAMPLES)
            .map(|_| {
                let start = self.sample_point_in_voxel(a);
                let end = self.sample_point_in_voxel(b);

                let ray = Ray {
                    origin: start,
                    direction: end - start,
                };

                let occluded = raytracer.is_ray_obstructed(ray, 1.0, accel);

                u8::from(!occluded)
            })
            .sum()
    }

    pub fn dump_visibility_image<P>(&self, path: P)
    where
        P: AsRef<Path>,
    {
        assert_eq!(self.visibility.len(), self.table_size);

        let image_size = vec2(self.voxel_count, self.voxel_count);
        let mut buffer = vec![RGBf32::new(0.0, 0.0, 0.0); image_size.x * image_size.y];

        for b in 0..self.voxel_count {
            for a in b..self.voxel_count {
                let i = self.get_table_index(a, b);
                buffer[a + b * self.voxel_count] =
                    RGBf32::from_grayscale(self.visibility[i] as f32 / VISIBILITY_SAMPLES as f32);
            }
        }

        save_image(&buffer, image_size, path);
    }

    pub fn dump_materials_as_rgb<P>(&self, path: P) -> Result<(), std::io::Error>
    where
        P: AsRef<Path>,
    {
        assert_eq!(self.materials.len(), self.voxel_count);

        let mut file = File::create(path)?;
        writeln!(file, "resolution {}", self.resolution)?;

        for x in 0..self.resolution {
            for y in 0..self.resolution {
                for z in 0..self.resolution {
                    let i = x + y * self.resolution + z * self.resolution * self.resolution;

                    if let Some(spectrum) = self.materials[i] {
                        let RGBf32 { r, g, b } = spectrum.to_srgb();
                        writeln!(file, "{x} {y} {z} {r} {g} {b}")?;
                    }
                }
            }
        }

        Ok(())
    }

    fn get_grid_position(&self, point: Point3<f32>) -> Point3<usize> {
        let relative_pos = (point - self.scene_bounds.min).div_element_wise(self.scene_extent);
        let v = relative_pos * self.resolution as f32;
        point3(
            (v.x as usize).min(self.resolution - 1),
            (v.y as usize).min(self.resolution - 1),
            (v.z as usize).min(self.resolution - 1),
        )
    }

    fn min_corner_of_voxel(&self, position: Point3<usize>) -> Point3<f32> {
        self.scene_bounds.min
            + vec3(
                position.x as f32 * self.voxel_extent.x,
                position.y as f32 * self.voxel_extent.y,
                position.z as f32 * self.voxel_extent.z,
            )
    }

    fn max_corner_of_voxel(&self, position: Point3<usize>) -> Point3<f32> {
        self.min_corner_of_voxel(position + vec3(1, 1, 1))
    }

    fn bounds_from_grid_position(&self, position: Point3<usize>) -> BoundingBox {
        BoundingBox {
            min: self.min_corner_of_voxel(position),
            max: self.max_corner_of_voxel(position),
        }
    }

    pub fn get_voxel_index(&self, point: Point3<f32>) -> usize {
        let v = self.get_grid_position(point);
        v.x + v.y * self.resolution + v.z * self.resolution * self.resolution
    }

    fn sample_point_in_voxel(&self, voxel_index: usize) -> Point3<f32> {
        let x = (voxel_index % self.resolution) as f32;
        let y = ((voxel_index % (self.resolution * self.resolution)) / self.resolution) as f32;
        let z = (voxel_index / (self.resolution * self.resolution)) as f32;
        let voxel_offset = self.voxel_extent.mul_element_wise(vec3(x, y, z));

        let mut rng = rand::thread_rng();
        let random_offset = vec3(rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>());

        self.scene_bounds.min + voxel_offset + self.voxel_extent.mul_element_wise(random_offset)
    }

    fn get_table_index(&self, first_index: usize, second_index: usize) -> usize {
        let a = first_index.max(second_index);
        let b = first_index.min(second_index);
        a + b * self.voxel_count - b * (b + 1) / 2
    }

    pub fn compute_light_voxel_distribution(&mut self, scene: &Scene) {
        for (i, light) in scene.lights.iter().enumerate() {
            // TODO: Account for radius
            if let Some(position) = light.position() {
                let voxel_index = self.get_voxel_index(position);

                if let Some(v) = self
                    .voxels_with_lights
                    .iter_mut()
                    .find(|v| v.voxel_index == voxel_index)
                {
                    v.light_indices.push(i);
                } else {
                    self.voxels_with_lights.push(VoxelLights {
                        voxel_index,
                        light_indices: vec![i],
                    });
                };
            } else {
                self.positionless_lights.push(i);
            }
        }
    }

    pub fn compute_visibility_weighted_material_sums(&mut self) {
        assert_eq!(self.visibility.len(), self.table_size);
        assert_eq!(self.materials.len(), self.voxel_count);

        self.spectral_distributions = self
            .materials
            .iter()
            .enumerate()
            .map(|(i, material)| {
                material.as_ref().map(|spectrum| {
                    let mut neighbour_vis_sum = 0.0;
                    let mut neigbour_material_sum = Spectrumf32::constant(0.0);

                    for (j, n) in self.materials.iter().enumerate() {
                        if i == j {
                            continue;
                        }

                        if let Some(n_spec) = n {
                            let vis = self.visibility[self.get_table_index(i, j)] as f32 / VISIBILITY_SAMPLES as f32;
                            neighbour_vis_sum += vis;
                            neigbour_material_sum += vis * *n_spec;
                        }
                    }

                    let neighbour_materials_mean = neigbour_material_sum / neighbour_vis_sum;
                    let base_chance = 0.1;
                    let weights = Spectrumf32::constant(base_chance) + spectrum * neighbour_materials_mean;
                    let probabilities = weights / weights.data.iter().sum::<f32>();

                    let mut cumulative_probabilities = Spectrumf32::constant(0.0);
                    let mut acc = 0.0;

                    for i in 0..Spectrumf32::RESOLUTION {
                        acc += probabilities.data[i];

                        cumulative_probabilities.data[i] = acc;
                    }

                    Distribution {
                        probabilities,
                        cumulative_probabilities,
                    }
                })
            })
            .collect::<Vec<_>>();
    }

    pub fn sample_materials(&mut self, raytracer: &Raytracer) {
        self.materials = self
            .split_triangles_into_voxels(&raytracer.scene)
            .into_iter()
            .map(|tris| sample_triangle_materials(tris, raytracer))
            .collect();
    }

    fn split_triangles_into_voxels(&self, scene: &Scene) -> Vec<Vec<ClippedTri>> {
        let mut tris_per_voxel = Vec::new();
        tris_per_voxel.reserve(self.voxel_count);

        for _ in 0..self.voxel_count {
            tris_per_voxel.push(Vec::new());
        }

        let mut tris_to_sort = scene
            .instances
            .iter()
            .enumerate()
            .flat_map(|(instance_index, instance)| {
                let mesh = &scene.meshes[instance.mesh_index as usize];
                mesh.triangles.iter().enumerate().map(move |(tri_index, tri)| {
                    let verts = [
                        Vert {
                            position: Point3::from_homogeneous(
                                instance.transform * mesh.vertices.positions[tri.indices[0] as usize].to_homogeneous(),
                            ),
                            barycentric: point2(0.0, 0.0),
                        },
                        Vert {
                            position: Point3::from_homogeneous(
                                instance.transform * mesh.vertices.positions[tri.indices[1] as usize].to_homogeneous(),
                            ),
                            barycentric: point2(1.0, 0.0),
                        },
                        Vert {
                            position: Point3::from_homogeneous(
                                instance.transform * mesh.vertices.positions[tri.indices[2] as usize].to_homogeneous(),
                            ),
                            barycentric: point2(0.0, 1.0),
                        },
                    ];

                    let mut bounds = BoundingBox::new();
                    bounds.add(verts[0].position);
                    bounds.add(verts[1].position);
                    bounds.add(verts[2].position);

                    ClippedTri {
                        instance_index,
                        tri_index,
                        verts,
                        bounds,
                    }
                })
            })
            .collect::<Vec<_>>();

        while let Some(tri) = tris_to_sort.pop() {
            let center_grid_pos = self.get_grid_position(tri.bounds.center());
            let voxel_bounds = self.bounds_from_grid_position(center_grid_pos);

            let epsilon = 1e-5;
            let (min_contained, min_contained_axes) =
                voxel_bounds.within_min_bound_with_epsilon(tri.bounds.min, epsilon);
            let (max_contained, max_contained_axes) =
                voxel_bounds.within_max_bound_with_epsilon(tri.bounds.max, epsilon);

            if min_contained && max_contained {
                let voxel_index = center_grid_pos.x
                    + center_grid_pos.y * self.resolution
                    + center_grid_pos.z * self.resolution * self.resolution;
                tris_per_voxel[voxel_index].push(tri);
            } else {
                // Split the triangle on the voxel boundary and recursively handle the resulting triangles
                for axis_index in 0..3 {
                    if !min_contained_axes[axis_index] || !max_contained_axes[axis_index] {
                        // If the intersection is on the max bound, offset by 1
                        let offset = usize::from(min_contained_axes[axis_index]);

                        let clip_position = self.scene_bounds.min[axis_index]
                            + self.voxel_extent[axis_index] * (center_grid_pos[axis_index] + offset) as f32;

                        match clip_triangle(tri, Axis::from_index(axis_index), clip_position) {
                            ClipTriResult::Two(t1, t2) => {
                                tris_to_sort.push(t1);
                                tris_to_sort.push(t2);
                            }
                            ClipTriResult::Three(t1, t2, t3) => {
                                tris_to_sort.push(t1);
                                tris_to_sort.push(t2);
                                tris_to_sort.push(t3);
                            }
                        }
                        break;
                    }
                }
            }
        }

        tris_per_voxel
    }

    pub fn write_to_file<P>(&self, path: P) -> Result<(), Error>
    where
        P: AsRef<Path>,
    {
        use Error::{Serde, IO};
        let mut file = File::create(path).map_err(IO)?;

        let json = serde_json::to_string(self).map_err(Serde)?;
        let bytes_to_pad = align_to(json.len(), 4) - json.len();
        let json = format!("{}{}", json, " ".repeat(bytes_to_pad));

        let visibility_size = self.visibility.len() * std::mem::size_of::<Visibility>();
        let materials_size = self.materials.len() * std::mem::size_of::<Option<Spectrumf32>>();

        let size = FileHeader::SIZE
            + JSON_TAG.len()
            + VISIBILITY_TAG.len()
            + MATERIALS_TAG.len()
            + 3 * std::mem::size_of::<SectionHeader>()
            + json.len()
            + visibility_size
            + materials_size;

        let header = FileHeader {
            format_version: FORMAT_VERSION,
            total_size: size as u64,
            resolution: self.resolution as u32,
            visibility_samples: VISIBILITY_SAMPLES as u32,
            material_samples: MATERIAL_SAMPLES as u32,
            spectrum_resolution: Spectrumf32::RESOLUTION as u32,
        };

        header.to_writer(&mut file).map_err(IO)?;

        let json_header = SectionHeader::new(json.len(), JSON_TAG);
        json_header.to_writer(&mut file).map_err(IO)?;

        file.write_all(json.as_bytes()).map_err(IO)?;

        let visibility_header = SectionHeader::new(visibility_size, VISIBILITY_TAG);
        visibility_header.to_writer(&mut file).map_err(IO)?;
        file.write_all(&self.visibility).map_err(IO)?;

        let materials_header = SectionHeader::new(materials_size, MATERIALS_TAG);
        materials_header.to_writer(&mut file).map_err(IO)?;

        let materials_buffer: &[u8] =
            unsafe { std::slice::from_raw_parts(self.materials.as_ptr() as *const u8, materials_size) };

        file.write_all(materials_buffer).map_err(IO)?;

        Ok(())
    }

    pub fn read_from_file<P>(
        path: P,
        expected_scene_version: &SceneVersion,
        scene: &Scene,
        expected_resolution: usize,
    ) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        use Error::{Serde, IO};
        let mut file = File::open(path).map_err(IO)?;

        let header = FileHeader::from_reader(&mut file, expected_resolution)?;

        let json_header = SectionHeader::from_reader(&mut file, JSON_TAG, None)?;

        //Read json
        let mut json_buffer = vec![0; json_header.size as usize];
        file.read_exact(&mut json_buffer).map_err(IO)?;

        let mut stats = serde_json::from_slice::<Self>(&json_buffer).map_err(Serde)?;

        if stats.scene_version.hash != expected_scene_version.hash {
            return Err(Error::SceneMismatch);
        }

        let expected_voxel_count = (header.resolution * header.resolution * header.resolution) as usize;
        let expected_vis_count = (expected_voxel_count * (expected_voxel_count + 1)) / 2;

        let expected_vis_size = expected_vis_count * std::mem::size_of::<Visibility>();
        let expected_mats_size = expected_voxel_count * std::mem::size_of::<Option<Spectrumf32>>();

        let _visibility_header = SectionHeader::from_reader(&mut file, VISIBILITY_TAG, Some(expected_vis_size))?;

        stats.visibility = vec![0; expected_vis_count];

        file.read_exact(&mut stats.visibility).map_err(IO)?;

        let _materials_header = SectionHeader::from_reader(&mut file, MATERIALS_TAG, Some(expected_mats_size))?;

        stats.materials = vec![None; expected_voxel_count];

        unsafe {
            file.read_exact(std::slice::from_raw_parts_mut(
                stats.materials.as_ptr() as *mut u8,
                expected_mats_size,
            ))
            .map_err(IO)?;
        }

        // These aren't stored because recomputing them is cheap
        stats.compute_visibility_weighted_material_sums();
        stats.compute_light_voxel_distribution(scene);

        Ok(stats)
    }
}

#[derive(Clone, Copy)]
struct Vert {
    position: Point3<f32>,
    barycentric: Point2<f32>,
}

#[derive(Clone, Copy)]
struct ClippedTri {
    instance_index: usize,
    tri_index: usize,
    verts: [Vert; 3],
    bounds: BoundingBox,
}

impl ClippedTri {
    fn recalculate_bounds(&mut self) {
        self.bounds = BoundingBox::new();

        for vert in self.verts {
            self.bounds.add(vert.position);
        }
    }
}

enum ClipTriResult {
    Two(ClippedTri, ClippedTri),
    Three(ClippedTri, ClippedTri, ClippedTri),
}

#[derive(PartialEq, Eq)]
enum PositionRelativeToPlane {
    Left,
    Right,
    On,
}

impl PositionRelativeToPlane {
    fn test(position: f32, plane: f32) -> Self {
        if position < plane {
            Self::Left
        } else if position > plane {
            Self::Right
        } else {
            Self::On
        }
    }
}

fn clip_triangle(tri: ClippedTri, axis: Axis, position: f32) -> ClipTriResult {
    let pos_on_split_plane = |v: &Vert| PositionRelativeToPlane::test(v.position[axis.index()], position);
    let verts_on_split_plane = tri.verts.iter().map(pos_on_split_plane).collect::<ArrayVec<_, 3>>();
    let num_verts_on_split_plane: usize = verts_on_split_plane
        .iter()
        .map(|p| usize::from(*p == PositionRelativeToPlane::On))
        .sum();

    assert!(num_verts_on_split_plane < 2);

    let intersect_edge = |v0: Vert, v1: Vert| {
        let (t, split_point) = line_axis_plane_intersect(v0.position, v1.position, axis, position);

        let new_vert = Vert {
            position: split_point,
            barycentric: v0.barycentric + t * (v1.barycentric - v0.barycentric),
        };

        (t, new_vert)
    };

    if num_verts_on_split_plane == 1 {
        let (p1, p2) = if verts_on_split_plane[0] == PositionRelativeToPlane::On {
            (1, 2)
        } else if verts_on_split_plane[1] == PositionRelativeToPlane::On {
            (2, 0)
        } else {
            (0, 1)
        };

        let (t, new_vert) = intersect_edge(tri.verts[p1], tri.verts[p2]);

        assert!((0.0..1.0).contains(&t));

        let mut tri1 = tri;
        tri1.verts[p1] = new_vert;
        tri1.recalculate_bounds();

        let mut tri2 = tri;
        tri2.verts[p2] = new_vert;
        tri2.recalculate_bounds();

        ClipTriResult::Two(tri1, tri2)
    } else {
        assert!(num_verts_on_split_plane == 0);

        let (p0, p1, p2) = if verts_on_split_plane[0] == verts_on_split_plane[1] {
            (2, 0, 1)
        } else if verts_on_split_plane[0] == verts_on_split_plane[2] {
            (1, 2, 0)
        } else {
            assert!(verts_on_split_plane[1] == verts_on_split_plane[2]);
            (0, 1, 2)
        };

        let (t1, new_vert1) = intersect_edge(tri.verts[p0], tri.verts[p1]);
        let (t2, new_vert2) = intersect_edge(tri.verts[p0], tri.verts[p2]);

        // If both are on the boundary of the interval we create the same triangle again and get stuck in infinite recursion
        assert!((0.0..1.0).contains(&t1) || (0.0..1.0).contains(&t2));

        let mut tri1 = tri;
        tri1.verts[p1] = new_vert1;
        tri1.verts[p2] = new_vert2;
        tri1.recalculate_bounds();

        let mut tri2 = tri;
        tri2.verts[p0] = new_vert1;
        tri2.verts[p2] = new_vert2;
        tri2.recalculate_bounds();

        let mut tri3 = tri;
        tri3.verts[p0] = new_vert2;
        tri3.recalculate_bounds();

        ClipTriResult::Three(tri1, tri2, tri3)
    }
}

fn sample_triangle_materials(tris: Vec<ClippedTri>, raytracer: &Raytracer) -> Option<Spectrumf32> {
    if tris.is_empty() {
        return None;
    }

    let surface_areas = tris
        .iter()
        .map(|tri| triangle_area(tri.verts[0].position, tri.verts[1].position, tri.verts[2].position))
        .collect::<Vec<_>>();

    let cumulative_probabilities = cumulative_probabilities_from_weights(&surface_areas);
    let mut spectrum = Spectrumf32::constant(0.0);
    let mut samples = 0.0;

    for _ in 0..MATERIAL_SAMPLES {
        let triangle = tris[sample_item_from_cumulative_probabilities(&cumulative_probabilities).unwrap()];
        let instance = &raytracer.scene.instances[triangle.instance_index];
        let mesh = &raytracer.scene.meshes[instance.mesh_index as usize];
        let original_tri = &mesh.triangles[triangle.tri_index];

        if let Some(texture_ref) = instance.material.base_color_texture {
            let point = sample_coordinates_on_triangle();

            let barycentric = interpolate_point_on_triangle(
                point,
                triangle.verts[0].barycentric,
                triangle.verts[1].barycentric,
                triangle.verts[2].barycentric,
            );

            let tex_coords = &mesh.vertices.tex_coords[texture_ref.texture_coordinate_set];

            let v0 = tex_coords[original_tri.indices[0] as usize];
            let v1 = tex_coords[original_tri.indices[1] as usize];
            let v2 = tex_coords[original_tri.indices[2] as usize];

            let uv = interpolate_point_on_triangle(barycentric, v0, v1, v2);
            let Point2 { x: u, y: v } = texture_ref.transform_texture_coordinates(uv);
            let coeffs_sample = raytracer.scene.textures.base_color_coefficients[texture_ref.index].sample(u, v);

            let alpha = coeffs_sample.a;

            let base_color_spectrum = alpha
                * Spectrumf32::from_coefficients(instance.material.base_color_coefficients)
                * Spectrumf32::from_coefficients(coeffs_sample.rgb().into());

            spectrum += base_color_spectrum;
            samples += alpha;
        } else {
            spectrum += Spectrumf32::from_coefficients(instance.material.base_color_coefficients);
            samples += 1.0;
        }
    }

    Some(spectrum / samples)
}

struct FileHeader {
    pub format_version: u32,
    pub total_size: u64,
    pub resolution: u32,
    pub visibility_samples: u32,
    pub material_samples: u32,
    pub spectrum_resolution: u32,
}

impl FileHeader {
    const SIZE: usize = TAG.len() + std::mem::size_of::<Self>();

    pub fn to_writer<W>(self, writer: &mut W) -> Result<(), std::io::Error>
    where
        W: Write,
    {
        write!(writer, "{TAG}")?;
        writer.write_u32::<LittleEndian>(self.format_version)?;
        writer.write_u64::<LittleEndian>(self.total_size)?;
        writer.write_u32::<LittleEndian>(self.resolution)?;
        writer.write_u32::<LittleEndian>(self.visibility_samples)?;
        writer.write_u32::<LittleEndian>(self.material_samples)?;
        writer.write_u32::<LittleEndian>(self.spectrum_resolution)?;
        Ok(())
    }

    pub fn from_reader<R: Read>(reader: &mut R, expected_resolution: usize) -> Result<Self, Error> {
        use Error::IO;
        let mut tag = [0; TAG.len()];
        reader.read_exact(&mut tag).map_err(IO)?;

        if tag != TAG.as_bytes() {
            return Err(Error::TagMismatch {
                expected: TAG.as_bytes().to_vec(),
                actual: tag.to_vec(),
            });
        }

        let format_version = reader.read_u32::<LittleEndian>().map_err(IO)?;
        if format_version != FORMAT_VERSION {
            return Err(Error::FormatVersionMismatch {
                current: FORMAT_VERSION,
                file: format_version,
            });
        }

        let total_size = reader.read_u64::<LittleEndian>().map_err(IO)?;

        let resolution = reader.read_u32::<LittleEndian>().map_err(IO)?;
        if resolution as usize != expected_resolution {
            return Err(Error::ResolutionMismatch {
                current: expected_resolution,
                file: resolution as usize,
                name: "visibility".to_string(),
            });
        }

        let visibility_samples = reader.read_u32::<LittleEndian>().map_err(IO)?;
        if visibility_samples as usize != VISIBILITY_SAMPLES {
            return Err(Error::SamplesMismatch {
                current: VISIBILITY_SAMPLES,
                file: visibility_samples as usize,
                name: "visibility".to_string(),
            });
        }

        let material_samples = reader.read_u32::<LittleEndian>().map_err(IO)?;
        if material_samples as usize != MATERIAL_SAMPLES {
            return Err(Error::SamplesMismatch {
                current: MATERIAL_SAMPLES,
                file: material_samples as usize,
                name: "material".to_string(),
            });
        }

        let spectrum_resolution = reader.read_u32::<LittleEndian>().map_err(IO)?;
        if spectrum_resolution as usize != Spectrumf32::RESOLUTION {
            return Err(Error::ResolutionMismatch {
                current: Spectrumf32::RESOLUTION,
                file: spectrum_resolution as usize,
                name: "spectrum".to_string(),
            });
        }

        Ok(Self {
            format_version,
            total_size,
            resolution,
            visibility_samples,
            material_samples,
            spectrum_resolution,
        })
    }
}
