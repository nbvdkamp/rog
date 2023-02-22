use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fs::File,
    io::{Read, Write},
    path::Path,
};

use arrayvec::ArrayVec;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use fnv::FnvHashMap;
use rayon::{iter::ParallelIterator, prelude::IntoParallelRefIterator};
use serde::{Deserialize, Serialize};
use static_assertions::const_assert;

use cgmath::{point2, vec2, vec3, ElementWise, Point2, Point3, Vector3, Zero};
use itertools::Itertools;
use rand::Rng;

use crate::{
    cie_data as CIE,
    color::RGBf32,
    raytracer::{geometry::triangle_area, sampling::sample_coordinates_on_triangle},
    scene::Scene,
    scene_version::SceneVersion,
    spectrum::Spectrumf32,
    util::save_image,
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

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Visibility {
    pub passed_tests: u8,
}

impl Visibility {
    pub fn value(&self) -> f32 {
        self.passed_tests as f32 / VISIBILITY_SAMPLES as f32
    }
}

const_assert!(u8::MAX as usize >= VISIBILITY_SAMPLES);

#[derive(Serialize, Deserialize)]
pub struct Distribution {
    pub probabilities: Spectrumf32,
    pub cumulative_probabilities: Spectrumf32,
}

impl Distribution {
    pub fn sample_wavelength(&self) -> (f32, f32) {
        let i = sample_item_from_cumulative_probabilities(&self.cumulative_probabilities.data)
            .expect("data can't be empty");

        let pdf = self.probabilities.data[i] * Spectrumf32::RESOLUTION as f32;
        let value = CIE::LAMBDA_MIN + Spectrumf32::STEP_SIZE * (i as f32 + rand::thread_rng().gen::<f32>());
        (value, pdf)
    }
}

#[derive(Serialize, Deserialize)]
pub struct VoxelLights {
    pub voxel: VoxelId,
    pub light_indices: Vec<usize>,
}

pub type VoxelId = Point3<u8>;

#[derive(PartialEq, Eq, Hash, Clone, Copy, Serialize, Deserialize)]
pub struct VoxelPair {
    first: VoxelId,
    second: VoxelId,
}

impl VoxelPair {
    pub fn new(a: VoxelId, b: VoxelId) -> Self {
        fn cmp(a: VoxelId, b: VoxelId) -> Ordering {
            let x = a.x.cmp(&b.x);
            let Ordering::Equal = x  else {
                return x;
            };

            let y = a.y.cmp(&b.y);
            let Ordering::Equal = y else {
                return y;
            };

            return a.z.cmp(&b.z);
        }

        if cmp(a, b) == Ordering::Less {
            Self { first: a, second: b }
        } else {
            Self { first: b, second: a }
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct SceneStatistics {
    resolution: u8,
    voxel_count: usize,
    scene_bounds: BoundingBox,
    scene_extent: Vector3<f32>,
    voxel_extent: Vector3<f32>,
    scene_version: SceneVersion,
    pub voxels_with_lights: Vec<VoxelLights>,
    pub positionless_lights: Vec<usize>,
    visibility: FnvHashMap<VoxelPair, Visibility>,
    materials: HashMap<VoxelId, Spectrumf32>,
    pub spectral_distributions: HashMap<VoxelId, Distribution>,
}

impl SceneStatistics {
    pub fn new(scene_bounds: BoundingBox, scene_version: SceneVersion, resolution: u8) -> Self {
        let scene_extent = scene_bounds.max - scene_bounds.min;
        let cube = |x| x * x * x;

        SceneStatistics {
            resolution,
            voxel_count: cube(resolution as usize),
            scene_bounds,
            scene_extent,
            voxel_extent: scene_extent / resolution as f32,
            scene_version,
            voxels_with_lights: Vec::new(),
            positionless_lights: Vec::new(),
            visibility: FnvHashMap::default(),
            materials: HashMap::new(),
            spectral_distributions: HashMap::new(),
        }
    }

    pub fn get_estimated_visibility(&self, first: VoxelId, second: VoxelId) -> f32 {
        let vis = self.get_raw_visibility(VoxelPair::new(first, second)).value();
        // Ensure estimate is never 0 to prevent bias
        (0.99 * vis + 0.01).clamp(0.01, 1.0)
    }

    pub fn sample_visibility(&mut self, raytracer: &Raytracer, accel: Accel) {
        assert!(!self.materials.is_empty());
        assert!(!self.voxels_with_lights.is_empty() || !self.positionless_lights.is_empty());

        let mut nonempty_voxels = self.materials.keys().copied().collect::<HashSet<_>>();

        for v in &self.voxels_with_lights {
            let mut stack = vec![v.voxel];

            while let Some(voxel) = stack.pop() {
                let bounds = BoundingBox {
                    min: self.min_corner_of_voxel(voxel),
                    max: self.max_corner_of_voxel(voxel),
                };

                let mut intersects_light = false;
                for &light_index in &v.light_indices {
                    let light = &raytracer.scene.lights[light_index];
                    if bounds.intersects_sphere(light.position().unwrap(), light.radius().unwrap()) {
                        intersects_light = true;
                        break;
                    }
                }

                if intersects_light && nonempty_voxels.insert(voxel) {
                    stack.append(&mut self.voxel_neighbours(voxel));
                }
            }
        }

        self.visibility = nonempty_voxels
            .par_iter()
            .flat_map(|&b| {
                nonempty_voxels
                    .iter()
                    .flat_map(|&a| {
                        let vis = self.measure_visibility_between_voxels(a, b, raytracer, accel);
                        if vis.passed_tests > 0 {
                            Some((VoxelPair::new(a, b), vis))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
    }

    fn measure_visibility_between_voxels(
        &self,
        a: VoxelId,
        b: VoxelId,
        raytracer: &Raytracer,
        accel: Accel,
    ) -> Visibility {
        let passed_tests = (0..VISIBILITY_SAMPLES)
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
            .sum();

        Visibility { passed_tests }
    }

    pub fn dump_visibility_image<P>(&self, path: P)
    where
        P: AsRef<Path>,
    {
        let image_size = vec2(self.voxel_count, self.voxel_count);
        let mut buffer = vec![RGBf32::new(1.0, 0.0, 0.0); image_size.x * image_size.y];

        for (pair, vis) in &self.visibility {
            let resolution = self.resolution as usize;
            let index = |v: Point3<usize>| v.x + v.y * resolution + v.z * resolution * resolution;
            let a = index(pair.first.cast().unwrap()) as usize;
            let b = index(pair.second.cast().unwrap()) as usize;
            let x = a.max(b);
            let y = a.min(b);
            buffer[x + y * self.voxel_count] = RGBf32::from_grayscale(vis.value());
        }

        save_image(&buffer, image_size, path);
    }

    pub fn dump_materials_as_rgb<P>(&self, path: P) -> Result<(), std::io::Error>
    where
        P: AsRef<Path>,
    {
        let mut file = File::create(path)?;
        writeln!(file, "resolution {}", self.resolution)?;

        for (&VoxelId { x, y, z }, spectrum) in &self.materials {
            let RGBf32 { r, g, b } = spectrum.to_srgb();
            writeln!(file, "{x} {y} {z} {r} {g} {b}")?;
        }

        Ok(())
    }

    pub fn get_grid_position(&self, point: Point3<f32>) -> VoxelId {
        let relative_pos = (point - self.scene_bounds.min).div_element_wise(self.scene_extent);
        let v = relative_pos * self.resolution as f32;
        VoxelId {
            x: (v.x as u8).min(self.resolution - 1),
            y: (v.y as u8).min(self.resolution - 1),
            z: (v.z as u8).min(self.resolution - 1),
        }
    }

    fn min_corner_of_voxel(&self, position: VoxelId) -> Point3<f32> {
        self.scene_bounds.min
            + vec3(
                position.x as f32 * self.voxel_extent.x,
                position.y as f32 * self.voxel_extent.y,
                position.z as f32 * self.voxel_extent.z,
            )
    }

    fn max_corner_of_voxel(&self, position: VoxelId) -> Point3<f32> {
        let VoxelId { x, y, z } = position;
        self.min_corner_of_voxel(VoxelId {
            x: x + 1,
            y: y + 1,
            z: z + 1,
        })
    }

    fn voxel_neighbours(&self, voxel: VoxelId) -> Vec<VoxelId> {
        let grid_range = 0..self.resolution as i32;

        (-1..=1)
            .cartesian_product((-1..=1).cartesian_product(-1..=1))
            .flat_map(|(ox, (oy, oz))| {
                let offset = Vector3::new(ox, oy, oz);

                if offset == Vector3::zero() {
                    return None;
                }

                let Point3 { x, y, z } = voxel.cast::<i32>().unwrap() + offset;

                if !grid_range.contains(&x) || !grid_range.contains(&y) || !grid_range.contains(&z) {
                    None
                } else {
                    Some(VoxelId {
                        x: x as u8,
                        y: y as u8,
                        z: z as u8,
                    })
                }
            })
            .collect()
    }

    fn bounds_from_grid_position(&self, position: VoxelId) -> BoundingBox {
        BoundingBox {
            min: self.min_corner_of_voxel(position),
            max: self.max_corner_of_voxel(position),
        }
    }

    fn sample_point_in_voxel(&self, voxel: VoxelId) -> Point3<f32> {
        let VoxelId { x, y, z } = voxel;
        let voxel_offset = self.voxel_extent.mul_element_wise(vec3(x as f32, y as f32, z as f32));

        let mut rng = rand::thread_rng();
        let random_offset = vec3(rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>());

        self.scene_bounds.min + voxel_offset + self.voxel_extent.mul_element_wise(random_offset)
    }

    pub fn compute_light_voxel_distribution(&mut self, scene: &Scene) {
        for (i, light) in scene.lights.iter().enumerate() {
            // TODO: Account for radius
            if let Some(position) = light.position() {
                let voxel = self.get_grid_position(position);

                if let Some(v) = self.voxels_with_lights.iter_mut().find(|v| v.voxel == voxel) {
                    v.light_indices.push(i);
                } else {
                    self.voxels_with_lights.push(VoxelLights {
                        voxel,
                        light_indices: vec![i],
                    });
                };
            } else {
                self.positionless_lights.push(i);
            }
        }
    }

    fn get_raw_visibility(&self, pair: VoxelPair) -> Visibility {
        if let Some(vis) = self.visibility.get(&pair) {
            *vis
        } else {
            Visibility { passed_tests: 0 }
        }
    }

    pub fn compute_visibility_weighted_material_sums(&mut self) {
        self.spectral_distributions = self
            .materials
            .par_iter()
            .map(|(&voxel, spectrum)| {
                let mut neighbour_vis_sum = 0.0;
                let mut neigbour_material_sum = Spectrumf32::constant(0.0);

                for (&other_voxel, other_spectrum) in self.materials.iter() {
                    if voxel == other_voxel {
                        continue;
                    }

                    let pair = VoxelPair::new(voxel, other_voxel);
                    let vis = self.get_raw_visibility(pair).value();
                    neighbour_vis_sum += vis;
                    neigbour_material_sum += vis * *other_spectrum;
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

                (
                    voxel,
                    Distribution {
                        probabilities,
                        cumulative_probabilities,
                    },
                )
            })
            .collect();
    }

    pub fn sample_materials(&mut self, raytracer: &Raytracer) {
        self.materials = self
            .split_triangles_into_voxels(&raytracer.scene)
            .into_iter()
            .map(|(voxel, tris)| (voxel, sample_triangle_materials(tris, raytracer)))
            .collect();
    }

    fn split_triangles_into_voxels(&self, scene: &Scene) -> HashMap<VoxelId, Vec<ClippedTri>> {
        let mut tris_per_voxel: HashMap<VoxelId, Vec<ClippedTri>> = HashMap::new();

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
                if let Some(tris) = tris_per_voxel.get_mut(&center_grid_pos) {
                    tris.push(tri);
                } else {
                    tris_per_voxel.insert(center_grid_pos, vec![tri]);
                }
            } else {
                // Split the triangle on the voxel boundary and recursively handle the resulting triangles
                for axis_index in 0..3 {
                    if !min_contained_axes[axis_index] || !max_contained_axes[axis_index] {
                        // If the intersection is on the max bound, offset by 1
                        let offset = u8::from(min_contained_axes[axis_index]);

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
        use Error::{Postcard, IO};
        let mut file = File::create(path).map_err(IO)?;

        let data = postcard::to_allocvec(self).map_err(Postcard)?;

        let header = FileHeader {
            format_version: FORMAT_VERSION,
            size: data.len() as u64,
            resolution: self.resolution as u32,
            visibility_samples: VISIBILITY_SAMPLES as u32,
            material_samples: MATERIAL_SAMPLES as u32,
            spectrum_resolution: Spectrumf32::RESOLUTION as u32,
        };

        header.to_writer(&mut file).map_err(IO)?;
        file.write_all(&data).map_err(IO)
    }

    pub fn read_from_file<P>(
        path: P,
        expected_scene_version: &SceneVersion,
        expected_resolution: u8,
    ) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        use Error::{Postcard, IO};
        let mut file = File::open(path).map_err(IO)?;

        let header = FileHeader::from_reader(&mut file, expected_resolution)?;
        let mut buffer = vec![0u8; header.size as usize];
        file.read_exact(&mut buffer).map_err(IO)?;
        let stats: SceneStatistics = postcard::from_bytes(&buffer).map_err(Postcard)?;

        if expected_scene_version.hash != stats.scene_version.hash {
            return Err(Error::SceneMismatch);
        }

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

fn sample_triangle_materials(tris: Vec<ClippedTri>, raytracer: &Raytracer) -> Spectrumf32 {
    assert!(!tris.is_empty());

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

    spectrum / samples
}

const FORMAT_VERSION: u32 = 1;
const TAG: &str = "SCENE_STATISTICS";
static_assertions::const_assert_eq!(TAG.len() % 4, 0);

struct FileHeader {
    pub format_version: u32,
    pub size: u64,
    pub resolution: u32,
    pub visibility_samples: u32,
    pub material_samples: u32,
    pub spectrum_resolution: u32,
}

impl FileHeader {
    pub fn to_writer<W>(self, writer: &mut W) -> Result<(), std::io::Error>
    where
        W: Write,
    {
        write!(writer, "{TAG}")?;
        writer.write_u32::<LittleEndian>(self.format_version)?;
        writer.write_u64::<LittleEndian>(self.size)?;
        writer.write_u32::<LittleEndian>(self.resolution)?;
        writer.write_u32::<LittleEndian>(self.visibility_samples)?;
        writer.write_u32::<LittleEndian>(self.material_samples)?;
        writer.write_u32::<LittleEndian>(self.spectrum_resolution)?;
        Ok(())
    }

    pub fn from_reader<R: Read>(reader: &mut R, expected_resolution: u8) -> Result<Self, Error> {
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
        if resolution != expected_resolution as u32 {
            return Err(Error::ResolutionMismatch {
                current: expected_resolution as usize,
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
            size: total_size,
            resolution,
            visibility_samples,
            material_samples,
            spectrum_resolution,
        })
    }
}
