use std::{
    fs::File,
    io::{Result, Write},
    path::Path,
};

use arrayvec::ArrayVec;
use rayon::iter::ParallelIterator;
use static_assertions::const_assert;

use cgmath::{point2, point3, vec2, vec3, ElementWise, Point2, Point3, Vector3};
use rand::Rng;
use rayon::prelude::IntoParallelIterator;

use crate::{
    color::RGBf32,
    material::Material,
    mesh::Vertex,
    raytracer::{geometry::triangle_area, sampling::sample_coordinates_on_triangle},
    spectrum::Spectrumf32,
    util::save_image,
};

use super::{
    aabb::BoundingBox,
    acceleration::{structure::TraceResult, Accel, AccelerationStructures},
    axis::Axis,
    geometry::{interpolate_point_on_triangle, line_axis_plane_intersect},
    ray::Ray,
    sampling::{cumulative_probabilities_from_weights, sample_item_from_cumulative_probabilities},
    triangle::Triangle,
    Textures,
};

const RESOLUTION: usize = 1 << 4;
const CELL_COUNT: usize = RESOLUTION * RESOLUTION * RESOLUTION;
const TABLE_SIZE: usize = (CELL_COUNT * (CELL_COUNT + 1)) / 2;
const VISIBILITY_SAMPLES: usize = 16;
const MATERIAL_SAMPLES: usize = 128;

type Visibility = u8;

const_assert!(Visibility::MAX as usize >= VISIBILITY_SAMPLES);

pub struct SceneStatistics {
    scene_bounds: BoundingBox,
    scene_extent: Vector3<f32>,
    cell_extent: Vector3<f32>,
    visibility: Vec<Visibility>,
    materials: Vec<Option<Spectrumf32>>,
}

impl SceneStatistics {
    pub fn new(scene_bounds: BoundingBox) -> Self {
        let scene_extent = scene_bounds.max - scene_bounds.min;

        SceneStatistics {
            scene_bounds,
            scene_extent,
            cell_extent: scene_extent / RESOLUTION as f32,
            visibility: Vec::new(),
            materials: Vec::new(),
        }
    }

    pub fn sample_visibility(
        &mut self,
        structures: &AccelerationStructures,
        accel: Accel,
        verts: &[Vertex],
        triangles: &[Triangle],
    ) {
        self.visibility = (0..CELL_COUNT)
            .into_par_iter()
            .flat_map(|b| {
                let mut vis = Vec::new();
                vis.reserve(CELL_COUNT - b);

                for a in b..CELL_COUNT {
                    let mut v = 0;

                    for _ in 0..VISIBILITY_SAMPLES {
                        let start = self.sample_point_in_cell(a);
                        let end = self.sample_point_in_cell(b);

                        let ray = Ray {
                            origin: start,
                            direction: end - start,
                        };

                        let occluded = match structures.get(accel).intersect(&ray, verts, triangles) {
                            TraceResult::Hit { t, .. } => t < 1.0,
                            TraceResult::Miss => false,
                        };

                        if !occluded {
                            v += 1;
                        }
                    }

                    vis.push(v);
                }

                vis
            })
            .collect();
    }

    pub fn dump_visibility_image<P>(&self, path: P)
    where
        P: AsRef<Path>,
    {
        assert_eq!(self.visibility.len(), TABLE_SIZE);

        let image_size = vec2(CELL_COUNT, CELL_COUNT);
        let mut buffer = vec![RGBf32::new(0.0, 0.0, 0.0); CELL_COUNT * CELL_COUNT];

        for b in 0..CELL_COUNT {
            for a in b..CELL_COUNT {
                let i = self.get_table_index(a, b);
                buffer[a + b * CELL_COUNT] =
                    RGBf32::from_grayscale(self.visibility[i] as f32 / VISIBILITY_SAMPLES as f32);
            }
        }

        save_image(&buffer, image_size, path);
    }

    pub fn dump_materials_as_rgb<P>(&self, path: P) -> Result<()>
    where
        P: AsRef<Path>,
    {
        assert_eq!(self.materials.len(), CELL_COUNT);

        let mut file = File::create(path)?;
        write!(file, "resolution {RESOLUTION}\n")?;

        for x in 0..RESOLUTION {
            for y in 0..RESOLUTION {
                for z in 0..RESOLUTION {
                    let i = x + y * RESOLUTION + z * RESOLUTION * RESOLUTION;

                    if let Some(spectrum) = self.materials[i] {
                        let RGBf32 { r, g, b } = spectrum.to_srgb();
                        write!(file, "{x} {y} {z} {r} {g} {b}\n")?;
                    }
                }
            }
        }

        Ok(())
    }

    fn get_grid_position(&self, point: Point3<f32>) -> Point3<usize> {
        let relative_pos = (point - self.scene_bounds.min).div_element_wise(self.scene_extent);
        let v = relative_pos * RESOLUTION as f32;
        point3(
            (v.x as usize).min(RESOLUTION - 1),
            (v.y as usize).min(RESOLUTION - 1),
            (v.z as usize).min(RESOLUTION - 1),
        )
    }

    fn min_corner_of_voxel(&self, position: Point3<usize>) -> Point3<f32> {
        self.scene_bounds.min
            + vec3(
                position.x as f32 * self.cell_extent.x,
                position.y as f32 * self.cell_extent.y,
                position.z as f32 * self.cell_extent.z,
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

    fn get_cell_index(&self, point: Point3<f32>) -> usize {
        let v = self.get_grid_position(point);
        v.x + v.y * RESOLUTION + v.z * RESOLUTION * RESOLUTION
    }

    fn sample_point_in_cell(&self, cell_index: usize) -> Point3<f32> {
        let x = (cell_index % RESOLUTION) as f32;
        let y = ((cell_index % (RESOLUTION * RESOLUTION)) / RESOLUTION) as f32;
        let z = (cell_index / (RESOLUTION * RESOLUTION)) as f32;
        let cell_offset = self.cell_extent.mul_element_wise(vec3(x, y, z));

        let mut rng = rand::thread_rng();
        let random_offset = vec3(rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>());

        self.scene_bounds.min + cell_offset + self.cell_extent.mul_element_wise(random_offset)
    }

    fn get_table_index(&self, first_index: usize, second_index: usize) -> usize {
        let a = first_index.max(second_index);
        let b = first_index.min(second_index);
        a + b * CELL_COUNT - b * (b + 1) / 2
    }

    pub fn sample_materials(
        &mut self,
        verts: &[Vertex],
        triangles: &[Triangle],
        triangle_bounds: &[BoundingBox],
        materials: &[Material],
        textures: &Textures,
    ) {
        self.materials = self
            .split_triangles_into_cells(verts, triangles, triangle_bounds)
            .into_iter()
            .map(|tris| sample_triangle_materials(tris, verts, triangles, materials, textures))
            .collect();
    }

    fn split_triangles_into_cells(
        &self,
        verts: &[Vertex],
        triangles: &[Triangle],
        triangle_bounds: &[BoundingBox],
    ) -> Vec<Vec<ClippedTri>> {
        let mut tris_per_cell = Vec::new();
        tris_per_cell.reserve(CELL_COUNT);

        for _ in 0..CELL_COUNT {
            tris_per_cell.push(Vec::new());
        }

        let mut tris_to_sort = (0..triangles.len())
            .map(|i| {
                let tri = &triangles[i];
                ClippedTri {
                    original_index: i,
                    verts: [
                        Vert {
                            position: verts[tri.index1 as usize].position,
                            barycentric: point2(0.0, 0.0),
                        },
                        Vert {
                            position: verts[tri.index2 as usize].position,
                            barycentric: point2(1.0, 0.0),
                        },
                        Vert {
                            position: verts[tri.index3 as usize].position,
                            barycentric: point2(0.0, 1.0),
                        },
                    ],
                    bounds: triangle_bounds[i],
                }
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
                let cell_index =
                    center_grid_pos.x + center_grid_pos.y * RESOLUTION + center_grid_pos.z * RESOLUTION * RESOLUTION;
                tris_per_cell[cell_index].push(tri);
            } else {
                // Split the triangle on the voxel boundary and recursively handle the resulting triangles
                for axis_index in 0..3 {
                    if !min_contained_axes[axis_index] || !max_contained_axes[axis_index] {
                        let x = if !min_contained_axes[axis_index] { 0 } else { 1 };

                        let clip_position = self.scene_bounds.min[axis_index]
                            + self.cell_extent[axis_index] * (center_grid_pos[axis_index] + x) as f32;

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

        tris_per_cell
    }
}

#[derive(Clone, Copy)]
struct Vert {
    position: Point3<f32>,
    barycentric: Point2<f32>,
}

#[derive(Clone, Copy)]
struct ClippedTri {
    original_index: usize,
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
        .map(|p| if *p == PositionRelativeToPlane::On { 1 } else { 0 })
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

fn sample_triangle_materials(
    tris: Vec<ClippedTri>,
    verts: &[Vertex],
    original_triangles: &[Triangle],
    materials: &[Material],
    textures: &Textures,
) -> Option<Spectrumf32> {
    if tris.is_empty() {
        return None;
    }

    let surface_areas = tris
        .iter()
        .map(|tri| triangle_area(tri.verts[0].position, tri.verts[1].position, tri.verts[2].position))
        .collect::<Vec<_>>();

    let cumulative_probabilities = cumulative_probabilities_from_weights(surface_areas);
    let mut spectrum = Spectrumf32::constant(0.0);

    for _ in 0..MATERIAL_SAMPLES {
        let triangle = tris[sample_item_from_cumulative_probabilities(&cumulative_probabilities).unwrap()];
        let original_tri = &original_triangles[triangle.original_index];
        let material = &materials[original_tri.material_index as usize];
        let has_tex_coords = verts[original_tri.index1 as usize].tex_coord.is_some();

        if has_tex_coords {
            let point = sample_coordinates_on_triangle();

            let barycentric = interpolate_point_on_triangle(
                point,
                triangle.verts[0].barycentric,
                triangle.verts[1].barycentric,
                triangle.verts[2].barycentric,
            );

            let v0 = verts[original_tri.index1 as usize].tex_coord.unwrap();
            let v1 = verts[original_tri.index2 as usize].tex_coord.unwrap();
            let v2 = verts[original_tri.index3 as usize].tex_coord.unwrap();

            let texture_coordinates = interpolate_point_on_triangle(barycentric, v0, v1, v2);
            let sample = material.sample(texture_coordinates, textures);
            spectrum += sample.base_color_spectrum;
        } else {
            spectrum += Spectrumf32::from_coefficients(material.base_color_coefficients);
        }
    }

    Some(spectrum / MATERIAL_SAMPLES as f32)
}
