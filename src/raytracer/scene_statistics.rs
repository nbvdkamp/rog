use std::path::Path;

use rayon::iter::ParallelIterator;
use static_assertions::const_assert;

use cgmath::{vec2, vec3, ElementWise, Point3, Vector3};
use rand::Rng;
use rayon::prelude::IntoParallelIterator;

use crate::{color::RGBf32, mesh::Vertex, util::save_image};

use super::{
    aabb::BoundingBox,
    acceleration::{structure::TraceResult, Accel, AccelerationStructures},
    ray::Ray,
    triangle::Triangle,
};

const RESOLUTION: usize = 1 << 4;
const CELL_COUNT: usize = RESOLUTION * RESOLUTION * RESOLUTION;
const TABLE_SIZE: usize = (CELL_COUNT * (CELL_COUNT + 1)) / 2;
const VISIBILITY_SAMPLES: usize = 16;

type Visibility = u8;

const_assert!(Visibility::MAX as usize >= VISIBILITY_SAMPLES);

pub struct SceneStatistics {
    scene_bounds: BoundingBox,
    scene_extent: Vector3<f32>,
    cell_extent: Vector3<f32>,
    visibility: Vec<Visibility>,
}

impl SceneStatistics {
    pub fn new(scene_bounds: BoundingBox) -> Self {
        let scene_extent = scene_bounds.max - scene_bounds.min;

        SceneStatistics {
            scene_bounds,
            scene_extent,
            cell_extent: scene_extent / RESOLUTION as f32,
            visibility: Vec::new(),
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

    fn get_cell_index(&self, point: Point3<f32>) -> usize {
        let relative_pos = (point - self.scene_bounds.min).div_element_wise(self.scene_extent);
        let v = relative_pos * RESOLUTION as f32;
        let x = (v.x as usize).min(RESOLUTION - 1);
        let y = (v.y as usize).min(RESOLUTION - 1);
        let z = (v.z as usize).min(RESOLUTION - 1);
        x + RESOLUTION * y + RESOLUTION * RESOLUTION * z
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
}
