use cgmath::{ElementWise, Point2, Point3, Vector3};
use itertools::Itertools;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    raytracer::{
        aabb::BoundingBox,
        acceleration::{
            bvh_rec::BoundingVolumeHierarchyRec,
            structure::{AccelerationStructure, TraceResult},
        },
        geometry::interpolate_point_on_triangle,
        ray::Ray,
        sampling::sample_uniform_on_unit_sphere,
        triangle::Triangle,
        Raytracer,
    },
    spectrum::Spectrumf32,
};

use super::ClippedTri;

//TODO: rename
pub struct VoxelGeometry {
    bvh: BoundingVolumeHierarchyRec,
    bounds: BoundingBox,
    tris: Vec<ClippedTri>,
    vertex_positions: Vec<Point3<f32>>,
    raw_tris: Vec<Triangle>,
}

#[derive(Serialize, Deserialize)]
pub struct SampleResult {
    hits: f32,
    samples: f32,
    pub albedo_mean: Spectrumf32,
}

impl VoxelGeometry {
    pub fn new(bounds: BoundingBox, tris: Vec<ClippedTri>) -> Self {
        let triangle_bounds = tris.iter().map(|t| t.bounds).collect_vec();
        let bvh = BoundingVolumeHierarchyRec::new(&triangle_bounds);

        let vertex_positions = tris.iter().flat_map(|t| t.verts.iter()).map(|v| v.position).collect();
        let raw_tris = tris
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let i = 3 * i as u32;
                Triangle {
                    indices: [i, i + 1, i + 2],
                }
            })
            .collect();

        Self {
            bvh,
            bounds,
            tris,
            vertex_positions,
            raw_tris,
        }
    }

    pub fn sample_rays(&self, raytracer: &Raytracer) -> SampleResult {
        const SAMPLES: usize = 1 << 9;

        let mut hits = 0.0;
        let mut albedo = Spectrumf32::constant(0.0);

        for _ in 0..SAMPLES {
            let p = self.sample_point();
            let direction = sample_uniform_on_unit_sphere();

            let ray = Ray {
                origin: p - 2.0 * direction,
                direction,
            };

            match self.bvh.intersect(&ray, &self.vertex_positions, &self.raw_tris) {
                TraceResult::Hit {
                    triangle_index, u, v, ..
                } => {
                    let triangle = self.tris[triangle_index as usize];
                    let instance = &raytracer.scene.instances[triangle.instance_index];
                    let mesh = &raytracer.scene.meshes[instance.mesh_index as usize];
                    let original_tri = &mesh.triangles[triangle.tri_index];

                    if let Some(texture_ref) = instance.material.base_color_texture {
                        let barycentric = interpolate_point_on_triangle(
                            Point2::new(u, v),
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
                        let coeffs_sample =
                            raytracer.scene.textures.base_color_coefficients[texture_ref.index].sample(u, v);

                        let alpha = coeffs_sample.a;

                        let base_color_spectrum = alpha
                            * Spectrumf32::from_coefficients(instance.material.base_color_coefficients)
                            * Spectrumf32::from_coefficients(coeffs_sample.rgb().into());

                        albedo += base_color_spectrum;
                        hits += alpha;
                    } else {
                        albedo += Spectrumf32::from_coefficients(instance.material.base_color_coefficients);
                        hits += 1.0;
                    }
                }
                TraceResult::Miss => {}
            }
        }

        if hits > 0.0 {
            albedo /= hits;
        }

        SampleResult {
            hits,
            samples: SAMPLES as f32,
            albedo_mean: albedo,
        }
    }

    fn sample_point(&self) -> Point3<f32> {
        let mut rng = rand::thread_rng();
        let extent = self.bounds.max - self.bounds.min;

        self.bounds.min + Vector3::new(rng.gen::<f32>(), rng.gen(), rng.gen()).mul_element_wise(extent)
    }
}
