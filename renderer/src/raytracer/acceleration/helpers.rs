use cgmath::Point3;

use crate::raytracer::{
    aabb::BoundingBox,
    ray::{IntersectionResult, Ray},
    triangle::Triangle,
};

use super::{statistics::Statistics, structure::TraceResult};

pub fn compute_bounding_box(positions: &[Point3<f32>]) -> BoundingBox {
    let mut bounds = BoundingBox::new();

    for p in positions {
        bounds.add(*p);
    }

    bounds
}

pub fn compute_bounding_box_item_indexed(item_bounds: &[BoundingBox], indices: &[usize]) -> BoundingBox {
    let mut bounds = BoundingBox::new();

    for &i in indices {
        bounds = bounds.union(&item_bounds[i]);
    }

    bounds
}

pub fn intersect_triangles_indexed(
    triangle_indices: &[usize],
    ray: &Ray,
    positions: &[Point3<f32>],
    triangles: &[Triangle],
    stats: &Statistics,
) -> TraceResult {
    let mut result = TraceResult::Miss;

    for triangle_index in triangle_indices {
        let triangle = &triangles[*triangle_index];
        let p1 = positions[triangle.indices[0] as usize];
        let p2 = positions[triangle.indices[1] as usize];
        let p3 = positions[triangle.indices[2] as usize];

        stats.count_intersection_test();

        if let IntersectionResult::Hit { t, u, v } = ray.intersect_triangle(p1, p2, p3) {
            stats.count_intersection_hit();

            if result.is_farther_than(t) {
                result = TraceResult::Hit {
                    triangle_index: *triangle_index as u32,
                    t,
                    u,
                    v,
                };
            }
        }
    }

    result
}
