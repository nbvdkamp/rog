use crate::{
    mesh::Vertex,
    raytracer::{
        aabb::BoundingBox,
        ray::{IntersectionResult, Ray},
        triangle::Triangle,
    },
};

use super::{statistics::Statistics, structure::TraceResult};

pub fn compute_bounding_box(vertices: &[Vertex]) -> BoundingBox {
    let mut bounds = BoundingBox::new();

    for vertex in vertices {
        bounds.add(vertex.position);
    }

    bounds
}

pub fn compute_bounding_box_triangle_indexed(
    triangle_bounds: &[BoundingBox],
    triangle_indices: &[usize],
) -> BoundingBox {
    let mut bounds = BoundingBox::new();

    for i in triangle_indices {
        bounds = bounds.union(triangle_bounds[*i]);
    }

    bounds
}

pub fn intersect_triangles_indexed(
    triangle_indices: &[usize],
    ray: &Ray,
    verts: &[Vertex],
    triangles: &[Triangle],
    stats: &Statistics,
) -> TraceResult {
    let mut result = TraceResult::Miss;

    for triangle_index in triangle_indices {
        let triangle = &triangles[*triangle_index];
        let p1 = &verts[triangle.index1 as usize];
        let p2 = &verts[triangle.index2 as usize];
        let p3 = &verts[triangle.index3 as usize];

        stats.count_intersection_test();

        if let IntersectionResult::Hit { t, u, v } = ray.intersect_triangle(p1.position, p2.position, p3.position) {
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
