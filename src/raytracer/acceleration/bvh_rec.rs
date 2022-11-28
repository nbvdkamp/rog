use crate::raytracer::{aabb::Intersects, axis::Axis, triangle::Triangle, Ray};

use cgmath::{Point3, Vector3};

use super::{
    super::aabb::BoundingBox,
    helpers::{compute_bounding_box_item_indexed, intersect_triangles_indexed},
    sah::{surface_area_heuristic_bvh, SurfaceAreaHeuristicResultBvh},
    statistics::{Statistics, StatisticsStore},
    structure::{AccelerationStructure, TraceResult},
};

pub struct BoundingVolumeHierarchyRec {
    root: Option<Box<Node>>,
    bounds: BoundingBox,
    stats: Statistics,
}

enum Node {
    Leaf {
        triangle_indices: Vec<usize>,
        bounds: BoundingBox,
    },
    Inner {
        left_child: Option<Box<Node>>,
        right_child: Option<Box<Node>>,
        bounds: BoundingBox,
    },
}

impl AccelerationStructure for BoundingVolumeHierarchyRec {
    #[allow(clippy::only_used_in_recursion)]
    fn intersect(&self, ray: &Ray, positions: &[Point3<f32>], triangles: &[Triangle]) -> TraceResult {
        self.stats.count_ray();

        let inv_dir = 1.0 / ray.direction;

        if let Intersects::Yes { .. } = intersects_bounds(&self.root, ray, inv_dir) {
            self.intersect(&self.root, ray, inv_dir, positions, triangles)
        } else {
            TraceResult::Miss
        }
    }

    fn get_statistics(&self) -> StatisticsStore {
        self.stats.get_copy()
    }

    fn bounds(&self) -> BoundingBox {
        self.bounds
    }
}

impl BoundingVolumeHierarchyRec {
    pub fn new(triangle_count: usize, triangle_bounds: &[BoundingBox]) -> Self {
        let mut stats = Statistics::new();
        let mut bounds = BoundingBox::new();

        for &b in triangle_bounds {
            bounds = bounds.union(b);
        }

        BoundingVolumeHierarchyRec {
            root: create_node(triangle_bounds, (0..triangle_count).collect(), 0, &mut stats),
            bounds,
            stats,
        }
    }

    fn intersect(
        &self,
        node_opt: &Option<Box<Node>>,
        ray: &Ray,
        inv_dir: Vector3<f32>,
        positions: &[Point3<f32>],
        triangles: &[Triangle],
    ) -> TraceResult {
        match node_opt {
            Some(node) => match node.as_ref() {
                Node::Inner {
                    left_child,
                    right_child,
                    ..
                } => self.inner_intersect(left_child, right_child, ray, inv_dir, positions, triangles),
                Node::Leaf { triangle_indices, .. } => {
                    intersect_triangles_indexed(triangle_indices, ray, positions, triangles, &self.stats)
                }
            },
            None => TraceResult::Miss,
        }
    }

    fn inner_intersect(
        &self,
        left: &Option<Box<Node>>,
        right: &Option<Box<Node>>,
        ray: &Ray,
        inv_dir: Vector3<f32>,
        positions: &[Point3<f32>],
        triangles: &[Triangle],
    ) -> TraceResult {
        self.stats.count_inner_node_traversal();

        let hit_l_box = intersects_bounds(left, ray, inv_dir);
        let hit_r_box = intersects_bounds(right, ray, inv_dir);

        match (hit_l_box, hit_r_box) {
            (Intersects::No, Intersects::No) => TraceResult::Miss,
            (Intersects::Yes { .. }, Intersects::No) => self.intersect(left, ray, inv_dir, positions, triangles),
            (Intersects::No, Intersects::Yes { .. }) => self.intersect(right, ray, inv_dir, positions, triangles),
            (Intersects::Yes { distance: l_distance }, Intersects::Yes { distance: r_distance }) => {
                if l_distance < r_distance {
                    self.intersect_both_children_hit(left, right, r_distance, ray, inv_dir, positions, triangles)
                } else {
                    self.intersect_both_children_hit(right, left, l_distance, ray, inv_dir, positions, triangles)
                }
            }
        }
    }

    fn intersect_both_children_hit(
        &self,
        first_hit_child: &Option<Box<Node>>,
        second_hit_child: &Option<Box<Node>>,
        dist_to_second_box: f32,
        ray: &Ray,
        inv_dir: Vector3<f32>,
        positions: &[Point3<f32>],
        triangles: &[Triangle],
    ) -> TraceResult {
        // We hit both children's bounds, so check which is hit first
        // If there is an intersection in that one that is closer than the other child's bounds we can stop
        let first_result = self.intersect(first_hit_child, ray, inv_dir, positions, triangles);

        let TraceResult::Hit { t: t_first, .. } = first_result else {
            return self.intersect(second_hit_child, ray, inv_dir, positions, triangles)
        };

        if t_first < dist_to_second_box {
            first_result
        } else {
            let second_result = self.intersect(second_hit_child, ray, inv_dir, positions, triangles);

            if let TraceResult::Hit { t: t_second, .. } = second_result {
                if t_second < t_first {
                    second_result
                } else {
                    first_result
                }
            } else {
                first_result
            }
        }
    }
}

fn intersects_bounds(node_opt: &Option<Box<Node>>, ray: &Ray, inv_dir: Vector3<f32>) -> Intersects {
    match node_opt {
        Some(node) => {
            let node = node.as_ref();
            match node {
                Node::Inner { bounds, .. } => bounds.intersects_ray_not_inlined(ray, &inv_dir),
                Node::Leaf { bounds, .. } => bounds.intersects_ray_not_inlined(ray, &inv_dir),
            }
        }
        None => Intersects::No,
    }
}

fn create_node(
    triangle_bounds: &[BoundingBox],
    triangle_indices: Vec<usize>,
    depth: usize,
    stats: &mut Statistics,
) -> Option<Box<Node>> {
    if triangle_indices.is_empty() {
        return None;
    }

    stats.count_max_depth(depth);

    let bounds = compute_bounding_box_item_indexed(triangle_bounds, &triangle_indices);
    let axes_to_search = [Axis::X, Axis::Y, Axis::Z];
    let relative_traversal_cost = 1.2;

    match surface_area_heuristic_bvh(
        triangle_bounds,
        triangle_indices,
        bounds,
        &axes_to_search,
        relative_traversal_cost,
    ) {
        SurfaceAreaHeuristicResultBvh::MakeLeaf { indices } => {
            stats.count_leaf_node();

            Some(Box::new(Node::Leaf {
                triangle_indices: indices,
                bounds,
            }))
        }
        SurfaceAreaHeuristicResultBvh::MakeInner {
            left_indices,
            right_indices,
        } => {
            let left = create_node(triangle_bounds, left_indices, depth + 1, stats);
            let right = create_node(triangle_bounds, right_indices, depth + 1, stats);

            stats.count_inner_node();

            Some(Box::new(Node::Inner {
                left_child: left,
                right_child: right,
                bounds,
            }))
        }
    }
}
