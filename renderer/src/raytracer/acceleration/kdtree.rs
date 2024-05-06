use cgmath::{Point3, Vector3};

use crate::raytracer::{aabb::Intersects, triangle::Triangle, Ray};

use super::{
    super::{aabb::BoundingBox, axis::Axis},
    helpers::{compute_bounding_box, intersect_triangles_indexed},
    sah::{surface_area_heuristic_kd_tree, SurfaceAreaHeuristicResultKdTree},
    statistics::{Statistics, StatisticsStore},
    structure::{AccelerationStructure, TraceResult},
};

pub struct KdTree {
    root: Node,
    bounds: BoundingBox,
    stats: Statistics,
}

enum Node {
    Leaf {
        items: Vec<usize>,
    },
    Inner {
        left_child: Box<Node>,
        right_child: Box<Node>,
        plane: f32,
        axis: Axis,
    },
}

impl AccelerationStructure for KdTree {
    #[allow(clippy::only_used_in_recursion)]
    fn intersect(&self, ray: &Ray, positions: &[Point3<f32>], triangles: &[Triangle]) -> TraceResult {
        self.stats.count_ray();

        let inv_dir = 1.0 / ray.direction;
        self.intersect(&self.root, ray, inv_dir, positions, triangles, self.bounds)
    }

    fn get_statistics(&self) -> StatisticsStore {
        self.stats.get_copy()
    }
}

impl KdTree {
    pub fn new(positions: &[Point3<f32>], triangles: &[Triangle], triangle_bounds: &[BoundingBox]) -> Self {
        let mut stats = Statistics::default();
        let bounds = compute_bounding_box(positions);

        KdTree {
            root: create_node(triangle_bounds, (0..triangles.len()).collect(), 0, &bounds, &mut stats),
            bounds,
            stats,
        }
    }

    fn intersect(
        &self,
        node: &Node,
        ray: &Ray,
        inv_dir: Vector3<f32>,
        positions: &[Point3<f32>],
        triangles: &[Triangle],
        bounds: BoundingBox,
    ) -> TraceResult {
        match node {
            Node::Inner {
                left_child,
                right_child,
                plane,
                axis,
            } => self.inner_intersect(
                left_child,
                right_child,
                bounds,
                *plane,
                *axis,
                ray,
                inv_dir,
                positions,
                triangles,
            ),
            Node::Leaf { items } => intersect_triangles_indexed(items, ray, positions, triangles, &self.stats),
        }
    }

    fn inner_intersect(
        &self,
        left: &Node,
        right: &Node,
        bounds: BoundingBox,
        plane: f32,
        axis: Axis,
        ray: &Ray,
        inv_dir: Vector3<f32>,
        positions: &[Point3<f32>],
        triangles: &[Triangle],
    ) -> TraceResult {
        self.stats.count_inner_node_traversal();

        let mut left_bounds = bounds;
        left_bounds.set_max(axis, plane);
        let mut right_bounds = bounds;
        right_bounds.set_min(axis, plane);

        let hit_l_box = left_bounds.intersects_ray(ray.origin, inv_dir);
        let hit_r_box = right_bounds.intersects_ray(ray.origin, inv_dir);

        match (hit_l_box, hit_r_box) {
            (Intersects::No, Intersects::No) => TraceResult::Miss,
            (Intersects::Yes { .. }, Intersects::No) => {
                self.intersect(left, ray, inv_dir, positions, triangles, left_bounds)
            }
            (Intersects::No, Intersects::Yes { .. }) => {
                self.intersect(right, ray, inv_dir, positions, triangles, right_bounds)
            }
            (Intersects::Yes { distance: l_distance }, Intersects::Yes { distance: r_distance }) => {
                if l_distance < r_distance {
                    self.intersect_both_children_hit(
                        left,
                        left_bounds,
                        right,
                        right_bounds,
                        r_distance,
                        ray,
                        inv_dir,
                        positions,
                        triangles,
                    )
                } else {
                    self.intersect_both_children_hit(
                        right,
                        right_bounds,
                        left,
                        left_bounds,
                        l_distance,
                        ray,
                        inv_dir,
                        positions,
                        triangles,
                    )
                }
            }
        }
    }

    #[inline(always)]
    fn intersect_both_children_hit(
        &self,
        first_hit_child: &Node,
        first_bounds: BoundingBox,
        second_hit_child: &Node,
        second_bounds: BoundingBox,
        dist_to_second_box: f32,
        ray: &Ray,
        inv_dir: Vector3<f32>,
        positions: &[Point3<f32>],
        triangles: &[Triangle],
    ) -> TraceResult {
        // We hit both children's bounds, so check which is hit first
        // If there is an intersection in that one that is closer than the other child's bounds we can stop
        let first_result = self.intersect(first_hit_child, ray, inv_dir, positions, triangles, first_bounds);

        let TraceResult::Hit { t: t_first, .. } = first_result else {
            return self.intersect(second_hit_child, ray, inv_dir, positions, triangles, second_bounds);
        };

        if t_first < dist_to_second_box {
            first_result
        } else {
            let second_result = self.intersect(second_hit_child, ray, inv_dir, positions, triangles, second_bounds);

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

fn create_node(
    triangle_bounds: &[BoundingBox],
    triangle_indices: Vec<usize>,
    depth: usize,
    bounds: &BoundingBox,
    stats: &mut Statistics,
) -> Node {
    stats.count_max_depth(depth);

    match surface_area_heuristic_kd_tree(triangle_bounds, triangle_indices, *bounds) {
        SurfaceAreaHeuristicResultKdTree::MakeLeaf { indices } => {
            stats.count_leaf_node();
            Node::Leaf { items: indices }
        }
        SurfaceAreaHeuristicResultKdTree::MakeInner {
            split_axis,
            split_position,
            left_indices,
            right_indices,
        } => {
            let mut left_bounds = *bounds;
            left_bounds.set_max(split_axis, split_position);
            let mut right_bounds = *bounds;
            right_bounds.set_min(split_axis, split_position);

            let left = create_node(triangle_bounds, left_indices, depth + 1, &left_bounds, stats);
            let right = create_node(triangle_bounds, right_indices, depth + 1, &right_bounds, stats);

            stats.count_inner_node();

            Node::Inner {
                left_child: Box::new(left),
                right_child: Box::new(right),
                plane: split_position,
                axis: split_axis,
            }
        }
    }
}
