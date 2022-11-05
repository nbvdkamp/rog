use cgmath::Vector3;

use crate::{
    mesh::Vertex,
    raytracer::{triangle::Triangle, Ray},
};

use super::{
    super::{aabb::BoundingBox, axis::Axis},
    helpers::{compute_bounding_box, intersect_triangles_indexed},
    sah::{surface_area_heuristic_kd_tree, SurfaceAreaHeuristicResultKdTree},
    statistics::{Statistics, StatisticsStore},
    structure::{AccelerationStructure, TraceResult},
};

pub struct KdTree {
    root: Option<Box<Node>>,
    scene_bounds: BoundingBox,
    stats: Statistics,
}

enum Node {
    Leaf {
        items: Vec<usize>,
    },
    Inner {
        left_child: Option<Box<Node>>,
        right_child: Option<Box<Node>>,
        plane: f32,
        axis: Axis,
    },
}

impl AccelerationStructure for KdTree {
    #[allow(clippy::only_used_in_recursion)]
    fn intersect(&self, ray: &Ray, verts: &[Vertex], triangles: &[Triangle]) -> TraceResult {
        self.stats.count_ray();

        let inv_dir = 1.0 / ray.direction;

        if !self.scene_bounds.intersects_ray(ray, &inv_dir) {
            return TraceResult::Miss;
        }

        self.intersect(&self.root, ray, inv_dir, verts, triangles, self.scene_bounds)
    }

    fn get_name(&self) -> &str {
        "K-d Tree"
    }

    fn get_statistics(&self) -> StatisticsStore {
        self.stats.get_copy()
    }
}

impl KdTree {
    pub fn new(verts: &[Vertex], triangles: &[Triangle], triangle_bounds: &[BoundingBox]) -> Self {
        let mut item_indices = Vec::new();
        let mut stats = Statistics::new();

        for i in 0..triangles.len() {
            item_indices.push(i);
        }

        let scene_bounds = compute_bounding_box(verts);

        KdTree {
            root: create_node(triangle_bounds, item_indices, 0, &scene_bounds, &mut stats),
            scene_bounds,
            stats,
        }
    }

    fn intersect(
        &self,
        node_opt: &Option<Box<Node>>,
        ray: &Ray,
        inv_dir: Vector3<f32>,
        verts: &[Vertex],
        triangles: &[Triangle],
        bounds: BoundingBox,
    ) -> TraceResult {
        match node_opt {
            Some(node) => match node.as_ref() {
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
                    verts,
                    triangles,
                ),
                Node::Leaf { items } => intersect_triangles_indexed(items, ray, verts, triangles, &self.stats),
            },
            None => TraceResult::Miss,
        }
    }

    fn inner_intersect(
        &self,
        left: &Option<Box<Node>>,
        right: &Option<Box<Node>>,
        bounds: BoundingBox,
        plane: f32,
        axis: Axis,
        ray: &Ray,
        inv_dir: Vector3<f32>,
        verts: &[Vertex],
        triangles: &[Triangle],
    ) -> TraceResult {
        self.stats.count_inner_node_traversal();

        let mut left_bounds = bounds;
        left_bounds.set_max(axis, plane);
        let mut right_bounds = bounds;
        right_bounds.set_min(axis, plane);

        let hit_l_box = left_bounds.intersects_ray(ray, &inv_dir);
        let hit_r_box = right_bounds.intersects_ray(ray, &inv_dir);

        if !hit_l_box && !hit_r_box {
            return TraceResult::Miss;
        } else if hit_l_box && !hit_r_box {
            return self.intersect(left, ray, inv_dir, verts, triangles, left_bounds);
        } else if !hit_l_box && hit_r_box {
            return self.intersect(right, ray, inv_dir, verts, triangles, right_bounds);
        }

        // We hit both children's bounds, so check which is hit first
        // If there is an intersection in that one that is closer than the other child's bounds we can stop

        let dist_to_left_box = left_bounds.t_distance_from_ray(ray, &inv_dir);
        let dist_to_right_box = right_bounds.t_distance_from_ray(ray, &inv_dir);

        if dist_to_left_box < dist_to_right_box {
            self.intersect_both_children_hit(
                left,
                left_bounds,
                right,
                right_bounds,
                dist_to_right_box,
                ray,
                inv_dir,
                verts,
                triangles,
            )
        } else {
            self.intersect_both_children_hit(
                right,
                right_bounds,
                left,
                left_bounds,
                dist_to_left_box,
                ray,
                inv_dir,
                verts,
                triangles,
            )
        }
    }

    fn intersect_both_children_hit(
        &self,
        first_hit_child: &Option<Box<Node>>,
        first_bounds: BoundingBox,
        second_hit_child: &Option<Box<Node>>,
        second_bounds: BoundingBox,
        dist_to_second_box: f32,
        ray: &Ray,
        inv_dir: Vector3<f32>,
        verts: &[Vertex],
        triangles: &[Triangle],
    ) -> TraceResult {
        let first_result = self.intersect(first_hit_child, ray, inv_dir, verts, triangles, first_bounds);

        let TraceResult::Hit { t: t_first, .. } = first_result else {
            return self.intersect(second_hit_child, ray, inv_dir, verts, triangles, second_bounds)
        };

        if t_first < dist_to_second_box {
            first_result
        } else {
            let second_result = self.intersect(second_hit_child, ray, inv_dir, verts, triangles, second_bounds);

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
) -> Option<Box<Node>> {
    if triangle_indices.is_empty() {
        return None;
    }

    stats.count_max_depth(depth);

    match surface_area_heuristic_kd_tree(triangle_bounds, triangle_indices, *bounds) {
        SurfaceAreaHeuristicResultKdTree::MakeLeaf { indices } => {
            stats.count_leaf_node();
            Some(Box::new(Node::Leaf { items: indices }))
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

            Some(Box::new(Node::Inner {
                left_child: left,
                right_child: right,
                plane: split_position,
                axis: split_axis,
            }))
        }
    }
}
