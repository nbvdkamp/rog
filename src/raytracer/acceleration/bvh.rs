use std::num::NonZeroU32;

use crate::{
    mesh::Vertex,
    raytracer::{aabb::BoundingBox, axis::Axis, triangle::Triangle, Ray},
};

use super::{
    helpers::{compute_bounding_box, compute_bounding_box_triangle_indexed, intersect_triangles_indexed},
    sah::{surface_area_heuristic, SurfaceAreaHeuristicResult},
    statistics::{Statistics, StatisticsStore},
    structure::{AccelerationStructure, TraceResult},
};

pub struct BoundingVolumeHierarchy {
    nodes: Vec<Node>,
    stats: Statistics,
}

enum Node {
    Leaf {
        triangle_indices: Vec<usize>,
        bounds: BoundingBox,
    },
    Inner {
        left_child: Option<NonZeroU32>,
        right_child: Option<NonZeroU32>,
        bounds: BoundingBox,
    },
}

impl Node {
    fn new_leaf(triangle_indices: Vec<usize>, bounds: BoundingBox) -> Node {
        Node::Leaf {
            triangle_indices,
            bounds,
        }
    }

    fn new_inner(bounds: BoundingBox) -> Node {
        Node::Inner {
            left_child: None,
            right_child: None,
            bounds,
        }
    }
}

impl BoundingVolumeHierarchy {
    pub fn new(verts: &[Vertex], triangles: &[Triangle], triangle_bounds: &[BoundingBox]) -> Self {
        let mut nodes = Vec::new();
        let stats = Statistics::new();

        let bounds = compute_bounding_box(verts);
        let mut item_indices = Vec::new();

        for i in 0..triangles.len() {
            item_indices.push(i);
        }

        stats.count_inner_node();
        nodes.push(Node::new_inner(bounds));

        let mut stack = vec![(0, item_indices, 1, Axis::X)];

        while let Some((index, item_indices, depth, split_axis)) = stack.pop() {
            let new_left_index = nodes.len();
            let new_right_index = new_left_index + 1;
            assert!(new_left_index > 0 && new_right_index <= u32::MAX as usize);

            let left_is_leaf;
            let right_is_leaf;
            stats.count_max_depth(depth);

            let node = nodes.get_mut(index).unwrap();

            let left_indices: Vec<usize>;
            let right_indices: Vec<usize>;

            let left_bounds;
            let right_bounds;

            if let Node::Inner {
                left_child,
                right_child,
                bounds,
            } = node
            {
                match surface_area_heuristic(triangle_bounds, item_indices, split_axis, *bounds) {
                    SurfaceAreaHeuristicResult::MakeLeaf { mut indices } => {
                        left_indices = indices.split_off(indices.len() / 2);
                        right_indices = indices;
                        left_is_leaf = true;
                        right_is_leaf = true;
                    }
                    SurfaceAreaHeuristicResult::MakeInner {
                        left_indices: left,
                        right_indices: right,
                    } => {
                        left_indices = left;
                        right_indices = right;
                        left_is_leaf = left_indices.len() < 2;
                        right_is_leaf = right_indices.len() < 2;
                    }
                };

                left_bounds = compute_bounding_box_triangle_indexed(triangle_bounds, &left_indices);
                right_bounds = compute_bounding_box_triangle_indexed(triangle_bounds, &right_indices);

                *left_child = NonZeroU32::new(new_left_index as u32);
                *right_child = NonZeroU32::new(new_right_index as u32);
            } else {
                unreachable!();
            }

            if left_is_leaf {
                stats.count_leaf_node();
                nodes.push(Node::new_leaf(left_indices, left_bounds));
            } else {
                stack.push((new_left_index as usize, left_indices, depth + 1, split_axis.next()));
                stats.count_inner_node();
                nodes.push(Node::new_inner(left_bounds));
            }

            if right_is_leaf {
                stats.count_leaf_node();
                nodes.push(Node::new_leaf(right_indices, right_bounds));
            } else {
                stack.push((new_right_index as usize, right_indices, depth + 1, split_axis.next()));
                stats.count_inner_node();
                nodes.push(Node::new_inner(right_bounds));
            }
        }

        BoundingVolumeHierarchy { nodes, stats }
    }
}

impl AccelerationStructure for BoundingVolumeHierarchy {
    fn intersect(&self, ray: &Ray, verts: &[Vertex], triangles: &[Triangle]) -> TraceResult {
        self.stats.count_ray();

        let mut result = TraceResult::Miss;
        let inv_dir = 1.0 / ray.direction;

        // Replacing this with a vec! macro degrades performance somehow??
        let mut stack = Vec::new();
        stack.reserve(f32::log2(self.nodes.len() as f32) as usize);
        stack.push(0);

        while let Some(i) = stack.pop() {
            match &self.nodes[i] {
                Node::Inner {
                    left_child,
                    right_child,
                    bounds,
                } => {
                    if bounds.intersects_ray(ray, &inv_dir) {
                        self.stats.count_inner_node_traversal();
                        stack.push(left_child.unwrap().get() as usize);
                        stack.push(right_child.unwrap().get() as usize);
                    }
                }
                Node::Leaf {
                    triangle_indices,
                    bounds,
                } => {
                    if !triangle_indices.is_empty() && bounds.intersects_ray(ray, &inv_dir) {
                        let r = intersect_triangles_indexed(triangle_indices, ray, verts, triangles, &self.stats);

                        if r.is_closer_than(&result) {
                            result = r;
                        }
                    }
                }
            }
        }

        result
    }

    fn get_name(&self) -> &str {
        "BVH (iterative)"
    }

    fn get_statistics(&self) -> StatisticsStore {
        self.stats.get_copy()
    }
}
