use std::num::NonZeroU32;

use cgmath::Point3;

use crate::raytracer::{
    aabb::{BoundingBox, Intersects},
    axis::Axis,
    triangle::Triangle,
    Ray,
};

use super::{
    helpers::{compute_bounding_box, compute_bounding_box_item_indexed, intersect_triangles_indexed},
    sah::{surface_area_heuristic_bvh, SurfaceAreaHeuristicResultBvh},
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
    pub fn new(positions: &[Point3<f32>], triangles: &[Triangle], triangle_bounds: &[BoundingBox]) -> Self {
        let mut nodes = Vec::new();
        let stats = Statistics::new();

        let bounds = compute_bounding_box(positions);

        stats.count_inner_node();
        nodes.push(Node::new_inner(bounds));

        let mut stack = vec![(0, (0..triangles.len()).collect(), 1)];

        while let Some((index, triangle_indices, depth)) = stack.pop() {
            let new_left_index = nodes.len();
            let new_right_index = new_left_index + 1;
            assert!(new_left_index > 0 && new_right_index <= u32::MAX as usize);

            let left_is_leaf;
            let right_is_leaf;
            stats.count_max_depth(depth);

            let node = nodes.get_mut(index).unwrap();

            let left_indices: Vec<usize>;
            let right_indices: Vec<usize>;

            let Node::Inner {
                left_child,
                right_child,
                bounds,
            } = node else {
                unreachable!();
            };

            let axes_to_search = [Axis::from_index(depth)];
            let relative_traversal_cost = 1.2;

            match surface_area_heuristic_bvh(
                triangle_bounds,
                triangle_indices,
                *bounds,
                &axes_to_search,
                relative_traversal_cost,
            ) {
                SurfaceAreaHeuristicResultBvh::MakeLeaf { mut indices } => {
                    left_indices = indices.split_off(indices.len() / 2);
                    right_indices = indices;
                    left_is_leaf = true;
                    right_is_leaf = true;
                }
                SurfaceAreaHeuristicResultBvh::MakeInner {
                    left_indices: left,
                    right_indices: right,
                } => {
                    left_indices = left;
                    right_indices = right;
                    left_is_leaf = left_indices.len() < 2;
                    right_is_leaf = right_indices.len() < 2;
                }
            };

            let left_bounds = compute_bounding_box_item_indexed(triangle_bounds, &left_indices);
            let right_bounds = compute_bounding_box_item_indexed(triangle_bounds, &right_indices);

            *left_child = NonZeroU32::new(new_left_index as u32);
            *right_child = NonZeroU32::new(new_right_index as u32);

            if left_is_leaf {
                stats.count_leaf_node();
                nodes.push(Node::new_leaf(left_indices, left_bounds));
            } else {
                stack.push((new_left_index, left_indices, depth + 1));
                stats.count_inner_node();
                nodes.push(Node::new_inner(left_bounds));
            }

            if right_is_leaf {
                stats.count_leaf_node();
                nodes.push(Node::new_leaf(right_indices, right_bounds));
            } else {
                stack.push((new_right_index, right_indices, depth + 1));
                stats.count_inner_node();
                nodes.push(Node::new_inner(right_bounds));
            }
        }

        BoundingVolumeHierarchy { nodes, stats }
    }
}

impl AccelerationStructure for BoundingVolumeHierarchy {
    fn intersect(&self, ray: &Ray, positions: &[Point3<f32>], triangles: &[Triangle]) -> TraceResult {
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
                    if let Intersects::Yes { .. } = bounds.intersects_ray(ray.origin, inv_dir) {
                        self.stats.count_inner_node_traversal();
                        stack.push(left_child.unwrap().get() as usize);
                        stack.push(right_child.unwrap().get() as usize);
                    }
                }
                Node::Leaf {
                    triangle_indices,
                    bounds,
                } => {
                    if !triangle_indices.is_empty() {
                        if let Intersects::Yes { .. } = bounds.intersects_ray(ray.origin, inv_dir) {
                            let r =
                                intersect_triangles_indexed(triangle_indices, ray, positions, triangles, &self.stats);

                            if r.is_closer_than(&result) {
                                result = r;
                            }
                        }
                    }
                }
            }
        }

        result
    }

    fn get_statistics(&self) -> StatisticsStore {
        self.stats.get_copy()
    }
}
