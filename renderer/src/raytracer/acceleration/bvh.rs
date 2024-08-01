use std::{collections::VecDeque, num::NonZeroU32};

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
        child_index: Option<NonZeroU32>,
        bounds: BoundingBox,
    },
}

impl Node {
    fn bounds(&self) -> &BoundingBox {
        match self {
            Node::Leaf { bounds, .. } => bounds,
            Node::Inner { bounds, .. } => bounds,
        }
    }
}

impl BoundingVolumeHierarchy {
    pub fn new(positions: &[Point3<f32>], triangles: &[Triangle], triangle_bounds: &[BoundingBox]) -> Self {
        let mut nodes = Vec::new();
        let stats = Statistics::default();

        let bounds = compute_bounding_box(positions);

        let mut deq = VecDeque::from([((0..triangles.len()).collect(), 1, bounds)]);

        while let Some((triangle_indices, depth, bounds)) = deq.pop_front() {
            stats.count_max_depth(depth);

            let axes_to_search = [Axis::from_index(depth)];
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
                    nodes.push(Node::Leaf {
                        triangle_indices: indices,
                        bounds,
                    })
                }
                SurfaceAreaHeuristicResultBvh::MakeInner {
                    left_indices,
                    right_indices,
                } => {
                    let child_index = nodes.len() + 1 + deq.len();
                    assert!(child_index > 0 && child_index + 1 <= u32::MAX as usize);

                    stats.count_inner_node();
                    nodes.push(Node::Inner {
                        child_index: NonZeroU32::new(child_index as u32),
                        bounds,
                    });

                    let left_bounds = compute_bounding_box_item_indexed(triangle_bounds, &left_indices);
                    let right_bounds = compute_bounding_box_item_indexed(triangle_bounds, &right_indices);

                    deq.push_back((left_indices, depth + 1, left_bounds));
                    deq.push_back((right_indices, depth + 1, right_bounds));
                }
            };
        }

        BoundingVolumeHierarchy { nodes, stats }
    }
}

impl AccelerationStructure for BoundingVolumeHierarchy {
    fn intersect(&self, ray: &Ray, positions: &[Point3<f32>], triangles: &[Triangle]) -> TraceResult {
        self.stats.count_ray();

        let mut result = TraceResult::Miss;
        let inv_dir = 1.0 / ray.direction;

        let mut stack = Vec::with_capacity(f32::log2(self.nodes.len() as f32) as usize);
        stack.push(0);

        while let Some(i) = stack.pop() {
            let node = &self.nodes[i];

            if let Intersects::Yes { distance } = node.bounds().intersects_ray(ray.origin, inv_dir) {
                if !result.is_farther_than(distance) {
                    continue;
                }
            } else {
                continue;
            }

            match node {
                Node::Inner { child_index, .. } => {
                    self.stats.count_inner_node_traversal();
                    let i = child_index.unwrap().get() as usize;
                    stack.push(i);
                    stack.push(i + 1);
                }
                Node::Leaf { triangle_indices, .. } => {
                    let r = intersect_triangles_indexed(triangle_indices, ray, positions, triangles, &self.stats);

                    if r.is_closer_than(&result) {
                        result = r;
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
