use crate::{
    mesh::Vertex,
    raytracer::{axis::Axis, triangle::Triangle, IntersectionResult, Ray},
};

use super::{
    super::aabb::BoundingBox,
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
        left_child: i32,
        right_child: i32,
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
            left_child: -1,
            right_child: -1,
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

        while let Some((index, mut item_indices, depth, split_axis)) = stack.pop() {
            let new_left_index = nodes.len() as i32;
            let new_right_index = new_left_index + 1;
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
                let axis_index = split_axis.index();

                let mut centroid_bounds = BoundingBox::new();

                item_indices.iter().for_each(|index| {
                    centroid_bounds.add(triangle_bounds[*index].center());
                });

                const BUCKET_COUNT: usize = 12;

                #[derive(Clone, Copy)]
                struct Bucket {
                    count: u32,
                    bounds: BoundingBox,
                }

                let mut buckets = [Bucket {
                    count: 0,
                    bounds: BoundingBox::new(),
                }; BUCKET_COUNT];

                let bucket_index = |center| {
                    let x = (center - centroid_bounds.min[axis_index])
                        / (centroid_bounds.max[axis_index] - centroid_bounds.min[axis_index]);
                    ((BUCKET_COUNT as f32 * x) as usize).min(BUCKET_COUNT - 1)
                };

                item_indices.iter().for_each(|index| {
                    let bounds = triangle_bounds[*index];

                    let center = bounds.center()[axis_index];
                    let bucket = &mut buckets[bucket_index(center)];
                    bucket.count += 1;
                    bucket.bounds = bucket.bounds.union(bounds);
                });

                let mut costs = [0.0; BUCKET_COUNT - 1];

                for i in 0..BUCKET_COUNT - 1 {
                    let mut b0 = BoundingBox::new();
                    let mut b1 = BoundingBox::new();

                    let mut count0 = 0;
                    let mut count1 = 0;

                    for j in 0..=i {
                        b0 = b0.union(buckets[j].bounds);
                        count0 += buckets[j].count;
                    }
                    for j in i + 1..BUCKET_COUNT {
                        b1 = b1.union(buckets[j].bounds);
                        count1 += buckets[j].count;
                    }

                    const RELATIVE_TRAVERSAL_COST: f32 = 1.2;
                    let approx_children_cost =
                        (count0 as f32 * b0.surface_area() + count1 as f32 * b1.surface_area()) / bounds.surface_area();
                    costs[i] = RELATIVE_TRAVERSAL_COST + approx_children_cost;
                }

                let mut min_cost = costs[0];
                let mut min_index = 0;

                for i in 1..BUCKET_COUNT - 1 {
                    if costs[i] < min_cost {
                        min_cost = costs[i];
                        min_index = i;
                    }
                }

                const MAX_TRIS_IN_LEAF: usize = 255;

                if item_indices.len() > MAX_TRIS_IN_LEAF || min_cost < item_indices.len() as f32 {
                    (left_indices, right_indices) = item_indices
                        .into_iter()
                        .partition(|i| bucket_index(triangle_bounds[*i].center()[axis_index]) <= min_index);
                    left_is_leaf = left_indices.len() < 2;
                    right_is_leaf = right_indices.len() < 2;
                } else {
                    left_indices = item_indices.split_off(item_indices.len() / 2);
                    right_indices = item_indices;

                    left_is_leaf = true;
                    right_is_leaf = true;
                }

                left_bounds = compute_bounding_box_triangle_indexed(triangle_bounds, &left_indices);
                right_bounds = compute_bounding_box_triangle_indexed(triangle_bounds, &right_indices);

                *left_child = new_left_index;
                *right_child = new_right_index;
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
        let mut min_distance = f32::MAX;
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
                        stack.push(*left_child as usize);
                        stack.push(*right_child as usize);
                    }
                }
                Node::Leaf {
                    triangle_indices,
                    bounds,
                } => {
                    if !triangle_indices.is_empty() && bounds.intersects_ray(ray, &inv_dir) {
                        for i in triangle_indices {
                            let triangle = &triangles[*i];
                            let p1 = &verts[triangle.index1 as usize];
                            let p2 = &verts[triangle.index2 as usize];
                            let p3 = &verts[triangle.index3 as usize];

                            self.stats.count_intersection_test();

                            if let IntersectionResult::Hit { t, u, v } =
                                ray.intersect_triangle(p1.position, p2.position, p3.position)
                            {
                                self.stats.count_intersection_hit();

                                if t < min_distance {
                                    min_distance = t;
                                    result = TraceResult::Hit {
                                        triangle_index: *i as u32,
                                        t,
                                        u,
                                        v,
                                    };
                                }
                            }
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

fn compute_bounding_box(vertices: &[Vertex]) -> BoundingBox {
    let mut bounds = BoundingBox::new();

    for vertex in vertices {
        bounds.add(vertex.position);
    }

    bounds
}

fn compute_bounding_box_triangle_indexed(triangle_bounds: &[BoundingBox], triangle_indices: &[usize]) -> BoundingBox {
    let mut bounds = BoundingBox::new();

    for i in triangle_indices {
        bounds = bounds.union(triangle_bounds[*i]);
    }

    bounds
}
