use crate::mesh::Vertex;
use crate::raytracer::triangle::Triangle;
use crate::raytracer::{Ray, IntersectionResult};

use super::super::aabb::BoundingBox;
use super::statistics::{Statistics, StatisticsStore};
use super::structure::{AccelerationStructure, TraceResult};

pub struct BoundingVolumeHierarchy {
    nodes: Vec<Node>,
    stats: Statistics,
}

enum Node {
    Leaf {
        triangle_index: i32,
        bounds: BoundingBox,
    },
    Inner {
        left_child: i32,
        right_child: i32,
        bounds: BoundingBox,
    },
}

impl Node {
    fn new_leaf(triangle_index: i32, bounds: BoundingBox) -> Node {
        Node::Leaf { 
            triangle_index,
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
    pub fn new(verts: &[Vertex], triangles: &[Triangle]) -> Self {
        let mut nodes = Vec::new();

        let bounds = compute_bounding_box(verts);
        let mut item_indices = Vec::new();

        for i in 0..triangles.len() {
            item_indices.push(i);
        }

        nodes.push(Node::new_inner(bounds));

        let mut stack = vec![(0, item_indices)];

        while let Some((index, mut item_indices)) = stack.pop() {
            let new_left_index = nodes.len() as i32;
            let new_right_index = new_left_index + 1;
            let left_is_leaf;
            let right_is_leaf;

            let node = nodes.get_mut(index).unwrap();

            let mut left_indices = Vec::new();
            let mut right_indices = Vec::new();

            let left_bounds;
            let right_bounds;

            let mut left_triangle_index = -1;
            let mut right_triangle_index = -1;

            match node {
                Node::Inner { left_child, right_child, bounds } => {
                    let (split_axis, _) = bounds.find_split_plane();
                    let axis_index = split_axis.index();


                    item_indices.sort_by(|index_a, index_b| {
                        let triangle_a = &triangles[*index_a];
                        let triangle_b = &triangles[*index_b];
                        let mid_a = (verts[triangle_a.index1 as usize].position[axis_index] +
                                        verts[triangle_a.index2 as usize].position[axis_index] + 
                                        verts[triangle_a.index3 as usize].position[axis_index]) / 3.0;
                        let mid_b = (verts[triangle_b.index1 as usize].position[axis_index] +
                                        verts[triangle_b.index2 as usize].position[axis_index] + 
                                        verts[triangle_b.index3 as usize].position[axis_index]) / 3.0;

                        mid_a.partial_cmp(&mid_b).unwrap()
                    });

                    let mid_item_index = item_indices.len() / 2;

                    for item in item_indices.iter().take(mid_item_index) {
                        left_indices.push(*item);
                    }

                    for item in item_indices.iter().skip(mid_item_index) {
                        right_indices.push(*item);
                    }

                    left_bounds = compute_bounding_box_triangle_indexed(verts, triangles, &left_indices);
                    right_bounds = compute_bounding_box_triangle_indexed(verts,  triangles,&right_indices);

                    left_is_leaf = left_indices.len() < 2;
                    right_is_leaf = right_indices.len() < 2;

                    *left_child = new_left_index;
                    *right_child = new_right_index;

                    if !left_is_leaf {
                        stack.push((new_left_index as usize, left_indices));
                    } else if !left_indices.is_empty()  {
                        left_triangle_index = left_indices[0] as i32;
                    }

                    if !right_is_leaf {
                        stack.push((new_right_index as usize, right_indices));
                    } else if !right_indices.is_empty() {
                        right_triangle_index = right_indices[0] as i32;
                    }
                }
                _ => unreachable!()
            }


            if left_is_leaf {
                nodes.push(Node::new_leaf(left_triangle_index, left_bounds));
            } else {
                nodes.push(Node::new_inner(left_bounds));
            }

            if right_is_leaf {
                nodes.push(Node::new_leaf(right_triangle_index, right_bounds));
            } else {
                nodes.push(Node::new_inner(right_bounds));
            }
        }

        BoundingVolumeHierarchy { nodes, stats: Statistics::new() }
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
                Node::Inner { left_child, right_child, bounds } => {
                    if bounds.intersects_ray(ray, &inv_dir) {
                        self.stats.count_inner_node_traversal();
                        stack.push(*left_child as usize);
                        stack.push(*right_child as usize);
                    }
                }
                Node::Leaf { triangle_index, bounds } => {
                    if *triangle_index > -1 && bounds.intersects_ray(ray,  &inv_dir) {
                        let triangle = &triangles[*triangle_index as usize];
                        let p1 = &verts[triangle.index1 as usize];
                        let p2 = &verts[triangle.index2 as usize];
                        let p3 = &verts[triangle.index3 as usize];

                        self.stats.count_intersection_test();

                        if let IntersectionResult::Hit{ t, u, v } = ray.intersect_triangle(p1.position, p2.position, p3.position) {
                            self.stats.count_intersection_hit();

                            if t < min_distance {
                                min_distance = t;
                                result = TraceResult::Hit{ triangle_index: *triangle_index, t, u, v };
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
        bounds.add(&vertex.position);
    }

    bounds
}

fn compute_bounding_box_triangle_indexed(vertices: &[Vertex], triangles: &[Triangle], triangle_indices: &[usize]) -> BoundingBox {
    let mut bounds = BoundingBox::new();

    for i in triangle_indices {
        let triangle = &triangles[*i];
        bounds.add(&vertices[triangle.index1 as usize].position);
        bounds.add(&vertices[triangle.index2 as usize].position);
        bounds.add(&vertices[triangle.index3 as usize].position);
    }

    bounds
}