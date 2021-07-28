use crate::mesh::Vertex;
use crate::raytracer::triangle::Triangle;
use crate::raytracer::Ray;

use super::super::aabb::BoundingBox;
use super::structure::AccelerationStructure;

pub struct BoundingVolumeHierarchy {
    nodes: Vec<Node>,
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

impl AccelerationStructure for BoundingVolumeHierarchy {
    fn new(verts: &[Vertex], triangles: &[Triangle]) -> Self {
        let mut nodes = Vec::new();

        let bounds = compute_bounding_box(&verts);
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

                    left_bounds = compute_bounding_box_triangle_indexed(&verts, &triangles, &left_indices);
                    right_bounds = compute_bounding_box_triangle_indexed(&verts,  &triangles,&right_indices);

                    left_is_leaf = left_indices.len() < 2;
                    right_is_leaf = right_indices.len() < 2;

                    *left_child = new_left_index;
                    *right_child = new_right_index;

                    if !left_is_leaf {
                        stack.push((new_left_index as usize, left_indices));
                    } else {
                        left_triangle_index = left_indices[0] as i32;
                    }

                    if !right_is_leaf {
                        stack.push((new_right_index as usize, right_indices));
                    } else {
                        right_triangle_index = right_indices[0] as i32;
                    }
                }
                _ => panic!("Unreachable")
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

        BoundingVolumeHierarchy { nodes }
    }

    fn intersect(&self, ray: &Ray) -> Vec<usize> {
        let mut result = Vec::new();
        let inv_dir = 1.0 / ray.direction;

        // Replacing this with a vec! macro degrades performance somehow??
        let mut stack = Vec::new();
        stack.push(0);

        while let Some(i) = stack.pop() {
            match &self.nodes[i] {
                Node::Inner { left_child, right_child, bounds } => {
                    if bounds.intersects(ray, &inv_dir) {
                        stack.push(*left_child as usize);
                        stack.push(*right_child as usize);
                    }
                }
                Node::Leaf { triangle_index, bounds } => {
                    if bounds.intersects(ray,  &inv_dir) {
                        result.push(*triangle_index as usize);
                    }
                }
            }
        }

        result
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