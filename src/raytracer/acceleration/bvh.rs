use crate::mesh::Vertex;
use crate::raytracer::triangle::Triangle;
use crate::raytracer::Ray;

use super::super::axis::Axis;
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
    fn new_leaf(bounds: BoundingBox) -> Node {
        Node::Leaf { 
            triangle_index: -1,
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
    fn new(verts: &Vec<Vertex>, triangles: &Vec<Triangle>) -> Self {
        let mut nodes = Vec::new();

        let bounds = compute_bounding_box(&verts[..]);
        let mut item_indices = Vec::new();

        for i in 0..triangles.len() {
            item_indices.push(i);
        }

        nodes.push(Node::new_inner(bounds));

        let mut stack = Vec::new();
        stack.push((0, item_indices));

        while let Some((index, mut item_indices)) = stack.pop() {
            let new_left_index = nodes.len() as i32;
            let new_right_index = new_left_index + 1;
            let left_is_leaf;
            let right_is_leaf;

            // TODO: Figure out how to make item_indices of size 1 lists into a leaf right away?

            let node = nodes.get_mut(index).unwrap();

            let mut left_indices = Vec::new();
            let mut right_indices = Vec::new();

            let left_bounds;
            let right_bounds;

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
                        return mid_a.partial_cmp(&mid_b).unwrap();
                    });

                    let mid_item_index = item_indices.len() / 2;

                    for i in 0..mid_item_index {
                        left_indices.push(item_indices[i]);
                    }

                    for i in mid_item_index..item_indices.len() {
                        right_indices.push(item_indices[i]);
                    }

                    left_bounds = compute_bounding_box_triangle_indexed(&verts[..], &triangles[..], &left_indices[..]);
                    right_bounds = compute_bounding_box_triangle_indexed(&verts[..],  &triangles[..],&right_indices[..]);

                    left_is_leaf = left_indices.len() < 2;
                    right_is_leaf = right_indices.len() < 2;

                    *left_child = new_left_index;
                    *right_child = new_right_index;

                    if !left_is_leaf {
                        stack.push((new_left_index as usize, left_indices));
                    }

                    if !right_is_leaf {
                        stack.push((new_right_index as usize, right_indices));
                    }
                }
                _ => panic!("Unreachable")
            }


            if left_is_leaf {
                nodes.push(Node::new_leaf(left_bounds));
            } else {
                nodes.push(Node::new_inner(left_bounds));
            }

            if right_is_leaf {
                nodes.push(Node::new_leaf(right_bounds));
            } else {
                nodes.push(Node::new_inner(right_bounds));
            }
        }

        BoundingVolumeHierarchy { nodes: nodes }
    }

    fn intersect(&self, ray: Ray) -> Vec<usize> {
        let result = Vec::new();
        return result
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
        if *i >= triangles.len() {
            println!("{} > {}", *i, triangles.len());
        } else {
            let triangle = &triangles[*i];
            bounds.add(&vertices[triangle.index1 as usize].position);
            bounds.add(&vertices[triangle.index2 as usize].position);
            bounds.add(&vertices[triangle.index3 as usize].position);
        }
    }

    bounds
}