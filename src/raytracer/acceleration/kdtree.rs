use crate::mesh::Vertex;
use crate::raytracer::triangle::Triangle;
use crate::raytracer::Ray;

use super::super::axis::Axis;
use super::super::aabb::BoundingBox;
use super::structure::AccelerationStructure;

pub struct KdTree {
    nodes: Vec<Node>,
}

enum Node {
    Leaf {
        items: Vec<usize>
    },
    Inner {
        left_child: i32,
        right_child: i32,
        plane: f32,
        axis: Axis,
    },
}

impl Node {
    fn new_leaf(items: Vec<usize>) -> Node {
        Node::Leaf { items }
    }

    fn new_inner() -> Node {
        Node::Inner { 
            left_child: -1,
            right_child: -1,
            plane: 0.0,
            axis: Axis::X,
        }
    }
}

impl AccelerationStructure for KdTree {
    fn new(verts: &[Vertex], triangles: &[Triangle]) -> Self {
        let mut nodes = Vec::new();

        let bounds = compute_bounding_box(&verts);
        let mut item_indices = Vec::new();

        for i in 0..triangles.len() {
            item_indices.push(i);
        }

        nodes.push(Node::new_inner());

        let mut stack = Vec::new();
        stack.push((0, bounds, item_indices));

        while let Some((index, bounds, item_indices)) = stack.pop() {
            let new_left_index ;
            let new_right_index;
            let left_is_empty;
            let right_is_empty;
            let left_is_leaf;
            let right_is_leaf;

            let mut current_index = nodes.len() as i32;

            let node = nodes.get_mut(index).unwrap();

            let mut left_indices = Vec::new();
            let mut right_indices = Vec::new();

            match node {
                Node::Inner { left_child, right_child, plane, axis } => {
                    let (split_axis, mid) = bounds.find_split_plane();
                    *axis = split_axis;
                    *plane = mid;



                    for index in item_indices {
                        let v1 = &verts[triangles[index].index1 as usize].position[axis.index()];
                        let v2 = &verts[triangles[index].index2 as usize].position[axis.index()];
                        let v3 = &verts[triangles[index].index3 as usize].position[axis.index()];

                        if *v1 <= mid || *v2 <= mid || *v3 <= mid {
                            left_indices.push(index);
                        }
                        if *v1 >= mid || *v2 >= mid || *v3 >= mid {
                            right_indices.push(index);
                        }
                    }

                    let max_leaf_items = 10;

                    left_is_empty = left_indices.len() == 0;
                    right_is_empty = right_indices.len() == 0;
                    left_is_leaf = left_indices.len() <= max_leaf_items;
                    right_is_leaf = right_indices.len() <= max_leaf_items;


                    if !left_is_empty {
                        new_left_index = current_index;
                        current_index += 1;
                    } else {
                        new_left_index = -1;
                    }

                    if !right_is_empty {
                        new_right_index = current_index;
                    } else {
                        new_right_index = -1;
                    }

                    *left_child = new_left_index;
                    *right_child = new_right_index;

                    if !left_is_empty && !left_is_leaf {
                        let mut left_bounds = bounds;
                        left_bounds.set_max(axis, mid);
                        stack.push((new_left_index as usize, left_bounds, left_indices.clone()));
                    }

                    if !left_is_empty && !right_is_leaf {
                        let mut right_bounds = bounds;
                        right_bounds.set_min(axis, mid);
                        stack.push((new_right_index as usize, right_bounds, right_indices.clone()));
                    }
                }
                _ => panic!("Unreachable")
            }

            if !left_is_empty {
                if left_is_leaf {
                    nodes.push(Node::new_leaf(left_indices));
                } else {
                    nodes.push(Node::new_inner());
                }
            }

            if !right_is_empty {
                if right_is_leaf {
                    nodes.push(Node::new_leaf(right_indices));
                } else {
                    nodes.push(Node::new_inner());
                }
            }
        }

        KdTree { nodes }
    }

    fn intersect(&self, ray: &Ray) -> Vec<usize> {
        let result = Vec::new();
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