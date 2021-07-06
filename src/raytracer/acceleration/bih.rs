use crate::mesh::Vertex;
use crate::raytracer::triangle::Triangle;
use crate::raytracer::Ray;

use super::super::axis::Axis;
use super::super::aabb::BoundingBox;

pub struct BoundingIntervalHierarchy {
    nodes: Vec<Node>,
}

enum Node {
    Leaf {
        items: i32
    },
    Inner {
        left_child: i32,
        right_child: i32,
        clip_left: f32,
        clip_right: f32,
        axis: Axis,
    },
}

impl Node {
    fn new_leaf() -> Node {
        Node::Leaf { items: -1 }
    }

    fn new_inner() -> Node {
        Node::Inner { 
            left_child: -1,
            right_child: -1,
            clip_left: f32::MIN,
            clip_right: f32::MAX,
            axis: Axis::X,
        }
    }
}

impl BoundingIntervalHierarchy {
    pub fn new(verts: &Vec<Vertex>, triangles: &Vec<Triangle>) -> Self {
        let mut nodes = Vec::new();

        let bounds = compute_bounding_box(&verts[..]);
        let mut item_indices = Vec::new();

        for i in 0..triangles.len() {
            item_indices.push(i);
        }

        nodes.push(Node::new_inner());

        let mut stack = Vec::new();
        stack.push((0, bounds, item_indices));

        while let Some((index, bounds, item_indices)) = stack.pop() {
            let new_left_index = nodes.len() as i32;
            let new_right_index = new_left_index + 1;
            let left_is_leaf;
            let right_is_leaf;

            let node = nodes.get_mut(index).unwrap();

            match node {
                Node::Inner { left_child, right_child, clip_left, clip_right, axis } => {
                    let (split_axis, mid) = bounds.find_split_plane();
                    *axis = split_axis;


                    let mut left_indices = Vec::new();
                    let mut right_indices = Vec::new();

                    for index in item_indices {
                        let v1 = &verts[triangles[index].index1 as usize].position[axis.index()];
                        let v2 = &verts[triangles[index].index2 as usize].position[axis.index()];
                        let v3 = &verts[triangles[index].index3 as usize].position[axis.index()];

                        // Simplistic heuristic, probably better to divide by surface on each side
                        if *v1 >= mid && *v2 >= mid {
                            right_indices.push(index);

                            let min = f32::min(*v1, f32::min(*v2, *v3));
                            *clip_right = f32::min(*clip_right, min);
                        } else {
                            left_indices.push(index);

                            let max = f32::max(*v1, f32::max(*v2, *v3));
                            *clip_left = f32::max(*clip_left, max);
                        }
                    }

                    left_is_leaf = left_indices.len() < 2;
                    right_is_leaf = right_indices.len() < 2;

                    *left_child = new_left_index;
                    *right_child = new_right_index;

                    // TODO test if this can be left out for empty leaves

                    if !left_is_leaf {
                        let mut left_bounds = bounds.clone();
                        left_bounds.set_max(axis, *clip_left);
                        stack.push((new_left_index as usize, left_bounds, left_indices));
                    }

                    if !right_is_leaf {
                        let mut right_bounds = bounds.clone();
                        right_bounds.set_min(axis, *clip_right);
                        stack.push((new_right_index as usize, right_bounds, right_indices));
                    }
                }
                _ => panic!("Unreachable")
            }

            if left_is_leaf {
                nodes.push(Node::new_leaf());
            } else {
                nodes.push(Node::new_inner());
            }

            if right_is_leaf {
                nodes.push(Node::new_leaf());
            } else {
                nodes.push(Node::new_inner());
            }
        }

        nodes.push(Node::Leaf{ items: 10 });

        BoundingIntervalHierarchy { nodes: nodes }
    }

    pub fn Intersect(ray: Ray) -> Vec<usize> {
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