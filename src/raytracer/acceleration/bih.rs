use cgmath::Point3;

use crate::mesh::Vertex;
use crate::raytracer::triangle::Triangle;

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

enum Axis {
    X, Y, Z
}

impl Axis {
    fn index(&self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
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
}

#[derive(Debug, Copy, Clone)]
struct BoundingBox {
    x_max: f32,
    x_min: f32,
    y_max: f32,
    y_min: f32,
    z_max: f32,
    z_min: f32,
}

impl BoundingBox {
    fn new() -> Self {
        BoundingBox {
            x_max: f32::MIN,
            x_min: f32::MAX,
            y_max: f32::MIN,
            y_min: f32::MAX,
            z_max: f32::MIN,
            z_min: f32::MAX,
        }
    }

    fn add(&mut self, p: &Point3<f32>) {
        self.x_max = f32::max(self.x_max, p.x);
        self.x_min = f32::min(self.x_min, p.x);
        self.y_max = f32::max(self.y_max, p.y);
        self.y_min = f32::min(self.y_min, p.y);
        self.z_max = f32::max(self.z_max, p.z);
        self.z_min = f32::min(self.z_min, p.z);
    }

    fn find_split_plane(&self) -> (Axis, f32) {
        let x_size = self.x_max - self.x_min;
        let y_size = self.y_max - self.y_min;
        let z_size = self.z_max - self.z_min;

        if x_size >= y_size && x_size >= z_size {
            (Axis::X, self.x_min + x_size / 2.0)
        } else if y_size >= z_size {
            (Axis::Y, self.y_min + y_size / 2.0)
        } else {
            (Axis::Z, self.z_min + z_size / 2.0)
        }
    }

    fn set_min(&mut self, axis: &Axis, value: f32) {
        match axis {
            Axis::X => self.x_min = value,
            Axis::Y => self.y_min = value,
            Axis::Z => self.z_min = value,
        }
    }

    fn set_max(&mut self, axis: &Axis, value: f32) {
        match axis {
            Axis::X => self.x_max = value,
            Axis::Y => self.y_max = value,
            Axis::Z => self.z_max = value,
        }
    }
}

fn compute_bounding_box(vertices: &[Vertex]) -> BoundingBox {
    let mut bounds = BoundingBox::new();

    for vertex in vertices {
        bounds.add(&vertex.position);
    }

    bounds
}