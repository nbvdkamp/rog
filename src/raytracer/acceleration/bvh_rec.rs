use crate::mesh::Vertex;
use crate::raytracer::triangle::Triangle;
use crate::raytracer::Ray;

use cgmath::Vector3;

use super::super::aabb::BoundingBox;
use super::structure::AccelerationStructure;

pub struct BoundingVolumeHierarchyRec {
    root: Option<Box<Node>>,
}

enum Node {
    Leaf {
        triangle_index: i32,
        bounds: BoundingBox,
    },
    Inner {
        left_child: Option<Box<Node>>,
        right_child: Option<Box<Node>>,
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
            left_child: None,
            right_child: None,
            bounds,
        }
    }
}

impl AccelerationStructure for BoundingVolumeHierarchyRec {
    fn new(verts: &[Vertex], triangles: &[Triangle]) -> Self {
        let mut item_indices = Vec::new();

        for i in 0..triangles.len() {
            item_indices.push(i);
        }

        let root = create_node(verts, triangles, &mut item_indices);

        BoundingVolumeHierarchyRec { root }
    }

    fn intersect(&self, ray: &Ray) -> Vec<usize> {
        let inv_dir = 1.0 / ray.direction;
        intersect(&self.root, ray, inv_dir)
    }
}

fn intersect(node_opt: &Option<Box<Node>>, ray: &Ray, inv_dir: Vector3<f32>) -> Vec<usize> {
    match node_opt {
        Some(node) => {
            let node = node.as_ref();
            match node {
                Node::Inner { left_child, right_child, bounds} => {
                    if bounds.intersects(ray, &inv_dir) {
                        let mut l = intersect(left_child, ray, inv_dir);
                        let mut r = intersect(right_child, ray, inv_dir);
                        l.append(&mut r);
                        l
                    } else {
                        vec![]
                    }
                }
                Node::Leaf { triangle_index, bounds } => {
                    if bounds.intersects(ray, &inv_dir) {
                        vec![*triangle_index as usize]
                    } else {
                        vec![]
                    }
                }
            }
        }
        None => vec![]
    }
}

fn create_node(verts: &[Vertex], triangles: &[Triangle], triangle_indices: &mut Vec<usize>) -> Option<Box<Node>> {
    if triangle_indices.len() == 0 {
        return None
    }

    let bounds = compute_bounding_box_triangle_indexed(verts, triangles, triangle_indices);

    if triangle_indices.len() == 1 {
        return Some(Box::new(Node::new_leaf(triangle_indices[0] as i32, bounds)));
    }
    
    let mut node = Node::new_inner(bounds);

    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();

    let (split_axis, _) = bounds.find_split_plane();
    let axis_index = split_axis.index();


    triangle_indices.sort_by(|index_a, index_b| {
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

    let mid_item_index = triangle_indices.len() / 2;

    for item in triangle_indices.iter().take(mid_item_index) {
        left_indices.push(*item);
    }

    for item in triangle_indices.iter().skip(mid_item_index) {
        right_indices.push(*item);
    }

    let left = create_node(verts, triangles, &mut left_indices);
    let right = create_node(verts, triangles, &mut right_indices);

    Some(Box::new(Node::Inner {
        left_child: left,
        right_child: right,
        bounds
    }))
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