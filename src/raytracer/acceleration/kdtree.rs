use rand::{seq::IteratorRandom, thread_rng};

use crate::mesh::Vertex;
use crate::raytracer::triangle::Triangle;
use crate::raytracer::Ray;

use super::super::axis::Axis;
use super::super::aabb::BoundingBox;
use super::structure::{AccelerationStructure, TraceResult};

pub struct KdTree {
    root: Option<Box<Node>>
}

enum Node {
    Leaf {
        items: Vec<usize>
    },
    Inner {
        left_child: Option<Box<Node>>,
        right_child: Option<Box<Node>>,
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
            left_child: None,
            right_child: None,
            plane: 0.0,
            axis: Axis::X,
        }
    }
}

impl KdTree {
    pub fn new(verts: &[Vertex], triangles: &[Triangle]) -> Self {
        let mut item_indices = Vec::new();

        for i in 0..triangles.len() {
            item_indices.push(i);
        }

        KdTree { root: create_node(verts, triangles, item_indices, 0) }
    }
}

impl AccelerationStructure for KdTree {
    fn intersect(&self, ray: &Ray, verts: &[Vertex], triangles: &[Triangle]) -> TraceResult {
        TraceResult::Miss
    }

    fn get_name(&self) -> &str {
        "K-d Tree"
    }
}

fn create_node(verts: &[Vertex], triangles: &[Triangle], triangle_indices: Vec<usize>, depth: i32) -> Option<Box<Node>> {
    if triangle_indices.len() == 0 {
        return None
    }

    let early_term_leaf_size = 10;
    let max_depth = 10;

    if triangle_indices.len() <= early_term_leaf_size || depth > max_depth {
        return Some(Box::new(Node::Leaf {
            items: triangle_indices,
        }));
    }
    
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();

    let axis_index = (depth % 3) as usize;
    let axis = Axis::from_index(axis_index);

    let samples = triangle_indices.iter().choose_multiple(&mut thread_rng(), 10);

    let mut sample_centers = Vec::new();
    sample_centers.reserve(samples.len());

    for i in samples {
        let triangle = &triangles[*i];
        sample_centers.push(
            verts[triangle.index1 as usize].position[axis_index] +
            verts[triangle.index2 as usize].position[axis_index] + 
            verts[triangle.index3 as usize].position[axis_index]);
    }

    sample_centers.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sample_centers[sample_centers.len() / 2];

    for index in triangle_indices {
        let triangle = &triangles[index];
        let v1 = &verts[triangle.index1 as usize].position[axis_index];
        let v2 = &verts[triangle.index2 as usize].position[axis_index];
        let v3 = &verts[triangle.index3 as usize].position[axis_index];

        if *v1 <= median || *v2 <= median || *v3 <= median {
            left_indices.push(index);
        }
        if *v1 >= median || *v2 >= median || *v3 >= median {
            right_indices.push(index);
        }
    }

    let left = create_node(verts, triangles, left_indices, depth + 1);
    let right = create_node(verts, triangles, right_indices, depth + 1);

    Some(Box::new(Node::Inner {
        left_child: left,
        right_child: right,
        plane: median,
        axis
    }))
}

fn compute_bounding_box(vertices: &[Vertex]) -> BoundingBox {
    let mut bounds = BoundingBox::new();

    for vertex in vertices {
        bounds.add(&vertex.position);
    }

    bounds
}