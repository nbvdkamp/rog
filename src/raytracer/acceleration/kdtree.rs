use rand::{seq::IteratorRandom, thread_rng};

use cgmath::{MetricSpace, Vector3};

use crate::mesh::Vertex;
use crate::raytracer::triangle::Triangle;
use crate::raytracer::{Ray, IntersectionResult};

use super::super::axis::Axis;
use super::super::aabb::BoundingBox;
use super::structure::{AccelerationStructure, TraceResult};

pub struct KdTree {
    root: Option<Box<Node>>,
    scene_bounds: BoundingBox,
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

impl KdTree {
    pub fn new(verts: &[Vertex], triangles: &[Triangle]) -> Self {
        let mut item_indices = Vec::new();

        for i in 0..triangles.len() {
            item_indices.push(i);
        }

        KdTree { 
            root: create_node(verts, triangles, item_indices, 0),
            scene_bounds: compute_bounding_box(verts),
        }
    }
}

impl AccelerationStructure for KdTree {
    fn intersect(&self, ray: &Ray, verts: &[Vertex], triangles: &[Triangle]) -> TraceResult {
        let inv_dir = 1.0 / ray.direction;

        intersect(&self.root, ray, inv_dir, verts, triangles, self.scene_bounds)
    }

    fn get_name(&self) -> &str {
        "K-d Tree"
    }
}

fn intersect(node_opt: &Option<Box<Node>>, ray: &Ray, inv_dir: Vector3<f32>, verts: &[Vertex], triangles: &[Triangle], bounds: BoundingBox) -> TraceResult {
    match node_opt {
        Some(node) => {
            if !bounds.intersects_ray(ray, &inv_dir) {
                return TraceResult::Miss;
            }

            let node = node.as_ref();
            match node {
                Node::Inner { left_child, right_child, plane, axis } => 
                    inner_intersect(left_child, right_child, bounds, *plane, *axis, ray, inv_dir, verts, triangles),
                Node::Leaf { items } => 
                    leaf_intersect(items, ray, verts, triangles)
            }
        }
        None => TraceResult::Miss
    }
}

fn inner_intersect(left: &Option<Box<Node>>, right: &Option<Box<Node>>, bounds: BoundingBox, plane: f32, axis: Axis,
                ray: &Ray, inv_dir: Vector3<f32>, verts: &[Vertex], triangles: &[Triangle]) -> TraceResult {
    let mut left_bounds = bounds;
    left_bounds.set_max(&axis, plane);
    let mut right_bounds = bounds;
    right_bounds.set_min(&axis, plane);
    
    let l = intersect(left, ray, inv_dir, verts, triangles, left_bounds);
    let r = intersect(right, ray, inv_dir, verts, triangles, right_bounds);

    if let TraceResult::Hit(_, hit_pos_l) = l {
        if let TraceResult::Hit(_, hit_pos_r) = r {
            let distance_l = hit_pos_l.distance2(ray.origin);
            let distance_r = hit_pos_r.distance2(ray.origin);

            if distance_l <= distance_r {
                l
            } else {
                r
            }
        } else {
            l
        }
    } else {
        r
    }
}

fn leaf_intersect(triangle_indices: &Vec<usize>, ray: &Ray, verts: &[Vertex], triangles: &[Triangle]) -> TraceResult {
    let mut result = TraceResult::Miss;
    let mut min_dist = f32::MAX;

    for triangle_index in triangle_indices {
        let triangle = &triangles[*triangle_index as usize];
        let p1 = &verts[triangle.index1 as usize];
        let p2 = &verts[triangle.index2 as usize];
        let p3 = &verts[triangle.index3 as usize];

        if let IntersectionResult::Hit(hit_pos) = ray.intersect_triangle(p1.position, p2.position, p3.position) {
            let dist = hit_pos.distance2(ray.origin);

            if dist < min_dist {
                result = TraceResult::Hit(*triangle_index as i32, hit_pos);
                min_dist = dist;
            }
        }
    }

    result
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