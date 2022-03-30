use crate::mesh::Vertex;
use crate::raytracer::triangle::Triangle;
use crate::raytracer::{Ray, IntersectionResult};

use cgmath::Vector3;

use super::super::aabb::BoundingBox;
use super::statistics::{Statistics, StatisticsStore};
use super::structure::{AccelerationStructure, TraceResult};

pub struct BoundingVolumeHierarchyRec {
    root: Option<Box<Node>>,
    stats: Statistics,
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

impl AccelerationStructure for BoundingVolumeHierarchyRec {
    #[allow(clippy::only_used_in_recursion)]
    fn intersect(&self, ray: &Ray, verts: &[Vertex], triangles: &[Triangle]) -> TraceResult {
        self.stats.count_ray();

        let inv_dir = 1.0 / ray.direction;

        if intersects_bounds(&self.root, ray, inv_dir) {
            self.intersect(&self.root, ray, inv_dir, verts, triangles)
        } else {
            TraceResult::Miss
        }
    }

    fn get_name(&self) -> &str {
        "BVH (recursive)"
    }

    fn get_statistics(&self) -> StatisticsStore {
        self.stats.get_copy()
    }
}

impl BoundingVolumeHierarchyRec {
    pub fn new(verts: &[Vertex], triangles: &[Triangle]) -> Self {
        let mut item_indices = Vec::new();

        for i in 0..triangles.len() {
            item_indices.push(i);
        }

        BoundingVolumeHierarchyRec { 
            root: create_node(verts, triangles, &mut item_indices),
            stats: Statistics::new(),
        }
    }

    fn intersect(&self, node_opt: &Option<Box<Node>>, ray: &Ray, inv_dir: Vector3<f32>, verts: &[Vertex], triangles: &[Triangle]) -> TraceResult {
        match node_opt {
            Some(node) => {
                let node = node.as_ref();
                match node {
                    Node::Inner { left_child, right_child, .. } => 
                        self.inner_intersect(left_child, right_child, ray, inv_dir, verts, triangles),
                    Node::Leaf { triangle_index, .. } => 
                        self.leaf_intersect(*triangle_index, ray, verts, triangles)
                }
            }
            None => TraceResult::Miss
        }
    }

    fn inner_intersect(&self, left: &Option<Box<Node>>, right: &Option<Box<Node>>,
                    ray: &Ray, inv_dir: Vector3<f32>, verts: &[Vertex], triangles: &[Triangle]) -> TraceResult {
        self.stats.count_inner_node_traversal();

        let hit_l_box = intersects_bounds(left, ray, inv_dir);
        let hit_r_box = intersects_bounds(right, ray, inv_dir);

        if !hit_l_box && !hit_r_box {
            return TraceResult::Miss;
        } else if hit_l_box && !hit_r_box {
            return self.intersect(left, ray, inv_dir, verts, triangles);
        } else if !hit_l_box && hit_r_box {
            return self.intersect(right, ray, inv_dir, verts, triangles);
        }

        // Both children are intersected

        let dist_to_left_box = intersects_bounds_distance(left, ray, inv_dir);
        let dist_to_right_box = intersects_bounds_distance(right, ray, inv_dir);

        if dist_to_left_box < dist_to_right_box {
            self.intersect_both_children_hit(left, right, dist_to_right_box, ray, inv_dir, verts, triangles)
        } else {
            self.intersect_both_children_hit(right, left, dist_to_left_box, ray, inv_dir, verts, triangles)
        }
    }

    fn intersect_both_children_hit(&self, first_hit_child: &Option<Box<Node>>, second_hit_child: &Option<Box<Node>>, dist_to_second_box: f32,
                        ray: &Ray, inv_dir: Vector3<f32>, verts: &[Vertex], triangles: &[Triangle]) -> TraceResult {

        let first_result = self.intersect(first_hit_child, ray, inv_dir, verts, triangles);

        if let TraceResult::Hit{ t: t_first, .. } = first_result {
            if t_first < dist_to_second_box {
                first_result
            } else {
                let second_result = self.intersect(second_hit_child, ray, inv_dir, verts, triangles);

                if let TraceResult::Hit{ t: t_second, .. } = second_result {
                    if t_second < t_first {
                        second_result
                    } else {
                        first_result
                    }
                } else {
                    first_result
                }
            }
        } else {
            self.intersect(second_hit_child, ray, inv_dir, verts, triangles)
        }
    }

    fn leaf_intersect(&self, triangle_index: i32, ray: &Ray, verts: &[Vertex], triangles: &[Triangle]) -> TraceResult {
        let triangle = &triangles[triangle_index as usize];
        let p1 = &verts[triangle.index1 as usize];
        let p2 = &verts[triangle.index2 as usize];
        let p3 = &verts[triangle.index3 as usize];

        self.stats.count_intersection_test();

        if let IntersectionResult::Hit{ t, u, v } = ray.intersect_triangle(p1.position, p2.position, p3.position) {
            self.stats.count_intersection_hit();

            TraceResult::Hit{ triangle_index, t, u, v }
        } else {
            TraceResult::Miss
        }
    }
}

fn intersects_bounds(node_opt: &Option<Box<Node>>, ray: &Ray, inv_dir: Vector3<f32>) -> bool {
    match node_opt {
        Some(node) => {
            let node = node.as_ref();
            match node {
                Node::Inner { bounds, .. } => bounds.intersects_ray(ray, &inv_dir),
                Node::Leaf {  bounds , .. } => bounds.intersects_ray(ray, &inv_dir)
            }
        }
        None => false
    }
}

fn intersects_bounds_distance(node_opt: &Option<Box<Node>>, ray: &Ray, inv_dir: Vector3<f32>) -> f32 {
    match node_opt {
        Some(node) => {
            let node = node.as_ref();
            match node {
                Node::Inner { bounds, .. } => bounds.t_distance_from_ray(ray, &inv_dir),
                Node::Leaf {  bounds , .. } => bounds.t_distance_from_ray(ray, &inv_dir)
            }
        }
        None => { unreachable!() }
    }
}

fn create_node(verts: &[Vertex], triangles: &[Triangle], triangle_indices: &mut [usize]) -> Option<Box<Node>> {
    if triangle_indices.is_empty() {
        return None
    }

    let bounds = compute_bounding_box_triangle_indexed(verts, triangles, triangle_indices);

    if triangle_indices.len() == 1 {
        return Some(Box::new(Node::Leaf {
            triangle_index: triangle_indices[0] as i32,
            bounds,
        }));
    }
    
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