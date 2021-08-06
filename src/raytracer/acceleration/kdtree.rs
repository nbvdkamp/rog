use cgmath::{MetricSpace, Vector3};

use crate::mesh::Vertex;
use crate::raytracer::triangle::Triangle;
use crate::raytracer::{Ray, IntersectionResult};

use super::super::axis::Axis;
use super::super::aabb::BoundingBox;
use super::structure::{AccelerationStructure, TraceResult};
use super::statistics::{Statistics, StatisticsStore};

pub struct KdTree {
    root: Option<Box<Node>>,
    scene_bounds: BoundingBox,
    stats: Statistics,
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

impl AccelerationStructure for KdTree {
    fn intersect(&self, ray: &Ray, verts: &[Vertex], triangles: &[Triangle]) -> TraceResult {
        self.stats.count_ray();

        let inv_dir = 1.0 / ray.direction;

        if !self.scene_bounds.intersects_ray(ray, &inv_dir) {
            return TraceResult::Miss;
        }

        self.intersect(&self.root, ray, inv_dir, verts, triangles, self.scene_bounds)
    }

    fn get_name(&self) -> &str {
        "K-d Tree"
    }

    fn get_statistics(&self) -> StatisticsStore {
        self.stats.get_copy()
    }
}

impl KdTree {
    pub fn new(verts: &[Vertex], triangles: &[Triangle]) -> Self {
        let mut item_indices = Vec::new();

        for i in 0..triangles.len() {
            item_indices.push(i);
        }

        let scene_bounds = compute_bounding_box(verts);

        KdTree { 
            root: create_node(verts, triangles, item_indices, 0, &scene_bounds),
            scene_bounds,
            stats: Statistics::new(),
        }
    }

    fn intersect(&self, node_opt: &Option<Box<Node>>, ray: &Ray, inv_dir: Vector3<f32>, verts: &[Vertex], triangles: &[Triangle], bounds: BoundingBox) -> TraceResult {
        match node_opt {
            Some(node) => {
                let node = node.as_ref();
                match node {
                    Node::Inner { left_child, right_child, plane, axis } => 
                        self.inner_intersect(left_child, right_child, bounds, *plane, *axis, ray, inv_dir, verts, triangles),
                    Node::Leaf { items } => 
                        self.leaf_intersect(items, ray, verts, triangles)
                }
            }
            None => TraceResult::Miss
        }
    }

    fn inner_intersect(&self, left: &Option<Box<Node>>, right: &Option<Box<Node>>, bounds: BoundingBox, plane: f32, axis: Axis,
                    ray: &Ray, inv_dir: Vector3<f32>, verts: &[Vertex], triangles: &[Triangle]) -> TraceResult {
        self.stats.count_inner_node_traversal();

        let mut left_bounds = bounds;
        left_bounds.set_max(&axis, plane);
        let mut right_bounds = bounds;
        right_bounds.set_min(&axis, plane);
        
        let hit_l_box = left_bounds.intersects_ray(ray, &inv_dir);
        let hit_r_box = right_bounds.intersects_ray(ray, &inv_dir);

        if !hit_l_box && !hit_r_box {
            return TraceResult::Miss;
        } else if hit_l_box && !hit_r_box {
            return self.intersect(left, ray, inv_dir, verts, triangles, left_bounds);
        } else if !hit_l_box && hit_r_box {
            return self.intersect(right, ray, inv_dir, verts, triangles, right_bounds);
        } 

        // We hit both children's bounds, so check which is hit first
        // If there is an intersection in that one that is closer than the other child's bounds we can stop

        let dist_to_left_box = left_bounds.t_distance_from_ray(ray, &inv_dir);
        let dist_to_right_box = right_bounds.t_distance_from_ray(ray, &inv_dir);

        if dist_to_left_box < dist_to_right_box {
            self.intersect_both_children_hit(left, left_bounds, 
                right, right_bounds, dist_to_right_box, ray, inv_dir, verts, triangles)
        } else {
            self.intersect_both_children_hit(right, right_bounds, 
                left, left_bounds, dist_to_left_box, ray, inv_dir, verts, triangles)
        }
    }

    fn intersect_both_children_hit(&self, first_hit_child: &Option<Box<Node>>, first_bounds: BoundingBox,
                        second_hit_child: &Option<Box<Node>>, second_bounds: BoundingBox, dist_to_second_box: f32,
                        ray: &Ray, inv_dir: Vector3<f32>, verts: &[Vertex], triangles: &[Triangle]) -> TraceResult {

        let first_result = self.intersect(&first_hit_child, ray, inv_dir, verts, triangles, first_bounds);

        if let TraceResult::Hit(_, hit_pos_l) = first_result {
            let distance_l = hit_pos_l.distance2(ray.origin);

            if distance_l < dist_to_second_box * dist_to_second_box {
                first_result
            } else {
                let second_result = self.intersect(second_hit_child, ray, inv_dir, verts, triangles, second_bounds);

                if let TraceResult::Hit(_, hit_pos_r) = second_result {
                    let distance_r = hit_pos_r.distance2(ray.origin);

                    if distance_r < distance_l {
                        second_result
                    } else {
                        first_result
                    }
                } else {
                    first_result
                }
            }
        } else {
            self.intersect(second_hit_child, ray, inv_dir, verts, triangles, second_bounds)
        }
    }

    fn leaf_intersect(&self, triangle_indices: &Vec<usize>, ray: &Ray, verts: &[Vertex], triangles: &[Triangle]) -> TraceResult {
        let mut result = TraceResult::Miss;
        let mut min_dist = f32::MAX;

        for triangle_index in triangle_indices {
            let triangle = &triangles[*triangle_index as usize];
            let p1 = &verts[triangle.index1 as usize];
            let p2 = &verts[triangle.index2 as usize];
            let p3 = &verts[triangle.index3 as usize];

            self.stats.count_intersection_test();

            if let IntersectionResult::Hit(hit_pos) = ray.intersect_triangle(p1.position, p2.position, p3.position) {
                self.stats.count_intersection_hit();

                let dist = hit_pos.distance2(ray.origin);

                if dist < min_dist {
                    result = TraceResult::Hit(*triangle_index as i32, hit_pos);
                    min_dist = dist;
                }
            }
        }

        result
    }
}

fn create_node(verts: &[Vertex], triangles: &[Triangle], triangle_indices: Vec<usize>, depth: i32, bounds: &BoundingBox) -> Option<Box<Node>> {
    if triangle_indices.len() == 0 {
        return None
    }

    let early_term_leaf_size = 10;
    let max_depth = (f32::log2(triangles.len() as f32) * 1.2) as i32;

    if triangle_indices.len() <= early_term_leaf_size || depth > max_depth {
        return Some(Box::new(Node::Leaf {
            items: triangle_indices,
        }));
    }
    
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();

    let (axis, mid_plane) = bounds.find_split_plane();
    let axis_index = axis.index();

    let mut split_plane = f32::MAX;
    let mut least_dist = f32::MAX;

    // Perfect vertex split
    for index in &triangle_indices {
        let bounds = &triangles[*index].bounds;
        
        let p = bounds.max[axis_index];
        let dist = f32::abs(p - mid_plane);

        if  dist < least_dist {
            split_plane = p;
            least_dist = dist;
        }

        let p = bounds.min[axis_index];
        let dist = f32::abs(p - mid_plane);

        if  dist < least_dist {
            split_plane = p;
            least_dist = dist;
        }
    }

    let mut left_bounds = bounds.clone();
    left_bounds.set_max(&axis, split_plane);
    let mut right_bounds = bounds.clone();
    right_bounds.set_min(&axis, split_plane);


    for index in triangle_indices {
        let triangle = &triangles[index];
        let v1 = verts[triangle.index1 as usize].position[axis_index];
        let v2 = verts[triangle.index2 as usize].position[axis_index];
        let v3 = verts[triangle.index3 as usize].position[axis_index];

        if v1 <= split_plane || v2 <= split_plane || v3 <= split_plane {
            left_indices.push(index);
        }
        if v1 > split_plane || v2 > split_plane || v3 > split_plane {
            right_indices.push(index);
        }
    }

    let left = create_node(verts, triangles, left_indices, depth + 1, &left_bounds);
    let right = create_node(verts, triangles, right_indices, depth + 1, &right_bounds);

    Some(Box::new(Node::Inner {
        left_child: left,
        right_child: right,
        plane: split_plane,
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