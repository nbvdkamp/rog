use std::time::Instant;
use cgmath::{InnerSpace, Point3, Vector2, Vector4};

mod ray;
mod triangle;
mod acceleration;
mod color;
mod aabb;
mod axis;

use triangle::Triangle;
use ray::{Ray, IntersectionResult};
use color::{Color, ColorNormalizable};
use crate::{camera::PerspectiveCamera, material::Material, mesh::Vertex, scene::Scene};

use self::aabb::BoundingBox;
use self::acceleration::{
    bih::BoundingIntervalHierarchy,
    bvh::BoundingVolumeHierarchy,
    bvh_rec::BoundingVolumeHierarchyRec,
    kdtree::KdTree, 
    structure::AccelerationStructure,
    structure::TraceResult,
};

pub struct Raytracer {
    verts: Vec<Vertex>,
    triangles: Vec<Triangle>,
    materials: Vec<Material>,
    camera: PerspectiveCamera,
    pub accel_structures: Vec<Box<dyn AccelerationStructure>>,
}

impl Raytracer {
    pub fn new(scene: &Scene) -> Self {
        let mut verts = Vec::new();
        let mut triangles = Vec::new();
        let mut materials = Vec::new();

        // TODO: Fix borrowing to prevent having to clone everything
        for mesh in &scene.meshes {
            let start_index = verts.len() as u32;
            let material_index = materials.len() as u32;

            for v in &mesh.vertices {
                verts.push(v.clone());
            }

            materials.push(mesh.material.clone());

            for i in (0..mesh.indices.len()).step_by(3) {
                let index1 = mesh.indices[i] + start_index;
                let index2 = mesh.indices[i + 1] + start_index;
                let index3 = mesh.indices[i + 2] + start_index;
                
                let mut bounds = BoundingBox::new();
                bounds.add(&verts[index1 as usize].position);
                bounds.add(&verts[index2 as usize].position);
                bounds.add(&verts[index3 as usize].position);

                triangles.push(Triangle { index1, index2, index3, material_index, bounds });
            }
        }

        let mut result = Raytracer {
            verts,
            triangles,
            materials,
            camera: scene.camera.clone(),
            accel_structures: Vec::new(),
        };

        // result.accel_structures.push(Box::new(BoundingIntervalHierarchy::new(&result.verts, &result.triangles)));
        result.accel_structures.push(Box::new(BoundingVolumeHierarchy::new(&result.verts, &result.triangles)));
        result.accel_structures.push(Box::new(BoundingVolumeHierarchyRec::new(&result.verts, &result.triangles)));
        result.accel_structures.push(Box::new(KdTree::new(&result.verts, &result.triangles)));

        result
    }

    pub fn render(&self, image_size: Vector2<u32>, accel_index: usize) -> (Vec::<u8>, f64) {
        let aspect_ratio = image_size.x as f32 / image_size.y as f32;
        let fov_factor = (self.camera.y_fov / 2.).tan();

        let start = Instant::now();

        let cam_model = self.camera.model;
        let cam_pos4 =  cam_model * Vector4::new(0., 0., 0., 1.);
        let camera_pos = Point3::from_homogeneous(cam_pos4);
        let mut buffer = Vec::<u8>::new();
        buffer.resize(3 * (image_size.x * image_size.y) as usize, 0);

        for y in 0..image_size.y {
            for x in 0..image_size.x {
                let offset = Vector2::new(0.5, 0.5);
                let screen = self.pixel_to_screen(Vector2::new(x, y), offset, image_size, aspect_ratio, fov_factor);

                // Using w = 0 because this is a direction vector
                let dir4 = cam_model * Vector4::new(screen.x, screen.y, -1., 0.).normalize();
                let ray = Ray { origin: camera_pos, direction: dir4.truncate().normalize() };

                let color = self.trace(ray, accel_index);

                let pixel_index = 3 * (image_size.x * y + x) as usize;
                buffer[pixel_index] = color.r_normalized();
                buffer[pixel_index + 1] = color.g_normalized();
                buffer[pixel_index + 2] = color.b_normalized();
            }
        }

        (buffer, start.elapsed().as_millis() as f64 / 1000.0)
    }

    fn pixel_to_screen(&self, pixel: Vector2<u32>, offset: Vector2<f32>, image_size: Vector2<u32>, aspect_ratio: f32, fov_factor: f32) -> Vector2<f32> {
        let normalized_x = (pixel.x as f32 + offset.x) / image_size.x as f32;
        let normalized_y = (pixel.y as f32 + offset.y) / image_size.y as f32;
        let x = (2. * normalized_x - 1.) * fov_factor * aspect_ratio;
        let y = (1. - 2. * normalized_y) * fov_factor;
        Vector2::new(x, y)
    }

    pub fn get_num_tris(&self) -> usize {
        self.triangles.len()
    }

    fn trace(&self, ray: Ray, accel_index: usize) -> Color {
        let mut result = Color::new(0., 0., 0., 1.);

        if let TraceResult::Hit(triangle_index, hit_pos) = self.accel_structures[accel_index].intersect(&ray, &self.verts, &self.triangles) {
            let triangle = &self.triangles[triangle_index as usize];
            result = self.materials[triangle.material_index as usize].base_color_factor;
        }

        result
    }
}