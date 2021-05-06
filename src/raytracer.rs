use std::convert::TryFrom;
use std::time::Instant;

mod ray;
mod triangle;
use cgmath::{MetricSpace, InnerSpace, Point3, Vector4};
use triangle::Triangle;
use ray::{Ray, IntersectionResult};
use crate::{camera::PerspectiveCamera, material::Material, mesh::Vertex, scene::Scene};

pub struct Raytracer {
    verts: Vec<Vertex>,
    triangles: Vec<Triangle>,
    materials: Vec<Material>,
    camera: PerspectiveCamera,
}

type Color = Vector4<f32>;

trait ColorNormalizable {
    fn r_normalized(&self) -> u8;
    fn g_normalized(&self) -> u8;
    fn b_normalized(&self) -> u8;
    fn a_normalized(&self) -> u8;
}

impl ColorNormalizable for Color {
    fn r_normalized(&self) -> u8 {
        (self.x * 255.0) as u8
    }
    fn g_normalized(&self) -> u8 {
        (self.y * 255.0) as u8
    }
    fn b_normalized(&self) -> u8 {
        (self.z * 255.0) as u8
    }
    fn a_normalized(&self) -> u8 {
        (self.w * 255.0) as u8
    }
}

impl Raytracer {
    pub fn new(scene: &Scene) -> Self {
        let mut result = Raytracer {
            verts: Vec::new(),
            triangles: Vec::new(),
            materials: Vec::new(),
            camera: scene.camera.clone()
        };

        // TODO: Fix borrowing to prevent having to clone everything
        for mesh in &scene.meshes {
            let start_index = u32::try_from(result.verts.len()).unwrap();
            let material_index = u32::try_from(result.materials.len()).unwrap();

            for v in &mesh.vertices {
                result.verts.push(v.clone());
            }

            result.materials.push(mesh.material.clone());

            for i in (0..mesh.indices.len()).step_by(3) {
                result.triangles.push(Triangle {
                    index1: mesh.indices[i] + start_index,
                    index2: mesh.indices[i + 1] + start_index,
                    index3: mesh.indices[i + 2] + start_index,
                    material_index 
                });
            }
        };

        result
    }

    pub fn render(&self) {
        let image_width: u32 = 1920;
        let image_height: u32 = 1080;
        let aspect_ratio = image_width as f32 / image_height as f32;

        let start = Instant::now();

        let cam_model = self.camera.model;
        let cam_pos4 =  cam_model * Vector4::new(0., 0., 0., 1.);
        let camera_pos = Point3::from_homogeneous(cam_pos4);
        let size = image_width * image_height * 3;
        let mut buffer = Vec::<u8>::new();
        buffer.resize(size as usize, 0);

        for y in 0..image_height {
            for x in 0..image_width {
                let pixel_index = (image_width * y + x) * 3;

                let normalized_x = (x as f32 + 0.5) / image_width as f32;
                let normalized_y = (y as f32 + 0.5) / image_height as f32;
                let scale_factor = (self.camera.y_fov / 2.).tan();
                let screen_x = (2. * normalized_x - 1.) * scale_factor * aspect_ratio;
                let screen_y = (1. - 2. * normalized_y) * scale_factor;

                // Using w = 0 because this is a direction vector
                let dir4 = cam_model * Vector4::new(screen_x, screen_y, -1., 0.).normalize();
                let ray = Ray { origin: camera_pos, direction: dir4.truncate().normalize() };

                let color = self.trace(ray);

                buffer[pixel_index as usize] = color.r_normalized();
                buffer[pixel_index as usize + 1] = color.g_normalized();
                buffer[pixel_index as usize + 2] = color.b_normalized();
            }
        }
        println!("Finished rendering in {} seconds", start.elapsed().as_millis() as f64 / 1000.0);

        let save_result = image::save_buffer("output/result.png", &buffer, image_width, image_height, image::ColorType::Rgb8);

        match save_result {
            Ok(_) => println!("File was saved succesfully"),
            Err(e) => println!("Couldn't save file: {}", e),
        }
    }

    fn trace(&self, ray: Ray) -> Color {
        let mut result = Color::new(0., 0., 0., 1.);
        let mut min_distance = f32::MAX;

        for triangle in &self.triangles {
            let p1 = &self.verts[triangle.index1 as usize];
            let p2 = &self.verts[triangle.index2 as usize];
            let p3 = &self.verts[triangle.index3 as usize];

            if let IntersectionResult::Hit(hit_pos) = ray.intersect_triangle(p1.position, p2.position, p3.position) {
                let distance = hit_pos.distance2(ray.origin);

                if distance < min_distance {
                    min_distance = distance;
                    result = self.materials[triangle.material_index as usize].base_color_factor;
                }
            }
        }

        result
    }
}