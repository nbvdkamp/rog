use std::convert::TryFrom;
use std::time::Instant;

mod ray;
mod triangle;
use cgmath::Vector4;
use triangle::Triangle;
use crate::{material::Material, mesh::{Mesh, Vertex}};

pub struct Raytracer {
    verts: Vec<Vertex>,
    triangles: Vec<Triangle>,
    materials: Vec<Material>,
}

type Color = Vector4<f32>;

trait ColorNormalizable {
    fn rNormalized(&self) -> u8;
    fn gNormalized(&self) -> u8;
    fn bNormalized(&self) -> u8;
    fn aNormalized(&self) -> u8;
}

impl ColorNormalizable for Color {
    fn rNormalized(&self) -> u8 {
        (self.x * 255.0) as u8
    }
    fn gNormalized(&self) -> u8 {
        (self.y * 255.0) as u8
    }
    fn bNormalized(&self) -> u8 {
        (self.z * 255.0) as u8
    }
    fn aNormalized(&self) -> u8 {
        (self.w * 255.0) as u8
    }
}

impl Raytracer {
    pub fn new(meshes: &Vec<Mesh>) -> Self {
        let mut result = Raytracer { verts: Vec::new(), triangles: Vec::new(), materials: Vec::new() };

        for mesh in meshes {
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
        let size = image_width * image_height * 3;

        let start = Instant::now();

        //let camera_pos = self.camera.position;
        let mut buffer = Vec::<u8>::new();
        buffer.resize(size as usize, 0);

        for x in 0..image_width {
            for y in 0..image_height {
                let pixel_index = (image_width * y + x) * 3;

                //let ray = Ray { origin: camera_pos, direction: ? };
                let color = Color::new(x as f32 / image_width as f32, y as f32 / image_height as f32, 0.4, 1.0);

                buffer[pixel_index as usize] = color.rNormalized();
                buffer[pixel_index as usize + 1] = color.gNormalized();
                buffer[pixel_index as usize + 2] = color.bNormalized();
            }
        }
        println!("Finished rendering in {} seconds", start.elapsed().as_millis() as f64 / 1000.0);

        let save_result = image::save_buffer("output/result.png", &buffer, image_width, image_height, image::ColorType::Rgb8);

        match save_result {
            Ok(_) => println!("File was saved succesfully"),
            Err(e) => println!("Couldn't save file: {}", e),
        }
    }
}