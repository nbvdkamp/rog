use std::convert::TryFrom;
use std::time::Instant;

mod ray;
mod triangle;
use triangle::Triangle;
use crate::{material::Material, mesh::{Mesh, Vertex}};

pub struct Raytracer {
    verts: Vec<Vertex>,
    triangles: Vec<Triangle>,
    materials: Vec<Material>,
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
        buffer.resize(size as usize, 0xFF);

        //: [u8; image_size] = [0; image_size];

        /*for pixel in res {
            let ray = Ray { origin: camera_pos, direction: ? };
            buffer[pixel] = trace(ray);
        }*/
        println!("Finished rendering in {} seconds", start.elapsed().as_millis() as f64 / 1000.0);

        let result = image::save_buffer("output/result.png", &buffer, image_width, image_height, image::ColorType::Rgb8);

        match result {
            Ok(_) => println!("File was saved succesfully"),
            Err(e) => println!("Couldn't save file: {}", e),
        }
    }
}