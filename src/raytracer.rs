use std::time::Instant;
use std::sync::{Arc, Mutex};
use rand::Rng;

use crossbeam_utils::thread;
use cgmath::{InnerSpace, Point3, Vector2, vec2, Vector4};

mod ray;
mod triangle;
mod acceleration;
mod aabb;
mod axis;
mod bsdf;
mod shadingframe;

use triangle::Triangle;
use ray::{Ray, IntersectionResult};
use crate::{
    color::RGBf32,
    camera::PerspectiveCamera,
    material::Material,
    light::Light,
    texture::Texture,
    environment::Environment,
    mesh::Vertex,
    scene::Scene,
};

use aabb::BoundingBox;
use shadingframe::ShadingFrame;
use acceleration::{
    // bih::BoundingIntervalHierarchy,
    bvh::BoundingVolumeHierarchy,
    bvh_rec::BoundingVolumeHierarchyRec,
    kdtree::KdTree, 
    structure::AccelerationStructure,
    structure::TraceResult,
};
use bsdf::{
    brdf_sample,
    brdf_eval,
    mis2,
};

pub struct Raytracer {
    verts: Vec<Vertex>,
    triangles: Vec<Triangle>,
    materials: Vec<Material>,
    lights: Vec<Light>,
    textures: Vec<Texture>,
    environment: Environment,
    camera: PerspectiveCamera,
    max_depth: usize,
    pub accel_structures: Vec<Box<dyn AccelerationStructure + Sync>>,
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
            lights: scene.lights.clone(),
            textures: scene.textures.clone(), // FIXME: This wastes a lot of memory
            environment: scene.environment.clone(),
            camera: scene.camera.clone(),
            max_depth: 10,
            accel_structures: Vec::new(),
        };

        // result.accel_structures.push(Box::new(BoundingIntervalHierarchy::new(&result.verts, &result.triangles)));
        result.accel_structures.push(Box::new(BoundingVolumeHierarchy::new(&result.verts, &result.triangles)));
        result.accel_structures.push(Box::new(BoundingVolumeHierarchyRec::new(&result.verts, &result.triangles)));
        result.accel_structures.push(Box::new(KdTree::new(&result.verts, &result.triangles)));

        result
    }

    pub fn render(&self, image_size: Vector2<usize>, samples: usize, accel_index: usize) -> (Vec::<RGBf32>, f64) {
        let aspect_ratio = image_size.x as f32 / image_size.y as f32;
        let fov_factor = (self.camera.y_fov / 2.).tan();

        let start = Instant::now();

        let cam_model = self.camera.model;
        let cam_pos4 =  cam_model * Vector4::new(0., 0., 0., 1.);
        let camera_pos = Point3::from_homogeneous(cam_pos4);
        let mut buffer = Vec::<RGBf32>::new();
        buffer.resize(image_size.x * image_size.y, RGBf32::from_grayscale(0.0));

        let buffer= Arc::new(Mutex::new(buffer));

        let thread_count = usize::max(num_cpus::get() - 2, 1);
        let rows_per_thread = image_size.y as f32 / thread_count as f32;

        thread::scope(|s| {
            for thread_index in 0..thread_count {
                let buffer = Arc::clone(&buffer);

                s.spawn(move |_| {
                    let y_start = (thread_index as f32 * rows_per_thread) as usize;
                    let y_end = ((thread_index + 1) as f32 * rows_per_thread) as usize;

                    // Simple row-wise split of work
                    for y in y_start..y_end {
                        for x in 0..image_size.x {
                            let mut color = RGBf32::new(0.0, 0.0, 0.0);

                            for sample in 0..samples {
                                let mut rng = rand::thread_rng();

                                let mut tent_sample = || {
                                    let r = 2.0 * rng.gen::<f32>();

                                    if r < 1.0 {
                                        r.sqrt() - 1.0
                                    } else {
                                        1.0 - (2.0 - r).sqrt()
                                    }
                                };

                                let mut offset = vec2(0.5, 0.5);

                                if sample > 0 {
                                    offset += vec2(tent_sample(), tent_sample());
                                }

                                let screen = self.pixel_to_screen(Vector2::new(x, y), offset, image_size, aspect_ratio, fov_factor);

                                // Using w = 0 because this is a direction vector
                                let dir4 = cam_model * Vector4::new(screen.x, screen.y, -1., 0.).normalize();
                                let ray = Ray { origin: camera_pos, direction: dir4.truncate().normalize() };

                                color +=  self.radiance(&ray, 0, accel_index);
                            }

                            color = color / samples as f32;
                            let gamma = 2.2;
                            color = color.pow(1.0 / gamma);

                            let mut buffer = buffer.lock().unwrap();
                            buffer[image_size.x * y + x] = color;
                        }
                    }
                });
            }
        }).unwrap();

        let lock = Arc::try_unwrap(buffer).expect("Buffer lock has multiple owners");

        (lock.into_inner().expect("Cannot unlock buffer mutex"), start.elapsed().as_millis() as f64 / 1000.0)
    }

    fn pixel_to_screen(&self, pixel: Vector2<usize>, offset: Vector2<f32>, image_size: Vector2<usize>, aspect_ratio: f32, fov_factor: f32) -> Vector2<f32> {
        let normalized_x = (pixel.x as f32 + offset.x) / image_size.x as f32;
        let normalized_y = (pixel.y as f32 + offset.y) / image_size.y as f32;
        let x = (2. * normalized_x - 1.) * fov_factor * aspect_ratio;
        let y = (1. - 2. * normalized_y) * fov_factor;
        Vector2::new(x, y)
    }

    pub fn get_num_tris(&self) -> usize {
        self.triangles.len()
    }

    fn radiance(&self, ray: &Ray, depth: usize, accel_index: usize) -> RGBf32 {
        let mut result = RGBf32::new(0., 0., 0.);

        if let TraceResult::Hit{ triangle_index, t, u, v } = self.trace(ray, accel_index) {
            let hit_pos = ray.traverse(t);
            let triangle = &self.triangles[triangle_index as usize];
            let material = &self.materials[triangle.material_index as usize];

            let verts = [
                &self.verts[triangle.index1 as usize],
                &self.verts[triangle.index2 as usize],
                &self.verts[triangle.index3 as usize],
            ];

            // Interpolate the vertex normals
            let normal = (
                (1. - u - v) * verts[0].normal +
                u * verts[1].normal +
                v * verts[2].normal
            ).normalize();
            
            let has_texture_coords = self.verts[triangle.index1 as usize].tex_coord.is_some();

            let texture_coords = if has_texture_coords {
                (1. - u - v) * verts[0].tex_coord.unwrap() +
                u * verts[1].tex_coord.unwrap() +
                v * verts[2].tex_coord.unwrap()
            } else {
                vec2(0.0, 0.0)
            };

            let offset_hit_pos = hit_pos + 0.0001 * normal;

            // FIXME: Only evaluate these when required?
            let frame = ShadingFrame::new(normal);
            let local_incident = frame.to_local(-ray.direction);
            let mat_sample = material.sample(texture_coords, &self.textures);

            if depth < self.max_depth {
                let (local_bounce_dir, brdf, pdf) = brdf_sample(&mat_sample, local_incident);

                if brdf > RGBf32::from_grayscale(0.0) {
                    let mis_weight = mis2(100.0, pdf);
                    let bounce_ray = Ray { origin: offset_hit_pos, direction: frame.to_global(local_bounce_dir) };

                    result += material.base_color * brdf / pdf * mis_weight * self.radiance(&bounce_ray, depth + 1, accel_index);
                }
            }

            // Next event estimation (directly sampling lights)
            for light in &self.lights {
                // FIXME: Properly sample light types other than point

                let light_vec = light.pos - offset_hit_pos;
                let light_dist = light_vec.magnitude();

                if light_dist > light.range {
                    continue;
                }

                let light_dir = light_vec / light_dist;

                let shadow_ray = Ray { origin: offset_hit_pos, direction: light_dir };
                let mut shadowed = false;

                if let TraceResult::Hit{ t, .. } = self.trace(&shadow_ray, accel_index) {
                    if t < light_dist {
                        shadowed = true;
                    }
                }

                if !shadowed {
                    let local_outgoing = frame.to_local(light_dir);
                    let (brdf, pdf) = brdf_eval(&mat_sample, local_incident, local_outgoing);

                    if brdf > RGBf32::from_grayscale(0.0) {
                        let falloff = (1.0 + light_dist) * (1.0 + light_dist);
                        let intensity = light.intensity / falloff;
                        let light_pick_prob= 1.0;
                        let light_sample_pdf = 100.0; 
                        let mis_weight = mis2(light_pick_prob * light_sample_pdf, pdf);

                        result += normal.dot(light_dir) * intensity * brdf * mis_weight * light.color * material.base_color;
                    }
                }
            }
        } else {
            result += self.environment.sample(ray.direction);
        }

        result
    }

    fn trace(&self, ray: &Ray, accel_index: usize) -> TraceResult {
        self.accel_structures[accel_index].intersect(ray, &self.verts, &self.triangles)
    }
}