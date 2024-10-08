use rand::Rng;
use std::{
    cell::RefCell,
    io::Write,
    path::Path,
    sync::{
        mpsc::{Receiver, TryRecvError},
        Arc,
        Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use cgmath::{point3, vec2, vec3, InnerSpace, Point2, Point3, Vector2, Vector3};
use crossbeam_deque::{Injector, Steal};

pub mod aabb;
pub mod acceleration;
mod axis;
mod bsdf;
pub mod file_formatting;
pub mod geometry;
mod ray;
pub(crate) mod sampling;
mod shadingframe;
pub mod single_channel_image;
pub(crate) mod triangle;
pub mod working_image;

use crate::{
    light::Light,
    raytracer::working_image::Pixel,
    render_settings::{ImageSettings, RenderSettings, TerminationCondition},
    scene::Scene,
    small_thread_rng::{seed_thread_rng_for_path, thread_rng},
    spectrum::{Spectrumf32, Wavelength},
    texture::{CoefficientTexture, Texture},
    util::normal_transform_from_mat4,
};

use acceleration::{structure::TraceResultMesh, Accel, AccelerationStructures};
use bsdf::{mis2, Evaluation, Sample};
use geometry::{ensure_valid_reflection, orthogonal_vector};
use ray::Ray;
use sampling::{sample_value_from_slice_uniform, tent_sample};
use shadingframe::ShadingFrame;
use working_image::WorkingImage;

thread_local! {
    static VOXEL_WEIGHTS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

pub struct Textures {
    pub base_color_coefficients: Vec<CoefficientTexture>,
    pub metallic_roughness: Vec<Texture>,
    pub transimission: Vec<Texture>,
    pub emissive: Vec<CoefficientTexture>,
    pub normal: Vec<Texture>,
}

pub struct Raytracer {
    pub scene: Scene,
    pub accel_structures: AccelerationStructures,
    image_settings: ImageSettings,
}

pub struct RenderProgressReporting {
    pub report_interval: Duration,
    /// Completion, time elapsed
    pub report: Box<dyn Fn(f32, Duration)>,
}

pub struct ImageUpdateReporting {
    pub update_interval: Duration,
    /// Image, samples completed
    pub update: Box<dyn Fn(&WorkingImage, usize)>,
}

pub enum RenderMessage {
    Cancel,
}

enum RadianceResult {
    Spectrum(Spectrumf32),
    SingleValue { value: f32, wavelength: Wavelength },
}
pub enum WavelengthState {
    Undecided,
    Sampled { value: Wavelength },
}

impl Raytracer {
    pub fn new(scene: Scene, accel_structures_to_construct: &[Accel], image_settings: ImageSettings) -> Self {
        let mut result = Raytracer {
            scene,
            accel_structures: AccelerationStructures::default(),
            image_settings,
        };

        let start = Instant::now();

        for &accel in accel_structures_to_construct {
            result
                .accel_structures
                .construct(accel, &result.scene.meshes, result.scene.instances.clone())
                .unwrap();
        }

        let l = accel_structures_to_construct.len();
        println!(
            "Constructed {} acceleration structure{} in {} seconds",
            l,
            if l == 1 { "" } else { "s" },
            start.elapsed().as_secs_f32()
        );

        result
    }

    pub fn render(
        &self,
        settings: &RenderSettings,
        reporting: Option<RenderProgressReporting>,
        image_reporting: Option<ImageUpdateReporting>,
        message_receiver: Option<Receiver<RenderMessage>>,
        image: WorkingImage,
    ) -> (WorkingImage, f32) {
        let image_size = image.settings.size;
        let aspect = AspectRatio::new(image_size, self.scene.camera.y_fov);
        let camera_pos = Point3::from_homogeneous(self.scene.camera.model.w);

        let start = Instant::now();
        let mut last_progress_update: Option<Instant> = None;
        let mut last_image_update: Option<Instant> = None;

        let paths_sampled_per_pixel = image.paths_sampled_per_pixel as usize;
        let image = Arc::new(Mutex::new(image));

        let tile_size = 100;
        let tiles: &Injector<Tile> = &Injector::new();

        add_tiles_to_queue(tiles, image_size, tile_size, paths_sampled_per_pixel);
        let tiles_per_sample = tiles.len();

        match settings.termination_condition {
            TerminationCondition::SampleCount(samples) => {
                for sample in 1..samples {
                    add_tiles_to_queue(tiles, image_size, tile_size, paths_sampled_per_pixel + sample);
                }
            }
            TerminationCondition::Time(_) => {
                // Always have one extra set of tiles in the queue to prevent threads being idle
                add_tiles_to_queue(tiles, image_size, tile_size, paths_sampled_per_pixel + 1);
            }
        }

        let finished = Arc::new(Mutex::new(false));
        let current_sample = Arc::new(Mutex::new(paths_sampled_per_pixel));
        let tiles_completed_for_current_sample = Arc::new(Mutex::new(0usize));

        thread::scope(|s| {
            for i in 0..settings.thread_count {
                let image = Arc::clone(&image);
                let finished = Arc::clone(&finished);
                let current_sample = Arc::clone(&current_sample);
                let tiles_completed_for_current_sample = Arc::clone(&tiles_completed_for_current_sample);

                let work = move || {
                    'work: loop {
                        let tile = loop {
                            match tiles.steal() {
                                Steal::Success(tile) => break tile,
                                Steal::Empty => {
                                    if *finished.lock().unwrap() {
                                        break 'work;
                                    }
                                }
                                Steal::Retry => {}
                            }
                        };

                        let mut buffer = vec![
                            Pixel {
                                samples: 0,
                                spectrum: Spectrumf32::constant(0.0)
                            };
                            tile_size * tile_size
                        ];

                        for y in tile.start.y..tile.end.y {
                            for x in tile.start.x..tile.end.x {
                                let i = x - tile.start.x;
                                let j = y - tile.start.y;
                                let pixel = &mut buffer[tile_size * j + i];
                                let pixel_sample = PixelSample {
                                    pixel_pos: vec2(x, y),
                                    sample: tile.sample,
                                };

                                // Reseed rng to ensure deterministic rendering
                                seed_thread_rng_for_path(pixel_sample);

                                let ray = self.shoot_camera_ray(pixel_sample, aspect, camera_pos);

                                match self.radiance(ray, settings) {
                                    RadianceResult::Spectrum(spectrum) => {
                                        pixel.spectrum += spectrum;
                                        pixel.samples += Spectrumf32::RESOLUTION as u32;
                                    }
                                    RadianceResult::SingleValue { value, wavelength } => {
                                        // wavelength is sampled in range so no bounds checks necessary
                                        pixel.spectrum.add_at_wavelength(value, wavelength);
                                        pixel.samples += 1;
                                    }
                                }
                            }
                        }

                        while *current_sample.lock().unwrap() < tile.sample {
                            let sleep_duration = Duration::from_millis(10);
                            thread::sleep(sleep_duration);

                            if *finished.lock().unwrap() {
                                break 'work;
                            }
                        }

                        let mut image = image.lock().unwrap();

                        for y in tile.start.y..tile.end.y {
                            for x in tile.start.x..tile.end.x {
                                let i = x - tile.start.x;
                                let j = y - tile.start.y;
                                let pixel = &buffer[tile_size * j + i];
                                image.pixels[image_size.x * y + x].spectrum += pixel.spectrum;
                                image.pixels[image_size.x * y + x].samples += pixel.samples;
                            }
                        }

                        *tiles_completed_for_current_sample.lock().unwrap() += 1;
                    }
                };

                if let Err(e) = thread::Builder::new()
                    .name(format!("Render thread {i}"))
                    .spawn_scoped(s, work)
                {
                    eprintln!("Unable to spawn thread {i}: {e}");
                    std::process::exit(-1);
                };
            }

            let termination_condition_met = || match settings.termination_condition {
                TerminationCondition::SampleCount(_) => tiles.is_empty(),
                TerminationCondition::Time(time_limit) => start.elapsed() >= time_limit,
            };

            let cancel_received = || {
                if let Some(receiver) = &message_receiver {
                    match receiver.try_recv() {
                        Ok(RenderMessage::Cancel) => true,
                        Err(TryRecvError::Disconnected) => true,
                        Err(TryRecvError::Empty) => false,
                    }
                } else {
                    false
                }
            };

            let terminate = || termination_condition_met() || cancel_received();

            while !terminate() {
                let sleep_duration = Duration::from_millis(1);
                thread::sleep(sleep_duration);

                let mut completed = tiles_completed_for_current_sample.lock().unwrap();

                let mut current_sample = current_sample.lock().unwrap();
                let current_sample_finished = *completed == tiles_per_sample;

                if let Some(reporting) = &reporting {
                    let elapsed = start.elapsed();
                    let should_update = match last_progress_update {
                        None => true,
                        Some(i) => i.elapsed() > reporting.report_interval,
                    };

                    if should_update {
                        let completion = match settings.termination_condition {
                            TerminationCondition::SampleCount(samples) => {
                                let total_tiles = tiles_per_sample * samples;
                                let completed = total_tiles - tiles.len();
                                completed as f32 / total_tiles as f32
                            }
                            TerminationCondition::Time(time_limit) => elapsed.as_secs_f32() / time_limit.as_secs_f32(),
                        };
                        (reporting.report)(completion, elapsed);
                        last_progress_update = Some(Instant::now());
                    }
                }

                if let Some(reporting) = &image_reporting {
                    if current_sample_finished {
                        let should_update = match last_image_update {
                            None => true,
                            Some(i) => i.elapsed() > reporting.update_interval,
                        };

                        if should_update {
                            (reporting.update)(&image.lock().unwrap(), *current_sample);
                            last_image_update = Some(Instant::now());
                        }
                    }
                }

                if current_sample_finished {
                    *completed = 0;
                    *current_sample += 1;

                    if let TerminationCondition::Time(_) = settings.termination_condition {
                        add_tiles_to_queue(tiles, image_size, tile_size, *current_sample + 1);
                    }
                }
            }

            *finished.lock().unwrap() = true;
        });

        // Errors when the lock has multiple owners but the scope should guarantee that never happens
        let lock = Arc::try_unwrap(image).ok().unwrap();
        let mut image = lock.into_inner().expect("Cannot unlock image mutex");
        let time_spent = start.elapsed().as_secs_f32();

        image.paths_sampled_per_pixel = *current_sample.lock().unwrap() as u32 + 1;
        image.seconds_spent_rendering += time_spent;

        (image, time_spent)
    }

    pub fn single_path_render(
        &self,
        settings: &RenderSettings,
        image_settings: ImageSettings,
        pixel_sample: PixelSample,
    ) {
        let camera_pos = Point3::from_homogeneous(self.scene.camera.model.w);
        let aspect = AspectRatio::new(image_settings.size, self.scene.camera.y_fov);
        seed_thread_rng_for_path(pixel_sample);
        let ray = self.shoot_camera_ray(pixel_sample, aspect, camera_pos);

        match self.radiance(ray, settings) {
            RadianceResult::Spectrum(spectrum) => {
                println!("Full spectrum: {:?}", spectrum.data);
            }
            RadianceResult::SingleValue { value, wavelength } => {
                println!("Single wavelength: {value} at {}nm", wavelength.value());
            }
        }
    }

    fn pixel_to_screen(&self, pixel: Vector2<usize>, offset: Vector2<f32>, aspect: AspectRatio) -> Vector2<f32> {
        let normalized_x = (pixel.x as f32 + offset.x) / aspect.image_size.x as f32;
        let normalized_y = (pixel.y as f32 + offset.y) / aspect.image_size.y as f32;
        let x = (2. * normalized_x - 1.) * aspect.fov_factor * aspect.ratio;
        let y = (1. - 2. * normalized_y) * aspect.fov_factor;
        Vector2::new(x, y)
    }

    fn shoot_camera_ray(
        &self,
        PixelSample { pixel_pos, sample }: PixelSample,
        aspect: AspectRatio,
        camera_pos: Point3<f32>,
    ) -> Ray {
        let mut offset = vec2(0.5, 0.5);
        if sample > 0 {
            offset += vec2(tent_sample(), tent_sample());
        }

        let screen = self.pixel_to_screen(pixel_pos, offset, aspect);

        let direction = {
            let m = self.scene.camera.model;
            screen.x * m.x.truncate() + screen.y * m.y.truncate() + -1. * m.z.truncate()
        }
        .normalize();

        Ray {
            origin: camera_pos,
            direction,
        }
    }

    pub fn get_num_tris(&self) -> usize {
        self.scene.meshes.iter().map(|m| m.triangles.len()).sum()
    }

    fn radiance(&self, ray: Ray, settings: &RenderSettings) -> RadianceResult {
        let mut result = Spectrumf32::constant(0.0);
        let mut path_weight = Spectrumf32::constant(1.0);
        let mut ray = ray;

        let mut wavelength = WavelengthState::Undecided;
        let mut depth = 0;

        // Hard cap bounces to prevent endless bouncing inside perfectly reflective surfaces
        while !self.image_settings.max_depth_reached(depth) && depth < 100 {
            let TraceResultMesh::Hit {
                instance,
                triangle_index,
                barycentric,
                ..
            } = self.trace(&ray, settings.accel_structure)
            else {
                result += path_weight * self.scene.environment.sample(ray.direction);
                break;
            };

            let normal_transform = normal_transform_from_mat4(instance.transform);
            let intersected_mesh = &self.scene.meshes[instance.mesh_index as usize];

            let triangle = &intersected_mesh.triangles[triangle_index as usize];
            let material = &instance.material;
            let verts = intersected_mesh.vertices.get(triangle.indices.map(|i| i as usize));

            let hit_pos = barycentric.interpolate_point(verts.positions);
            let hit_pos = Point3::from_homogeneous(instance.transform * hit_pos.to_homogeneous());

            let texture_coordinates = verts
                .tex_coords
                .iter()
                .map(|t| barycentric.interpolate_point2(*t))
                .collect();

            let mut mat_sample = material.sample(texture_coordinates, &self.scene.textures);

            if thread_rng().gen::<f32>() > mat_sample.alpha {
                // Offset to the back of the triangle
                ray.origin = hit_pos + ray.direction * 0.0002;

                // Don't count alpha hits for max bounces
                continue;
            }

            if let Some(emission) = &mat_sample.emissive {
                result += path_weight * emission;
            }

            if self.image_settings.enable_dispersion && mat_sample.transmission > 0.0 {
                let lambda;

                match wavelength {
                    WavelengthState::Undecided => {
                        // Ignoring pdf as we it is constant with uniform sampling and we would have to correct by RESOLUTION again
                        let (w, _pdf) = Wavelength::sample_uniform_visible();
                        // path_weight /= pdf;
                        lambda = w.value();
                        wavelength = WavelengthState::Sampled { value: w }
                    }
                    WavelengthState::Sampled { value } => lambda = value.value(),
                }

                mat_sample.ior = mat_sample.cauchy_coefficients.ior_for_wavelength(lambda);
            }

            let edge1 = verts.positions[0] - verts.positions[1];
            let edge2 = verts.positions[0] - verts.positions[2];
            let mut geom_normal = (normal_transform * edge1.cross(edge2)).normalize();

            // Interpolate the vertex normals
            let mut normal = (normal_transform * barycentric.interpolate_vector(verts.normals)).normalize();
            let mut tangent = (normal_transform * barycentric.interpolate_vector(verts.tangents)).normalize();

            // Handle the rare case where a bad tangent (linearly dependent with the normal) causes NaNs
            if normal == tangent || -normal == tangent {
                tangent = orthogonal_vector(normal);
            }

            // Flip the computed geometric normal to the same side as the interpolated vertex normal.
            if geom_normal.dot(normal) < 0.0 {
                geom_normal = -geom_normal;
            }

            let backfacing = geom_normal.dot(-ray.direction) < 0.0;

            if backfacing {
                normal = -normal;
                geom_normal = -geom_normal;

                // Swap IORs
                (mat_sample.ior, mat_sample.medium_ior) = (mat_sample.medium_ior, mat_sample.ior);
            }

            let mut frame = ShadingFrame::new_with_tangent(normal, tangent);

            let shading_normal = mat_sample
                .shading_normal
                .map(|n| ensure_valid_reflection(geom_normal, -ray.direction, frame.to_global(n)).normalize());

            if let Some(shading_normal) = shading_normal {
                frame = ShadingFrame::new(shading_normal);
            }

            let local_outgoing = frame.to_local(-ray.direction);

            let sample = bsdf::sample(&mat_sample, local_outgoing);

            let mut nee_result = Spectrumf32::constant(0.0);

            // Next event estimation (directly sampling lights)
            if let Some((light, light_pick_pdf)) = self.sample_light() {
                let light_sample = light.sample(hit_pos);

                let offset_direction = light_sample.direction.dot(geom_normal).signum() * geom_normal;
                // FIXME: This offset prevents shadow acne but can cause light to leak
                // through seams between triangles with an acute angle.
                // The whole problem could be avoided by starting the shadow ray at the light
                // but this doesn't work for directional lights.
                let offset_hit_pos = offset_hit_pos(hit_pos, offset_direction);

                if light_sample.distance <= light.range {
                    let shadow_ray = Ray {
                        origin: offset_hit_pos,
                        direction: light_sample.direction,
                    };

                    let shadowed = self.is_ray_obstructed(&shadow_ray, light_sample.distance, settings.accel_structure);

                    if !shadowed {
                        let local_incident = frame.to_local(light_sample.direction);
                        let eval = bsdf::eval(&mat_sample, local_outgoing, local_incident);

                        if let Evaluation::Evaluation {
                            weight: bsdf,
                            pdf: bsdf_pdf,
                        } = eval
                        {
                            let mis_weight = if light_sample.use_mis {
                                mis2(light_sample.pdf, bsdf_pdf)
                            } else {
                                1.0
                            };

                            let shadow_terminator = shading_normal.map_or(1.0, |shading_normal| {
                                bump_shading_factor(normal, shading_normal, light_sample.direction)
                            });

                            nee_result += light.spectrum
                                * bsdf
                                * local_incident.z.abs()
                                * (mis_weight * light_sample.intensity * shadow_terminator
                                    / (light_pick_pdf * light_sample.pdf));
                        }
                    }
                }
            }

            if let WavelengthState::Undecided = wavelength {
                if self.image_settings.always_sample_single_wavelength {
                    let (value, _) = Wavelength::sample_uniform_visible();
                    wavelength = WavelengthState::Sampled { value };
                }
            }

            result += path_weight * nee_result;

            let Sample::Sample {
                incident: local_bounce_dir,
                weight: bsdf,
                pdf,
            } = sample
            else {
                break;
            };

            let bounce_dir = frame.to_global(local_bounce_dir);
            let offset_direction = bounce_dir.dot(geom_normal).signum() * geom_normal;
            let offset_hit_pos = offset_hit_pos(hit_pos, offset_direction);

            let shadow_terminator = shading_normal.map_or(1.0, |shading_normal| {
                bump_shading_factor(normal, shading_normal, bounce_dir)
            });

            path_weight *= bsdf * (shadow_terminator * local_bounce_dir.z.abs() / pdf);

            let max_weight = if let WavelengthState::Sampled { value } = wavelength {
                path_weight.at_wavelength(value)
            } else {
                path_weight.max_value()
            };

            let continue_probability = (5.0 * max_weight).min(1.0);

            if continue_probability < 1.0 {
                if thread_rng().gen_bool(continue_probability as f64) {
                    path_weight /= continue_probability;
                } else {
                    break;
                }
            }

            ray = Ray {
                origin: offset_hit_pos,
                direction: bounce_dir,
            };

            depth += 1;
        }

        match wavelength {
            WavelengthState::Undecided => RadianceResult::Spectrum(result),
            WavelengthState::Sampled { value: wavelength } => RadianceResult::SingleValue {
                value: result.at_wavelength(wavelength),
                wavelength,
            },
        }
    }

    fn sample_light(&self) -> Option<(&Light, f32)> {
        let lights = &self.scene.lights;

        if lights.is_empty() {
            return None;
        }

        if lights.len() == 1 {
            return Some((&self.scene.lights[0], 1.0));
        }

        Some(sample_value_from_slice_uniform(lights))
    }

    /// Checks if distance to the nearest obstructing triangle is less than the distance
    /// Handles alpha by checking if R ~ U(0, 1) is greater than the texture's alpha and ignoring
    /// the triangle if it is.
    fn is_ray_obstructed(&self, ray: &Ray, distance: f32, accel: Accel) -> bool {
        let mut distance = distance;
        let mut ray = *ray;

        while let TraceResultMesh::Hit {
            instance,
            triangle_index,
            t,
            barycentric,
        } = self.trace(&ray, accel)
        {
            // Never shadowed
            if t > distance {
                break;
            }

            let intersected_mesh = &self.scene.meshes[instance.mesh_index as usize];
            let verts = &intersected_mesh.vertices;

            let triangle = &intersected_mesh.triangles[triangle_index as usize];
            let material = &instance.material;

            if let Some(tex_ref) = material.base_color_texture {
                if !self.scene.textures.base_color_coefficients[tex_ref.index].has_alpha {
                    return true;
                }
            } else {
                return true;
            }

            let indices = triangle.indices.map(|i| i as usize);

            let alpha = match material.base_color_texture {
                Some(tex) => {
                    let t = &intersected_mesh.vertices.tex_coords[tex.texture_coordinate_set];
                    let uv = barycentric.interpolate_point2(indices.map(|i| t[i]));
                    let Point2 { x: u, y: v } = tex.transform_texture_coordinates(uv);
                    self.scene.textures.base_color_coefficients[tex.index].sample_alpha(u, v)
                }
                None => 1.0,
            };

            if thread_rng().gen::<f32>() > alpha {
                // Cast another ray from slightly further than where we hit
                let hit_pos = barycentric.interpolate_point(indices.map(|i| verts.positions[i]));
                let hit_pos = Point3::from_homogeneous(instance.transform * hit_pos.to_homogeneous());
                let offset = 0.0002;
                ray.origin = hit_pos + offset * ray.direction;
                distance -= t + offset;
            } else {
                return true;
            }
        }

        false
    }

    fn trace(&self, ray: &Ray, accel: Accel) -> TraceResultMesh {
        self.accel_structures.get(accel).intersect(ray, &self.scene.meshes)
    }
}

pub fn render_and_save<P>(
    raytracer: &Raytracer,
    render_settings: &RenderSettings,
    image: WorkingImage,
    path: P,
    image_reporting: Option<ImageUpdateReporting>,
    message_receiver: Option<Receiver<RenderMessage>>,
) where
    P: AsRef<Path>,
{
    let report_progress = |completion, time_elapsed: Duration| {
        let completion_percentage = 100.0 * completion;
        let total_time = time_elapsed.as_secs_f32() / completion;
        let time_remaining = total_time * (1.0 - completion);
        print!("\r\x1b[2K Approximately {completion_percentage:.2}% complete, {time_remaining:.2} seconds remaining");
        std::io::stdout().flush().unwrap();
    };

    let progress = Some(RenderProgressReporting {
        report_interval: Duration::from_secs(1),
        report: Box::new(report_progress),
    });

    let (image, time_elapsed) = raytracer.render(render_settings, progress, image_reporting, message_receiver, image);
    println!("\r\x1b[2KFinished rendering in {time_elapsed} seconds");

    image.save_as_rgb(path);

    if let Some(path) = &render_settings.intermediate_write_path {
        match image.write_to_file(path) {
            Ok(()) => println!("Saved intermediate file successfully"),
            Err(e) => {
                eprintln!("Failed to save intermediate file: {e}");
            }
        };
    }
}

/// From Chiang et al. 2019, 'Taming the Shadow Terminator'
fn bump_shading_factor(
    geometric_normal: Vector3<f32>,
    shading_normal: Vector3<f32>,
    light_direction: Vector3<f32>,
) -> f32 {
    let g_dot_i = geometric_normal.dot(light_direction);
    let s_dot_i = shading_normal.dot(light_direction);
    let g_dot_s = geometric_normal.dot(shading_normal);
    let g = (g_dot_i / (s_dot_i * g_dot_s)).clamp(0.0, 1.0);

    // Hermite interpolation
    -(g * g * g) + (g * g) + g
}

/// From Ray Tracing Gems chapter 6: "A Fast and Robust Method for Avoiding Self-Intersection"
fn offset_hit_pos(pos: Point3<f32>, normal: Vector3<f32>) -> Point3<f32> {
    const ORIGIN: f32 = 1.0 / 32.0;
    const FLOAT_SCALE: f32 = 1.0 / 65536.0;
    const INT_SCALE: f32 = 256.0;

    let offset_int = vec3(
        (INT_SCALE * normal.x) as i32,
        (INT_SCALE * normal.y) as i32,
        (INT_SCALE * normal.z) as i32,
    );

    let scale = |p: f32, o: i32| {
        use crate::util::bit_hacks::{f32_as_i32, i32_as_f32};

        i32_as_f32(f32_as_i32(p) + p.signum() as i32 * o)
    };

    let p_i = vec3(
        scale(pos.x, offset_int.x),
        scale(pos.y, offset_int.y),
        scale(pos.z, offset_int.z),
    );

    let e = |pos: f32, normal, p_i| {
        if pos.abs() < ORIGIN {
            pos + FLOAT_SCALE * normal
        } else {
            p_i
        }
    };

    point3(
        e(pos.x, normal.x, p_i.x),
        e(pos.y, normal.y, p_i.y),
        e(pos.z, normal.z, p_i.z),
    )
}

struct Tile {
    start: Vector2<usize>,
    end: Vector2<usize>,
    sample: usize,
}

fn add_tiles_to_queue(tiles: &Injector<Tile>, image_size: Vector2<usize>, tile_size: usize, sample: usize) {
    for x_start in (0..image_size.x).step_by(tile_size) {
        let x_end = (x_start + tile_size).min(image_size.x);

        for y_start in (0..image_size.y).step_by(tile_size) {
            let y_end = (y_start + tile_size).min(image_size.y);

            tiles.push(Tile {
                start: vec2(x_start, y_start),
                end: vec2(x_end, y_end),
                sample,
            });
        }
    }
}

#[derive(Clone, Copy)]
struct AspectRatio {
    image_size: Vector2<usize>,
    ratio: f32,
    fov_factor: f32,
}

impl AspectRatio {
    pub fn new(image_size: Vector2<usize>, y_fov: f32) -> Self {
        Self {
            image_size,
            ratio: image_size.x as f32 / image_size.y as f32,
            fov_factor: (y_fov / 2.).tan(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct PixelSample {
    pub pixel_pos: Vector2<usize>,
    pub sample: usize,
}
