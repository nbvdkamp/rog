use std::path::Path;

use eframe::{
    egui::{
        global_dark_light_mode_switch,
        load::SizedTexture,
        vec2,
        CentralPanel,
        Color32,
        Context,
        ScrollArea,
        SidePanel,
        Slider,
        TextureFilter,
        TextureOptions,
        TextureWrapMode,
        TopBottomPanel,
        Vec2,
        ViewportBuilder,
    },
    epaint::{ColorImage, TextureHandle},
    run_native,
    App,
    NativeOptions,
};
use egui_plot::{Bar, BarChart, Plot};
use renderer::{
    color::RGBu8,
    raytracer::working_image::{Pixel, WorkingImage},
};

pub fn run(image: Option<WorkingImage>) -> Result<(), eframe::Error> {
    let options = NativeOptions {
        viewport: ViewportBuilder::default().with_inner_size([900.0, 600.0]),
        ..Default::default()
    };

    run_native(
        "Image Inspector",
        options,
        Box::new(move |cc| {
            Box::new(ImageInspectorApp {
                image_data: image.map(|image| ImageData::new(image, &cc.egui_ctx)),
                hovered_pixel: None,
                zoom: 1.0,
                brightness_factor: 1.0,
            })
        }),
    )
}

struct ImageData {
    image: WorkingImage,
    texture: Texture,
    nan_count: usize,
}

impl ImageData {
    fn new(image: WorkingImage, ctx: &Context) -> Self {
        let texture = Texture::from_image_and_ctx(&image, ctx);
        let nan_count = image
            .pixels
            .iter()
            .filter(|p| p.spectrum.data.iter().any(|v| v.is_nan()))
            .count();

        ImageData {
            image,
            texture,
            nan_count,
        }
    }
}

struct ImageInspectorApp {
    image_data: Option<ImageData>,
    hovered_pixel: Option<usize>,
    zoom: f32,
    brightness_factor: f32,
}

impl App for ImageInspectorApp {
    fn update(&mut self, ctx: &Context, _: &mut eframe::Frame) {
        TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                global_dark_light_mode_switch(ui);
                if ui.button("Open file...").clicked() {
                    if let Some(path) = rfd::FileDialog::new().add_filter("Image", &["specimg"]).pick_file() {
                        self.try_open_image(path, ctx);
                    }
                }

                if let Some(ImageData { image, .. }) = &self.image_data {
                    if ui.button("Save as...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Image", &["png", "bmp", "jpg"])
                            .save_file()
                        {
                            image.save_as_rgb(path);
                        }
                    }
                }

                ctx.input(|i| {
                    if let Some(file) = i.raw.dropped_files.first() {
                        if let Some(path) = &file.path {
                            self.try_open_image(path, ctx);
                        }
                    }
                });

                ui.separator();
                ui.add(Slider::new(&mut self.zoom, 1.0..=16.0).text("zoom"));
                ui.separator();
                ui.add(Slider::new(&mut self.brightness_factor, 0.001..=1000.0).text("brightness"));

                if ui.button("Update brightness").clicked() {
                    if let Some(ImageData { image, texture, .. }) = &mut self.image_data {
                        texture.update(&WorkingImage {
                            pixels: image
                                .pixels
                                .iter()
                                .map(|p| Pixel {
                                    spectrum: p.spectrum * self.brightness_factor,
                                    samples: p.samples,
                                })
                                .collect(),
                            settings: image.settings.clone(),
                            paths_sampled_per_pixel: image.paths_sampled_per_pixel,
                            seconds_spent_rendering: image.seconds_spent_rendering,
                        });
                    }
                }
            })
        });

        SidePanel::right("side_panel").resizable(false).show(ctx, |ui| {
            let scroll_area = ScrollArea::vertical();
            let tray_width = 250.0;

            scroll_area.show(ui, |ui| {
                ui.set_min_width(tray_width);
                if let Some(ImageData { image, nan_count, .. }) = &self.image_data {
                    if let Some(index) = self.hovered_pixel {
                        let pixel = &image.pixels[index];
                        let bar = BarChart::new(
                            pixel
                                .result_spectrum()
                                .data
                                .iter()
                                .enumerate()
                                .map(|(i, v)| {
                                    if !v.is_finite() {
                                        let mut b = Bar::new(i as f64, 10.0);
                                        b.stroke.color = Color32::from_rgb(255, 128, 0);
                                        b
                                    } else {
                                        Bar::new(i as f64, *v as f64)
                                    }
                                })
                                .collect(),
                        );
                        Plot::new("Pixel spectrum")
                            .height(tray_width)
                            .show(ui, |plot_ui| plot_ui.bar_chart(bar));
                    } else {
                        ui.add_space(tray_width + ui.spacing().item_spacing.y);
                    }

                    ui.scope(|ui| {
                        if *nan_count > 0 {
                            ui.visuals_mut().override_text_color = Some(Color32::RED);
                            ui.label(format!("Pixels with NaN values: {nan_count}"));
                        }
                    });
                    ui.label(format!("{:#?}", image));
                }
            });
        });

        CentralPanel::default().show(ctx, |ui| {
            let scroll_area = ScrollArea::both();

            scroll_area.show(ui, |ui| {
                if let Some(ImageData { image, texture, .. }) = &self.image_data {
                    let panel_rect = ui.max_rect();
                    let x = panel_rect.width() / texture.size.x;
                    let y = panel_rect.height() / texture.size.y;
                    let mut t = SizedTexture::from_handle(&texture.handle);
                    t.size = self.zoom * texture.size * x.min(y);
                    let r = ui.image(t);

                    self.hovered_pixel = None;
                    if let Some(pointer_pos) = r.hover_pos() {
                        if r.rect.contains(pointer_pos) {
                            let relative_vec =
                                (pointer_pos - r.rect.left_top()) / (r.rect.right_bottom() - r.rect.left_top());
                            let pixel_pos = relative_vec * texture.size;

                            let size = image.settings.size;
                            let x = (pixel_pos.x as usize).clamp(0, size.x - 1);
                            let y = (pixel_pos.y as usize).clamp(0, size.y - 1);
                            let index = y * size.x + x;
                            self.hovered_pixel = Some(index);

                            if secondary_clicked(ctx) {
                                let pixel = &image.pixels[index];
                                println!(
                                    "x: {x}, y: {y}, samples: {}\nspectrum: {:?}",
                                    pixel.samples,
                                    pixel.result_spectrum().data
                                )
                            }
                        }
                    }
                }
            });
        });
    }
}

impl ImageInspectorApp {
    fn try_open_image<P>(&mut self, path: P, ctx: &Context)
    where
        P: AsRef<Path>,
    {
        match WorkingImage::read_from_file(path, &None) {
            Ok(image) => {
                self.image_data = Some(ImageData::new(image, ctx));
            }
            Err(e) => {
                self.image_data = None;
                eprintln!("Error: {e:?}");
            }
        };
    }
}

struct Texture {
    handle: TextureHandle,
    size: Vec2,
}

impl Texture {
    fn from_image_and_ctx(image: &WorkingImage, ctx: &Context) -> Self {
        let options = TextureOptions {
            magnification: TextureFilter::Nearest,
            minification: TextureFilter::Linear,
            wrap_mode: TextureWrapMode::ClampToEdge,
        };
        let s = image.settings.size;

        Texture {
            handle: ctx.load_texture("name", to_egui_color_image(image), options),
            size: vec2(s.x as f32, s.y as f32),
        }
    }

    fn update(&mut self, image: &WorkingImage) {
        let options = TextureOptions {
            magnification: TextureFilter::Nearest,
            minification: TextureFilter::Linear,
            wrap_mode: TextureWrapMode::ClampToEdge,
        };
        let s = image.settings.size;

        self.handle.set(to_egui_color_image(image), options);
        self.size = vec2(s.x as f32, s.y as f32);
    }
}

fn to_egui_color_image(image: &WorkingImage) -> ColorImage {
    let pixels: Vec<RGBu8> = image.to_rgb_buffer().iter().map(|c| c.normalized()).collect();
    let data_ptr = pixels.as_ptr() as *const u8;
    let len = image.pixels.len() * 3;
    let rgb_buffer = unsafe { std::slice::from_raw_parts(data_ptr, len) };
    ColorImage::from_rgb([image.settings.size.x, image.settings.size.y], rgb_buffer)
}

// Workaround for ScrollArea not allowing inner elements to be clicked
fn secondary_clicked(ctx: &Context) -> bool {
    ctx.input(|i| {
        for event in &i.events {
            match event {
                eframe::egui::Event::PointerButton {
                    pos: _,
                    button,
                    pressed,
                    modifiers,
                } if *button == eframe::egui::PointerButton::Secondary && !pressed && modifiers.is_none() => {
                    return true
                }
                _ => (),
            }
        }
        false
    })
}
