use std::path::Path;

use eframe::{
    egui::{
        global_dark_light_mode_switch,
        load::SizedTexture,
        vec2,
        CentralPanel,
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
use renderer::{color::RGBu8, raytracer::working_image::WorkingImage};

pub fn run(image: Option<WorkingImage>) -> Result<(), eframe::Error> {
    let options = NativeOptions {
        viewport: ViewportBuilder::default().with_inner_size([900.0, 600.0]),
        ..Default::default()
    };

    run_native(
        "Pixel Sort",
        options,
        Box::new(move |cc| {
            let texture = image
                .as_ref()
                .map(|image| Texture::from_image_and_ctx(image, &cc.egui_ctx));
            Box::new(ImageInspectorApp {
                image,
                texture,
                zoom: 1.0,
            })
        }),
    )
}

struct ImageInspectorApp {
    image: Option<WorkingImage>,
    texture: Option<Texture>,
    zoom: f32,
}

impl App for ImageInspectorApp {
    fn update(&mut self, ctx: &Context, _: &mut eframe::Frame) {
        TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                global_dark_light_mode_switch(ui);

                if ui.button("Sort!").clicked() {}

                if ui.button("Open file...").clicked() {
                    if let Some(path) = rfd::FileDialog::new().add_filter("Image", &["specimg"]).pick_file() {
                        self.try_open_image(path, ctx);
                    }
                }

                ctx.input(|i| {
                    if let Some(file) = i.raw.dropped_files.first() {
                        if let Some(path) = &file.path {
                            self.try_open_image(path, ctx);
                        }
                    }
                });

                ui.add(Slider::new(&mut self.zoom, 1.0..=4.0).text("zoom"));
            })
        });

        SidePanel::right("filters_panel").resizable(false).show(ctx, |ui| {
            let scroll_area = ScrollArea::vertical();
            let tray_width = 250.0;

            scroll_area.show(ui, |ui| {
                ui.set_min_width(tray_width);
            });
        });

        CentralPanel::default().show(ctx, |ui| {
            let scroll_area = ScrollArea::both();

            scroll_area.show(ui, |ui| {
                if let Some(texture) = &self.texture {
                    let panel_rect = ui.max_rect();
                    let x = panel_rect.width() / texture.size.x;
                    let y = panel_rect.height() / texture.size.y;
                    let mut t = SizedTexture::from_handle(&texture.handle);
                    t.size = self.zoom * texture.size * x.min(y);
                    let r = ui.image(t);

                    if let Some(pointer_pos) = r.hover_pos() {
                        if r.rect.contains(pointer_pos) {
                            let relative_vec =
                                (pointer_pos - r.rect.left_top()) / (r.rect.right_bottom() - r.rect.left_top());

                            let pixel_pos = relative_vec * texture.size;
                            ui.label(format!("pixel: {} {}", pixel_pos.x as usize, pixel_pos.y as usize));
                        }
                    };
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
                self.texture = Some(Texture::from_image_and_ctx(&image, ctx));
                self.image = Some(image);
            }
            Err(e) => {
                self.image = None;
                self.texture = None;
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
        self.size = vec2(s.x as f32, s.y as f32);
        self.handle.set(to_egui_color_image(image), options);
    }
}

fn to_egui_color_image(image: &WorkingImage) -> ColorImage {
    let pixels: Vec<RGBu8> = image.to_rgb_buffer().iter().map(|c| c.normalized()).collect();
    let data_ptr = pixels.as_ptr() as *const u8;
    let len = image.pixels.len() * 3;
    let rgb_buffer = unsafe { std::slice::from_raw_parts(data_ptr, len) };
    ColorImage::from_rgb([image.settings.size.x, image.settings.size.y], rgb_buffer)
}
