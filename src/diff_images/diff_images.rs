mod structural_similarity;
use std::{fs::create_dir_all, path::PathBuf};

use cgmath::vec2;
use itertools::Itertools;
use renderer::{
    color::RGBu8,
    raytracer::{single_channel_image::SingleChannelImage, working_image::WorkingImage},
    util::save_u8_image,
};
use structural_similarity::structural_similarity;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Not enough arguments, usage: diff_images image.specimg reference.specimg");
        std::process::exit(-1);
    }

    let path = &args[1];
    let reference_path = &args[2];
    let normalize_images = false;

    let image = match WorkingImage::read_from_file(path, &None) {
        Ok(image) => image,
        Err(e) => {
            eprintln!("Error while opening image: {e}");
            std::process::exit(-1);
        }
    };

    let r = match WorkingImage::read_from_file(reference_path, &image.settings.scene_version) {
        Ok(image) => image,
        Err(e) => {
            eprintln!("Error while opening reference image: {e}");
            std::process::exit(-1);
        }
    };

    let image = if normalize_images { image.normalized() } else { image };
    let reference = if normalize_images { r.normalized() } else { r };

    if image.settings.size != reference.settings.size {
        eprintln!(
            "Dimension mismatch: {:?} != {:?}",
            image.settings.size, reference.settings.size
        );
        std::process::exit(-1);
    }

    let mse = image.mean_square_error(&reference);
    let rgb_mse = image.rgb_mean_square_error(&reference);
    let rel_mse = image.relative_mean_square_error(&reference);
    output_error("MSE", mse, range(0.00, 1.0), path);
    output_error("rgb_MSE", rgb_mse, range(0.0, 0.5), path);
    output_error("relMSE", rel_mse, range(0.0, 10.0), path);

    let img_grayscale = image.to_grayscale();
    let ref_grayscale = reference.to_grayscale();
    let grayscale_diff = &img_grayscale - &ref_grayscale;
    let grayscale_mse = &grayscale_diff * &grayscale_diff;
    let mssim = structural_similarity(img_grayscale, ref_grayscale);
    output_error("grayscale_MSE", grayscale_mse, range(0.0, 0.05), path);
    output_error("MSSIM", mssim, range(1.0, 0.0), path);
}

struct Range {
    start: f32,
    end: f32,
}

fn range(start: f32, end: f32) -> Range {
    Range { start, end }
}

fn output_error<P>(name: &str, error_image: SingleChannelImage, r: Range, path: P)
where
    PathBuf: From<P>,
{
    let mean = error_image.mean();
    println!("{name} {mean}");

    let image_size = vec2(error_image.width, error_image.height);

    let pixels = error_image
        .data
        .into_iter()
        .map(|v| {
            let x = (v - r.start) / (r.end - r.start);
            let c = colorous::VIRIDIS.eval_continuous(x.clamp(0.0, 1.0) as f64);
            RGBu8::new(c.r, c.g, c.b)
        })
        .collect_vec();

    let mut path = PathBuf::from(path);
    let mut file_name = path.file_stem().unwrap().to_owned();
    file_name.push(".png");
    path.pop();
    path.push("diff/");
    path.push(format!("{name}/"));
    path.push(file_name);

    match create_dir_all(path.parent().unwrap()) {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Can't create directories for path: {path:?}, {e}");
            std::process::exit(-1);
        }
    }
    save_u8_image(&pixels, image_size, path, false);
}
