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

    let image = match WorkingImage::read_from_file(path, &None) {
        Ok(image) => image,
        Err(e) => {
            eprintln!("Error while opening image: {e}");
            std::process::exit(-1);
        }
    };
    let reference = match WorkingImage::read_from_file(&args[2], &image.settings.scene_version) {
        Ok(image) => image,
        Err(e) => {
            eprintln!("Error while opening reference image: {e}");
            std::process::exit(-1);
        }
    };

    if image.settings.size != reference.settings.size {
        eprintln!(
            "Dimension mismatch: {:?} != {:?}",
            image.settings.size, reference.settings.size
        );
        std::process::exit(-1);
    }

    output_error("MSE", image.mean_square_error(&reference), path);
    output_error("rgb_MSE", image.rgb_mean_square_error(&reference), path);
    output_error("relMSE", image.relative_mean_square_error(&reference), path);

    let img_grayscale = image.to_grayscale();
    let ref_grayscale = reference.to_grayscale();
    let grayscale_diff = &img_grayscale - &ref_grayscale;
    output_error("grayscale_MSE", &grayscale_diff * &grayscale_diff, path);
    output_error("MSSIM", structural_similarity(img_grayscale, ref_grayscale), path);
}

fn output_error<P>(name: &str, error_image: SingleChannelImage, path: P)
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
            let c = colorous::VIRIDIS.eval_continuous(v.clamp(0.0, 1.0) as f64);
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
