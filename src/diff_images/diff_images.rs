mod structural_similarity;
use renderer::raytracer::{single_channel_image::SingleChannelImage, working_image::WorkingImage};
use structural_similarity::structural_similarity;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Not enough arguments, useage: diff_images image.specimg reference.specimg");
        std::process::exit(-1);
    }

    let image = match WorkingImage::read_from_file(&args[1], &None) {
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

    output_error("MSE", image.mean_square_error(&reference));
    output_error("rgb_MSE", image.rgb_mean_square_error(&reference));
    output_error("relMSE", image.relative_mean_square_error(&reference));

    let img_grayscale = image.to_grayscale();
    let ref_grayscale = reference.to_grayscale();
    let grayscale_diff = &img_grayscale - &ref_grayscale;
    output_error("grayscale_MSE", &grayscale_diff * &grayscale_diff);
    output_error("MSSIM", structural_similarity(img_grayscale, ref_grayscale));
}

fn output_error(name: &str, error_image: SingleChannelImage) {
    let mean = error_image.mean();
    println!("{name} {mean}");
}
