use renderer::raytracer::single_channel_image::SingleChannelImage;

/// An implementation of the structural similarity index from Wang et al. 2003
pub fn structural_similarity(image_x: SingleChannelImage, image_y: SingleChannelImage) -> SingleChannelImage {
    let k_1 = 0.01;
    let k_2 = 0.03;
    let kernel_radius = 5;

    let (x_min, x_max) = image_x.extent();
    let (y_min, y_max) = image_y.extent();
    let min = x_min.min(y_min);
    let max = x_max.max(y_max);
    let dynamic_range = max - min;

    let c_1 = square(k_1 * dynamic_range);
    let c_2 = square(k_2 * dynamic_range);

    let image_x_squared = &image_x * &image_x;
    let image_y_squared = &image_y * &image_y;
    let image_x_image_y = &image_x * &image_y;

    let mu_x = image_x.gaussian_blurred(kernel_radius);
    let mu_y = image_y.gaussian_blurred(kernel_radius);

    let mu_x_squared = &mu_x * &mu_x;
    let mu_y_squared = &mu_y * &mu_y;
    let mu_x_mu_y = &mu_x * &mu_y;

    let sigma_x_squared = image_x_squared.gaussian_blurred(kernel_radius) - &mu_x_squared;
    let sigma_y_squared = image_y_squared.gaussian_blurred(kernel_radius) - &mu_y_squared;
    let sigma_xy = image_x_image_y.gaussian_blurred(kernel_radius) - &mu_x_mu_y;

    ((2.0 * mu_x_mu_y + c_1) * (2.0 * sigma_xy + c_2))
        / ((mu_x_squared + mu_y_squared + c_1) * (sigma_x_squared + sigma_y_squared + c_2))
}

fn square(x: f32) -> f32 {
    x * x
}
