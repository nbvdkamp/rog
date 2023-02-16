use renderer::raytracer::{file_formatting::Error as FileFormatError, working_image::WorkingImage};

#[derive(Debug)]
enum Error {
    File(FileFormatError),
    ArgumentCount(usize),
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        return Err(Error::ArgumentCount(args.len()));
    }

    use Error::File;
    let image = WorkingImage::read_from_file(&args[1], &None).map_err(File)?;

    println!("{image:#?}");

    if args.len() >= 3 {
        image.save_as_rgb(&args[2]);
    }

    Ok(())
}
