use std::num::ParseIntError;

use renderer::raytracer::{file_formatting::Error as FileFormatError, working_image::WorkingImage};

#[derive(Debug)]
enum Error {
    File(FileFormatError),
    ArgumentCount(usize),
    InvalidIndex(ParseIntError),
    IndexOutOfRange,
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 4 {
        return Err(Error::ArgumentCount(args.len()));
    }

    use Error::{File, IndexOutOfRange, InvalidIndex};
    let image = WorkingImage::read_from_file(&args[1], &None).map_err(File)?;
    let size = image.settings.size;
    let x: usize = args[2].parse().map_err(InvalidIndex)?;
    let y: usize = args[3].parse().map_err(InvalidIndex)?;

    if x >= size.x {
        return Err(IndexOutOfRange);
    }
    if y >= size.y {
        return Err(IndexOutOfRange);
    }

    let spectrum = image.pixels[y * size.x + x].spectrum.data;
    println!("{spectrum:#?}");

    Ok(())
}
