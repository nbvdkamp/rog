use rayon::prelude::*;

use std::{
    fmt,
    fs::File,
    io::{Read, Write},
    path::Path,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};

use crate::{
    color::RGBf32,
    render_settings::ImageSettings,
    scene_version::SceneVersion,
    spectrum::Spectrumf32,
    util::save_image,
};

const FORMAT_VERSION: u32 = 1;
const TAG: &str = "SPECTRAL IMG";
const JSON_TAG: &str = "JSON";
const PIXELS_TAG: &str = "PIX ";
static_assertions::const_assert_eq!(TAG.len() % 4, 0);
static_assertions::const_assert_eq!(JSON_TAG.len() % 4, 0);
static_assertions::const_assert_eq!(PIXELS_TAG.len() % 4, 0);

#[derive(Serialize, Deserialize)]
pub struct WorkingImage {
    pub settings: ImageSettings,
    #[serde(skip)]
    pub pixels: Vec<Pixel>,
}

#[derive(Clone)]
pub struct Pixel {
    pub spectrum: Spectrumf32,
    pub samples: u32,
}

impl WorkingImage {
    pub fn new(settings: ImageSettings) -> Self {
        WorkingImage {
            pixels: vec![
                Pixel {
                    spectrum: Spectrumf32::constant(0.0),
                    samples: 0
                };
                settings.width * settings.height
            ],
            settings,
        }
    }

    pub fn save_as_rgb<P>(&self, path: P)
    where
        P: AsRef<Path>,
    {
        let buffer = self
            .pixels
            .par_iter()
            .map(|pixel| {
                if pixel.spectrum.data.iter().any(|v| v.is_nan()) {
                    return RGBf32::new(1.0, 0.5, 0.0);
                }
                let spectrum = pixel.spectrum * Spectrumf32::RESOLUTION as f32 / pixel.samples as f32;

                let c = spectrum.to_srgb().linear_to_srgb();
                if c.has_nan_component() {
                    RGBf32::new(0.0, 1.0, 0.0)
                } else {
                    c
                }
            })
            .collect::<Vec<_>>();

        save_image(&buffer, self.settings.size(), path)
    }

    pub fn write_to_file<P>(&self, path: P) -> Result<(), Error>
    where
        P: AsRef<Path>,
    {
        use Error::{Serde, IO};
        let mut file = File::create(path).map_err(IO)?;

        let json = serde_json::to_string(self).map_err(Serde)?;
        let bytes_to_pad = align_to(json.len(), 4) - json.len();
        let json = format!("{}{}", json, " ".repeat(bytes_to_pad));

        let pixels_size = self.pixels.len() * std::mem::size_of::<Pixel>();

        let size = FileHeader::SIZE + JsonSectionHeader::SIZE + json.len() + PixelsSectionHeader::SIZE + pixels_size;

        let header = FileHeader {
            format_version: FORMAT_VERSION,
            total_size: size as u64,
        };

        let json_header = JsonSectionHeader {
            size: json.len() as u64,
        };

        let pixels_header = PixelsSectionHeader {
            size: pixels_size as u64,
            spectrum_resolution: Spectrumf32::RESOLUTION as u64,
        };

        header.to_writer(&mut file).map_err(IO)?;
        json_header.to_writer(&mut file).map_err(IO)?;

        file.write_all(json.as_bytes()).map_err(IO)?;

        pixels_header.to_writer(&mut file).map_err(IO)?;

        let pixels_buffer: &[u8] =
            unsafe { std::slice::from_raw_parts(self.pixels.as_ptr() as *const u8, pixels_size) };
        file.write_all(pixels_buffer).map_err(IO)?;

        Ok(())
    }

    pub fn read_from_file<P>(path: P, expected_scene_version: &SceneVersion) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        use Error::{Serde, IO};
        let mut file = File::open(path).map_err(IO)?;

        let _header = FileHeader::from_reader(&mut file)?;

        let json_header = JsonSectionHeader::from_reader(&mut file)?;

        //Read json
        let mut json_buffer = vec![0; json_header.size as usize];
        file.read_exact(&mut json_buffer).map_err(IO)?;

        let mut image = serde_json::from_slice::<WorkingImage>(&json_buffer).map_err(Serde)?;

        if image
            .settings
            .scene_version
            .as_ref()
            .map_or(true, |v| v.hash != expected_scene_version.hash)
        {
            return Err(Error::SceneMismatch);
        }

        let expected_pixel_count = image.settings.width * image.settings.height;

        let pixels_header = PixelsSectionHeader::from_reader(&mut file)?;

        let spectrum_resolution = pixels_header.spectrum_resolution as usize;

        if spectrum_resolution != Spectrumf32::RESOLUTION {
            return Err(Error::SpectrumResolutionMismatch(spectrum_resolution));
        }

        let expected_pixels_size = expected_pixel_count * std::mem::size_of::<Pixel>();

        if expected_pixels_size != pixels_header.size as usize {
            return Err(Error::PixelSectionSizeMismatch);
        }

        image.pixels = vec![
            Pixel {
                spectrum: Spectrumf32::constant(0.0),
                samples: 0
            };
            expected_pixel_count
        ];

        unsafe {
            file.read_exact(std::slice::from_raw_parts_mut(
                image.pixels.as_ptr() as *mut u8,
                expected_pixels_size,
            ))
            .map_err(IO)?;
        }

        Ok(image)
    }
}

pub enum Error {
    IO(std::io::Error),
    Serde(serde_json::Error),
    TagMismatch { expected: Vec<u8>, actual: Vec<u8> },
    FormatVersionMismatch(u32),
    SceneMismatch,
    SpectrumResolutionMismatch(usize),
    PixelSectionSizeMismatch,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Error::IO(e) => return e.fmt(f),
                Error::Serde(e) => return e.fmt(f),
                Error::TagMismatch { expected, actual } => format!("wrong binary tag found, expected {expected:?} but got {actual:?}"),
                Error::FormatVersionMismatch(version) =>
                    format!("current format version ({}) does not match the version of the file ({version})", FORMAT_VERSION),
                Error::SceneMismatch => "hash of provided scene file does not match that of the scene used to render itermediate image".to_string(),
                Error::SpectrumResolutionMismatch(resolution) =>
                    format!("current spectrum resolution ({}) does not match the resolution used when rendering the intermediate image ({resolution})", Spectrumf32::RESOLUTION),
                Error::PixelSectionSizeMismatch => "size of pixels section does not match the expected value".to_string(),
            }
        )
    }
}

struct FileHeader {
    pub format_version: u32,
    pub total_size: u64,
}

impl FileHeader {
    const SIZE: usize = TAG.len() + std::mem::size_of::<Self>();

    pub fn to_writer<W>(self, writer: &mut W) -> Result<(), std::io::Error>
    where
        W: Write,
    {
        write!(writer, "{}", TAG)?;
        writer.write_u32::<LittleEndian>(self.format_version)?;
        writer.write_u64::<LittleEndian>(self.total_size)?;
        Ok(())
    }

    pub fn from_reader<R: Read>(reader: &mut R) -> Result<Self, Error> {
        use Error::IO;
        let mut tag = [0; TAG.len()];
        reader.read_exact(&mut tag).map_err(IO)?;

        if tag == TAG.as_bytes() {
            let format_version = reader.read_u32::<LittleEndian>().map_err(IO)?;
            if format_version != FORMAT_VERSION {
                return Err(Error::FormatVersionMismatch(format_version));
            }

            let total_size = reader.read_u64::<LittleEndian>().map_err(IO)?;
            Ok(Self {
                format_version,
                total_size,
            })
        } else {
            Err(Error::TagMismatch {
                expected: TAG.as_bytes().to_vec(),
                actual: tag.to_vec(),
            })
        }
    }
}

struct JsonSectionHeader {
    pub size: u64,
}

impl JsonSectionHeader {
    const SIZE: usize = JSON_TAG.len() + std::mem::size_of::<Self>();

    pub fn to_writer<W>(self, writer: &mut W) -> Result<(), std::io::Error>
    where
        W: Write,
    {
        write!(writer, "{}", JSON_TAG)?;
        writer.write_u64::<LittleEndian>(self.size)?;
        Ok(())
    }

    pub fn from_reader<R: Read>(reader: &mut R) -> Result<Self, Error> {
        use Error::IO;
        let mut tag = [0; JSON_TAG.len()];
        reader.read_exact(&mut tag).map_err(IO)?;

        if tag == JSON_TAG.as_bytes() {
            let size = reader.read_u64::<LittleEndian>().map_err(IO)?;
            Ok(Self { size })
        } else {
            Err(Error::TagMismatch {
                expected: JSON_TAG.as_bytes().to_vec(),
                actual: tag.to_vec(),
            })
        }
    }
}

struct PixelsSectionHeader {
    pub size: u64,
    pub spectrum_resolution: u64,
}

impl PixelsSectionHeader {
    const SIZE: usize = PIXELS_TAG.len() + std::mem::size_of::<Self>();

    pub fn to_writer<W>(self, writer: &mut W) -> Result<(), std::io::Error>
    where
        W: Write,
    {
        write!(writer, "{}", PIXELS_TAG)?;
        writer.write_u64::<LittleEndian>(self.size)?;
        writer.write_u64::<LittleEndian>(self.spectrum_resolution)?;
        Ok(())
    }

    pub fn from_reader<R: Read>(reader: &mut R) -> Result<Self, Error> {
        use Error::IO;
        let mut tag = [0; PIXELS_TAG.len()];
        reader.read_exact(&mut tag).map_err(IO)?;

        if tag == PIXELS_TAG.as_bytes() {
            let size = reader.read_u64::<LittleEndian>().map_err(IO)?;
            let spectrum_resolution = reader.read_u64::<LittleEndian>().map_err(IO)?;
            Ok(Self {
                size,
                spectrum_resolution,
            })
        } else {
            Err(Error::TagMismatch {
                expected: PIXELS_TAG.as_bytes().to_vec(),
                actual: tag.to_vec(),
            })
        }
    }
}

fn align_to(x: usize, alignment: usize) -> usize {
    assert!(alignment.count_ones() == 1);

    let a = alignment - 1;
    (x + a) & !a
}
