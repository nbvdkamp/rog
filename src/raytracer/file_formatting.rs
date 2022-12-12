use std::{
    fmt,
    io::{Read, Write},
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

#[derive(Debug)]
pub enum Error {
    IO(std::io::Error),
    Serde(serde_json::Error),
    TagMismatch { expected: Vec<u8>, actual: Vec<u8> },
    FormatVersionMismatch { current: u32, file: u32 },
    SceneMismatch,
    ResolutionMismatch { current: usize, file: usize, name: String },
    SamplesMismatch { current: usize, file: usize, name: String },
    SectionSizeMismatch { expected: usize, file: usize, name: String },
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
                Error::FormatVersionMismatch {current, file} =>
                    format!("current format version ({current}) does not match the version of the file ({file})"),
                Error::SceneMismatch => "hash of provided scene file does not match that of the scene used to render itermediate image".to_string(),
                Error::ResolutionMismatch{current, file, name} =>
                    format!("current {name} resolution ({current}) does not match the resolution used when rendering the intermediate image ({file})"),
                Error::SamplesMismatch {current, file, name} =>
                    format!("current {name} sample count ({current}) does not match the count used when rendering the intermediate image ({file})"),
                Error::SectionSizeMismatch  {expected, file, name}  => format!("size of {name} binary data section = {file} does not match the expected value of {expected}"),
            }
        )
    }
}

pub struct SectionHeader {
    pub size: u64,
    pub tag: String,
}

impl SectionHeader {
    pub fn new(size: usize, tag: &str) -> Self {
        SectionHeader {
            size: size as u64,
            tag: tag.to_string(),
        }
    }

    pub fn to_writer<W>(self, writer: &mut W) -> Result<(), std::io::Error>
    where
        W: Write,
    {
        write!(writer, "{}", self.tag)?;
        writer.write_u64::<LittleEndian>(self.size)?;
        Ok(())
    }

    pub fn from_reader<R: Read>(
        reader: &mut R,
        expected_tag: &str,
        expected_size: Option<usize>,
    ) -> Result<Self, Error> {
        use Error::IO;
        let mut tag = vec![0; expected_tag.len()];
        reader.read_exact(&mut tag).map_err(IO)?;

        if tag == expected_tag.as_bytes() {
            let size = reader.read_u64::<LittleEndian>().map_err(IO)?;

            if let Some(expected) = expected_size {
                if size as usize != expected {
                    return Err(Error::SectionSizeMismatch {
                        expected,
                        file: size as usize,
                        name: expected_tag.to_string(),
                    });
                }
            }

            Ok(Self {
                size,
                tag: expected_tag.to_string(),
            })
        } else {
            Err(Error::TagMismatch {
                expected: expected_tag.as_bytes().to_vec(),
                actual: tag.to_vec(),
            })
        }
    }
}
