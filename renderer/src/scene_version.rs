use std::{fmt, fs::File, path::PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{digest::DynDigest, Digest, Sha256};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SceneVersion {
    pub filepath: PathBuf,
    pub hash: Hash,
}

impl SceneVersion {
    pub fn new(filepath: PathBuf) -> Result<Self, std::io::Error> {
        let mut file = File::open(&filepath)?;
        let mut hasher = Sha256::new();

        let _ = std::io::copy(&mut file, &mut hasher)?;

        assert_eq!(hasher.output_size(), 32);
        let mut hash_buf = [0; 32];
        hash_buf.copy_from_slice(&hasher.finalize());
        let hash = Hash(hash_buf);

        Ok(SceneVersion { filepath, hash })
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct Hash([u8; 32]);

impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("Hash {{ {:x?} }}", self.0))
    }
}
