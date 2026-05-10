//! Error type for `oxideav-mesh3d`.
//!
//! Under the default `registry` feature this is a re-export of
//! [`oxideav_core::Error`] / [`oxideav_core::Result`] so format
//! crates and downstream consumers share one error vocabulary across
//! the framework. With `--no-default-features` we fall back to a
//! crate-local enum that mirrors the relevant variants — enough for
//! decoders/encoders to report invalid input or unsupported features
//! without dragging in `oxideav-core`.

#[cfg(feature = "registry")]
pub use oxideav_core::{Error, Result};

#[cfg(not(feature = "registry"))]
mod standalone {
    use std::fmt;

    /// Crate-local error type used when the `registry` feature is off.
    #[derive(Debug)]
    pub enum Error {
        /// Bytes don't match what the decoder expected.
        InvalidData(String),
        /// Spec construct that this crate or format doesn't yet handle.
        Unsupported(String),
        /// I/O failure while reading or writing.
        Io(std::io::Error),
        /// Catch-all for ad-hoc error messages.
        Other(String),
    }

    impl Error {
        pub fn invalid(msg: impl Into<String>) -> Self {
            Self::InvalidData(msg.into())
        }

        pub fn unsupported(msg: impl Into<String>) -> Self {
            Self::Unsupported(msg.into())
        }

        pub fn other(msg: impl Into<String>) -> Self {
            Self::Other(msg.into())
        }
    }

    impl fmt::Display for Error {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::InvalidData(s) => write!(f, "invalid data: {s}"),
                Self::Unsupported(s) => write!(f, "unsupported: {s}"),
                Self::Io(e) => write!(f, "I/O error: {e}"),
                Self::Other(s) => write!(f, "{s}"),
            }
        }
    }

    impl std::error::Error for Error {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            match self {
                Self::Io(e) => Some(e),
                _ => None,
            }
        }
    }

    impl From<std::io::Error> for Error {
        fn from(e: std::io::Error) -> Self {
            Self::Io(e)
        }
    }

    pub type Result<T> = std::result::Result<T, Error>;
}

#[cfg(not(feature = "registry"))]
pub use standalone::{Error, Result};
