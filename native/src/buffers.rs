//! NumPy arrays as Rust slices, through the buffer protocol.
//!
//! Python keeps the arrays alive and unchanged while a kernel runs (the `PyBuffer`s
//! taken here are held until it returns); the kernels read and write them through
//! these raw slices with the GIL released. Every array must be C-contiguous with the
//! element type the kernel expects, outputs writable, and no output may overlap another
//! array: anything else is a `ValueError`, never undefined behavior.

use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A read-only view of an array's elements (usable without the GIL).
pub struct In<T> {
    ptr: *const T,
    len: usize,
}

/// A writable view of an array's elements (usable without the GIL).
pub struct Out<T> {
    ptr: *mut T,
    len: usize,
}

// The views are only used while the arrays are pinned by their `PyBuffer`s, and kernels
// share them read-only (inputs) or write disjoint elements (outputs) across threads.
unsafe impl<T: Sync> Send for In<T> {}
unsafe impl<T: Sync> Sync for In<T> {}
unsafe impl<T: Send> Send for Out<T> {}
unsafe impl<T: Send> Sync for Out<T> {}

impl<T> In<T> {
    /// The elements. Safety: the array's `PyBuffer` is alive.
    pub unsafe fn slice(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        std::slice::from_raw_parts(self.ptr, self.len)
    }
}

impl<T> Out<T> {
    /// The elements. Safety: the array's `PyBuffer` is alive, and no other slice of these
    /// elements is in use (each output is taken once, or split between threads).
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice(&self) -> &mut [T] {
        if self.len == 0 {
            return &mut [];
        }
        std::slice::from_raw_parts_mut(self.ptr, self.len)
    }

    /// Element `i`. Safety: as `slice`, and `i < len`.
    pub unsafe fn set(&self, i: usize, value: T) {
        assert!(i < self.len);
        *self.ptr.add(i) = value;
    }
}

/// The arrays of one call: their buffers (pinned until the call returns) and byte ranges.
pub struct Arrays {
    ranges: Vec<(usize, usize, String, bool)>, // (start, end, name, written)
    held: Vec<Box<dyn std::any::Any>>,
}

impl Arrays {
    pub fn new() -> Self {
        Arrays {
            ranges: Vec::new(),
            held: Vec::new(),
        }
    }

    fn take<T: Element + 'static>(
        &mut self,
        obj: &Bound<'_, PyAny>,
        name: &str,
        len: Option<usize>,
        written: bool,
    ) -> PyResult<(*mut T, usize)> {
        let buffer = PyBuffer::<T>::get(obj).map_err(|e| {
            PyValueError::new_err(format!("{name}: not an array of the expected type ({e})"))
        })?;
        if !buffer.is_c_contiguous() {
            return Err(PyValueError::new_err(format!("{name}: not C-contiguous")));
        }
        if written && buffer.readonly() {
            return Err(PyValueError::new_err(format!("{name}: read-only")));
        }
        let count = buffer.item_count();
        if let Some(expected) = len {
            if count != expected {
                return Err(PyValueError::new_err(format!(
                    "{name}: {count} elements, expected {expected}"
                )));
            }
        }
        let ptr = buffer.buf_ptr() as *mut T;
        let start = ptr as usize;
        if count > 0 {
            let end = start + count * std::mem::size_of::<T>();
            self.ranges.push((start, end, name.to_string(), written));
        }
        self.held.push(Box::new(buffer));
        Ok((ptr, count))
    }

    /// An input array (``len`` elements, if given).
    pub fn input<T: Element + 'static>(
        &mut self,
        obj: &Bound<'_, PyAny>,
        name: &str,
        len: Option<usize>,
    ) -> PyResult<In<T>> {
        let (ptr, len) = self.take::<T>(obj, name, len, false)?;
        Ok(In { ptr, len })
    }

    /// An output array (``len`` elements, if given).
    pub fn output<T: Element + 'static>(
        &mut self,
        obj: &Bound<'_, PyAny>,
        name: &str,
        len: Option<usize>,
    ) -> PyResult<Out<T>> {
        let (ptr, len) = self.take::<T>(obj, name, len, true)?;
        Ok(Out { ptr, len })
    }

    /// Fails if an output shares memory with any other array of the call.
    pub fn check_overlaps(&self) -> PyResult<()> {
        for (i, a) in self.ranges.iter().enumerate() {
            for b in self.ranges.iter().skip(i + 1) {
                if (a.3 || b.3) && a.0 < b.1 && b.0 < a.1 {
                    return Err(PyValueError::new_err(format!(
                        "{} and {} share memory (an output must be an array of its own)",
                        a.2, b.2
                    )));
                }
            }
        }
        Ok(())
    }
}
