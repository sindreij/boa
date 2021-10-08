#![warn(unsafe_op_in_unsafe_fn)]

use core::hash::Hash;
use std::{
    alloc::{dealloc, Layout},
    borrow::Borrow,
    cell::Cell,
    fmt,
    marker::PhantomData,
    ops::Deref,
    ptr::{self, NonNull},
};

use boa_gc::{unsafe_empty_trace, Finalize, Trace};
use const_utf16::encode as utf16;

/// The inner representation of a [`JsString`].
#[repr(C)]
struct Inner {
    /// The utf16 length.
    len: usize,

    /// The number of references to the string.
    ///
    /// When this reaches `0` the string is deallocated.
    refcount: Cell<usize>,

    /// An empty array which is used to get the offset of string data.
    data: [u16; 0],
}

#[derive(Finalize)]
pub struct JsString {
    ptr: NonNull<Inner>,
    phantom: PhantomData<Inner>,
}

impl JsString {
    #[inline(always)]
    fn inner(&self) -> &Inner {
        unsafe { self.ptr.as_ref() }
    }

    fn from_inner(ptr: NonNull<Inner>) -> Self {
        Self {
            ptr,
            phantom: PhantomData,
        }
    }

    unsafe fn from_ptr(ptr: *mut Inner) -> Self {
        Self::from_inner(unsafe { NonNull::new_unchecked(ptr) })
    }

    unsafe fn allocate(len: usize) -> *mut Inner {
        // We get the layout of the `Inner` type and we extend by the size
        // of the string array.
        let inner_layout = Layout::new::<Inner>();
        let (layout, offset) = inner_layout
            .extend(Layout::array::<u16>(len).unwrap())
            .unwrap();

        unsafe {
            let inner = std::alloc::alloc(layout) as *mut Inner;

            // Write the first part, the Inner.
            inner.write(Inner {
                len,
                refcount: Cell::new(1),
                data: [0; 0],
            });

            // Get offset into the string data.
            let data = (*inner).data.as_mut_ptr();

            debug_assert!(ptr::eq(inner.cast::<u8>().add(offset).cast(), data));

            inner
        }
    }

    pub fn from_slice(data: &[u16]) -> Self {
        unsafe {
            let ptr = Self::allocate(data.len());
            ptr::copy_nonoverlapping(data.as_ptr(), (*ptr).data.as_mut_ptr(), data.len());
            Self::from_ptr(ptr)
        }
    }

    pub fn as_slice(&self) -> &[u16] {
        unsafe { std::slice::from_raw_parts(self.inner().data.as_ptr(), self.inner().len) }
    }
}

impl AsRef<[u16]> for JsString {
    fn as_ref(&self) -> &[u16] {
        self.as_slice()
    }
}

impl Borrow<[u16]> for JsString {
    fn borrow(&self) -> &[u16] {
        self.as_slice()
    }
}

// Safety: [`JsString`] does not contain any objects which require trace,
// so this is safe.
unsafe impl Trace for JsString {
    unsafe_empty_trace!();
}

impl Clone for JsString {
    #[inline]
    fn clone(&self) -> Self {
        self.inner().refcount.set(self.inner().refcount.get() + 1);

        JsString {
            ptr: self.ptr,
            phantom: PhantomData,
        }
    }
}

impl Default for JsString {
    fn default() -> Self {
        Self::from_slice(utf16!(""))
    }
}

impl Drop for JsString {
    #[inline]
    fn drop(&mut self) {
        self.inner().refcount.set(self.inner().refcount.get() - 1);
        if self.inner().refcount.get() == 0 {
            // Safety: If refcount is 0 and we call drop, that means this is the last
            // JsString which points to this memory allocation, so deallocating it is safe.
            unsafe {
                dealloc(
                    self.ptr.as_ptr().cast(),
                    Layout::for_value(self.ptr.as_ref()),
                );
            }
        }
    }
}

impl Deref for JsString {
    type Target = [u16];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl PartialEq for JsString {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr || self.as_slice() == other.as_slice()
    }
}

impl Eq for JsString {}

impl PartialOrd for JsString {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl Ord for JsString {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl PartialEq<[u16]> for JsString {
    fn eq(&self, other: &[u16]) -> bool {
        self.as_slice() == other
    }
}

impl PartialEq<JsString> for [u16] {
    fn eq(&self, other: &JsString) -> bool {
        self == other.as_slice()
    }
}

impl Hash for JsString {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl From<&[u16]> for JsString {
    #[inline]
    fn from(s: &[u16]) -> Self {
        Self::from_slice(s)
    }
}

impl fmt::Debug for JsString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        std::char::decode_utf16(self.as_slice().to_owned())
            .map(|r| {
                r.map_or_else(
                    |err| format!("<0x{:04x}>", err.unpaired_surrogate()),
                    String::from,
                )
            })
            .collect::<String>()
            .fmt(f)
    }
}

impl fmt::Display for JsString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        String::from_utf16_lossy(self.as_slice()).fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::JsString;
    use const_utf16::encode as utf16;
    use std::mem::size_of;

    impl JsString {
        /// Gets the number of `JsString`s which point to this allocation.
        #[inline]
        fn refcount(&self) -> usize {
            self.inner().refcount.get()
        }
    }

    #[test]
    fn empty() {
        let s = JsString::from_slice(utf16!(""));
        assert_eq!(*s, "".encode_utf16().collect::<Vec<u16>>())
    }

    #[test]
    fn pointer_size() {
        assert_eq!(size_of::<JsString>(), size_of::<*const u8>());
        assert_eq!(size_of::<Option<JsString>>(), size_of::<*const u8>());
    }

    #[test]
    fn refcount() {
        let x = JsString::from_slice(utf16!("Hello world"));
        assert_eq!(JsString::refcount(&x), 1);

        {
            let y = x.clone();
            assert_eq!(JsString::refcount(&x), 2);
            assert_eq!(JsString::refcount(&y), 2);

            {
                let z = y.clone();
                assert_eq!(JsString::refcount(&x), 3);
                assert_eq!(JsString::refcount(&y), 3);
                assert_eq!(JsString::refcount(&z), 3);
            }

            assert_eq!(JsString::refcount(&x), 2);
            assert_eq!(JsString::refcount(&y), 2);
        }

        assert_eq!(JsString::refcount(&x), 1);
    }

    #[test]
    fn ptr_eq() {
        let x = JsString::from_slice(utf16!("Hello"));
        let y = x.clone();

        assert_eq!(x.ptr, y.ptr);

        let z = JsString::from_slice(utf16!("Hello"));
        assert_ne!(x.ptr, z.ptr);
        assert_ne!(y.ptr, z.ptr);
    }

    #[test]
    fn as_str() {
        const HELLO: &str = "Hello";
        let x = JsString::from_slice(utf16!(HELLO));

        assert_eq!(*x, HELLO.encode_utf16().collect::<Vec<u16>>());
    }

    #[test]
    fn hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        const HELLOWORLD: &[u16] = utf16!("Hello World!");
        let x = JsString::from_slice(HELLOWORLD);

        assert_eq!(&*x, HELLOWORLD);

        let mut hasher = DefaultHasher::new();
        HELLOWORLD.hash(&mut hasher);
        let s_hash = hasher.finish();

        let mut hasher = DefaultHasher::new();
        x.hash(&mut hasher);
        let x_hash = hasher.finish();

        assert_eq!(s_hash, x_hash);
    }
}
