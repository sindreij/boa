#![warn(unsafe_op_in_unsafe_fn)]

use core::hash::Hash;
use std::{
    alloc::{dealloc, Layout},
    borrow::Borrow,
    cell::Cell,
    fmt,
    ops::{Deref, Index},
    ptr::{self, NonNull},
    slice::SliceIndex,
};

use crate::builtins::string::is_trimmable_whitespace;
use boa_gc::{unsafe_empty_trace, Finalize, Trace};
use const_utf16::encode as utf16;
use rustc_hash::FxHashSet;

#[macro_export]
macro_rules! js_string {
    ($s:literal) => {
        JsString::from(utf16!($s))
    };
    ($s:expr) => {
        JsString::from($s)
    };
}

const CONSTANTS_ARRAY: [&[u16]; 120] = [
    // Empty string
    utf16!(""),
    // Misc
    utf16!(","),
    utf16!(":"),
    // Generic use
    utf16!("name"),
    utf16!("length"),
    utf16!("arguments"),
    utf16!("prototype"),
    utf16!("constructor"),
    // typeof
    utf16!("null"),
    utf16!("undefined"),
    utf16!("number"),
    utf16!("string"),
    utf16!("symbol"),
    utf16!("bigint"),
    utf16!("object"),
    utf16!("function"),
    // Property descriptor
    utf16!("value"),
    utf16!("get"),
    utf16!("set"),
    utf16!("writable"),
    utf16!("enumerable"),
    utf16!("configurable"),
    // Object object
    utf16!("Object"),
    utf16!("assign"),
    utf16!("create"),
    utf16!("toString"),
    utf16!("valueOf"),
    utf16!("is"),
    utf16!("seal"),
    utf16!("isSealed"),
    utf16!("freeze"),
    utf16!("isFrozen"),
    utf16!("keys"),
    utf16!("values"),
    utf16!("entries"),
    // Function object
    utf16!("Function"),
    utf16!("apply"),
    utf16!("bind"),
    utf16!("call"),
    // Array object
    utf16!("Array"),
    utf16!("from"),
    utf16!("isArray"),
    utf16!("of"),
    utf16!("get [Symbol.species]"),
    utf16!("copyWithin"),
    utf16!("every"),
    utf16!("fill"),
    utf16!("filter"),
    utf16!("find"),
    utf16!("findIndex"),
    utf16!("flat"),
    utf16!("flatMap"),
    utf16!("forEach"),
    utf16!("includes"),
    utf16!("indexOf"),
    utf16!("join"),
    utf16!("map"),
    utf16!("reduce"),
    utf16!("reduceRight"),
    utf16!("reverse"),
    utf16!("shift"),
    utf16!("slice"),
    utf16!("some"),
    utf16!("sort"),
    utf16!("unshift"),
    utf16!("push"),
    utf16!("pop"),
    // String object
    utf16!("String"),
    utf16!("charAt"),
    utf16!("charCodeAt"),
    utf16!("concat"),
    utf16!("endsWith"),
    utf16!("lastIndexOf"),
    utf16!("match"),
    utf16!("matchAll"),
    utf16!("normalize"),
    utf16!("padEnd"),
    utf16!("padStart"),
    utf16!("repeat"),
    utf16!("replace"),
    utf16!("replaceAll"),
    utf16!("search"),
    utf16!("split"),
    utf16!("startsWith"),
    utf16!("substring"),
    utf16!("toLowerString"),
    utf16!("toUpperString"),
    utf16!("trim"),
    utf16!("trimEnd"),
    utf16!("trimStart"),
    // Number object
    utf16!("Number"),
    // Boolean object
    utf16!("Boolean"),
    // RegExp object
    utf16!("RegExp"),
    utf16!("exec"),
    utf16!("test"),
    utf16!("flags"),
    utf16!("index"),
    utf16!("lastIndex"),
    // Symbol object
    utf16!("Symbol"),
    utf16!("for"),
    utf16!("keyFor"),
    utf16!("description"),
    utf16!("[Symbol.toPrimitive]"),
    // Map object
    utf16!("Map"),
    utf16!("clear"),
    utf16!("delete"),
    utf16!("has"),
    utf16!("size"),
    // Set object
    utf16!("Set"),
    // Reflect object
    utf16!("Reflect"),
    // Error objects
    utf16!("Error"),
    utf16!("TypeError"),
    utf16!("RangeError"),
    utf16!("SyntaxError"),
    utf16!("ReferenceError"),
    utf16!("EvalError"),
    utf16!("URIError"),
    utf16!("message"),
    // Date object
    utf16!("Date"),
    utf16!("toJSON"),
];

const MAX_CONSTANT_STRING_LENGTH: usize = {
    let mut max = 0;
    let mut i = 0;
    while i < CONSTANTS_ARRAY.len() {
        let len = CONSTANTS_ARRAY[i].len();
        if len > max {
            max = len;
        }
        i += 1;
    }
    max
};

thread_local! {
    static CONSTANTS: FxHashSet<JsString> = {
        let mut constants = FxHashSet::default();

        for s in CONSTANTS_ARRAY.iter() {
            let s = JsString::from_slice_skip_interning(s);
            constants.insert(s);
        }

        constants
    };
}

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

/// This represents a JavaScript primitive string.
///
/// This is similar to `Rc<str>`. But unlike `Rc<str>` which stores the length
/// on the stack and a pointer to the data (this is also known as fat pointers).
/// The `JsString` length and data is stored on the heap. and just an non-null
/// pointer is kept, so its size is the size of a pointer.
#[derive(Finalize)]
pub struct JsString {
    ptr: NonNull<Inner>,
}

impl JsString {
    fn inner(&self) -> &Inner {
        unsafe { self.ptr.as_ref() }
    }

    unsafe fn from_ptr(ptr: *mut Inner) -> Self {
        Self {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
        }
    }

    unsafe fn allocate(len: usize) -> *mut Inner {
        // We get the layout of the `Inner` type and we extend by the size
        // of the string array.
        let (layout, offset) = Layout::array::<u16>(len)
            .and_then(|arr| Layout::new::<Inner>().extend(arr))
            .map(|(layout, offset)| (layout.pad_to_align(), offset))
            .unwrap();

        unsafe {
            let inner = std::alloc::alloc(layout).cast::<Inner>();

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

    fn from_slice_skip_interning(data: &[u16]) -> Self {
        unsafe {
            let ptr = Self::allocate(data.len());
            ptr::copy_nonoverlapping(data.as_ptr(), (*ptr).data.as_mut_ptr(), data.len());
            Self::from_ptr(ptr)
        }
    }

    pub fn as_slice(&self) -> &[u16] {
        self
    }

    /// Concatenate two string.
    pub fn concat(x: &[u16], y: &[u16]) -> Self {
        Self::concat_array(&[x, y])
    }

    /// Concatenate array of string.
    pub fn concat_array(strings: &[&[u16]]) -> Self {
        let len = strings.iter().fold(0, |len, s| len + s.len());

        let string = unsafe {
            let ptr = Self::allocate(len);
            let data = (*ptr).data.as_mut_ptr();
            let mut offset = 0;
            for string in strings {
                ptr::copy_nonoverlapping(string.as_ptr(), data.add(offset), string.len());
                offset += string.len();
            }
            Self::from_ptr(ptr)
        };

        if string.len() <= MAX_CONSTANT_STRING_LENGTH {
            if let Some(constant) = CONSTANTS.with(|c| c.get(&string).cloned()) {
                return constant;
            }
        }

        string
    }

    /// Decode a `JsString` into a [`String`], replacing invalid data with the replacement character (`U+FFFD`).
    pub fn as_string_lossy(&self) -> String {
        String::from_utf16_lossy(self)
    }

    /// Decode a `JsString` into a [`String`], returning [`Err`] if it contains any invalid data.
    pub fn as_string(&self) -> Result<String, std::string::FromUtf16Error> {
        String::from_utf16(self)
    }

    /// `6.1.4.1 StringIndexOf ( string, searchValue, fromIndex )`
    ///
    /// Note: Instead of returning an isize with `-1` as the "not found" value,
    /// We make use of the type system and return Option<usize> with None as the "not found" value.
    ///
    /// More information:
    ///  - [ECMAScript reference][spec]
    ///
    /// [spec]: https://tc39.es/ecma262/#sec-stringindexof
    pub(crate) fn index_of(&self, search_value: &Self, from_index: usize) -> Option<usize> {
        // 1. Assert: Type(string) is String.
        // 2. Assert: Type(searchValue) is String.
        // 3. Assert: fromIndex is a non-negative integer.

        // 4. Let len be the length of string.
        let len = self.len();

        // 5. If searchValue is the empty String and fromIndex ≤ len, return fromIndex.
        if search_value.is_empty() && from_index <= len {
            return Some(from_index);
        }

        // 6. Let searchLen be the length of searchValue.
        let search_len = search_value.len();

        let range = len.checked_sub(search_len)?;

        // 7. For each integer i starting with fromIndex such that i ≤ len - searchLen, in ascending order, do
        for i in from_index..=range {
            // a. Let candidate be the substring of string from i to i + searchLen.
            let candidate = &self[i..i + search_len];

            // b. If candidate is the same sequence of code units as searchValue, return i.
            if candidate == search_value {
                return Some(i);
            }
        }

        // 8. Return -1.
        None
    }

    pub(crate) fn trim(&self) -> &[u16] {
        let chars = char::decode_utf16(self.iter().copied()).collect::<Vec<_>>();
        let left = chars
            .iter()
            .position(|r| {
                !r.as_ref()
                    .map(|&c| is_trimmable_whitespace(c))
                    .unwrap_or_default()
            })
            .unwrap_or(0);
        let right = chars
            .iter()
            .rposition(|r| {
                !r.as_ref()
                    .map(|&c| is_trimmable_whitespace(c))
                    .unwrap_or_default()
            })
            .unwrap_or_else(|| self.len());

        &self[left..right]
    }

    #[allow(clippy::question_mark)]
    pub(crate) fn string_to_number(&self) -> f64 {
        // TODO: to optimize this we would need to create our own version of `trim_matches` but for utf16
        let string = if let Ok(s) = self.as_string() {
            s
        } else {
            return f64::NAN;
        };

        let string = string.trim_matches(is_trimmable_whitespace);

        // TODO: write our own lexer to match syntax StrDecimalLiteral
        match string {
            "" => 0.0,
            "Infinity" | "+Infinity" => f64::INFINITY,
            "-Infinity" => f64::NEG_INFINITY,
            _ if matches!(
                string
                    .chars()
                    .take(4)
                    .map(|c| char::to_ascii_lowercase(&c))
                    .collect::<String>()
                    .as_str(),
                "inf" | "+inf" | "-inf" | "nan" | "+nan" | "-nan"
            ) =>
            {
                // Prevent fast_float from parsing "inf", "+inf" as Infinity and "-inf" as -Infinity
                f64::NAN
            }
            _ => fast_float::parse(string).unwrap_or(f64::NAN),
        }
    }
}

impl AsRef<[u16]> for JsString {
    fn as_ref(&self) -> &[u16] {
        self
    }
}

impl Borrow<[u16]> for JsString {
    fn borrow(&self) -> &[u16] {
        self
    }
}

impl Clone for JsString {
    #[inline]
    fn clone(&self) -> Self {
        self.inner().refcount.set(self.inner().refcount.get() + 1);

        Self { ptr: self.ptr }
    }
}

impl Default for JsString {
    fn default() -> Self {
        js_string!("")
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

impl Deref for JsString {
    type Target = [u16];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.inner().data.as_ptr(), self.inner().len) }
    }
}

impl fmt::Display for JsString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        String::from_utf16_lossy(self.as_slice()).fmt(f)
    }
}

impl Eq for JsString {}

impl From<&[u16]> for JsString {
    fn from(s: &[u16]) -> Self {
        if s.len() <= MAX_CONSTANT_STRING_LENGTH {
            if let Some(constant) = CONSTANTS.with(|c| c.get(s).cloned()) {
                return constant;
            }
        }
        Self::from_slice_skip_interning(s)
    }
}

impl From<&str> for JsString {
    #[inline]
    fn from(s: &str) -> Self {
        let s = s.encode_utf16().collect::<Vec<_>>();

        Self::from(&s[..])
    }
}

impl<const N: usize> From<&[u16; N]> for JsString {
    #[inline]
    fn from(s: &[u16; N]) -> Self {
        Self::from(&s[..])
    }
}

impl Hash for JsString {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self[..].hash(state);
    }
}

impl<I: SliceIndex<[u16]>> Index<I> for JsString {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&**self, index)
    }
}

impl Ord for JsString {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self[..].cmp(other)
    }
}

impl PartialEq for JsString {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr || self[..] == other[..]
    }
}

impl PartialEq<JsString> for [u16] {
    fn eq(&self, other: &JsString) -> bool {
        self == &**other
    }
}

impl<const N: usize> PartialEq<JsString> for [u16; N] {
    fn eq(&self, other: &JsString) -> bool {
        self[..] == *other
    }
}

impl PartialEq<[u16]> for JsString {
    fn eq(&self, other: &[u16]) -> bool {
        &**self == other
    }
}

impl<const N: usize> PartialEq<[u16; N]> for JsString {
    fn eq(&self, other: &[u16; N]) -> bool {
        *self == other[..]
    }
}

impl PartialOrd for JsString {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self[..].partial_cmp(other)
    }
}

// Safety: [`JsString`] does not contain any objects which require trace,
// so this is safe.
unsafe impl Trace for JsString {
    unsafe_empty_trace!();
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
        let s = js_string!("");
        assert_eq!(*s, "".encode_utf16().collect::<Vec<u16>>())
    }

    #[test]
    fn pointer_size() {
        assert_eq!(size_of::<JsString>(), size_of::<*const ()>());
        assert_eq!(size_of::<Option<JsString>>(), size_of::<*const ()>());
    }

    #[test]
    fn refcount() {
        let x = js_string!("Hello world");
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
        let x = js_string!("Hello");
        let y = x.clone();

        assert_eq!(x.ptr, y.ptr);

        let z = js_string!("Hello");
        assert_ne!(x.ptr, z.ptr);
        assert_ne!(y.ptr, z.ptr);
    }

    #[test]
    fn as_str() {
        const HELLO: &str = "Hello";
        let x = js_string!(HELLO);

        assert_eq!(*x, HELLO.encode_utf16().collect::<Vec<u16>>());
    }

    #[test]
    fn hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        const HELLOWORLD: &[u16] = utf16!("Hello World!");
        let x = js_string!(HELLOWORLD);

        assert_eq!(&*x, HELLOWORLD);

        let mut hasher = DefaultHasher::new();
        HELLOWORLD.hash(&mut hasher);
        let s_hash = hasher.finish();

        let mut hasher = DefaultHasher::new();
        x.hash(&mut hasher);
        let x_hash = hasher.finish();

        assert_eq!(s_hash, x_hash);
    }

    #[test]
    fn concat() {
        const Y: &[u16] = utf16!(", ");
        const W: &[u16] = utf16!("!");

        let x = js_string!("hello");
        let z = js_string!("world");

        let xy = JsString::concat(&x, Y);
        assert_eq!(xy, *utf16!("hello, "));
        assert_eq!(JsString::refcount(&xy), 1);

        let xyz = JsString::concat(&xy, &z);
        assert_eq!(xyz, *utf16!("hello, world"));
        assert_eq!(JsString::refcount(&xyz), 1);

        let xyzw = JsString::concat(&xyz, W);
        assert_eq!(xyzw, *utf16!("hello, world!"));
        assert_eq!(JsString::refcount(&xyzw), 1);
    }
}
