#![warn(unsafe_op_in_unsafe_fn)]

//! A UTF-16–encoded, reference counted, immutable string.
//!
//! This module contains the [`JsString`] type, the [`js_string`] macro and
//! the [`utf16`] macro.
//!
//! The [`js_string`] macro is almost always used when you need to create a new
//! [`JsString`], and the [`utf16`] macro is used for const conversions of
//! string literals to UTF-16.

use crate::{builtins::string::is_trimmable_whitespace, JsBigInt};
use boa_gc::{unsafe_empty_trace, Finalize, Trace};
pub use const_utf16::encode as utf16;
use rustc_hash::FxHashSet;
use std::{
    alloc::{alloc, dealloc, Layout},
    borrow::Borrow,
    cell::Cell,
    fmt,
    hash::{Hash, Hasher},
    ops::{Deref, Index},
    ptr::{self, NonNull},
    slice::SliceIndex,
};

/// Utility macro to create a `JsString`.
///
/// # Examples
///
/// You can call the macro without arguments to create an empty [`JsString`]:
///
/// ```
/// use boa_engine::js_string;
/// use boa_engine::string::utf16;
///
/// let empty_str = js_string!();
/// assert!(empty_str.is_empty());
/// ```
///
///
/// You can create a [`JsString`] from a string literal, which completely skips
/// the runtime conversion from [`&str`] to [`&\[u16\]`]:
///
/// ```
/// # use boa_engine::js_string;
/// # use boa_engine::string::utf16;
/// let hw = js_string!("Hello, world!");
/// assert_eq!(&hw, utf16!("Hello, world!"));
/// ```
///
/// Any [`\[u16\]`] slice is a valid [`JsString`], including unpaired surrogates:
///
/// ```
/// # use boa_engine::js_string;
/// let array = js_string!(&[0xD8AFu16, 0x00A0, 0xD8FF, 0x00F0]);
/// ```
///
/// You can also pass it any number of [`&\[u16\]`] as arguments to create a new
/// [`JsString`] with the concatenation of every slice:
///
/// ```
/// # use boa_engine::js_string;
/// # use boa_engine::string::utf16;
/// const NAME: &[u16]  = utf16!("human! ");
/// let greeting = js_string!("Hello, ");
/// let msg = js_string!(&greeting, &NAME, utf16!("Nice to meet you!"));
///
/// assert_eq!(&msg, utf16!("Hello, human! Nice to meet you!"));
/// ```
#[macro_export]
macro_rules! js_string {
    () => {
        $crate::JsString::default()
    };
    ($s:literal) => {
        $crate::JsString::from($crate::string::utf16!($s))
    };
    ($s:expr) => {
        $crate::JsString::from($s)
    };
    ( $x:expr, $y:expr ) => {
        $crate::JsString::concat($x, $y)
    };
    ( $( $s:expr ),+ ) => {
        $crate::JsString::concat_array(&[ $( $s ),+ ])
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

/// Represents a codepoint within a [`JsString`], which could be a valid
/// Unicode code point, or an unpaired surrogate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CodePoint {
    Unicode(char),
    UnpairedSurrogate(u16),
}

impl CodePoint {
    /// Get the number of UTF-16 code units needed to encode
    /// this code point.
    pub(crate) fn code_unit_count(self) -> usize {
        match self {
            Self::Unicode(c) => c.len_utf16(),
            Self::UnpairedSurrogate(_) => 1,
        }
    }

    /// Convert the code point to its [`u32`] representation.
    pub(crate) fn as_u32(self) -> u32 {
        match self {
            Self::Unicode(c) => c as u32,
            Self::UnpairedSurrogate(surr) => u32::from(surr),
        }
    }

    /// If the code point represents a valid Unicode code point, return
    /// its [`char`] representation.
    pub(crate) fn as_char(self) -> Option<char> {
        match self {
            Self::Unicode(c) => Some(c),
            Self::UnpairedSurrogate(_) => None,
        }
    }

    /// Encodes this code point as UTF-16 into the provided u16 buffer, and then
    /// returns the subslice of the buffer that contains the encoded character.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not large enough. A buffer of length 2 is large
    /// enough to encode any code point.
    pub(crate) fn encode_utf16(self, dst: &mut [u16]) -> &[u16] {
        match self {
            CodePoint::Unicode(c) => c.encode_utf16(dst),
            CodePoint::UnpairedSurrogate(surr) => {
                dst[0] = surr;
                &dst[0..=0]
            }
        }
    }
}

/// The inner representation of a [`JsString`].
#[repr(C)]
struct Inner {
    /// The UTF-16 length.
    len: usize,

    /// The number of references to the string.
    ///
    /// When this reaches `0` the string is deallocated.
    refcount: Cell<usize>,

    /// An empty array which is used to get the offset of string data.
    data: [u16; 0],
}

/// A UTF-16–encoded, reference counted, immutable string.
///
/// This is pretty similar to a [`Rc<\[u16\]>`], but without the length metadata
/// associated with the `Rc` fat pointer. Instead, the length of every string
/// is stored on the heap, along with its reference counter and its data.
///
/// # Deref
///
/// `JsString` implements <code>[Deref]<Target = [\[u16\]]></code>, inheriting
/// all of [`\[u16\]`]'s methods.
#[derive(Finalize)]
pub struct JsString {
    ptr: NonNull<Inner>,
}

impl JsString {
    /// Obtain the inner reference to the
    fn inner(&self) -> &Inner {
        // SAFETY: Initialization should leave `ptr` in a correct state, pointing
        //         to a valid `Inner`.
        unsafe { self.ptr.as_ref() }
    }

    // This is marked as safe because it is always valid to call this function to request
    // any number of `u16`, since this function ought to fail on an OOM error.
    /// Allocate a new `Inner` with an internal capacity of `str_len` chars.
    fn allocate_inner(str_len: usize) -> NonNull<Inner> {
        // We get the layout of the `Inner` type and we extend by the size
        // of the string array.
        let (layout, offset) = Layout::array::<u16>(str_len)
            .and_then(|arr| Layout::new::<Inner>().extend(arr))
            .map(|(layout, offset)| (layout.pad_to_align(), offset))
            .expect("failed to create memory layout");

        // SAFETY:
        // - The layout size of `Inner` is never zero, since it has to store
        // the length of the string and the reference count.
        let inner = unsafe { alloc(layout).cast::<Inner>() };

        // We need to verify that the pointer returned by `alloc` is not null, otherwise
        // we should abort, since an allocation error is pretty unrecoverable for us
        // right now.
        let inner = NonNull::new(inner).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));

        // SAFETY:
        // `NonNull` verified for us that the pointer returned by `alloc` is valid,
        // meaning we can write to its pointed memory.
        unsafe {
            // Write the first part, the Inner.
            inner.as_ptr().write(Inner {
                len: str_len,
                refcount: Cell::new(1),
                data: [0; 0],
            });
        }

        debug_assert!({
            let inner = inner.as_ptr();
            // SAFETY:
            // - `inner` must be a valid pointer, since it comes from a `NonNull`,
            // meaning we can safely dereference it to `Inner`.
            // - `offset` should point us to the beginning of the array,
            // and since we requested an `Inner` layout with a trailing
            // `[u16; str_len]`, the memory of the array must be in the `usize`
            // range for the allocation to succeed.
            unsafe {
                let data = (*inner).data.as_ptr();
                ptr::eq(inner.cast::<u8>().add(offset).cast(), data)
            }
        });

        inner
    }

    /// Create a new `JsString` from `data`, without checking if the string is
    /// in the interner.
    fn from_slice_skip_interning(data: &[u16]) -> Self {
        let count = data.len();
        let ptr = Self::allocate_inner(count);
        // SAFETY:
        // - We read `count = data.len()` elements from `data`, which is within the bounds
        //   of the slice.
        // - `allocate_inner` must allocate at least `count` elements, which
        //   allows us to safely write at least `count` elements.
        // - `allocate_inner` should already take care of the alignment of `ptr`,
        //   and `data` must be aligned to be a valid slice.
        // - `allocate_inner` must return a valid pointer to newly allocated memory,
        //    meaning `ptr` and `data` should never overlap.
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), (*ptr.as_ptr()).data.as_mut_ptr(), count);
        }
        Self { ptr }
    }

    /// Obtain the underlying `&[u16]` slice of a `JsString`
    pub fn as_slice(&self) -> &[u16] {
        self
    }

    /// Create a new `JsString` from the concatenation of `x` and `y`.
    pub fn concat(x: &[u16], y: &[u16]) -> Self {
        Self::concat_array(&[x, y])
    }

    /// Create a new `JsString` from the concatenation of every element of
    /// `strings`.
    pub fn concat_array(strings: &[&[u16]]) -> Self {
        let full_count = strings.iter().fold(0, |len, s| len + s.len());

        let ptr = Self::allocate_inner(full_count);

        let string = {
            // SAFETY:
            // - `ptr` being a `NonNull` ensures that a dereference of its underlying
            // pointer is always valid.
            let mut data = unsafe { (*ptr.as_ptr()).data.as_mut_ptr() };
            for string in strings {
                let count = string.len();
                // SAFETY:
                // - The sum of all `count` for each `string` equals `full_count`,
                // and since we're iteratively writing each of them to `data`,
                // `copy_non_overlapping` always stays in-bounds for `count` reads
                // of each string and `full_count` writes to `data`.
                //
                // - Each `string` must be properly aligned to be a valid
                // slice, and `data` must be properly aligned by `allocate_inner`.
                //
                // - `allocate_inner` must return a valid pointer to newly allocated memory,
                //    meaning `ptr` and all `string`s should never overlap.
                unsafe {
                    ptr::copy_nonoverlapping(string.as_ptr(), data, count);
                    data = data.add(count);
                }
            }
            Self { ptr }
        };

        if string.len() <= MAX_CONSTANT_STRING_LENGTH {
            if let Some(constant) = CONSTANTS.with(|c| c.get(&string).cloned()) {
                return constant;
            }
        }

        string
    }

    /// Decode a `JsString` into a [`String`], replacing invalid data with
    /// its escaped representation in 4 digit hexadecimal.
    pub fn to_std_string_escaped(&self) -> String {
        self.to_string_escaped()
    }

    /// Decode a `JsString` into a [`String`], returning [`Err`] if it contains any invalid data.
    pub fn to_std_string(&self) -> Result<String, std::string::FromUtf16Error> {
        String::from_utf16(self)
    }

    pub(crate) fn to_code_points(&self) -> impl Iterator<Item = CodePoint> + '_ {
        char::decode_utf16(self.iter().copied()).map(|res| match res {
            Ok(c) => CodePoint::Unicode(c),
            Err(e) => CodePoint::UnpairedSurrogate(e.unpaired_surrogate()),
        })
    }

    /// Abstract operation `StringIndexOf ( string, searchValue, fromIndex )`
    ///
    /// Note: Instead of returning an isize with `-1` as the "not found" value,
    /// we make use of the type system and return Option<usize> with None as the "not found" value.
    ///
    /// More information:
    ///  - [ECMAScript reference][spec]
    ///
    /// [spec]: https://tc39.es/ecma262/#sec-stringindexof
    pub(crate) fn index_of(&self, search_value: &[u16], from_index: usize) -> Option<usize> {
        // 1. Assert: Type(string) is String.
        // 2. Assert: Type(searchValue) is String.
        // 3. Assert: fromIndex is a non-negative integer.

        // 4. Let len be the length of string.
        let len = self.len();

        // 5. If searchValue is the empty String and fromIndex ≤ len, return fromIndex.
        if search_value.is_empty() {
            return if from_index <= len {
                Some(from_index)
            } else {
                None
            };
        }

        // 6. Let searchLen be the length of searchValue.
        // 7. For each integer i starting with fromIndex such that i ≤ len - searchLen, in ascending order, do
        // a. Let candidate be the substring of string from i to i + searchLen.
        // b. If candidate is the same sequence of code units as searchValue, return i.
        // 8. Return -1.
        self.windows(search_value.len())
            .skip(from_index)
            .position(|s| s == search_value)
            .map(|i| i + from_index)
    }

    /// Abstract operation `CodePointAt( string, position )`.
    ///
    /// The abstract operation `CodePointAt` interprets `string` as a sequence of
    /// UTF-16 encoded code points and reads from it a single code point starting
    /// with the code unit at index `position`.
    ///
    /// More information:
    ///  - [ECMAScript reference][spec]
    ///
    /// [spec]: https://tc39.es/ecma262/#sec-codepointat
    pub(crate) fn code_point_at(&self, position: usize) -> CodePoint {
        // 1. Let size be the length of string.
        let size = self.len();

        // 2. Assert: position ≥ 0 and position < size.
        // position >= 0 ensured by position: usize
        assert!(position < size);

        // 3. Let first be the code unit at index position within string.
        // 4. Let cp be the code point whose numeric value is that of first.
        // 5. If first is not a leading surrogate or trailing surrogate, then
        // a. Return the Record { [[CodePoint]]: cp, [[CodeUnitCount]]: 1, [[IsUnpairedSurrogate]]: false }.
        // 6. If first is a trailing surrogate or position + 1 = size, then
        // a. Return the Record { [[CodePoint]]: cp, [[CodeUnitCount]]: 1, [[IsUnpairedSurrogate]]: true }.
        // 7. Let second be the code unit at index position + 1 within string.
        // 8. If second is not a trailing surrogate, then
        // a. Return the Record { [[CodePoint]]: cp, [[CodeUnitCount]]: 1, [[IsUnpairedSurrogate]]: true }.
        // 9. Set cp to ! UTF16SurrogatePairToCodePoint(first, second).

        // We can skip the checks and instead use the `char::decode_utf16` function to take care of that for us.
        let code_point = self
            .get(position..=position + 1)
            .unwrap_or(&self[position..=position]);

        match char::decode_utf16(code_point.iter().copied())
            .next()
            .expect("code_point always has a value")
        {
            Ok(c) => CodePoint::Unicode(c),
            Err(e) => CodePoint::UnpairedSurrogate(e.unpaired_surrogate()),
        }
    }

    /// Abstract operation `StringToNumber ( str )`
    ///
    /// More information:
    /// - [ECMAScript reference][spec]
    ///
    /// [spec]: https://tc39.es/ecma262/#sec-stringtonumber
    #[allow(clippy::question_mark)]
    pub(crate) fn to_number(&self) -> f64 {
        // 1. Let text be ! StringToCodePoints(str).
        // 2. Let literal be ParseText(text, StringNumericLiteral).
        let string = if let Ok(string) = self.to_std_string() {
            string
        } else {
            // 3. If literal is a List of errors, return NaN.
            return f64::NAN;
        };
        // 4. Return StringNumericValue of literal.
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

    /// Abstract operation `StringToBigInt ( str )`
    ///
    /// More information:
    /// - [ECMAScript reference][spec]
    ///
    /// [spec]: https://tc39.es/ecma262/#sec-stringtobigint
    pub(crate) fn to_big_int(&self) -> Option<JsBigInt> {
        // 1. Let text be ! StringToCodePoints(str).
        // 2. Let literal be ParseText(text, StringIntegerLiteral).
        // 3. If literal is a List of errors, return undefined.
        // 4. Let mv be the MV of literal.
        // 5. Assert: mv is an integer.
        // 6. Return ℤ(mv).
        JsBigInt::from_string(self.to_std_string().ok().as_ref()?)
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
        Self::from(utf16!(""))
    }
}

impl Drop for JsString {
    #[inline]
    fn drop(&mut self) {
        self.inner().refcount.set(self.inner().refcount.get() - 1);
        if self.inner().refcount.get() == 0 {
            // Safety:
            // - If refcount is 0 and we call drop, that means this is the last JsString
            // which points to this memory allocation, so deallocating it is safe.
            unsafe {
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                    (*self.ptr.as_ptr()).data.as_mut_ptr(),
                    self.inner().len,
                ));
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

impl From<String> for JsString {
    #[inline]
    fn from(s: String) -> Self {
        Self::from(s.as_str())
    }
}

impl<const N: usize> From<&[u16; N]> for JsString {
    #[inline]
    fn from(s: &[u16; N]) -> Self {
        Self::from(&s[..])
    }
}

impl Hash for JsString {
    fn hash<H: Hasher>(&self, state: &mut H) {
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

pub(crate) trait Utf16Trim {
    fn trim(&self) -> &Self {
        self.trim_start().trim_end()
    }
    fn trim_start(&self) -> &Self;

    fn trim_end(&self) -> &Self;
}

impl Utf16Trim for [u16] {
    fn trim_start(&self) -> &Self {
        if let Some(left) = self.iter().copied().position(|r| {
            !char::from_u32(u32::from(r))
                .map(is_trimmable_whitespace)
                .unwrap_or_default()
        }) {
            &self[left..]
        } else {
            &[]
        }
    }
    fn trim_end(&self) -> &Self {
        if let Some(right) = self.iter().copied().rposition(|r| {
            !char::from_u32(u32::from(r))
                .map(is_trimmable_whitespace)
                .unwrap_or_default()
        }) {
            &self[..=right]
        } else {
            &[]
        }
    }
}

pub(crate) trait ToStringEscaped {
    fn to_string_escaped(&self) -> String;
}

impl ToStringEscaped for [u16] {
    fn to_string_escaped(&self) -> String {
        char::decode_utf16(self.iter().copied())
            .map(|r| match r {
                Ok(c) => String::from(c),
                Err(e) => format!("\\u{:04X}", e.unpaired_surrogate()),
            })
            .collect()
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
        let s = js_string!();
        assert_eq!(*s, "".encode_utf16().collect::<Vec<u16>>());
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

        let xy = js_string!(&x, Y);
        assert_eq!(xy, *utf16!("hello, "));
        assert_eq!(JsString::refcount(&xy), 1);

        let xyz = js_string!(&xy, &z);
        assert_eq!(xyz, *utf16!("hello, world"));
        assert_eq!(JsString::refcount(&xyz), 1);

        let xyzw = js_string!(&xyz, W);
        assert_eq!(xyzw, *utf16!("hello, world!"));
        assert_eq!(JsString::refcount(&xyzw), 1);
    }
}
