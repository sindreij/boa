//! String interner for Boa.
//!
//! The idea behind using a string interner is that in most of the code, strings such as
//! identifiers and literals are often repeated. This causes extra burden when comparing them and
//! storing them. A string interner stores a unique `usize` symbol for each string, making sure
//! that there are no duplicates. This makes it much easier to compare, since it's just comparing
//! to `usize`, and also it's easier to store, since instead of a heap-allocated string, you only
//! need to store a `usize`. This reduces memory consumption and improves performance in the
//! compiler.

#![doc(
    html_logo_url = "https://raw.githubusercontent.com/boa-dev/boa/main/assets/logo.svg",
    html_favicon_url = "https://raw.githubusercontent.com/boa-dev/boa/main/assets/logo.svg"
)]
#![cfg_attr(not(test), forbid(clippy::unwrap_used))]
#![warn(
    clippy::perf,
    clippy::single_match_else,
    clippy::dbg_macro,
    clippy::doc_markdown,
    clippy::wildcard_imports,
    clippy::struct_excessive_bools,
    clippy::doc_markdown,
    clippy::semicolon_if_nothing_returned,
    clippy::pedantic
)]
#![deny(
    clippy::all,
    clippy::cast_lossless,
    clippy::redundant_closure_for_method_calls,
    clippy::use_self,
    clippy::unnested_or_patterns,
    clippy::trivially_copy_pass_by_ref,
    clippy::needless_pass_by_value,
    clippy::match_wildcard_for_single_variants,
    clippy::map_unwrap_or,
    unused_qualifications,
    unused_import_braces,
    unused_lifetimes,
    unreachable_pub,
    trivial_numeric_casts,
    // rustdoc,
    missing_debug_implementations,
    missing_copy_implementations,
    deprecated_in_future,
    meta_variable_misuse,
    non_ascii_idents,
    rust_2018_compatibility,
    rust_2018_idioms,
    future_incompatible,
    nonstandard_style,
)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_ptr_alignment,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::missing_inline_in_public_items,
    clippy::cognitive_complexity,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::as_conversions,
    clippy::let_unit_value,
    rustdoc::missing_doc_code_examples
)]

#[cfg(test)]
mod tests;

use core::fmt;
use std::{fmt::Display, num::NonZeroUsize};

use const_utf16::encode as utf16;
use gc::{unsafe_empty_trace, Finalize, Trace};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use string_interner::{backend::BucketBackend, symbol::SymbolUsize, StringInterner, Symbol};

/// Backend of the string interner.
type UTF8Backend = BucketBackend<str, UTF8Sym>;
type UTF16Backend = BucketBackend<[u16], UTF16Sym>;

/// The string interner for Boa.
///
/// This is a type alias that makes it easier to reference it in the code.
#[derive(Debug)]
pub struct Interner {
    utf8: StringInterner<UTF8Backend>,
    utf16: StringInterner<UTF16Backend>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InternedStr<'a> {
    Str(&'a str),
    JStr(&'a [u16]),
}

impl<'a> InternedStr<'a> {
    pub fn join<F, G, T>(self, f: F, g: G) -> T
    where
        F: FnOnce(&'a str) -> T,
        G: FnOnce(&'a [u16]) -> T,
    {
        match self {
            InternedStr::Str(s) => f(s),
            InternedStr::JStr(js) => g(js),
        }
    }

    pub fn join_with_context<C, F, G, T>(self, ctx: C, f: F, g: G) -> T
    where
        F: FnOnce(C, &'a str) -> T,
        G: FnOnce(C, &'a [u16]) -> T,
    {
        match self {
            InternedStr::Str(s) => f(ctx, s),
            InternedStr::JStr(js) => g(ctx, js),
        }
    }

    pub fn into_common<C>(self) -> C
    where
        C: From<&'a str> + From<&'a [u16]>,
    {
        match self {
            InternedStr::Str(s) => s.into(),
            InternedStr::JStr(js) => js.into(),
        }
    }
}

impl<'a> From<&'a str> for InternedStr<'a> {
    fn from(s: &'a str) -> Self {
        InternedStr::Str(s)
    }
}

impl<'a> From<&'a [u16]> for InternedStr<'a> {
    fn from(s: &'a [u16]) -> Self {
        InternedStr::JStr(s)
    }
}

impl<'a> Display for InternedStr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.join_with_context(
            f,
            |f, s| s.fmt(f),
            |f, js| {
                char::decode_utf16(js.iter().copied())
                    .map(|r| match r {
                        Ok(c) => String::from(c),
                        Err(e) => format!("\\u{:04X}", e.unpaired_surrogate()),
                    })
                    .collect::<String>()
                    .fmt(f)
            },
        )
    }
}

pub trait Internable<'a> {
    fn as_internable(&'a self) -> InternedStr<'a>;
}

impl<'a> Internable<'a> for str {
    fn as_internable(&'a self) -> InternedStr<'a> {
        InternedStr::Str(self)
    }
}

impl<'a> Internable<'a> for [u16] {
    fn as_internable(&'a self) -> InternedStr<'a> {
        InternedStr::JStr(self)
    }
}

impl Interner {
    /// Creates a new `StringInterner` with the given initial capacity.
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            utf8: StringInterner::with_capacity(cap),
            utf16: StringInterner::with_capacity(cap),
        }
    }

    /// Returns the number of strings interned by the interner.
    #[inline]
    pub fn len(&self) -> usize {
        self.utf8.len() + self.utf16.len()
    }

    /// Returns `true` if the string interner has no interned strings.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.utf8.is_empty() && self.utf16.is_empty()
    }

    /// Returns the symbol for the given string if any.
    ///
    /// Can be used to query if a string has already been interned without interning.
    pub fn get<'a, T>(&self, string: &'a T) -> Option<Sym>
    where
        T: Internable<'a> + ?Sized,
    {
        let internable = string.as_internable();
        if let Some(sym) = Self::get_static(internable) {
            return Some(sym);
        }
        string.as_internable().join(
            |s| self.utf8.get(s).map(Sym::from),
            |js| self.utf16.get(js).map(Sym::from),
        )
    }

    /// Interns the given string.
    ///
    /// Returns a symbol for resolution into the original string.
    ///
    /// # Panics
    /// If the interner already interns the maximum number of strings possible by the chosen symbol type.
    pub fn get_or_intern<'a, T>(&mut self, string: &'a T) -> Sym
    where
        T: Internable<'a> + ?Sized,
    {
        let internable = string.as_internable();
        if let Some(sym) = Self::get_static(internable) {
            return sym;
        }
        string.as_internable().join(
            |s| self.utf8.get_or_intern(s).into(),
            |js| self.utf16.get_or_intern(js).into(),
        )
    }

    /// Interns the given `'static` string.
    ///
    /// Returns a symbol for resolution into the original string.
    ///
    /// # Note
    ///
    /// This is more efficient than [`StringInterner::get_or_intern`] since it might
    /// avoid some memory allocations if the backends supports this.
    ///
    /// # Panics
    ///
    /// If the interner already interns the maximum number of strings possible
    /// by the chosen symbol type.
    pub fn get_or_intern_static<T>(&mut self, string: &'static T) -> Sym
    where
        T: Internable<'static> + ?Sized,
    {
        let internable = string.as_internable();
        if let Some(sym) = Self::get_static(internable) {
            return sym;
        }
        internable.join(
            |s| self.utf8.get_or_intern_static(s).into(),
            |js| self.utf16.get_or_intern_static(js).into(),
        )
    }

    /// Shrink backend capacity to fit the interned strings exactly.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.utf8.shrink_to_fit();
        self.utf16.shrink_to_fit();
    }

    /// Returns the string for the given symbol if any.
    #[inline]
    pub fn resolve(&self, symbol: Sym) -> Option<InternedStr<'_>> {
        match symbol.into_internal_symbol() {
            IntoSymbol::Static(index) => Some(InternedStr::Str(Self::STATIC_STRINGS[index].0)),
            IntoSymbol::UTF8Sym(s) => self.utf8.resolve(s).map(InternedStr::from),
            IntoSymbol::UTF16Sym(js) => self.utf16.resolve(js).map(InternedStr::from),
        }
    }

    /// Returns the string for the given symbol.
    ///
    /// # Panics
    ///
    /// If the interner cannot resolve the given symbol.
    #[inline]
    pub fn resolve_expect(&self, symbol: Sym) -> InternedStr<'_> {
        self.resolve(symbol).expect("string disappeared")
    }

    /// Gets the symbol of the static string if one of them
    fn get_static(string: InternedStr<'_>) -> Option<Sym> {
        Self::STATIC_STRINGS
            .into_iter()
            .enumerate()
            .find(|&(_i, s)| match string {
                InternedStr::Str(str) => str == s.0,
                InternedStr::JStr(jstr) => jstr == s.1,
            })
            .map(|(i, _str)| {
                let raw = NonZeroUsize::new(i.wrapping_add(1)).expect("static array too big");
                Sym::from_raw(raw)
            })
    }
}

impl<'a> FromIterator<&'a str> for Interner {
    #[inline]
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = &'a str>,
    {
        Self {
            utf8: StringInterner::from_iter(iter),
            utf16: StringInterner::default(),
        }
    }
}

impl FromIterator<String> for Interner {
    #[inline]
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = String>,
    {
        Self {
            utf8: StringInterner::from_iter(iter),
            utf16: StringInterner::default(),
        }
    }
}

impl<'a> FromIterator<&'a [u16]> for Interner {
    #[inline]
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = &'a [u16]>,
    {
        Self {
            utf8: StringInterner::default(),
            utf16: StringInterner::from_iter(iter),
        }
    }
}

impl FromIterator<Vec<u16>> for Interner {
    #[inline]
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Vec<u16>>,
    {
        Self {
            utf8: StringInterner::default(),
            utf16: StringInterner::from_iter(iter),
        }
    }
}
impl<'a> Extend<&'a str> for Interner {
    #[inline]
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = &'a str>,
    {
        self.utf8.extend(iter);
    }
}

impl Extend<String> for Interner {
    #[inline]
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = String>,
    {
        self.utf8.extend(iter);
    }
}

impl<'a> Extend<&'a [u16]> for Interner {
    #[inline]
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = &'a [u16]>,
    {
        self.utf16.extend(iter);
    }
}

impl Extend<Vec<u16>> for Interner {
    #[inline]
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Vec<u16>>,
    {
        self.utf16.extend(iter);
    }
}

pub struct Iter<'a> {
    utf8: <&'a UTF8Backend as IntoIterator>::IntoIter,
    utf16: <&'a UTF16Backend as IntoIterator>::IntoIter,
}

#[allow(clippy::type_repetition_in_bounds)]
impl<'a> fmt::Debug for Iter<'a>
where
    <&'a UTF8Backend as IntoIterator>::IntoIter: fmt::Debug,
    <&'a UTF16Backend as IntoIterator>::IntoIter: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Iter")
            .field("utf8", &self.utf8)
            .field("utf16", &self.utf16)
            .finish()
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = (Sym, InternedStr<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        self.utf8
            .next()
            .map(|(sym, str)| (sym.into(), str.into()))
            .or_else(|| self.utf16.next().map(|(sym, str)| (sym.into(), str.into())))
    }
}

impl<'a> IntoIterator for &'a Interner {
    type Item = (Sym, InternedStr<'a>);
    type IntoIter = Iter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Iter {
            utf8: self.utf8.into_iter(),
            utf16: self.utf16.into_iter(),
        }
    }
}

impl Default for Interner {
    fn default() -> Self {
        Self {
            utf8: StringInterner::new(),
            utf16: StringInterner::new(),
        }
    }
}

/// The string symbol type for Boa.
///
/// This symbol type is internally a `NonZeroUsize`, which makes it pointer-width in size and it's
/// optimized so that it can occupy 1 pointer width even in an `Option` type.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Finalize)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
#[allow(clippy::unsafe_derive_deserialize)]
pub struct Sym {
    value: NonZeroUsize,
}

impl Sym {
    /// Padding for the symbol internal value.
    const PADDING: usize = Interner::STATIC_STRINGS.len() + 1;

    /// Symbol for the empty string (`""`).
    pub const EMPTY_STRING: Self = unsafe { Self::from_raw(NonZeroUsize::new_unchecked(1)) };

    /// Symbol for the `"arguments"` string.
    pub const ARGUMENTS: Self = unsafe { Self::from_raw(NonZeroUsize::new_unchecked(2)) };

    /// Symbol for the `"await"` string.
    pub const AWAIT: Self = unsafe { Self::from_raw(NonZeroUsize::new_unchecked(3)) };

    /// Symbol for the `"yield"` string.
    pub const YIELD: Self = unsafe { Self::from_raw(NonZeroUsize::new_unchecked(4)) };

    /// Symbol for the `"eval"` string.
    pub const EVAL: Self = unsafe { Self::from_raw(NonZeroUsize::new_unchecked(5)) };

    /// Symbol for the `"default"` string.
    pub const DEFAULT: Self = unsafe { Self::from_raw(NonZeroUsize::new_unchecked(6)) };

    /// Symbol for the `"null"` string.
    pub const NULL: Self = unsafe { Self::from_raw(NonZeroUsize::new_unchecked(7)) };

    /// Symbol for the `"RegExp"` string.
    pub const REGEXP: Self = unsafe { Self::from_raw(NonZeroUsize::new_unchecked(8)) };

    /// Symbol for the `"get"` string.
    pub const GET: Self = unsafe { Self::from_raw(NonZeroUsize::new_unchecked(9)) };

    /// Symbol for the `"set"` string.
    pub const SET: Self = unsafe { Self::from_raw(NonZeroUsize::new_unchecked(10)) };

    /// Symbol for the `"<main>"` string.
    pub const MAIN: Self = unsafe { Self::from_raw(NonZeroUsize::new_unchecked(11)) };

    /// Symbol for the `"raw"` string.
    pub const RAW: Self = unsafe { Self::from_raw(NonZeroUsize::new_unchecked(12)) };

    /// Creates a `Sym` from a raw `NonZeroUsize`.
    const fn from_raw(value: NonZeroUsize) -> Self {
        Self { value }
    }

    /// Retrieves the raw `NonZeroUsize` for this symbol.
    const fn as_raw(self) -> NonZeroUsize {
        self.value
    }

    fn into_internal_symbol(self) -> IntoSymbol {
        let unpadded = match self.as_raw().get() {
            val if val < Self::PADDING => return IntoSymbol::Static(val - 1),
            val => val - Self::PADDING,
        };
        if unpadded % 2 == 0 {
            IntoSymbol::UTF8Sym(UTF8Sym(
                SymbolUsize::try_from_usize(unpadded / 2)
                    .expect("Symbol index overflowed the interner size!"),
            ))
        } else {
            IntoSymbol::UTF16Sym(UTF16Sym(
                SymbolUsize::try_from_usize((unpadded - 1) / 2)
                    .expect("Symbol index overflowed the interner size!"),
            ))
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct UTF8Sym(SymbolUsize);

impl From<UTF8Sym> for Sym {
    fn from(sym: UTF8Sym) -> Self {
        // SAFETY:
        // The definition of `Self::PADDING` guarantees that it's >= 1,
        // so sym.0.to_usize() * 2 + Self::PADDING > 0.
        unsafe {
            let value = sym.0.to_usize() * 2 + Self::PADDING;
            Self::from_raw(NonZeroUsize::new_unchecked(value))
        }
    }
}

impl Symbol for UTF8Sym {
    #[inline]
    fn try_from_usize(index: usize) -> Option<Self> {
        SymbolUsize::try_from_usize(index).map(Self)
    }

    #[inline]
    fn to_usize(self) -> usize {
        self.0.to_usize()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct UTF16Sym(SymbolUsize);

impl From<UTF16Sym> for Sym {
    fn from(sym: UTF16Sym) -> Self {
        // SAFETY:
        // 1 > 0, so sym.value.get() * 2 + Self::PADDING + 1 > 1 > 0,
        // meaning that sym.value.get() * 2 + Self::PADDING + 1 > 0
        unsafe {
            let value = sym.0.to_usize() * 2 + Self::PADDING + 1;
            Self::from_raw(NonZeroUsize::new_unchecked(value))
        }
    }
}

impl Symbol for UTF16Sym {
    #[inline]
    fn try_from_usize(index: usize) -> Option<Self> {
        SymbolUsize::try_from_usize(index).map(Self)
    }

    #[inline]
    fn to_usize(self) -> usize {
        self.0.to_usize()
    }
}

enum IntoSymbol {
    Static(usize),
    UTF8Sym(UTF8Sym),
    UTF16Sym(UTF16Sym),
}

// Safe because `Sym` implements `Copy`.
unsafe impl Trace for Sym {
    unsafe_empty_trace!();
}

/// Converts a given element to a string using an interner.
pub trait ToInternedString {
    /// Converts a given element to a string using an interner.
    fn to_interned_string(&self, interner: &Interner) -> String;
}

impl<T> ToInternedString for T
where
    T: Display,
{
    fn to_interned_string(&self, _interner: &Interner) -> String {
        self.to_string()
    }
}

macro_rules! static_strings {
    (
        $($string:literal),*;
        $size:literal
    ) => {
        impl Interner {
            /// List of commonly used static strings.
            ///
            /// Make sure that any string added as a `Sym` constant is also added here.
            const STATIC_STRINGS: [(&'static str, &'static [u16]); $size] = [
                $(
                    ($string, utf16!($string))
                ),*
            ];
        }
    };
}

static_strings![
    "",
    "arguments",
    "await",
    "yield",
    "eval",
    "default",
    "null",
    "RegExp",
    "get",
    "set",
    "<main>",
    "raw";
    12
];
