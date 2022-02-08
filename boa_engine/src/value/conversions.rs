use super::{Display, JsValue};

impl From<&Self> for JsValue {
    #[inline]
    fn from(value: &Self) -> Self {
        value.clone()
    }
}

impl From<char> for JsValue {
    #[inline]
    fn from(value: char) -> Self {
        Self::new(value.to_string())
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct TryFromCharError;

impl Display for TryFromCharError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Could not convert value to a char type")
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct TryFromObjectError;

impl Display for TryFromObjectError {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Could not convert value to an Object type")
    }
}

pub(crate) trait IntoOrUndefined {
    fn into_or_undefined(self) -> JsValue;
}

impl<T> IntoOrUndefined for Option<T>
where
    T: Into<JsValue>,
{
    fn into_or_undefined(self) -> JsValue {
        self.map_or_else(JsValue::undefined, Into::into)
    }
}
