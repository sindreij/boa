//! Execution of the AST, this is where the interpreter actually runs

#[cfg(test)]
mod tests;

use crate::{Context, JsResult, JsValue};

pub trait Executable {
    /// Runs this executable in the given context.
    fn run(&self, context: &mut Context) -> JsResult<JsValue>;
}

/// Enum representing all the possible completion results that can be returned
/// when interpreting Javascript code.
pub(crate) enum CompletionKind {
    Normal,
    Return,
    Break(Option<Box<str>>),
    Continue(Option<Box<str>>),
    Throw,
}

/// The `Completion Record` Specification Type
///
/// The `Completion` type is a `Record` used to explain the runtime propagation of
/// values and control flow such as the behaviour of statements
/// (break, continue, return and throw) that perform nonlocal transfers of control.
///
/// More information:
///  - [ECMAScript reference][spec]
///
/// [spec]: https://tc39.es/ecma262/#sec-completion-record-specification-type
pub(crate) struct CompletionRecord {
    pub(crate) kind: CompletionKind,
    pub(crate) value: Option<JsValue>,
}

impl CompletionRecord {
    /// Abstract operation `UpdateEmpty ( completionRecord, value )`
    ///
    /// More information:
    ///  - [ECMAScript reference][spec]
    ///
    /// [spec]: https://tc39.es/ecma262/#sec-updateempty
    pub(crate) fn update_empty(mut self, value: JsValue) -> Self {
        if self.value.is_none() {
            self.value = Some(value)
        }
        self
    }
}

impl From<JsResult<JsValue>> for CompletionRecord {
    fn from(result: JsResult<JsValue>) -> Self {
        match result {
            Ok(val) => Self {
                value: Some(val),
                kind: CompletionKind::Normal,
            },
            Err(err) => Self {
                value: Some(err),
                kind: CompletionKind::Throw,
            },
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum InterpreterState {
    Executing,
    Return,
    Break(Option<Box<str>>),
    Continue(Option<Box<str>>),
}

/// A Javascript intepreter
#[derive(Debug)]
pub struct Interpreter {
    /// the current state of the interpreter.
    state: InterpreterState,
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl Interpreter {
    /// Creates a new interpreter.
    pub fn new() -> Self {
        Self {
            state: InterpreterState::Executing,
        }
    }

    #[inline]
    pub(crate) fn set_current_state(&mut self, new_state: InterpreterState) {
        self.state = new_state
    }

    #[inline]
    pub(crate) fn get_current_state(&self) -> &InterpreterState {
        &self.state
    }
}
