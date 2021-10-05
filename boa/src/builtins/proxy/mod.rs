//! This module implements the global `Number` object.
//!
//! The `Number` JavaScript object is a wrapper object allowing you to work with numerical values.
//! A `Number` object is created using the `Number()` constructor. A primitive type object number is created using the `Number()` **function**.
//!
//! The JavaScript `Number` type is double-precision 64-bit binary format IEEE 754 value. In more recent implementations,
//! JavaScript also supports integers with arbitrary precision using the BigInt type.
//!
//! More information:
//!  - [ECMAScript reference][spec]
//!  - [MDN documentation][mdn]
//!
//! [spec]: https://tc39.es/ecma262/#sec-proxy-objects
//! [mdn]: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Proxy

use crate::{
    builtins::{function::make_builtin_fn, BuiltIn},
    object::ConstructorBuilder,
    profiler::BoaProfiler,
    property::Attribute,
    Context, JsResult, JsValue,
};

/// `Proxy` implementation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Proxy;

impl BuiltIn for Proxy {
    const NAME: &'static str = "Number";

    const ATTRIBUTE: Attribute = Attribute::WRITABLE
        .union(Attribute::NON_ENUMERABLE)
        .union(Attribute::CONFIGURABLE);

    fn init(context: &mut Context) -> JsValue {
        let _timer = BoaProfiler::global().start_event(Self::NAME, "init");

        // let attribute = Attribute::READONLY | Attribute::NON_ENUMERABLE | Attribute::PERMANENT;
        let proxy_object = ConstructorBuilder::with_standard_object(
            context,
            Self::constructor,
            context.standard_objects().proxy_object().clone(),
        )
        .name(Self::NAME)
        .length(Self::LENGTH)
        .build();

        proxy_object.into()
    }
}

impl Proxy {
    /// The amount of arguments this function object takes.
    pub(crate) const LENGTH: usize = 2;

    /// `Proxy ( target, handler )`
    pub(crate) fn constructor(
        new_target: &JsValue,
        args: &[JsValue],
        context: &mut Context,
    ) -> JsResult<JsValue> {
        // let data = match args.get(0) {
        //     Some(value) => value.to_numeric_number(context)?,
        //     None => 0.0,
        // };
        // if new_target.is_undefined() {
        //     return Ok(JsValue::new(data));
        // }
        // let prototype =
        //     get_prototype_from_constructor(new_target, StandardObjects::proxy_object, context)?;
        // let this = JsObject::from_proto_and_data(prototype, ObjectData::number(data));
        // Ok(this.into())
        todo!()
    }
}
