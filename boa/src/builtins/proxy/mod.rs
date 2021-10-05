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
    builtins::{BuiltIn, JsArgs},
    gc::{Finalize, Trace},
    object::{ConstructorBuilder, JsObject, ObjectData},
    profiler::BoaProfiler,
    property::Attribute,
    Context, JsResult, JsValue,
};

/// `Proxy` implementation.
#[derive(Debug, Clone, Trace, Finalize)]
pub struct Proxy {
    handler: JsObject,
    target: JsObject,
}

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
        // 1. If NewTarget is undefined, throw a TypeError exception.
        if new_target.is_undefined() {
            context.throw_type_error("NewTarget was undefined")?;
        }

        let target = args.get_or_undefined(0);
        let handler = args.get_or_undefined(1);

        // 2. Return ? ProxyCreate(target, handler).
        Self::create(target, handler, context)
    }

    /// `ProxyCreate ( target, handler )`
    /// More information:
    ///  - [ECMAScript reference][spec]
    ///
    /// [spec]: https://tc39.es/ecma262/#sec-proxycreate
    fn create(target: &JsValue, handler: &JsValue, context: &mut Context) -> JsResult<JsValue> {
        // 1. If Type(target) is not Object, throw a TypeError exception.
        let target = if let Some(obj) = target.as_object() {
            obj
        } else {
            return context.throw_type_error("");
        };

        // 2. If Type(handler) is not Object, throw a TypeError exception.
        let handler = if let Some(obj) = handler.as_object() {
            obj
        } else {
            return context.throw_type_error("");
        };

        // 4. Set P's essential internal methods, except for [[Call]] and [[Construct]], to the definitions specified in 10.5.
        // 5. If IsCallable(target) is true, then
        if target.is_callable() {
            // a. Set P.[[Call]] as specified in 10.5.12.
            // TODO

            // b. If IsConstructor(target) is true, then
            if target.is_constructable() {
                // i. Set P.[[Construct]] as specified in 10.5.13.
                // TODO
            }
        }

        // 3. Let P be ! MakeBasicObject(« [[ProxyHandler]], [[ProxyTarget]] »).
        // 6. Set P.[[ProxyTarget]] to target.
        // 7. Set P.[[ProxyHandler]] to handler.
        let p = JsObject::from_proto_and_data(None, ObjectData::proxy(Self { handler, target }));

        // 8. Return P.
        Ok(p.into())
    }
}
