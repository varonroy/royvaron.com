use mini_builder_rs::value::Value;

fn math(values: &[Value]) -> Value {
    /*if let Value::Text(text) = &values[0] {
        let out = latex_to_mathml(text, latex2mathml::DisplayStyle::Inline).unwrap();

        // <math
        // 01234
        Value::text(format!("{} class=\"math-inline\" {}", &out[..5], &out[6..],))
    // Value::text(format!("{}", out.escape_debug()))
    } else {
        Value::text("error")
    }*/

    if let Value::Text(text) = &values[0] {
        if let Some(mml) = itex2mml::MML::parse(&format!("${}$", text)) {
            // <math
            // 01234
            // return Value::text(format!("{} class=\"math-inline\" {}", &out[..5], &out[6..],));
            let out = mml.as_str();
            return Value::text(format!("{} class=\"math-inline\" {}", &out[..5], &out[6..]));
            // return Value::text(text.as_str());
        }
    }
    Value::text("error")
}

fn math_itex(values: &[Value]) -> Value {
    if let Value::Text(text) = &values[0] {
        if let Some(mml) = itex2mml::MML::parse(&format!("$${}$$", text)) {
            // <math
            // 01234
            // return Value::text(format!("{} class=\"math-inline\" {}", &out[..5], &out[6..],));
            let out = mml.as_str();
            return Value::text(format!("{} class=\"math-block\" {}", &out[..5], &out[6..]));
            // return Value::text(text.as_str());
        }
    }
    Value::text("error")
}

fn remove_newline(values: &[Value]) -> Value {
    if let Some(Value::Text(text)) = values.get(0) {
        Value::text(
            text.lines()
                .map(|l| l.trim_start())
                .filter(|l| !l.starts_with("//"))
                .into_iter()
                .collect::<Vec<_>>()
                .join(""),
        )
    } else {
        Value::text("error")
    }
}

fn main() {
    mini_builder_rs::builder::Builder::new(
        Some("../html-src".into()),
        Some("../html-templates".into()),
        Some("../html-generated".into()),
        None,
        Default::default(),
    )
    .unwrap()
    .add_function("math", Box::new(math) as _)
    .add_function("math_itex", Box::new(math_itex) as _)
    .add_function("remove_newline", Box::new(remove_newline) as _)
    .watch()
    .unwrap();
}
