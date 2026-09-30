//! The playground: a page that is mostly its script.

use crate::markdown::{Body, Context};

pub fn render(context: &Context) -> Body {
    let source = std::fs::read_to_string(context.site.root.join("site/content/playground.md"))
        .expect("playground.md");
    let mut body = crate::markdown::render(&source, context);
    body.scripts.push("playground.js".to_string());
    body
}
