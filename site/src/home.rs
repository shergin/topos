//! The front page.

use crate::markdown::{Body, Context};

pub fn render(context: &Context) -> Body {
    let source =
        std::fs::read_to_string(context.site.root.join("site/content/home.md")).expect("home.md");
    crate::markdown::render(&source, context)
}
