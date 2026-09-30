//! The page shell: masthead, navigation, the article, its table of
//! contents, and the footer. One template, every page.

use topos_plates::escape;

use crate::Site;
use crate::markdown::Body;
use crate::pages::{self, Page, SECTIONS, Source};

const REPO: &str = "https://github.com/shergin/topos";

/// Wraps a rendered body in the site shell.
pub fn page(site: &Site, section: &str, page: &Page, body: &Body) -> String {
    let root = page.root();
    let href = |url: &str| format!("{root}{}", url.trim_start_matches('/'));
    let home = page.source == Source::Home;
    let title = if home {
        "topos — an autodiff compiler stack in Rust".to_string()
    } else {
        format!("{} — topos", page.title)
    };
    let mut out = String::with_capacity(body.html.len() + 8192);
    out.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    out.push_str(&format!("<title>{}</title>\n", escape(&title)));
    out.push_str(&format!(
        "<meta name=\"description\" content=\"{}\">\n",
        escape(page.blurb)
    ));
    out.push_str("<meta name=\"color-scheme\" content=\"light dark\">\n");
    out.push_str(&format!(
        "<link rel=\"icon\" href=\"{root}favicon.svg\" type=\"image/svg+xml\">\n"
    ));
    out.push_str(&format!(
        "<link rel=\"stylesheet\" href=\"{root}site.css\">\n"
    ));
    out.push_str("<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">\n<link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>\n");
    out.push_str("<link href=\"https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;1,400&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400;1,8..60,600&display=swap\" rel=\"stylesheet\">\n");
    out.push_str(&format!(
        "<script>window.TOPOS_ROOT = \"{root}\";</script>\n"
    ));
    out.push_str("</head>\n");
    out.push_str(&format!(
        "<body class=\"{}\">\n",
        if home { "home" } else { "doc" }
    ));
    out.push_str("<a class=\"skip\" href=\"#content\">Skip to content</a>\n");

    // Masthead.
    out.push_str("<header class=\"top\">\n");
    out.push_str(&format!(
        "<a class=\"brand\" href=\"{root}\" aria-label=\"topos, home\"><img class=\"mark\" src=\"{root}topos.png\" alt=\"\" width=\"704\" height=\"464\">topos</a>\n"
    ));
    out.push_str("<button class=\"nav-toggle\" type=\"button\" aria-controls=\"sidebar\" aria-expanded=\"false\">Contents</button>\n");
    out.push_str("<div class=\"search\"><input id=\"search\" type=\"search\" placeholder=\"Search the docs\" aria-label=\"Search the docs\" autocomplete=\"off\"><div class=\"results\" id=\"results\" hidden></div></div>\n");
    out.push_str(&format!(
        "<nav class=\"links\" aria-label=\"Elsewhere\"><a href=\"{REPO}\">GitHub</a><a href=\"https://docs.rs/topos\">docs.rs</a><a href=\"https://crates.io/crates/topos\">crates.io</a></nav>\n"
    ));
    out.push_str("</header>\n");

    out.push_str("<div class=\"shell\">\n");
    // Sidebar.
    out.push_str("<nav class=\"sidebar\" id=\"sidebar\" aria-label=\"Documentation\">\n");
    for group in SECTIONS {
        out.push_str(&format!("<section><h2>{}</h2><ul>\n", escape(group.title)));
        for entry in group.pages {
            let current = if entry.url == page.url {
                " aria-current=\"page\""
            } else {
                ""
            };
            let label = if entry.source == Source::Home {
                "Home"
            } else {
                entry.title
            };
            out.push_str(&format!(
                "<li><a href=\"{}\"{current}>{}</a></li>\n",
                href(entry.url),
                escape(label)
            ));
        }
        out.push_str("</ul></section>\n");
    }
    out.push_str(&format!(
        "<section><h2>Elsewhere</h2><ul><li><a href=\"https://docs.rs/topos\">API reference</a></li><li><a href=\"{REPO}\">Source on GitHub</a></li><li><a href=\"https://crates.io/crates/topos\">crates.io</a></li></ul></section>\n"
    ));
    out.push_str("</nav>\n");

    // Article.
    out.push_str("<main id=\"content\">\n<article>\n");
    if !home {
        out.push_str("<header class=\"page-head\">\n");
        out.push_str(&format!("<p class=\"kicker\">{}</p>\n", escape(section)));
        out.push_str(&format!("<h1>{}</h1>\n", escape(page.title)));
        out.push_str(&format!(
            "<p class=\"standfirst\">{}</p>\n",
            escape(page.blurb)
        ));
        out.push_str("</header>\n");
    }
    out.push_str(&body.html);
    if !home {
        out.push_str("<footer class=\"page-foot\">\n");
        let all: Vec<&Page> = pages::all().collect();
        let index = all
            .iter()
            .position(|entry| entry.url == page.url)
            .unwrap_or(0);
        out.push_str("<nav class=\"pager\" aria-label=\"Previous and next\">\n");
        if index > 0 {
            let previous = all[index - 1];
            out.push_str(&format!(
                "<a class=\"prev\" href=\"{}\"><span>Previous</span>{}</a>\n",
                href(previous.url),
                escape(if previous.source == Source::Home {
                    "Home"
                } else {
                    previous.title
                })
            ));
        } else {
            out.push_str("<span></span>\n");
        }
        if index + 1 < all.len() {
            let next = all[index + 1];
            out.push_str(&format!(
                "<a class=\"next\" href=\"{}\"><span>Next</span>{}</a>\n",
                href(next.url),
                escape(next.title)
            ));
        }
        out.push_str("</nav>\n");
        if let Some(source) = source_link(page) {
            out.push_str(&format!(
                "<p class=\"edit\"><a href=\"{source}\">Edit this page on GitHub</a></p>\n"
            ));
        }
        out.push_str("</footer>\n");
    }
    out.push_str("</article>\n");

    // Table of contents.
    let toc: Vec<_> = body
        .headings
        .iter()
        .filter(|heading| heading.level == 2)
        .collect();
    if !home && toc.len() >= 3 {
        out.push_str(
            "<aside class=\"toc\" aria-label=\"On this page\"><h2>On this page</h2><ol>\n",
        );
        for heading in toc {
            out.push_str(&format!(
                "<li><a href=\"#{}\">{}</a></li>\n",
                heading.id,
                escape(&heading.text)
            ));
        }
        out.push_str("</ol></aside>\n");
    }
    out.push_str("</main>\n</div>\n");

    // Footer.
    out.push_str("<footer class=\"site-foot\">\n");
    out.push_str(&format!(
        "<p>topos {} · Rust 1.88 or newer · MIT or Apache-2.0.</p>\n",
        escape(&site.version)
    ));
    out.push_str("<p>Every reading on this site is program output: recorded, differentiated, scheduled, emitted, and run by the crate when the page was built, never typed by hand. A topos is a place — here, one where the whole compiler stack stays in view.</p>\n");
    out.push_str("</footer>\n");
    out.push_str(&format!("<script src=\"{root}site.js\" defer></script>\n"));
    for script in &body.scripts {
        out.push_str(&format!(
            "<script type=\"module\" src=\"{root}{script}\"></script>\n"
        ));
    }
    out.push_str("</body>\n</html>\n");
    out
}

fn source_link(page: &Page) -> Option<String> {
    match page.source {
        Source::Content(path) => Some(format!("{REPO}/blob/master/site/content/{path}")),
        Source::Repo(path) | Source::RepoWith(path, _) => {
            Some(format!("{REPO}/blob/master/{path}"))
        }
        Source::Gallery => Some(format!("{REPO}/tree/master/examples")),
        _ => None,
    }
}
