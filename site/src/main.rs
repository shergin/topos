//! The generator behind <https://shergin.github.io/topos/>.
//!
//! One rule from the repository applies here too: no reading on the
//! site is typed by hand. Every plate is a block of Rust in
//! `site/figures/*.rs` that records a graph and asks the crate for one
//! of its readings — the spec, the derivative, the plan, the StableHLO,
//! a run, a training curve — rendered at build time, with the code
//! that ran shown beside it, included verbatim.
//!
//! ```sh
//! cargo run -p topos-site            # writes site/dist
//! ./site/build.sh                    # the same, plus the wasm the playground uses
//! ```

mod figures;
mod gallery;
mod home;
mod layout;
mod markdown;
mod pages;
mod playground;
mod show;

use std::fs;
use std::path::{Path, PathBuf};

use pages::{Page, Source};

/// Everything a page needs to know about where it lives.
pub struct Site {
    /// The repository root.
    pub root: PathBuf,
    /// The output directory.
    pub out: PathBuf,
    /// The crate version, from the root manifest.
    pub version: String,
}

/// One entry in the client-side search index.
#[derive(serde::Serialize)]
struct SearchEntry {
    url: String,
    title: String,
    section: String,
    headings: Vec<String>,
    text: String,
}

fn main() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .expect("site sits inside the repo")
        .to_path_buf();
    let mut out = manifest.join("dist");
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--out" => out = PathBuf::from(arguments.next().expect("--out takes a directory")),
            other => panic!("unknown argument {other}"),
        }
    }

    let site = Site {
        version: manifest_version(&root.join("Cargo.toml")),
        root,
        out,
    };

    prepare(&site.out);
    copy_dir(&site.root.join("site/static"), &site.out);
    fs::copy(site.root.join("topos.png"), site.out.join("topos.png")).expect("copy the logo");
    let figures = figures::Registry::new();
    let mut index = Vec::new();
    let mut written = 0;
    for section in pages::SECTIONS {
        for page in section.pages {
            let rendered = render_page(&site, &figures, section.title, page);
            write(&site.out.join(page.path()), &rendered.html);
            written += 1;
            if page.searchable() {
                index.push(SearchEntry {
                    url: page.url.to_string(),
                    title: page.title.to_string(),
                    section: section.title.to_string(),
                    headings: rendered.headings,
                    text: rendered.summary,
                });
            }
        }
    }
    // A contact sheet of every figure, for proofing: SITE_CONTACT_SHEET=1.
    if std::env::var_os("SITE_CONTACT_SHEET").is_some() {
        let page = Page {
            url: "/contact/",
            title: "Every figure",
            blurb: "The contact sheet.",
            source: Source::Content("contact.md"),
        };
        let mut source = String::new();
        for name in figures.names() {
            source.push_str(&format!("## {name}\n\n{{{{figure {name}}}}}\n\n"));
        }
        let context = markdown::Context {
            site: &site,
            figures: &figures,
            page: &page,
            source_dir: "",
        };
        let body = markdown::render(&source, &context);
        write(
            &site.out.join("contact/index.html"),
            &layout::page(&site, "Proofing", &page, &body),
        );
    }
    let index = serde_json::to_string(&index).expect("search index serializes");
    write(&site.out.join("search.json"), &index);
    write(&site.out.join(".nojekyll"), "");
    println!(
        "wrote {written} pages and {} plates to {}",
        figures.rendered(),
        site.out.display()
    );
}

/// A rendered page plus what the search index keeps of it.
struct RenderedPage {
    html: String,
    headings: Vec<String>,
    summary: String,
}

fn render_page(
    site: &Site,
    figures: &figures::Registry,
    section: &str,
    page: &Page,
) -> RenderedPage {
    let context = markdown::Context {
        site,
        figures,
        page,
        source_dir: page.source_dir(),
    };
    let body = match page.source {
        Source::Content(path) => {
            markdown::render(&read(&site.root.join("site/content").join(path)), &context)
        }
        Source::Repo(path) => markdown::render(&read(&site.root.join(path)), &context),
        Source::RepoWith(path, inserts) => {
            markdown::render(&illustrate(&read(&site.root.join(path)), inserts), &context)
        }
        Source::Home => home::render(&context),
        Source::Gallery => gallery::render(&context),
        Source::Playground => playground::render(&context),
    };
    let html = layout::page(site, section, page, &body);
    RenderedPage {
        html,
        headings: body
            .headings
            .iter()
            .map(|heading| heading.text.clone())
            .collect(),
        summary: body.summary,
    }
}

/// Inserts site material after the named headings of a repository
/// document. A heading that does not exist fails the build: the
/// insert would vanish silently.
fn illustrate(source: &str, inserts: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(source.len() + 1024);
    let mut pending: Vec<&(&str, &str)> = inserts.iter().collect();
    for line in source.lines() {
        out.push_str(line);
        out.push('\n');
        if let Some(index) = pending
            .iter()
            .position(|(heading, _)| *heading == line.trim_end())
        {
            let (_, material) = pending.remove(index);
            out.push('\n');
            out.push_str(material);
            out.push('\n');
        }
    }
    assert!(
        pending.is_empty(),
        "headings not found for inserts: {:?}",
        pending.iter().map(|(h, _)| h).collect::<Vec<_>>()
    );
    out
}

fn prepare(out: &Path) {
    // Keep `wasm/` — build.sh fills it after this generator runs, and a
    // local iteration should not have to rebuild it — and replace
    // everything else.
    if out.exists() {
        for entry in fs::read_dir(out).expect("output directory") {
            let entry = entry.expect("directory entry");
            if entry.file_name() == "wasm" {
                continue;
            }
            let path = entry.path();
            if path.is_dir() {
                fs::remove_dir_all(&path).expect("remove stale directory");
            } else {
                fs::remove_file(&path).expect("remove stale file");
            }
        }
    }
    fs::create_dir_all(out).expect("create output directory");
}

fn copy_dir(from: &Path, to: &Path) {
    for entry in fs::read_dir(from).unwrap_or_else(|error| panic!("{}: {error}", from.display())) {
        let entry = entry.expect("directory entry");
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            fs::create_dir_all(&target).expect("create directory");
            copy_dir(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), &target).expect("copy static file");
        }
    }
}

pub fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
}

fn write(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create page directory");
    }
    fs::write(path, content).unwrap_or_else(|error| panic!("writing {}: {error}", path.display()));
}

fn manifest_version(path: &Path) -> String {
    read(path)
        .lines()
        .find_map(|line| {
            line.strip_prefix("version = \"")
                .and_then(|rest| rest.strip_suffix('"'))
        })
        .expect("manifest has a version")
        .to_string()
}
