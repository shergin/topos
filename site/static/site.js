// Site chrome: the contents toggle, the search box, copy buttons, the table
// of contents that follows the reader, tabs, and the two hovers that make a
// reading legible — an IR line lights up what it reads and who reads it,
// and a graph node lights up its edges.

(() => {
  const root = window.TOPOS_ROOT ?? "./";

  // Contents toggle on narrow screens.
  const toggle = document.querySelector(".nav-toggle");
  const sidebar = document.getElementById("sidebar");
  if (toggle && sidebar) {
    toggle.addEventListener("click", () => {
      const open = sidebar.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(open));
    });
  }

  // Copy buttons on code blocks.
  for (const block of document.querySelectorAll(".code")) {
    const pre = block.querySelector("pre");
    if (!pre) continue;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy";
    button.textContent = "copy";
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(pre.textContent ?? "");
        button.textContent = "copied";
      } catch {
        button.textContent = "select it";
      }
      window.setTimeout(() => {
        button.textContent = "copy";
      }, 1400);
    });
    block.append(button);
  }

  // Tabs: several readings of one graph under one plate.
  const wireTabs = (scope) => {
    for (const panels of scope.querySelectorAll(".panels")) {
      const tabs = [...panels.querySelectorAll(":scope > .tabs > button")];
      const bodies = [...panels.querySelectorAll(":scope > .panel")];
      for (const tab of tabs) {
        tab.addEventListener("click", () => {
          for (const other of tabs) other.setAttribute("aria-selected", String(other === tab));
          for (const body of bodies) body.hidden = body.dataset.panel !== tab.dataset.panel;
        });
      }
    }
  };

  // The IR plate: hovering a line marks the lines it reads (operands) and
  // the lines that read it (consumers).
  const wireIr = (scope) => {
    for (const plate of scope.querySelectorAll("pre.ir")) {
      const lines = [...plate.querySelectorAll(".ir-line")];
      if (lines.length === 0) continue;
      const byNode = new Map(lines.map((line) => [line.dataset.node, line]));
      const consumers = new Map();
      for (const line of lines) {
        for (const ref of line.querySelectorAll(".ir-ref")) {
          const list = consumers.get(ref.dataset.ref) ?? [];
          list.push(line);
          consumers.set(ref.dataset.ref, list);
        }
      }
      const clear = () => {
        for (const line of lines) line.classList.remove("is-hover", "is-operand", "is-consumer");
      };
      for (const line of lines) {
        line.addEventListener("mouseenter", () => {
          clear();
          line.classList.add("is-hover");
          for (const ref of line.querySelectorAll(".ir-ref")) byNode.get(ref.dataset.ref)?.classList.add("is-operand");
          for (const consumer of consumers.get(line.dataset.node) ?? []) consumer.classList.add("is-consumer");
        });
      }
      plate.addEventListener("mouseleave", clear);
      if (!plate.nextElementSibling?.classList.contains("ir-hint")) {
        const hint = document.createElement("p");
        hint.className = "ir-hint";
        hint.textContent = "hover a line: green is what it reads, orange is who reads it";
        plate.after(hint);
      }
    }
  };

  // The graph picture: hovering a node lights its edges and neighbors.
  const wireGraphs = (scope) => {
    for (const graph of scope.querySelectorAll("svg.graph")) {
      const nodes = [...graph.querySelectorAll("g.node")];
      const edges = [...graph.querySelectorAll("path.edge")];
      const clear = () => {
        for (const edge of edges) edge.classList.remove("is-hover");
        for (const node of nodes) node.classList.remove("is-hover");
      };
      for (const node of nodes) {
        node.addEventListener("mouseenter", () => {
          clear();
          const id = node.dataset.node;
          for (const edge of edges) {
            if (edge.dataset.from === id || edge.dataset.to === id) {
              edge.classList.add("is-hover");
              const other = edge.dataset.from === id ? edge.dataset.to : edge.dataset.from;
              graph.querySelector(`g.node[data-node="${other}"]`)?.classList.add("is-hover");
            }
          }
        });
      }
      graph.addEventListener("mouseleave", clear);
    }
  };

  window.toposWire = (scope) => {
    wireTabs(scope);
    wireIr(scope);
    wireGraphs(scope);
  };
  window.toposWire(document);

  // The table of contents follows the section in view.
  const toc = document.querySelector(".toc");
  if (toc && "IntersectionObserver" in window) {
    const links = new Map();
    for (const link of toc.querySelectorAll("a[href^='#']")) {
      links.set(decodeURIComponent(link.getAttribute("href").slice(1)), link);
    }
    const headings = [...links.keys()].map((id) => document.getElementById(id)).filter(Boolean);
    let current = null;
    const mark = (id) => {
      if (current) current.removeAttribute("aria-current");
      current = links.get(id) ?? null;
      if (current) current.setAttribute("aria-current", "true");
    };
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) mark(entry.target.id);
        }
      },
      { rootMargin: "-10% 0px -80% 0px" },
    );
    for (const heading of headings) observer.observe(heading);
  }

  // Search: a small index, substring matching, keyboard navigable.
  const input = document.getElementById("search");
  const results = document.getElementById("results");
  if (!input || !results) return;
  let index = null;
  let selected = -1;
  const load = async () => {
    if (index) return index;
    const response = await fetch(`${root}search.json`);
    index = await response.json();
    return index;
  };
  const escape = (text) =>
    text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const snippet = (text, query) => {
    const at = text.toLowerCase().indexOf(query);
    if (at < 0) return text.slice(0, 110);
    const start = Math.max(0, at - 40);
    return (start > 0 ? "…" : "") + text.slice(start, start + 120) + "…";
  };
  const score = (entry, query) => {
    const title = entry.title.toLowerCase();
    if (title === query) return 100;
    if (title.startsWith(query)) return 60;
    if (title.includes(query)) return 40;
    const heading = entry.headings.find((h) => h.toLowerCase().includes(query));
    if (heading) return 25;
    if (entry.text.toLowerCase().includes(query)) return 10;
    return 0;
  };
  const slug = (text) =>
    text
      .toLowerCase()
      .replace(/[^\p{L}\p{N} -]/gu, "")
      .replace(/[ ]/g, "-");
  const render = async () => {
    const query = input.value.trim().toLowerCase();
    selected = -1;
    if (query.length < 2) {
      results.hidden = true;
      results.innerHTML = "";
      return;
    }
    const entries = await load();
    const hits = entries
      .map((entry) => ({ entry, score: score(entry, query) }))
      .filter((hit) => hit.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 8);
    if (hits.length === 0) {
      results.innerHTML = `<p class="empty">Nothing matches “${escape(input.value.trim())}”.</p>`;
      results.hidden = false;
      return;
    }
    results.innerHTML = hits
      .map(({ entry }) => {
        const heading = entry.headings.find((h) => h.toLowerCase().includes(query));
        const anchor = heading ? `#${slug(heading)}` : "";
        const where = heading ? `${entry.section} › ${escape(entry.title)}` : entry.section;
        const label = heading ? escape(heading) : escape(entry.title);
        return `<a href="${root}${entry.url.replace(/^\//, "")}${anchor}"><span class="where">${where}</span>${label}<span class="snippet">${escape(snippet(entry.text, query))}</span></a>`;
      })
      .join("");
    results.hidden = false;
  };
  input.addEventListener("input", render);
  input.addEventListener("focus", render);
  input.addEventListener("keydown", (event) => {
    const items = [...results.querySelectorAll("a")];
    if (event.key === "Escape") {
      results.hidden = true;
      input.blur();
      return;
    }
    if (items.length === 0) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      items[selected]?.removeAttribute("aria-selected");
      selected = event.key === "ArrowDown" ? (selected + 1) % items.length : (selected - 1 + items.length) % items.length;
      items[selected].setAttribute("aria-selected", "true");
      items[selected].scrollIntoView({ block: "nearest" });
    }
    if (event.key === "Enter" && selected >= 0) {
      event.preventDefault();
      items[selected].click();
    }
  });
  document.addEventListener("click", (event) => {
    if (!results.contains(event.target) && event.target !== input) results.hidden = true;
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement !== input && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName ?? "")) {
      event.preventDefault();
      input.focus();
    }
  });
})();
