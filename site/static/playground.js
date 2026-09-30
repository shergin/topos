// The playground: a live editor over the wasm build. The program on the
// left records onto a real tape; every tab on the right is one reading of
// that recording, remade as you type. The state lives in the URL hash, so
// a playground can be linked.

import init, { readings } from "./wasm/topos_playground.js";

const SAMPLES = {
  readme: {
    program: [
      "# The README's loss: one weight fit to y = 2x.",
      "w = parameter 0.0",
      "x = input 2.0",
      "y = input 4.0",
      "error = w * x - y",
      "loss = error * error",
    ].join("\n"),
    target: "loss",
    observe: "error",
  },
  "dense layer": {
    program: [
      "# A matrix product, a tanh, and a loss: the walkthrough's graph.",
      "weights = parameter [2, 2]",
      "input = input [1, 2]",
      "hidden = tanh(input @ weights)",
      "loss = sum(hidden * hidden)",
    ].join("\n"),
    target: "loss",
    observe: "hidden",
  },
  classifier: {
    program: [
      "# A softmax classifier over four samples and three classes.",
      "w = parameter [5, 3]",
      "b = parameter [3]",
      "x = input [4, 5]",
      "logits = x @ w + broadcast(0.0, [4, 3])",
      "scores = log_softmax(logits, 1)",
      "loss = -sum(scores) / 4.0",
    ].join("\n"),
    target: "loss",
    observe: "",
  },
  "two layers": {
    program: [
      "# Two tanh layers, and the mean of the output as a loss.",
      "w1 = parameter [3, 8]",
      "w2 = parameter [8, 2]",
      "x = input [6, 3]",
      "h = tanh(x @ w1)",
      "y = tanh(h @ w2)",
      "loss = sum(y * y)",
    ].join("\n"),
    target: "loss",
    observe: "h",
  },
  "not yet a scalar": {
    program: [
      "# A gradient needs a rank-0 target; watch the derivative tab.",
      "w = parameter [2, 2]",
      "x = input [1, 2]",
      "y = relu(x @ w)",
    ].join("\n"),
    target: "y",
    observe: "",
  },
};

const TABS = [
  ["spec", "spec"],
  ["derivative", "derivative"],
  ["plan", "plan"],
  ["StableHLO", "hlo"],
  ["values", "values"],
  ["graph", "graph"],
];

const program = document.getElementById("program");
const error = document.getElementById("error");
const targetSelect = document.getElementById("target");
const observeSelect = document.getElementById("observe");
const exact = document.getElementById("exact");
const samples = document.getElementById("samples");
const readingsBox = document.getElementById("readings");

let currentTab = "spec";
let ready = false;

// The state in the URL hash, so a playground can be linked.
function readHash() {
  const hash = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  return {
    program: hash.get("program"),
    target: hash.get("target") ?? "",
    observe: hash.get("observe") ?? "",
    exact: hash.get("exact") === "1",
    tab: hash.get("tab") ?? "spec",
  };
}

function writeHash() {
  const hash = new URLSearchParams();
  hash.set("program", program.value);
  if (targetSelect.value) hash.set("target", targetSelect.value);
  if (observeSelect.value) hash.set("observe", observeSelect.value);
  if (exact.checked) hash.set("exact", "1");
  if (currentTab !== "spec") hash.set("tab", currentTab);
  window.history.replaceState(null, "", `#${hash.toString()}`);
}

function fillSelect(select, names, wanted, placeholder) {
  const previous = wanted ?? select.value;
  select.innerHTML = "";
  const none = document.createElement("option");
  none.value = "";
  none.textContent = placeholder;
  select.append(none);
  for (const named of names) {
    const option = document.createElement("option");
    option.value = named.name;
    option.textContent = `${named.name}  ${named.shape}`;
    select.append(option);
  }
  select.value = names.some((named) => named.name === previous) ? previous : "";
}

function render() {
  if (!ready) return;
  let result;
  try {
    result = JSON.parse(readings(program.value, targetSelect.value, observeSelect.value, exact.checked));
  } catch (failure) {
    // A trap leaves the module in an unknown state: reload it, then say so.
    error.hidden = false;
    error.textContent = `the engine stopped: ${failure.message ?? failure}. Reloading it.`;
    ready = false;
    init().then(() => {
      ready = true;
      render();
    });
    return;
  }
  if (result.error) {
    error.hidden = false;
    error.textContent = result.error;
    return;
  }
  error.hidden = true;
  const scalars = result.names.filter((named) => named.scalar);
  fillSelect(targetSelect, scalars, targetSelect.value, "the last scalar");
  fillSelect(observeSelect, result.names, observeSelect.value, "nothing extra");

  const tabs = TABS.map(
    ([label, key]) =>
      `<button type="button" role="tab" aria-selected="${key === currentTab}" data-panel="${key}">${label}</button>`,
  ).join("");
  const panels = TABS.map(
    ([, key]) =>
      `<div class="panel" role="tabpanel" data-panel="${key}"${key === currentTab ? "" : " hidden"}>${result[key]}</div>`,
  ).join("");
  readingsBox.innerHTML = `<div class="panels"><div class="tabs" role="tablist">${tabs}</div>${panels}</div><p class="readout">${result.summary}</p>`;
  for (const tab of readingsBox.querySelectorAll(".tabs button")) {
    tab.addEventListener("click", () => {
      currentTab = tab.dataset.panel;
      writeHash();
    });
  }
  window.toposWire?.(readingsBox);
  writeHash();
}

let pending = null;
function schedule() {
  window.clearTimeout(pending);
  pending = window.setTimeout(render, 120);
}

function load(name) {
  const sample = SAMPLES[name];
  program.value = sample.program;
  targetSelect.value = "";
  observeSelect.value = "";
  for (const button of samples.querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.textContent === name));
  }
  // The selects are refilled by the first render; then the sample's
  // choices apply.
  render();
  targetSelect.value = sample.target;
  observeSelect.value = sample.observe;
  render();
}

for (const name of Object.keys(SAMPLES)) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = name;
  button.setAttribute("aria-pressed", "false");
  button.addEventListener("click", () => load(name));
  samples.append(button);
}

program.addEventListener("input", () => {
  for (const button of samples.querySelectorAll("button")) button.setAttribute("aria-pressed", "false");
  schedule();
});
program.addEventListener("keydown", (event) => {
  // Tab indents instead of leaving the editor.
  if (event.key === "Tab") {
    event.preventDefault();
    const { selectionStart, selectionEnd, value } = program;
    program.value = `${value.slice(0, selectionStart)}    ${value.slice(selectionEnd)}`;
    program.selectionStart = program.selectionEnd = selectionStart + 4;
    schedule();
  }
});
targetSelect.addEventListener("change", render);
observeSelect.addEventListener("change", render);
exact.addEventListener("change", render);

init()
  .then(() => {
    ready = true;
    const state = readHash();
    currentTab = TABS.some(([, key]) => key === state.tab) ? state.tab : "spec";
    if (state.program) {
      program.value = state.program;
      render();
      targetSelect.value = state.target;
      observeSelect.value = state.observe;
      exact.checked = state.exact;
      render();
    } else {
      load("readme");
    }
  })
  .catch((failure) => {
    readingsBox.innerHTML = `<p class="status error">The engine did not load: ${failure.message ?? failure}</p>`;
  });
