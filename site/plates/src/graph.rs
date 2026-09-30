//! A recorded graph as a picture: the nodes `Network::nodes` answers,
//! laid out in ranks by dependency and drawn as an SVG the page
//! styles. Edges run from an operand down to the node that reads it,
//! entering in operand order, so `MatMul 1, 0` shows which side is
//! which.
//!
//! The layout is the classic layered one: rank by longest path from
//! the sources, sources sunk to just above their first reader, and a
//! few barycenter sweeps to keep edges short. Small graphs — the ones
//! a page can teach from — come out readable; nothing here aims at a
//! thousand nodes.

use std::fmt::Write;

use topos::{Node, Opcode};

use crate::escape;

/// One node to draw.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// The node's index on its tape.
    pub index: usize,
    /// The opcode's display name.
    pub name: String,
    /// The opcode's parameters, as `describe` prints them.
    pub detail: String,
    /// The shape, as `describe` prints it.
    pub shape: String,
    /// The operands' indices, in positional order.
    pub operands: Vec<usize>,
    /// Whether the node is supplied rather than computed.
    pub source: bool,
    /// Extra CSS classes: `derived`, `kept`, `skipped`, and so on.
    pub classes: Vec<String>,
}

/// A graph to draw.
#[derive(Debug, Clone, Default)]
pub struct Graph {
    pub nodes: Vec<GraphNode>,
}

impl Graph {
    /// Builds the picture's nodes from a spec's nodes.
    pub fn from_nodes(nodes: impl IntoIterator<Item = Node>) -> Graph {
        Graph {
            nodes: nodes.into_iter().map(GraphNode::from).collect(),
        }
    }

    /// Adds a class to every node whose index is at or past `split`.
    pub fn classify_from(mut self, split: usize, class: &str) -> Graph {
        for node in &mut self.nodes {
            if node.index >= split {
                node.classes.push(class.to_string());
            }
        }
        self
    }

    /// Adds a class to the nodes with the given indices.
    pub fn classify(mut self, indices: &[usize], class: &str) -> Graph {
        for node in &mut self.nodes {
            if indices.contains(&node.index) {
                node.classes.push(class.to_string());
            }
        }
        self
    }

    /// Adds a class to every node whose index is *not* in `indices`.
    pub fn classify_others(mut self, indices: &[usize], class: &str) -> Graph {
        for node in &mut self.nodes {
            if !indices.contains(&node.index) {
                node.classes.push(class.to_string());
            }
        }
        self
    }
}

impl From<Node> for GraphNode {
    fn from(node: Node) -> GraphNode {
        GraphNode {
            index: node.symbol().index(),
            name: node.name().to_string(),
            detail: detail(node.opcode()),
            shape: node.shape().to_string(),
            operands: node
                .operands()
                .iter()
                .map(|symbol| symbol.index())
                .collect(),
            source: node.is_source(),
            classes: Vec::new(),
        }
    }
}

/// The parameters a describe line prints after the operands.
fn detail(opcode: &Opcode) -> String {
    match opcode {
        Opcode::SumAlong { axis } | Opcode::LogSoftmax { axis } | Opcode::LogSumExp { axis } => {
            format!("axis={axis}")
        }
        Opcode::Broadcast { shape } | Opcode::Reshape { shape } => format!("shape={shape}"),
        Opcode::BroadcastAlong { axis, extent } => format!("axis={axis} extent={extent}"),
        Opcode::Permute { order } => {
            let axes: Vec<String> = order.iter().map(usize::to_string).collect();
            format!("order=[{}]", axes.join(", "))
        }
        Opcode::Narrow { axis, start, len } => format!("axis={axis} start={start} len={len}"),
        Opcode::Pad {
            axis,
            start,
            full_extent,
        } => format!("axis={axis} start={start} full_extent={full_extent}"),
        Opcode::Unfold {
            axis,
            size,
            step,
            dilation,
        } => format!("axis={axis} size={size} step={step} dilation={dilation}"),
        Opcode::Fold {
            axis,
            size,
            step,
            dilation,
            extent,
        } => format!("axis={axis} size={size} step={step} dilation={dilation} extent={extent}"),
        _ => String::new(),
    }
}

/// Geometry, in SVG user units.
const NODE_HEIGHT: f64 = 36.0;
const CHAR_WIDTH: f64 = 7.2;
const NODE_PADDING: f64 = 18.0;
const NODE_GAP: f64 = 16.0;
const RANK_GAP: f64 = 40.0;
const MARGIN: f64 = 10.0;
const PORT_SPREAD: f64 = 12.0;

struct Placed {
    x: f64,
    y: f64,
    width: f64,
}

/// Ranks the nodes: computed nodes one past their deepest operand,
/// sources just above their first reader.
fn ranks(graph: &Graph) -> Vec<usize> {
    let count = graph.nodes.len();
    let position: std::collections::HashMap<usize, usize> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(at, node)| (node.index, at))
        .collect();
    let mut rank = vec![0usize; count];
    for (at, node) in graph.nodes.iter().enumerate() {
        if node.source {
            continue;
        }
        rank[at] = node
            .operands
            .iter()
            .filter_map(|operand| position.get(operand))
            .map(|&operand| rank[operand] + 1)
            .max()
            .unwrap_or(0);
    }
    // Sink each source to one rank above its earliest reader.
    for (at, node) in graph.nodes.iter().enumerate() {
        if !node.source {
            continue;
        }
        let first_reader = graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, reader)| reader.operands.contains(&node.index))
            .map(|(reader, _)| rank[reader])
            .min();
        if let Some(reader) = first_reader {
            rank[at] = reader.saturating_sub(1);
        }
    }
    // Close the gaps.
    let mut used: Vec<usize> = rank.clone();
    used.sort_unstable();
    used.dedup();
    for value in &mut rank {
        *value = used.iter().position(|u| u == value).unwrap_or(0);
    }
    rank
}

/// Orders each rank by the barycenter of the neighbors in the rank
/// before it, then the rank after it, a few times over.
fn order(graph: &Graph, rank: &[usize]) -> Vec<Vec<usize>> {
    let depth = rank.iter().copied().max().map_or(0, |max| max + 1);
    let mut layers: Vec<Vec<usize>> = vec![Vec::new(); depth];
    for (at, &r) in rank.iter().enumerate() {
        layers[r].push(at);
    }
    let position: std::collections::HashMap<usize, usize> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(at, node)| (node.index, at))
        .collect();
    let readers: Vec<Vec<usize>> = graph
        .nodes
        .iter()
        .map(|node| {
            graph
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, reader)| reader.operands.contains(&node.index))
                .map(|(reader, _)| reader)
                .collect()
        })
        .collect();
    let slot = |layers: &Vec<Vec<usize>>, at: usize| -> f64 {
        let layer = &layers[rank[at]];
        layer.iter().position(|&n| n == at).unwrap_or(0) as f64
    };
    for sweep in 0..4 {
        let downward = sweep % 2 == 0;
        let sequence: Vec<usize> = if downward {
            (1..depth).collect()
        } else {
            (0..depth.saturating_sub(1)).rev().collect()
        };
        for r in sequence {
            let mut keyed: Vec<(f64, usize)> = layers[r]
                .iter()
                .map(|&at| {
                    let neighbors: Vec<f64> = if downward {
                        graph.nodes[at]
                            .operands
                            .iter()
                            .filter_map(|operand| position.get(operand))
                            .filter(|&&operand| rank[operand] < r)
                            .map(|&operand| slot(&layers, operand))
                            .collect()
                    } else {
                        readers[at]
                            .iter()
                            .filter(|&&reader| rank[reader] > r)
                            .map(|&reader| slot(&layers, reader))
                            .collect()
                    };
                    let key = if neighbors.is_empty() {
                        slot(&layers, at)
                    } else {
                        neighbors.iter().sum::<f64>() / neighbors.len() as f64
                    };
                    (key, at)
                })
                .collect();
            keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            layers[r] = keyed.into_iter().map(|(_, at)| at).collect();
        }
    }
    layers
}

fn label(node: &GraphNode) -> (String, String) {
    let head = format!("{} {}", node.index, node.name);
    let foot = if node.detail.is_empty() {
        node.shape.clone()
    } else {
        format!("{} {}", node.shape, node.detail)
    };
    (head, foot)
}

/// Renders the graph as an inline SVG.
pub fn svg(graph: &Graph, title: &str) -> String {
    if graph.nodes.is_empty() {
        return String::from("<svg class=\"graph\" viewBox=\"0 0 10 10\"></svg>");
    }
    let rank = ranks(graph);
    let layers = order(graph, &rank);
    let widths: Vec<f64> = graph
        .nodes
        .iter()
        .map(|node| {
            let (head, foot) = label(node);
            let longest = head.chars().count().max(foot.chars().count()) as f64;
            longest * CHAR_WIDTH + NODE_PADDING
        })
        .collect();
    let layer_width = |layer: &[usize]| -> f64 {
        layer.iter().map(|&at| widths[at]).sum::<f64>()
            + NODE_GAP * layer.len().saturating_sub(1) as f64
    };
    let total_width = layers
        .iter()
        .map(|layer| layer_width(layer))
        .fold(0.0, f64::max)
        + 2.0 * MARGIN;
    let total_height = layers.len() as f64 * (NODE_HEIGHT + RANK_GAP) - RANK_GAP + 2.0 * MARGIN;
    let mut placed: Vec<Placed> = graph
        .nodes
        .iter()
        .map(|_| Placed {
            x: 0.0,
            y: 0.0,
            width: 0.0,
        })
        .collect();
    for (r, layer) in layers.iter().enumerate() {
        let mut x = (total_width - layer_width(layer)) / 2.0;
        let y = MARGIN + r as f64 * (NODE_HEIGHT + RANK_GAP);
        for &at in layer {
            placed[at] = Placed {
                x,
                y,
                width: widths[at],
            };
            x += widths[at] + NODE_GAP;
        }
    }
    let position: std::collections::HashMap<usize, usize> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(at, node)| (node.index, at))
        .collect();

    let mut out = String::new();
    let _ = write!(
        out,
        "<svg class=\"graph\" role=\"img\" aria-label=\"{}\" viewBox=\"0 0 {:.0} {:.0}\" style=\"max-width:{:.0}px\">",
        escape(title),
        total_width,
        total_height,
        total_width * 1.15
    );
    out.push_str("<defs><marker id=\"arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"7\" markerHeight=\"7\" orient=\"auto-start-reverse\"><path d=\"M0,0 L10,5 L0,10 z\"/></marker></defs>");
    // Edges first, so nodes paint over them.
    out.push_str("<g class=\"edges\">");
    for (at, node) in graph.nodes.iter().enumerate() {
        let count = node.operands.len();
        for (slot, operand) in node.operands.iter().enumerate() {
            let Some(&from) = position.get(operand) else {
                continue;
            };
            let source = &placed[from];
            let target = &placed[at];
            let x1 = source.x + source.width / 2.0;
            let y1 = source.y + NODE_HEIGHT;
            let offset = (slot as f64 - (count as f64 - 1.0) / 2.0) * PORT_SPREAD;
            let x2 = target.x + target.width / 2.0 + offset;
            let y2 = target.y;
            let bend = ((y2 - y1) / 2.0).max(14.0);
            let _ = write!(
                out,
                "<path class=\"edge\" data-from=\"{}\" data-to=\"{}\" d=\"M{x1:.1},{y1:.1} C{x1:.1},{:.1} {x2:.1},{:.1} {x2:.1},{:.1}\" marker-end=\"url(#arrow)\"/>",
                node_index_of(graph, from),
                node.index,
                y1 + bend,
                y2 - bend,
                y2 - 1.0
            );
        }
    }
    out.push_str("</g><g class=\"nodes\">");
    for (at, node) in graph.nodes.iter().enumerate() {
        let place = &placed[at];
        let (head, foot) = label(node);
        let mut classes = vec!["node"];
        if node.source {
            classes.push("source");
        }
        let extra: Vec<&str> = node.classes.iter().map(String::as_str).collect();
        classes.extend(extra);
        let _ = write!(
            out,
            "<g class=\"{}\" data-node=\"{}\" transform=\"translate({:.1},{:.1})\"><rect width=\"{:.1}\" height=\"{NODE_HEIGHT}\" rx=\"5\"/><text class=\"head\" x=\"{:.1}\" y=\"15\" text-anchor=\"middle\">{}</text><text class=\"foot\" x=\"{:.1}\" y=\"29\" text-anchor=\"middle\">{}</text><title>{}</title></g>",
            classes.join(" "),
            node.index,
            place.x,
            place.y,
            place.width,
            place.width / 2.0,
            escape(&head),
            place.width / 2.0,
            escape(&foot),
            escape(&format!("{head}  {foot}"))
        );
    }
    out.push_str("</g></svg>");
    out
}

fn node_index_of(graph: &Graph, at: usize) -> usize {
    graph.nodes[at].index
}

#[cfg(test)]
mod tests {
    use super::*;
    use topos::{Detach, Tape};

    #[test]
    fn a_small_spec_draws_every_node_and_edge() {
        let (network, _) = Tape::record(|tape| {
            let w = tape.parameter(1.0_f64);
            let x = tape.input(2.0);
            [(w * x - x) * (w * x - x)].detach()
        });
        let graph = Graph::from_nodes(network.nodes());
        let picture = svg(&graph, "test");
        assert_eq!(picture.matches("data-node=\"").count(), network.len());
        let edges: usize = network.nodes().map(|node| node.operands().len()).sum();
        assert_eq!(picture.matches("class=\"edge\"").count(), edges);
    }

    #[test]
    fn sources_sit_above_their_first_reader() {
        let (network, _) = Tape::record(|tape| {
            let a = tape.parameter(1.0_f64);
            let b = tape.parameter(2.0);
            let c = tape.leaf(3.0);
            [(a * b) * c].detach()
        });
        let graph = Graph::from_nodes(network.nodes());
        let rank = ranks(&graph);
        // `c` is read by the second product, one rank below the first.
        assert_eq!(rank[2], rank[3]);
        assert_eq!(rank[0], rank[1]);
        assert!(rank[4] > rank[3]);
    }
}
