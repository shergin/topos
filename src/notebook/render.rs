//! Payload rendering: the one place that turns a `Differentiable`
//! into a table, a chart, or a number.
//!
//! Every display in this module ultimately shows payloads, so the
//! choice of form lives here rather than in each type's display. Small
//! payloads become tables because exact values matter at that size;
//! large ones become `malevich` charts because shape matters more than
//! any individual element.

use malevich::{Frame, Theme};

use super::html;
use crate::{Element, Emittable, Shape, Tensor};

/// The largest payload rendered as an exact table rather than a chart.
const TABLE_LIMIT: usize = 144;

/// The longest row a one-dimensional table prints before the payload
/// becomes a chart instead.
const ROW_LIMIT: usize = 24;

/// Most points a chart will hold. Larger payloads are subsampled
/// so a weight tensor cannot clone itself into `f64`.
const CHART_SAMPLE: usize = 2048;

/// The frame every chart in a notebook card is drawn into.
fn chart_frame(theme: Theme) -> Frame {
    let mut frame = Frame::plain(72, 20);
    frame.theme = theme;
    frame
}

/// Returns a tensor's elements in row-major order as the `f64` cells
/// a small table prints.
///
/// Callers must only use this when the payload is small enough to
/// tabulate; large payloads stream through [`extrema`] and
/// [`chart_cells`] instead.
pub(crate) fn cells<E: Element>(tensor: &Tensor<E>) -> Vec<f64>
where
    f64: From<E>,
{
    tensor.iter().map(f64::from).collect()
}

/// Minimum, maximum, and mean of the finite elements, plus how many
/// were not finite. A constant fill is O(1); a dense payload streams
/// without an extra buffer.
pub(crate) fn extrema<E: Element>(tensor: &Tensor<E>) -> (Option<(f64, f64, f64)>, usize)
where
    f64: From<E>,
{
    let volume = tensor.shape().volume();
    if volume == 0 {
        return (None, 0);
    }
    if let Some(value) = tensor.as_constant() {
        let cell = f64::from(value.clone());
        return if cell.is_finite() {
            (Some((cell, cell, cell)), 0)
        } else {
            (None, volume)
        };
    }
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    let mut sum = 0.0;
    let mut finite = 0usize;
    let mut unusual = 0usize;
    for element in tensor.iter() {
        let cell = f64::from(element);
        if cell.is_finite() {
            minimum = minimum.min(cell);
            maximum = maximum.max(cell);
            sum += cell;
            finite += 1;
        } else {
            unusual += 1;
        }
    }
    let summary = (finite > 0).then_some((minimum, maximum, sum / finite as f64));
    (summary, unusual)
}

/// Euclidean norm of a payload, streamed. A constant fill is O(1).
pub(crate) fn euclidean_norm<E: Element>(tensor: &Tensor<E>) -> f64
where
    f64: From<E>,
{
    if let Some(value) = tensor.as_constant() {
        let cell = f64::from(value.clone());
        if !cell.is_finite() {
            return cell.abs();
        }
        return cell.abs() * (tensor.shape().volume() as f64).sqrt();
    }
    tensor
        .iter()
        .map(|element| {
            let cell = f64::from(element);
            cell * cell
        })
        .sum::<f64>()
        .sqrt()
}

/// Points for a chart: the full payload when it is small, an even
/// subsample when it is not.
fn chart_cells<E: Element>(tensor: &Tensor<E>) -> Vec<f64>
where
    f64: From<E>,
{
    let volume = tensor.shape().volume();
    if volume == 0 {
        return Vec::new();
    }
    if let Some(value) = tensor.as_constant() {
        return vec![f64::from(value.clone())];
    }
    if volume <= CHART_SAMPLE {
        return tensor.iter().map(f64::from).collect();
    }
    let step = volume.div_ceil(CHART_SAMPLE).max(1);
    if let Some(slice) = tensor.as_slice() {
        return (0..volume)
            .step_by(step)
            .map(|index| f64::from(slice[index].clone()))
            .collect();
    }
    tensor.iter().step_by(step).map(f64::from).collect()
}

/// Formats one number at a width a table can align, without trailing
/// zeros and without scientific notation for ordinary magnitudes.
pub(crate) fn number(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let magnitude = value.abs();
    if !(1e-4..1e6).contains(&magnitude) {
        return format!("{value:.3e}");
    }
    let text = format!("{value:.4}");
    let trimmed = text.trim_end_matches('0').trim_end_matches('.');
    trimmed.to_string()
}

/// Renders `[2, 3]`-style axis text for a shape.
pub(crate) fn shape_text(shape: &Shape) -> String {
    let axes: Vec<String> = shape.axes().iter().map(usize::to_string).collect();
    format!("[{}]", axes.join(", "))
}

/// Extremes of an already-materialized cell list, for table tinting.
fn extremes_of(cells: &[f64]) -> Option<(f64, f64, f64)> {
    let finite: Vec<f64> = cells
        .iter()
        .copied()
        .filter(|cell| cell.is_finite())
        .collect();
    if finite.is_empty() {
        return None;
    }
    let minimum = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    Some((minimum, maximum, mean))
}

/// The card header for a payload: shape, element type, and the
/// summary statistics that say whether the numbers are sane.
pub(crate) fn header<E: Emittable>(
    shape: &Shape,
    summary: Option<(f64, f64, f64)>,
    unusual: usize,
) -> String {
    let mut parts = vec![shape_text(shape), E::ELEMENT.to_string()];
    if let Some((minimum, maximum, mean)) = summary {
        parts.push(format!(
            "min {} max {} mean {}",
            number(minimum),
            number(maximum),
            number(mean)
        ));
    }
    if unusual > 0 {
        parts.push(format!("{unusual} non-finite"));
    }
    html::escape(&parts.join("  \u{b7}  "))
}

/// Renders elements as an HTML table of `columns` per row, shading each
/// cell by where its value falls between the payload's extremes.
fn table(theme: Theme, cells: &[f64], columns: usize) -> String {
    use std::fmt::Write as _;

    let (low, high) = match extremes_of(cells) {
        Some((minimum, maximum, _)) => (minimum, maximum),
        None => (0.0, 0.0),
    };
    let span = high - low;
    let muted = html::muted_color(theme);
    let mut markup = String::from("<table style=\"border-collapse:collapse\">");
    for (index, cell) in cells.iter().enumerate() {
        if index % columns == 0 {
            let _ = write!(markup, "{}<tr>", if index == 0 { "" } else { "</tr>" });
        }
        // A flat payload gets no shading: every cell would tint identically
        // and the color would imply a variation that is not there.
        let weight = if span > 0.0 && cell.is_finite() {
            (cell - low) / span
        } else {
            0.0
        };
        let tint = format!("rgba(88,166,255,{:.3})", 0.10 + 0.55 * weight);
        let _ = write!(
            markup,
            "<td style=\"padding:2px 8px;text-align:right;background-color:{};\
             border:1px solid {muted}33\">{}</td>",
            if span > 0.0 {
                tint
            } else {
                "transparent".to_string()
            },
            html::escape(&number(*cell))
        );
    }
    markup.push_str("</tr></table>");
    markup
}

/// Renders elements as plain text laid out in rows of `columns`.
fn table_text(cells: &[f64], columns: usize) -> String {
    let rendered: Vec<String> = cells.iter().copied().map(number).collect();
    let width = rendered.iter().map(String::len).max().unwrap_or(1);
    let mut text = String::new();
    for (index, cell) in rendered.iter().enumerate() {
        if index > 0 && index % columns == 0 {
            text.push('\n');
        } else if index > 0 {
            text.push(' ');
        }
        let _ = std::fmt::Write::write_fmt(&mut text, format_args!("{cell:>width$}"));
    }
    text
}

/// The number of columns a payload's table prints, which is its last
/// axis for a shaped payload and its whole length for a flat one.
fn columns_of(shape: &Shape) -> usize {
    shape.axes().last().copied().unwrap_or(1).max(1)
}

/// Renders a payload's body: the exact values when they are few, and a
/// chart of them when they are many.
pub(crate) fn body<E: Element>(theme: Theme, data: &Tensor<E>) -> (String, String)
where
    f64: From<E>,
{
    let shape = data.shape();
    let volume = shape.volume();
    let columns = columns_of(&shape);

    if volume == 1 {
        let value = match data.as_constant() {
            Some(element) => f64::from(element.clone()),
            None => data.iter().next().map(f64::from).unwrap_or(0.0),
        };
        let text = number(value);
        return (
            format!(
                "<div style=\"font-size:20px\">{}</div>",
                html::escape(&text)
            ),
            text,
        );
    }

    let small_row = shape.rank() <= 1 && volume <= ROW_LIMIT;
    let small_grid = shape.rank() >= 2 && volume <= TABLE_LIMIT;
    if small_row || small_grid {
        let cells = cells(data);
        return (table(theme, &cells, columns), table_text(&cells, columns));
    }

    let sampled = chart_cells(data);
    let frame = chart_frame(theme);
    let full_heatmap = shape.rank() >= 2 && volume <= CHART_SAMPLE;
    let plot = if full_heatmap {
        malevich::heatmap(columns, &sampled[..])
    } else {
        malevich::Plot::new().layer(malevich::Line::y(&sampled[..]))
    };
    (plot.to_html(&frame), plot.render(&frame))
}

/// Renders a complete payload card: header, then body.
pub(crate) fn payload_card<E: Element + Emittable>(
    theme: Theme,
    label: &str,
    data: &Tensor<E>,
) -> String
where
    f64: From<E>,
{
    let shape = data.shape();
    let (summary, unusual) = extrema(data);
    let (body_html, _) = body(theme, data);
    let head = format!(
        "{}  \u{b7}  {}",
        html::escape(label),
        header::<E>(&shape, summary, unusual)
    );
    html::card(theme, &head, &body_html)
}

/// Renders a complete payload as plain text: header, then body.
pub(crate) fn payload_text<E: Element + Emittable>(label: &str, data: &Tensor<E>) -> String
where
    f64: From<E>,
{
    let shape = data.shape();
    let (summary, _) = extrema(data);
    let (_, body_text) = body(Theme::DARK, data);
    let mut parts = vec![shape_text(&shape), E::ELEMENT.to_string()];
    if let Some((minimum, maximum, mean)) = summary {
        parts.push(format!(
            "min {} max {} mean {}",
            number(minimum),
            number(maximum),
            number(mean)
        ));
    }
    format!("{label}  {}\n{body_text}", parts.join("  "))
}

#[cfg(test)]
#[path = "tests/render_tests.rs"]
mod tests;
