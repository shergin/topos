//! The tape language: a list of `name = expression` lines, each
//! recording onto a `Tape<f32>`. The grammar is small on purpose —
//! sources with a shape or a value, arithmetic, the matrix product,
//! and calls for the maps, reductions, and shape changes — and every
//! shape is checked here, before the crate is asked to record, so a
//! mistake is a message with both shapes in it.

use std::collections::HashMap;

use topos::{Shape, Symbol, Tape, Tensor, Value, init};

/// What a program recorded: its names in statement order, and the
/// parameters among them.
#[derive(Debug)]
pub struct Recorded {
    pub names: Vec<(String, Symbol, Shape)>,
    pub parameters: Vec<Symbol>,
}

impl Recorded {
    /// The symbol a name refers to: the last assignment of it.
    pub fn symbol(&self, name: &str) -> Option<Symbol> {
        self.names
            .iter()
            .rev()
            .find(|(candidate, _, _)| candidate == name)
            .map(|(_, symbol, _)| *symbol)
    }

    /// The name of a symbol, or `#index` for an unnamed node.
    pub fn name_of(&self, symbol: Symbol) -> String {
        self.names
            .iter()
            .rev()
            .find(|(_, candidate, _)| *candidate == symbol)
            .map(|(name, _, _)| name.clone())
            .unwrap_or_else(|| format!("#{}", symbol.index()))
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Name(String),
    Number(f32),
    Shape(Vec<usize>),
    Symbol(char),
    Newline,
}

fn lex(source: &str) -> Result<Vec<(Token, usize)>, String> {
    let mut tokens = Vec::new();
    for (line_number, line) in source.lines().enumerate() {
        let line_number = line_number + 1;
        let chars: Vec<char> = line.chars().collect();
        let mut at = 0;
        while at < chars.len() {
            let c = chars[at];
            if c == '#' {
                break;
            }
            if c.is_whitespace() {
                at += 1;
                continue;
            }
            if c.is_ascii_digit()
                || (c == '.' && chars.get(at + 1).is_some_and(|d| d.is_ascii_digit()))
            {
                let start = at;
                while at < chars.len()
                    && (chars[at].is_ascii_digit()
                        || chars[at] == '.'
                        || chars[at] == 'e'
                        || chars[at] == 'E'
                        || ((chars[at] == '-' || chars[at] == '+')
                            && matches!(chars.get(at - 1), Some('e') | Some('E'))))
                {
                    at += 1;
                }
                let text: String = chars[start..at].iter().collect();
                let number: f32 = text
                    .parse()
                    .map_err(|_| format!("line {line_number}: `{text}` is not a number"))?;
                tokens.push((Token::Number(number), line_number));
                continue;
            }
            if c.is_alphabetic() || c == '_' {
                let start = at;
                while at < chars.len() && (chars[at].is_alphanumeric() || chars[at] == '_') {
                    at += 1;
                }
                tokens.push((Token::Name(chars[start..at].iter().collect()), line_number));
                continue;
            }
            if c == '[' {
                let mut end = at + 1;
                while end < chars.len() && chars[end] != ']' {
                    end += 1;
                }
                if end >= chars.len() {
                    return Err(format!("line {line_number}: a shape needs a closing `]`"));
                }
                let inner: String = chars[at + 1..end].iter().collect();
                let mut axes = Vec::new();
                for part in inner.split(',') {
                    let part = part.trim();
                    if part.is_empty() {
                        continue;
                    }
                    let extent: usize = part.parse().map_err(|_| {
                        format!("line {line_number}: `{part}` is not an axis extent")
                    })?;
                    if extent == 0 {
                        return Err(format!(
                            "line {line_number}: an axis extent must be at least 1"
                        ));
                    }
                    axes.push(extent);
                }
                tokens.push((Token::Shape(axes), line_number));
                at = end + 1;
                continue;
            }
            if "=+-*/@(),".contains(c) {
                tokens.push((Token::Symbol(c), line_number));
                at += 1;
                continue;
            }
            return Err(format!("line {line_number}: unexpected `{c}`"));
        }
        tokens.push((Token::Newline, line_number));
    }
    Ok(tokens)
}

/// The recorder: the tape, the names so far, and the cursor over the
/// tokens.
struct Recorder<'tape> {
    tape: &'tape Tape<f32>,
    tokens: Vec<(Token, usize)>,
    at: usize,
    names: HashMap<String, Value<'tape, f32>>,
    order: Vec<(String, Symbol, Shape)>,
    parameters: Vec<Symbol>,
    seed: u64,
}

const SOURCES: [&str; 3] = ["parameter", "input", "leaf"];

const UNARY: [&str; 8] = [
    "tanh",
    "exp",
    "ln",
    "sqrt",
    "sin",
    "cos",
    "relu",
    "transpose",
];

const ALONG: [&str; 5] = [
    "sum_along",
    "log_softmax",
    "logsumexp",
    "softmax",
    "mean_along",
];

const SHAPED: [&str; 2] = ["broadcast", "reshape"];

/// Records `program` on `tape`.
pub fn record(tape: &Tape<f32>, program: &str) -> Result<Recorded, String> {
    let tokens = lex(program)?;
    let mut recorder = Recorder {
        tape,
        tokens,
        at: 0,
        names: HashMap::new(),
        order: Vec::new(),
        parameters: Vec::new(),
        seed: 1,
    };
    recorder.program()?;
    Ok(Recorded {
        names: recorder.order,
        parameters: recorder.parameters,
    })
}

impl<'tape> Recorder<'tape> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.at).map(|(token, _)| token)
    }

    fn line(&self) -> usize {
        self.tokens
            .get(self.at)
            .or(self.tokens.last())
            .map_or(1, |(_, line)| *line)
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.at).map(|(token, _)| token.clone());
        self.at += 1;
        token
    }

    fn expect(&mut self, symbol: char) -> Result<(), String> {
        match self.next() {
            Some(Token::Symbol(c)) if c == symbol => Ok(()),
            other => Err(format!(
                "line {}: expected `{symbol}`, found {}",
                self.line(),
                describe(other.as_ref())
            )),
        }
    }

    fn program(&mut self) -> Result<(), String> {
        while self.at < self.tokens.len() {
            if self.peek() == Some(&Token::Newline) {
                self.at += 1;
                continue;
            }
            let line = self.line();
            let name = match self.next() {
                Some(Token::Name(name)) => name,
                other => {
                    return Err(format!(
                        "line {line}: a line is `name = expression`, found {}",
                        describe(other.as_ref())
                    ));
                }
            };
            if SOURCES.contains(&name.as_str()) || is_function(&name) {
                return Err(format!(
                    "line {line}: `{name}` is a keyword; pick another name"
                ));
            }
            self.expect('=')?;
            let value = self.expression()?;
            match self.peek() {
                Some(Token::Newline) | None => {
                    self.at += 1;
                }
                other => {
                    return Err(format!(
                        "line {line}: unexpected {} after the expression",
                        describe(other)
                    ));
                }
            }
            self.names.insert(name.clone(), value);
            self.order.push((name, value.symbol(), value.shape()));
        }
        Ok(())
    }

    fn expression(&mut self) -> Result<Value<'tape, f32>, String> {
        let mut left = self.term()?;
        loop {
            match self.peek() {
                Some(Token::Symbol('+')) => {
                    self.at += 1;
                    let right = self.term()?;
                    let (left_value, right_value) = self.align(left, right, "+")?;
                    left = left_value + right_value;
                }
                Some(Token::Symbol('-')) => {
                    self.at += 1;
                    let right = self.term()?;
                    let (left_value, right_value) = self.align(left, right, "-")?;
                    left = left_value - right_value;
                }
                _ => return Ok(left),
            }
        }
    }

    fn term(&mut self) -> Result<Value<'tape, f32>, String> {
        let mut left = self.unary()?;
        loop {
            match self.peek() {
                Some(Token::Symbol('*')) => {
                    self.at += 1;
                    let right = self.unary()?;
                    let (left_value, right_value) = self.align(left, right, "*")?;
                    left = left_value * right_value;
                }
                Some(Token::Symbol('/')) => {
                    self.at += 1;
                    let right = self.unary()?;
                    let (left_value, right_value) = self.align(left, right, "/")?;
                    left = left_value / right_value;
                }
                Some(Token::Symbol('@')) => {
                    self.at += 1;
                    let right = self.unary()?;
                    let (lhs, rhs) = (left.shape(), right.shape());
                    if lhs.rank() != 2 || rhs.rank() != 2 {
                        return Err(format!(
                            "line {}: `@` multiplies two matrices, got {lhs} @ {rhs}",
                            self.line()
                        ));
                    }
                    if lhs.axes()[1] != rhs.axes()[0] {
                        return Err(format!(
                            "line {}: matmul cannot multiply {lhs} by {rhs}: the inner extents differ",
                            self.line()
                        ));
                    }
                    left = left.matmul(right);
                }
                _ => return Ok(left),
            }
        }
    }

    fn unary(&mut self) -> Result<Value<'tape, f32>, String> {
        if self.peek() == Some(&Token::Symbol('-')) {
            self.at += 1;
            let operand = self.unary()?;
            return Ok(-operand);
        }
        self.primary()
    }

    fn primary(&mut self) -> Result<Value<'tape, f32>, String> {
        let line = self.line();
        match self.next() {
            Some(Token::Number(number)) => Ok(self.tape.leaf(number)),
            Some(Token::Symbol('(')) => {
                let inner = self.expression()?;
                self.expect(')')?;
                Ok(inner)
            }
            Some(Token::Name(name)) if SOURCES.contains(&name.as_str()) => self.source(&name),
            Some(Token::Name(name)) if is_function(&name) => self.call(&name),
            Some(Token::Name(name)) => self
                .names
                .get(&name)
                .copied()
                .ok_or_else(|| format!("line {line}: `{name}` is not defined yet")),
            other => Err(format!(
                "line {line}: expected a value, found {}",
                describe(other.as_ref())
            )),
        }
    }

    /// `parameter [2, 2]`, `input 0.5`, `leaf [3]`: a source with a
    /// seeded payload of the shape, or a rank-0 payload of the value.
    fn source(&mut self, kind: &str) -> Result<Value<'tape, f32>, String> {
        let line = self.line();
        let payload = match self.next() {
            Some(Token::Shape(axes)) => {
                let shape = Shape::new(axes);
                self.seed += 1;
                if shape.rank() == 0 {
                    Tensor::from(0.0_f32)
                } else {
                    init::uniform(self.seed, 1.0)(&shape)
                }
            }
            Some(Token::Number(number)) => Tensor::from(number),
            other => {
                return Err(format!(
                    "line {line}: `{kind}` takes a shape like `[2, 2]` or a number, found {}",
                    describe(other.as_ref())
                ));
            }
        };
        let value = match kind {
            "parameter" => {
                let value = self.tape.parameter(payload);
                self.parameters.push(value.symbol());
                value
            }
            "input" => self.tape.input(payload),
            _ => self.tape.leaf(payload),
        };
        Ok(value)
    }

    fn call(&mut self, function: &str) -> Result<Value<'tape, f32>, String> {
        let line = self.line();
        self.expect('(')?;
        let operand = self.expression()?;
        let shape = operand.shape();
        let value = if UNARY.contains(&function) {
            self.expect(')')?;
            match function {
                "tanh" => operand.tanh(),
                "exp" => operand.exp(),
                "ln" => operand.ln(),
                "sqrt" => operand.sqrt(),
                "sin" => operand.sin(),
                "cos" => operand.cos(),
                "relu" => operand.relu(),
                _ => {
                    if shape.rank() != 2 {
                        return Err(format!(
                            "line {line}: `transpose` takes a matrix, got {shape}"
                        ));
                    }
                    operand.transpose()
                }
            }
        } else if function == "sum" {
            self.expect(')')?;
            operand.sum()
        } else if ALONG.contains(&function) {
            self.expect(',')?;
            let axis = match self.next() {
                Some(Token::Number(number)) if number >= 0.0 && number.fract() == 0.0 => {
                    number as usize
                }
                other => {
                    return Err(format!(
                        "line {line}: `{function}` takes an axis number second, found {}",
                        describe(other.as_ref())
                    ));
                }
            };
            self.expect(')')?;
            if axis >= shape.rank() {
                return Err(format!(
                    "line {line}: axis {axis} is out of range for {shape} (rank {})",
                    shape.rank()
                ));
            }
            match function {
                "sum_along" => operand.sum_along(axis),
                "log_softmax" => operand.log_softmax(axis),
                "logsumexp" => operand.logsumexp(axis),
                "softmax" => operand.softmax(axis),
                _ => operand.mean_along(axis),
            }
        } else if SHAPED.contains(&function) {
            self.expect(',')?;
            let target = match self.next() {
                Some(Token::Shape(axes)) => Shape::new(axes),
                other => {
                    return Err(format!(
                        "line {line}: `{function}` takes a shape second, found {}",
                        describe(other.as_ref())
                    ));
                }
            };
            self.expect(')')?;
            if function == "broadcast" {
                if shape.rank() != 0 {
                    return Err(format!(
                        "line {line}: `broadcast` spreads a rank-0 value, got {shape}"
                    ));
                }
                operand.broadcast(target)
            } else {
                if shape.volume() != target.volume() {
                    return Err(format!(
                        "line {line}: `reshape` from {shape} to {target} changes the number of elements"
                    ));
                }
                operand.reshape(target)
            }
        } else {
            return Err(format!("line {line}: `{function}` is not a function"));
        };
        Ok(value)
    }

    /// Two elementwise operands: equal shapes pass, a rank-0 side is
    /// broadcast explicitly, anything else is refused.
    fn align(
        &self,
        left: Value<'tape, f32>,
        right: Value<'tape, f32>,
        operator: &str,
    ) -> Result<(Value<'tape, f32>, Value<'tape, f32>), String> {
        let (lhs, rhs) = (left.shape(), right.shape());
        if lhs == rhs {
            return Ok((left, right));
        }
        if lhs.rank() == 0 {
            return Ok((left.broadcast(rhs), right));
        }
        if rhs.rank() == 0 {
            return Ok((left, right.broadcast(lhs)));
        }
        Err(format!(
            "line {}: `{operator}` needs equal shapes, got {lhs} {operator} {rhs}; nothing broadcasts by accident",
            self.line()
        ))
    }
}

fn is_function(name: &str) -> bool {
    name == "sum" || UNARY.contains(&name) || ALONG.contains(&name) || SHAPED.contains(&name)
}

fn describe(token: Option<&Token>) -> String {
    match token {
        Some(Token::Name(name)) => format!("`{name}`"),
        Some(Token::Number(number)) => format!("`{number}`"),
        Some(Token::Shape(axes)) => format!("the shape {axes:?}"),
        Some(Token::Symbol(c)) => format!("`{c}`"),
        Some(Token::Newline) | None => "the end of the line".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_readme_records_six_nodes() {
        let tape: Tape<f32> = Tape::new();
        let recorded = record(
            &tape,
            "w = parameter 0.0\nx = input 0.0\ny = input 0.0\nerror = w * x - y\nloss = error * error\n",
        )
        .expect("records");
        assert_eq!(tape.len(), 6);
        assert_eq!(recorded.parameters.len(), 1);
        assert_eq!(recorded.names.len(), 5);
    }

    #[test]
    fn a_scalar_is_broadcast_explicitly() {
        let tape: Tape<f32> = Tape::new();
        record(&tape, "x = input [2, 3]\ny = x * 2.0\n").expect("records");
        assert!(tape.describe().contains("Broadcast"));
    }

    #[test]
    fn a_bad_product_is_a_message_not_a_panic() {
        let tape: Tape<f32> = Tape::new();
        let error = record(&tape, "a = input [3, 2]\nb = input [4, 5]\nc = a @ b\n").unwrap_err();
        assert!(error.contains("[3, 2]"), "{error}");
        assert!(error.contains("[4, 5]"), "{error}");
    }

    #[test]
    fn functions_take_their_arguments() {
        let tape: Tape<f32> = Tape::new();
        record(
            &tape,
            "w = parameter [2, 3]\nx = input [4, 2]\nh = tanh(x @ w)\np = log_softmax(h, 1)\nl = sum(p * p)\n",
        )
        .expect("records");
        assert!(tape.describe().contains("LogSoftmax"));
    }
}
