mod tokenizer;
use std::{collections::BTreeMap, fmt::Write, iter::Peekable};

use tokenizer::{Token, TokenKind};

use crate::throw;

#[derive(Debug, Clone)]
pub enum JsonValue {
    Object(BTreeMap<String, JsonValue>),
    List(Vec<JsonValue>),
    String(String),
    Integer(i64),
    Float(f64),
}

pub fn parse_impl<'a>(tokens: &mut Peekable<impl Iterator<Item = Token<'a>>>) -> JsonValue {
    let unexpected_eof = || throw!("unexpected EOF");

    macro_rules! expect_kind {
        ($token: expr, $kind: ident) => {{
            let token = $token;
            if token.kind != TokenKind::$kind {
                throw!("expected {} but got {:?}", TokenKind::$kind, token.text)
            }
        }};
    }

    let process_string = |text: &str| {
        if !text.ends_with('\"') {
            throw!("unterminated string literal")
        }

        let mut result = String::with_capacity(text.len());
        let mut chars = text[1..text.len() - 1].chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars
                    .next()
                    .unwrap_or_else(|| throw!("incomplete escape sequence"))
                {
                    'n' => result.push('\n'),
                    'r' => result.push('\r'),
                    '\\' => result.push('\\'),
                    e => throw!("invalid escape sequence: \\{e}"), // TODO: \u \U \x
                }
            } else {
                result.push(c)
            }
        }

        result
    };

    match tokens.next().unwrap_or_else(unexpected_eof) {
        Token {
            kind: TokenKind::String,
            text,
        } => JsonValue::String(process_string(text)),
        Token {
            kind: TokenKind::Number,
            text,
        } => {
            if text.contains('.') {
                JsonValue::Float(
                    text.parse()
                        .unwrap_or_else(|e| throw!("failed to parse json float: {e}")),
                )
            } else {
                JsonValue::Integer(
                    text.parse()
                        .unwrap_or_else(|e| throw!("failed to parse json integer: {e}")),
                )
            }
        }
        Token {
            kind: TokenKind::LBracket,
            ..
        } => {
            let mut result = vec![];
            loop {
                if tokens.peek().is_some_and(|x| x.kind == TokenKind::RBracket) {
                    tokens.next();
                    break;
                }

                result.push(parse_impl(tokens));

                match tokens.next().unwrap_or_else(unexpected_eof) {
                    Token {
                        kind: TokenKind::Comma,
                        ..
                    } => continue,
                    Token {
                        kind: TokenKind::RBracket,
                        ..
                    } => break,
                    Token { text, .. } => throw!("expected ',' or ']' but found {text:?}"),
                }
            }
            JsonValue::List(result)
        }
        Token {
            kind: TokenKind::LBrace,
            ..
        } => {
            let mut result = BTreeMap::default();
            loop {
                let name = tokens.next().unwrap_or_else(unexpected_eof);
                if name.kind == TokenKind::RBrace {
                    break;
                }
                expect_kind!(name, String);
                expect_kind!(tokens.next().unwrap_or_else(unexpected_eof), Colon);
                let value = parse_impl(tokens);
                if result.insert(process_string(name.text), value).is_some() {
                    throw!("duplicate object key {:?}", name.text)
                }

                match tokens.next().unwrap_or_else(unexpected_eof) {
                    Token {
                        kind: TokenKind::Comma,
                        ..
                    } => continue,
                    Token {
                        kind: TokenKind::RBrace,
                        ..
                    } => break,
                    Token { text, .. } => throw!("expected ',' or '}}' but found {text:?}"),
                }
            }
            JsonValue::Object(result)
        }
        Token { text, .. } => {
            throw!("expected a string, '[' or '{{' but found {text:?}")
        }
    }
}

pub fn parse(text: &str, allow_trailing_tokens: bool) -> JsonValue {
    let mut tokens = tokenizer::tokenize(text).into_iter().peekable();

    let result = parse_impl(&mut tokens);

    if !allow_trailing_tokens && tokens.next().is_some() {
        throw!("trailing tokens in json string")
    }

    result
}

pub struct JsonPrinter {
    output: String,
    depth: u64,
    suppress_comma: bool,
}

impl JsonPrinter {
    pub fn new() -> Self {
        Self {
            output: String::new(),
            depth: 0,
            suppress_comma: false,
        }
    }

    pub fn finish(self) -> String {
        assert_eq!(self.depth, 0);
        self.output
    }

    pub fn start_object(&mut self) -> &mut Self {
        self.write_comma();
        self.output.push('{');
        self.depth += 1;
        self.suppress_comma = true;
        self
    }

    pub fn key(&mut self, key: &str) -> &mut Self {
        self.string(key);
        self.output.push(':');
        self.suppress_comma = true;
        self
    }

    pub fn end_object(&mut self) -> &mut Self {
        self.output.push('}');
        self.depth -= 1;
        self.suppress_comma = false;
        self
    }

    pub fn start_list(&mut self) -> &mut Self {
        self.write_comma();
        self.output.push('[');
        self.depth += 1;
        self.suppress_comma = true;
        self
    }

    pub fn end_list(&mut self) -> &mut Self {
        self.output.push(']');
        self.depth -= 1;
        self.suppress_comma = false;
        self
    }

    fn write_comma(&mut self) {
        if self.depth > 0 && !self.suppress_comma {
            self.output.push(',')
        }
        self.suppress_comma = false;
    }

    pub fn string(&mut self, string: &str) -> &mut Self {
        self.write_comma();
        self.output.push('"');
        for c in string.chars() {
            if c == '\"' {
                self.output.push('\\')
            }
            self.output.push(c);
        }
        self.output.push('"');
        self
    }

    pub fn integer(&mut self, value: i64) -> &mut Self {
        self.write_comma();
        write!(self.output, "{value}").unwrap();
        self
    }

    pub fn float(&mut self, value: f64) -> &mut Self {
        self.write_comma();
        write!(self.output, "{value}").unwrap();
        self
    }

    pub fn bool(&mut self, value: bool) -> &mut Self {
        self.write_comma();
        if value {
            self.output.push_str("true");
        } else {
            self.output.push_str("false");
        }
        self
    }

    pub fn null(&mut self) -> &mut Self {
        self.write_comma();
        self.output.push_str("null");
        self
    }
}
