use std::fmt::Display;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    String,
    Number,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Error,
}

impl Display for TokenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            TokenKind::String => "a string",
            TokenKind::Number => "a number",
            TokenKind::LBrace => "'{'",
            TokenKind::RBrace => "'}'",
            TokenKind::LBracket => "'['",
            TokenKind::RBracket => "']'",
            TokenKind::Comma => "','",
            TokenKind::Colon => "':'",
            TokenKind::Error => "an invalid token",
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Token<'a> {
    pub kind: TokenKind,
    pub text: &'a str,
}

pub fn tokenize(text: &str) -> Vec<Token<'_>> {
    let mut chars = text.char_indices().peekable();
    let mut tokens = vec![];

    loop {
        while chars.next_if(|c| c.1.is_whitespace()).is_some() {}

        macro_rules! next_offset {
            () => { chars.peek().map(|x| x.0).unwrap_or(text.len()) };
        }

        macro_rules! simple_token {
            ($start: ident, $kind: ident) => {{
                chars.next();
                tokens.push(Token {
                    kind: TokenKind::$kind,
                    text: &text[$start..next_offset!()],
                })
            }};
        }

        match chars.peek().copied() {
            Some((start, '{')) => simple_token!(start, LBrace),
            Some((start, '}')) => simple_token!(start, RBrace),
            Some((start, '[')) => simple_token!(start, LBracket),
            Some((start, ']')) => simple_token!(start, RBracket),
            Some((start, ',')) => simple_token!(start, Comma),
            Some((start, ':')) => simple_token!(start, Colon),
            // Some((start, c)) if c.is_alphabetic() => {
            //     while chars.next_if(|(_, c)| c.is_alphanumeric()).is_some() {}
            //     tokens.push(Token {
            //         kind: TokenKind::Identifier,
            //         text: &text[start..next_offset!()]
            //     })
            // },
            Some((start, c)) if c.is_ascii_digit() || c == '.' => {
                while chars.next_if(|(_, c)| c.is_ascii_digit()).is_some() {}
                chars.next_if(|(_, c)| *c == '.');
                while chars.next_if(|(_, c)| c.is_ascii_digit()).is_some() {}
                tokens.push(Token {
                    kind: TokenKind::Number,
                    text: &text[start..next_offset!()]
                })
            }
            Some((start, '"')) => {
                chars.next();
                loop {
                    match chars.next() {
                        Some((_, '\\')) => _ = chars.next(),
                        Some((_, '"')) => break,
                        Some((_, _)) => continue,
                        None => break,
                    }
                }
                tokens.push(Token {
                    kind: TokenKind::String,
                    text: &text[start..next_offset!()]
                })
            },
            Some((start, _)) => {
                while chars.next_if(|(_, c)| !c.is_ascii_whitespace()).is_some() {}
                tokens.push(Token {
                    kind: TokenKind::Error,
                    text: &text[start..next_offset!()]
                })
            }
            None => break,
        }
    }

    tokens
}
