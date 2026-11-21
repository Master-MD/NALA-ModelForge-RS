#![forbid(unsafe_code)]

use std::collections::BTreeMap;

/// Zero-dependency strict RFC 8259 JSON AST for NALA-ModelForge-RS.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    pub fn obj(entries: Vec<(&str, Json)>) -> Self {
        let mut map = BTreeMap::new();
        for (k, v) in entries {
            map.insert(k.to_string(), v);
        }
        Json::Obj(map)
    }

    pub fn s(v: impl Into<String>) -> Self {
        Json::Str(v.into())
    }

    pub fn n(v: f64) -> Self {
        if v.is_finite() { Json::Num(v) } else { Json::Num(0.0) }
    }

    pub fn u(v: u64) -> Self {
        Json::Num(v as f64)
    }

    pub fn i(v: i64) -> Self {
        Json::Num(v as f64)
    }

    pub fn b(v: bool) -> Self {
        Json::Bool(v)
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.get(key),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Num(n) if *n >= 0.0 && n.is_finite() => Some(*n as u64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&BTreeMap<String, Json>> {
        match self {
            Json::Obj(m) => Some(m),
            _ => None,
        }
    }

    pub fn to_string_compact(&self) -> String {
        let mut out = String::with_capacity(256);
        self.write_into(&mut out);
        out
    }

    fn write_into(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(true) => out.push_str("true"),
            Json::Bool(false) => out.push_str("false"),
            Json::Num(n) => {
                if !n.is_finite() {
                    out.push_str("0");
                } else if n.fract() == 0.0 && n.abs() < 9_007_199_254_740_991.0 {
                    out.push_str(&format!("{}", *n as i64));
                } else {
                    out.push_str(&format!("{n}"));
                }
            }
            Json::Str(s) => write_escaped_str(s, out),
            Json::Arr(arr) => {
                out.push('[');
                for (idx, item) in arr.iter().enumerate() {
                    if idx > 0 {
                        out.push(',');
                    }
                    item.write_into(out);
                }
                out.push(']');
            }
            Json::Obj(map) => {
                out.push('{');
                for (idx, (k, v)) in map.iter().enumerate() {
                    if idx > 0 {
                        out.push(',');
                    }
                    write_escaped_str(k, out);
                    out.push(':');
                    v.write_into(out);
                }
                out.push('}');
            }
        }
    }
}

pub fn write_escaped_str(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn parse_json(input: &str) -> Result<Json, String> {
    let mut parser = Parser {
        bytes: input.as_bytes(),
        pos: 0,
        depth: 0,
    };
    let val = parser.parse_value()?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(format!("trailing bytes at offset {}", parser.pos));
    }
    Ok(val)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b' ' | b'\t' | b'\r' | b'\n' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self) -> Result<Json, String> {
        if self.depth > 64 {
            return Err("JSON nesting depth limit exceeded".into());
        }
        self.skip_ws();
        let b = self.peek().ok_or_else(|| "unexpected EOF".to_string())?;
        match b {
            b'{' => self.parse_obj(),
            b'[' => self.parse_arr(),
            b'"' => self.parse_string().map(Json::Str),
            b't' => self.expect_lit(b"true", Json::Bool(true)),
            b'f' => self.expect_lit(b"false", Json::Bool(false)),
            b'n' => self.expect_lit(b"null", Json::Null),
            b'-' | b'0'..=b'9' => self.parse_number(),
            other => Err(format!("unexpected byte '{}' at {}", other as char, self.pos)),
        }
    }

    fn expect_lit(&mut self, lit: &[u8], out: Json) -> Result<Json, String> {
        if self.bytes[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            Ok(out)
        } else {
            Err(format!("invalid literal at offset {}", self.pos))
        }
    }

    fn parse_obj(&mut self) -> Result<Json, String> {
        self.depth += 1;
        self.pos += 1; // '{'
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(Json::Obj(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(format!("expected ':' at {}", self.pos));
            }
            self.pos += 1;
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    self.depth -= 1;
                    return Ok(Json::Obj(map));
                }
                _ => return Err(format!("expected ',' or '}}' at {}", self.pos)),
            }
        }
    }

    fn parse_arr(&mut self) -> Result<Json, String> {
        self.depth += 1;
        self.pos += 1; // '['
        let mut arr = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(Json::Arr(arr));
        }
        loop {
            let val = self.parse_value()?;
            arr.push(val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    self.depth -= 1;
                    return Ok(Json::Arr(arr));
                }
                _ => return Err(format!("expected ',' or ']' at {}", self.pos)),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        if self.peek() != Some(b'"') {
            return Err(format!("expected '\"' at {}", self.pos));
        }
        self.pos += 1;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == b'"' {
                self.pos += 1;
                return Ok(out);
            }
            if b == b'\\' {
                self.pos += 1;
                let esc = *self
                    .bytes
                    .get(self.pos)
                    .ok_or_else(|| "unterminated escape".to_string())?;
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\x08'),
                    b'f' => out.push('\x0c'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        if self.pos + 4 > self.bytes.len() {
                            return Err("short unicode escape".into());
                        }
                        let hex = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
                            .map_err(|_| "invalid utf8 in unicode escape".to_string())?;
                        let code = u32::from_str_radix(hex, 16)
                            .map_err(|_| "invalid hex in unicode escape".to_string())?;
                        self.pos += 4;
                        if let Some(c) = char::from_u32(code) {
                            out.push(c);
                        }
                    }
                    _ => return Err(format!("unknown escape '\\{}'", esc as char)),
                }
            } else {
                let rest = &self.bytes[self.pos..];
                let s = std::str::from_utf8(rest)
                    .map_err(|_| "invalid utf8 in json string".to_string())?;
                let ch = s.chars().next().ok_or_else(|| "eof in string".to_string())?;
                out.push(ch);
                self.pos += ch.len_utf8();
            }
        }
        Err("unterminated string".into())
    }

    fn parse_number(&mut self) -> Result<Json, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(b'0'..=b'9') = self.peek() {
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
        }
        let slice = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "invalid number utf8".to_string())?;
        let val = slice
            .parse::<f64>()
            .map_err(|_| format!("invalid number '{slice}'"))?;
        Ok(Json::Num(val))
    }
}
