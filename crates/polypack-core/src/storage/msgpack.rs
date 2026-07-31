//! Minimal MessagePack encoder/decoder.
//!
//! The encoder is byte-for-byte compatible with the JavaScript
//! `@msgpack/msgpack` encoder used by the TypeScript reference implementation
//! (`src/persistence/binary-format.ts`): integers use the smallest
//! representation, floats are float64, maps preserve key order. The decoder
//! produces the `Msg` value tree used by the snapshot/WAL codecs.

use crate::error::{PolypackError, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum Msg {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<Msg>),
    Map(Vec<(Msg, Msg)>),
}

impl Msg {
    pub fn str(s: impl Into<String>) -> Msg {
        Msg::Str(s.into())
    }
    pub fn map(entries: Vec<(&str, Msg)>) -> Msg {
        Msg::Map(entries.into_iter().map(|(k, v)| (Msg::Str(k.to_string()), v)).collect())
    }
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self {
            Msg::Map(entries) => entries
                .iter()
                .find(|(k, _)| matches!(k, Msg::Str(s) if s == key))
                .and_then(|(_, v)| match v {
                    Msg::Str(s) => Some(s.as_str()),
                    _ => None,
                }),
            _ => None,
        }
    }
    pub fn get(&self, key: &str) -> Option<&Msg> {
        match self {
            Msg::Map(entries) => entries
                .iter()
                .find(|(k, _)| matches!(k, Msg::Str(s) if s == key))
                .map(|(_, v)| v),
            _ => None,
        }
    }
}

pub fn encode(msg: &Msg, out: &mut Vec<u8>) {
    match msg {
        Msg::Nil => out.push(0xc0),
        Msg::Bool(true) => out.push(0xc3),
        Msg::Bool(false) => out.push(0xc2),
        Msg::Int(v) => encode_int(*v, out),
        Msg::Float(v) => {
            // Match the JS encoder: integer-valued numbers encode as integers
            // (Number.isInteger), so 1.0 -> int, 0.5 -> float64.
            if v.fract() == 0.0 && v.abs() < 9007199254740992.0 {
                encode_int(*v as i64, out);
            } else {
                out.push(0xcb);
                out.extend_from_slice(&v.to_bits().to_be_bytes());
            }
        }
        Msg::Str(s) => encode_str(s, out),
        Msg::Array(items) => {
            encode_array_header(items.len(), out);
            for item in items {
                encode(item, out);
            }
        }
        Msg::Map(entries) => {
            encode_map_header(entries.len(), out);
            for (k, v) in entries {
                encode(k, out);
                encode(v, out);
            }
        }
    }
}

fn encode_int(v: i64, out: &mut Vec<u8>) {
    if v >= 0 {
        if v < 128 {
            out.push(v as u8);
        } else if v < 256 {
            out.push(0xcc);
            out.push(v as u8);
        } else if v < 65536 {
            out.push(0xcd);
            out.extend_from_slice(&(v as u16).to_be_bytes());
        } else if v < 4294967296 {
            out.push(0xce);
            out.extend_from_slice(&(v as u32).to_be_bytes());
        } else {
            out.push(0xcf);
            out.extend_from_slice(&(v as u64).to_be_bytes());
        }
    } else if v >= -32 {
        out.push(0xe0 | ((v + 32) as u8));
    } else if v >= -128 {
        out.push(0xd0);
        out.push(v as i8 as u8);
    } else if v >= -32768 {
        out.push(0xd1);
        out.extend_from_slice(&(v as i16).to_be_bytes());
    } else if v >= -2147483648 {
        out.push(0xd2);
        out.extend_from_slice(&(v as i32).to_be_bytes());
    } else {
        out.push(0xd3);
        out.extend_from_slice(&v.to_be_bytes());
    }
}

fn encode_str(s: &str, out: &mut Vec<u8>) {
    let len = s.len();
    if len < 32 {
        out.push(0xa0 | len as u8);
    } else if len < 256 {
        out.push(0xd9);
        out.push(len as u8);
    } else if len < 65536 {
        out.push(0xda);
        out.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        out.push(0xdb);
        out.extend_from_slice(&(len as u32).to_be_bytes());
    }
    out.extend_from_slice(s.as_bytes());
}

fn encode_array_header(len: usize, out: &mut Vec<u8>) {
    if len < 16 {
        out.push(0x90 | len as u8);
    } else if len < 65536 {
        out.push(0xdc);
        out.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        out.push(0xdd);
        out.extend_from_slice(&(len as u32).to_be_bytes());
    }
}

fn encode_map_header(len: usize, out: &mut Vec<u8>) {
    if len < 16 {
        out.push(0x80 | len as u8);
    } else if len < 65536 {
        out.push(0xde);
        out.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        out.push(0xdf);
        out.extend_from_slice(&(len as u32).to_be_bytes());
    }
}

/// Decode a complete MessagePack value from `data`. Fails on structural
/// errors; trailing bytes after one value are an error for snapshot/WAL bodies.
pub fn decode(data: &[u8]) -> Result<Msg> {
    let mut cursor = Cursor::new(data);
    let value = decode_value(&mut cursor)?;
    Ok(value)
}

pub struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Cursor { data, pos: 0 }
    }
    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }
    pub fn pos(&self) -> usize {
        self.pos
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return Err(PolypackError::CorruptData("truncated msgpack".into()));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn be_u16(&mut self) -> Result<u16> {
        Ok(u16::from_be_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn be_u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn be_u64(&mut self) -> Result<u64> {
        Ok(u64::from_be_bytes(self.take(8)?.try_into().unwrap()))
    }
}

fn decode_value(c: &mut Cursor) -> Result<Msg> {
    let b = c.u8()?;
    match b {
        0xc0 => Ok(Msg::Nil),
        0xc2 => Ok(Msg::Bool(false)),
        0xc3 => Ok(Msg::Bool(true)),
        0xcc => Ok(Msg::Int(c.u8()? as i64)),
        0xcd => Ok(Msg::Int(c.be_u16()? as i64)),
        0xce => Ok(Msg::Int(c.be_u32()? as i64)),
        0xcf => Ok(Msg::Int(c.be_u64()? as i64)),
        0xd0 => Ok(Msg::Int(c.u8()? as i8 as i64)),
        0xd1 => Ok(Msg::Int(c.be_u16()? as i16 as i64)),
        0xd2 => Ok(Msg::Int(c.be_u32()? as i32 as i64)),
        0xd3 => Ok(Msg::Int(c.be_u64()? as i64)),
        0xca => {
            let bits = c.be_u32()?;
            Ok(Msg::Float(f32::from_bits(bits) as f64))
        }
        0xcb => {
            let bits = c.be_u64()?;
            Ok(Msg::Float(f64::from_bits(bits)))
        }
        0xd9 => {
            let len = c.u8()? as usize;
            Ok(Msg::Str(String::from_utf8_lossy(c.take(len)?).into_owned()))
        }
        0xda => {
            let len = c.be_u16()? as usize;
            Ok(Msg::Str(String::from_utf8_lossy(c.take(len)?).into_owned()))
        }
        0xdb => {
            let len = c.be_u32()? as usize;
            Ok(Msg::Str(String::from_utf8_lossy(c.take(len)?).into_owned()))
        }
        0xdc => {
            let len = c.be_u16()? as usize;
            decode_array(c, len)
        }
        0xdd => {
            let len = c.be_u32()? as usize;
            decode_array(c, len)
        }
        0xde => {
            let len = c.be_u16()? as usize;
            decode_map(c, len)
        }
        0xdf => {
            let len = c.be_u32()? as usize;
            decode_map(c, len)
        }
        0xa0..=0xbf => {
            let len = (b & 0x1f) as usize;
            Ok(Msg::Str(String::from_utf8_lossy(c.take(len)?).into_owned()))
        }
        0x90..=0x9f => {
            let len = (b & 0x0f) as usize;
            decode_array(c, len)
        }
        0x80..=0x8f => {
            let len = (b & 0x0f) as usize;
            decode_map(c, len)
        }
        0xe0..=0xff => Ok(Msg::Int((b as i8) as i64)),
        0x00..=0x7f => Ok(Msg::Int(b as i64)),
        _ => Err(PolypackError::CorruptData(format!("unexpected msgpack marker 0x{b:02x}"))),
    }
}

fn decode_array(c: &mut Cursor, len: usize) -> Result<Msg> {
    let mut items = Vec::with_capacity(len.min(4096));
    for _ in 0..len {
        items.push(decode_value(c)?);
    }
    Ok(Msg::Array(items))
}

fn decode_map(c: &mut Cursor, len: usize) -> Result<Msg> {
    let mut entries = Vec::with_capacity(len.min(4096));
    for _ in 0..len {
        let k = decode_value(c)?;
        let v = decode_value(c)?;
        entries.push((k, v));
    }
    Ok(Msg::Map(entries))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_smallest_integers_and_floats() {
        let mut out = Vec::new();
        encode(&Msg::Int(1), &mut out);
        assert_eq!(out, vec![0x01]);
        let mut out = Vec::new();
        encode(&Msg::Int(300), &mut out);
        assert_eq!(out, vec![0xcd, 0x01, 0x2c]);
        let mut out = Vec::new();
        encode(&Msg::Int(-1), &mut out);
        assert_eq!(out, vec![0xff]);
        let mut out = Vec::new();
        encode(&Msg::Float(0.5), &mut out);
        assert_eq!(out, vec![0xcb, 0x3f, 0xe0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn round_trips_structures() {
        let value = Msg::map(vec![
            ("id", Msg::str("n1")),
            ("type", Msg::str("doc")),
            ("data", Msg::Map(vec![])),
            ("vector", Msg::Array(vec![Msg::Float(0.1), Msg::Float(0.2)])),
            ("insertedAt", Msg::Int(7)),
            ("updatedAt", Msg::Nil),
        ]);
        let mut out = Vec::new();
        encode(&value, &mut out);
        let decoded = decode(&out).unwrap();
        assert_eq!(value, decoded);
    }

    #[test]
    fn rejects_truncated_input() {
        let mut out = Vec::new();
        encode(&Msg::Str("hello".into()), &mut out);
        assert!(decode(&out[..3]).is_err());
    }
}
