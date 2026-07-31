//! `mulberry32`, a small deterministic PRNG.
//!
//! Bit-for-bit compatible with the JavaScript implementation used by the
//! TypeScript benchmark harness (`benchmarks/run-ts.ts`), so both engines can
//! generate identical datasets from the same seed.

pub struct Mulberry32(u32);

impl Mulberry32 {
    pub fn new(seed: u32) -> Self {
        Mulberry32(seed)
    }

    /// Returns a float in `[0, 1)`, identical to the JS `mulberry32`.
    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x6d2b79f5);
        let a = self.0 as i32;

        let mut t = i32::wrapping_mul(a ^ ((a as u32) >> 15) as i32, 1 | a);
        t = (t.wrapping_add(i32::wrapping_mul(
            t ^ ((t as u32) >> 7) as i32,
            61 | t,
        ))) ^ t;

        ((t ^ ((t as u32) >> 14) as i32) as u32) as f64 / 4294967296.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_javascript_mulberry32_seed_42() {
        // Expected values captured from the JS implementation (seed 42).
        let expected = [
            0.6011037519201636,
            0.44829055899754167,
            0.8524657934904099,
            0.6697340414393693,
            0.17481389874592423,
            0.5265925421845168,
            0.2732279943302274,
            0.6247446539346129,
            0.8654746483080089,
            0.4723170551005751,
        ];
        let mut rng = Mulberry32::new(42);
        for (i, e) in expected.iter().enumerate() {
            let got = rng.next_f64();
            assert!((got - e).abs() < f64::EPSILON, "value {i}: got {got}, expected {e}");
        }
    }
}
