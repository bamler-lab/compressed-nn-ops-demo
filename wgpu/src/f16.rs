/// Simple struct to emulate a 16-bit floating point number, mostly for I/O purposes.
///
/// This is not optimized and does not even implement any arithmetic operations.
/// It's only meant for converting to and from `f32` and serializing / deserializing
/// to and from bytes (e.g., using the `byteorder` crate). Even the conversions are not
/// optimized, they don't preserve NaN payload bits, and they truncate (i.e., round towards
/// zero), rather than round to nearest for simplicity.
#[derive(Clone, Copy, Debug)]
pub struct SimpleF16 {
    /// The bits, in IEEE 754 representation of a 16-bit float:
    /// - most significant bit: sign
    /// - next 5 bits: exponent
    /// - least significant 10 bits: mantissa
    ///
    /// These represent the following value:
    /// - if `exponent == (00000)_2`, then
    ///   `logical_value = (-1)^sign * 2^(-14) * (mantissa / 2^10)`
    /// - if `exponent == (11111)_2` and `mantissa == (0000000000)_2`, then
    ///   `logical_value = (-1)^sign * ∞`
    /// - if `exponent == (11111)_2` and `mantissa != (0000000000)_2`, then
    ///   `logical_value = NaN`
    /// - else, `logical_value = (-1)^sign * 2^(exponent - 15) * (1 + mantissa / 2^10)`.
    ///
    /// By comparison, the IEEE 754 representation of a **32-bit float** is:
    /// - most significant bit: sign
    /// - next 8 bits: exponent
    /// - least significant 23 bits: mantissa
    ///
    /// These represent the following value:
    /// - if `exponent == (00000000)_2`, then
    ///   `logical_value = (-1)^sign * 2^(-126) * (mantissa / 2^23)`
    /// - if `exponent == (11111111)_2` and `mantissa == (00000000000000000000000)_2`, then
    ///   `logical_value = (-1)^sign * ∞`
    /// - if `exponent == (11111111)_2` and `mantissa != (00000000000000000000000)_2`, then
    ///   `logical_value = NaN`
    /// - else, `logical_value = (-1)^sign * 2^(exponent - 127) * (1 + mantissa / 2^23)`.
    bits: u16,
}

impl PartialEq for SimpleF16 {
    fn eq(&self, other: &Self) -> bool {
        (self.bits == other.bits && !self.is_nan()) || (self.is_zero() && other.is_zero())
    }
}

impl SimpleF16 {
    pub fn from_f32(x: f32) -> Self {
        let f32_bits = x.to_bits();
        let f32_sign = f32_bits >> 31;
        let f32_exponent = (f32_bits >> 23) & 0b1111_1111;
        let f32_mantissa = f32_bits & 0b0111_1111_1111_1111_1111_1111;

        let (f16_exponent, f16_mantissa) = match f32_exponent {
            0..103 => {
                // `x` is smaller (in absolute value) than the smallest normal `f16` value, so we convert to zero (but preserve the sign).
                (0, 0)
            }
            103..113 => {
                // `103 <= f32_exponent <= 112`, so the *logical* exponent ranges from `-24` to `-15`.
                // This is precisely the range of values that can be represented by subnormal `f16`s:
                // `2^(-24)` to `(1.111111111...)_2 * 2^(-15)`.
                let f32_mantissa_with_implicit_1 = f32_mantissa | (1 << 23);
                let shift = 126 - f32_exponent;
                // `14 <= shift <= 23`, and `f32_mantissa_with_implicit_1` is exactly 24 bit long.
                // Thus, after the shift below, it is between `10` and `1` bit long (both inclusive).
                let f16_mantissa = (f32_mantissa_with_implicit_1 >> shift) as u16; // We truncate rather than round. Not sure if this is technically correct.

                // Example: if `f32_exponent == 107`, then
                // ```
                // f32_logical_value = (-1)^sign * 2^(107 - 127) * (1 + f32_mantissa / 2^23)
                //                   = (-1)^sign * 2^(-20) * (2^23 + f32_mantissa) / 2^23
                //                   = (-1)^sign * 2^(-14) * (2^23 + f32_mantissa) / 2^29
                //                   = (-1)^sign * 2^(-14) * (2^23 + f32_mantissa) / 2^(126 - 107 + 10)
                //                   = (-1)^sign * 2^(-14) * f16_mantissa / 2^10
                // ```
                //
                // where `f16_mantissa = (2^23 + f32_mantissa) / 2^shift` and `shift = 126 - 107`.

                (0, f16_mantissa)
            }
            113..143 => {
                // `113 <= f32_exponent <= 142`, so the *logical* exponent ranges from `-14` to `15`.
                // This corresponds to `f16` values with `1 <= exponent <= 30`, which is precisely
                // the range of normal `f16` exponents.

                let f16_exponent = f32_exponent as u16 - (127 - 15);
                let f16_mantissa = (f32_mantissa >> 13) as u16; // We truncate rather than round. Not sure if this is technically correct.
                (f16_exponent, f16_mantissa)
            }
            0b1111_1111 => (0b1_1111, (f32_mantissa != 0) as u16), // NaN or ∞
            _ => {
                // `f32_exponent >= 143`, so the *logical* exponent is `>= 16`, which is too large for `f16`.
                // Thus, we map to infinity (preserving the sign).
                (0b1_1111, 0)
            }
        };

        let f16_bits = ((f32_sign as u16) << 15) | (f16_exponent << 10) | f16_mantissa;

        Self::from_bits(f16_bits)
    }

    pub fn to_f32(self) -> f32 {
        let sign = self.sign();
        let exponent = self.exponent();
        let mantissa = self.mantissa();

        let (f32_exponent, f32_mantissa) = match exponent {
            0 => {
                if mantissa == 0 {
                    (0, 0)
                } else {
                    let leading_zeros = mantissa.leading_zeros() as i32 - 6;
                    let logical_exponent = -15 - leading_zeros;
                    // `-24 <= logical_exponent <= -15`, which is in the range of normal `f32` exponents.
                    let f32_exponent = (logical_exponent + 127) as u32;
                    let f32_mantissa = ((mantissa as u32) << (leading_zeros + 1 + 13))
                        & 0b0111_1111_1111_1111_1111_1111;
                    (f32_exponent, f32_mantissa)
                }
            }
            0b1_1111 => (0b1111_1111, mantissa as u32), // NaN or ∞
            _ => {
                // `1 <= exponent <= 30`, so the *logical* exponent ranges from
                // -14 to 15. This corresponds to `f32` values with
                // `113 <= exponent <= 157`, which is the range of normal `f32` exponents.

                let f32_exponent = exponent as u32 + (127 - 15);
                let f32_mantissa = (mantissa as u32) << 13;
                (f32_exponent, f32_mantissa)
            }
        };

        let f32_bits = ((sign as u32) << 31) | (f32_exponent << 23) | f32_mantissa;

        f32::from_bits(f32_bits)
    }

    pub fn from_bits(bits: u16) -> Self {
        Self { bits }
    }

    pub fn to_bits(self) -> u16 {
        self.bits
    }

    pub fn is_nan(self) -> bool {
        self.exponent() == 0b1_1111 && self.mantissa() != 0
    }

    pub fn is_sign_positive(self) -> bool {
        self.sign() == 0
    }

    pub fn is_sign_negative(self) -> bool {
        self.sign() == 1
    }

    /// Returns `true` if this value is positive infinity or negative infinity, and `false` otherwise.
    pub fn is_infinite(self) -> bool {
        self.exponent() == 0b1_1111 && self.mantissa() == 0
    }

    fn sign(self) -> u16 {
        self.bits >> 15
    }

    fn exponent(self) -> u16 {
        (self.bits >> 10) & 0b1_1111
    }

    fn mantissa(self) -> u16 {
        self.bits & 0b11_1111_1111
    }

    fn is_zero(self) -> bool {
        // Ignore sign bit.
        self.bits << 1 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::SimpleF16;

    #[test]
    fn exhaustive_f16_to_f32_and_back() {
        for bits in 0..=0xFFFF {
            let my_f16 = SimpleF16::from_bits(bits);
            let my_f32 = my_f16.to_f32();
            let reconstruction = SimpleF16::from_f32(my_f32);

            assert_eq!(my_f16.is_nan(), my_f32.is_nan());
            assert_eq!(my_f16.is_nan(), reconstruction.is_nan());

            assert_eq!(my_f16.is_infinite(), my_f32.is_infinite());
            assert_eq!(my_f16.is_infinite(), reconstruction.is_infinite());

            assert_eq!(my_f16.is_sign_positive(), my_f32.is_sign_positive());
            assert_eq!(my_f16.is_sign_positive(), reconstruction.is_sign_positive());

            assert_eq!(my_f16.is_sign_negative(), my_f32.is_sign_negative());
            assert_eq!(my_f16.is_sign_negative(), reconstruction.is_sign_negative());

            if !my_f16.is_nan() {
                assert_eq!(my_f16.to_bits(), reconstruction.to_bits());
            } else {
                // We don't expect NaN payload bits to be preserved. The only things we can
                // check are that the sign and NaN-ness are preserved (see above), and that
                // NaNs are not infinite.
                assert!(!my_f16.is_infinite());
            }
        }
    }

    #[test]
    fn exact_f32_to_f16_and_back() {
        let exactly_representable = [
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.0,
            -1.0,
            0.5,
            -0.5,
            1.0 / 1024.0,
            -1.0 / 1024.0,
            123.625,
            -123.625,
            f32::NAN,
        ];

        for &my_f32 in &exactly_representable {
            let my_f16 = SimpleF16::from_f32(my_f32);
            let reconstruction = my_f16.to_f32();

            assert_eq!(my_f32.is_sign_positive(), reconstruction.is_sign_positive());
            assert_eq!(my_f32.is_nan(), reconstruction.is_nan());

            if !my_f32.is_nan() {
                assert_eq!(my_f32, reconstruction);
            }
        }
    }

    #[test]
    fn approximate_f32_to_f16_and_back() {
        for &sign in &[-1.0, 1.0] {
            for exponent in -100..101 {
                let my_f32 = sign * (exponent as f32 * 0.1).exp();
                let my_f16 = SimpleF16::from_f32(my_f32);
                let reconstruction = my_f16.to_f32();

                assert!(!my_f32.is_nan());
                assert!(!my_f16.is_nan());
                assert!(!reconstruction.is_nan());

                assert!(!my_f32.is_infinite());
                assert!(!my_f16.is_infinite());
                assert!(!reconstruction.is_infinite());

                assert_eq!(my_f32.is_sign_positive(), my_f16.is_sign_positive());
                assert_eq!(my_f32.is_sign_positive(), reconstruction.is_sign_positive());

                assert_eq!(my_f32.is_sign_negative(), my_f16.is_sign_negative());
                assert_eq!(my_f32.is_sign_negative(), reconstruction.is_sign_negative());

                assert!(my_f32 != 0.0);
                assert!(!my_f16.is_zero()); // (remove sign bit)
                assert!(reconstruction != 0.0);

                let diff = (my_f32 - reconstruction).abs();
                let rel_diff = diff / my_f32;
                assert!(rel_diff < 1e-3);
            }
        }

        for very_small in [-1e-10f32, 1e-10f32] {
            assert!(very_small != 0.0);
            let my_f16 = SimpleF16::from_f32(very_small);
            assert!(my_f16.is_zero());
            let reconstruction = my_f16.to_f32();
            assert!(reconstruction == 0.0);
        }
    }
}
