// The elementwise map kernels: one thread per element, one kernel per
// transcendental, compiled with fast math off for parity with the CPU
// paths. The count rides as a small constant so a grid rounded up to
// the threadgroup width never writes past the buffer.

#include <metal_stdlib>

using namespace metal;

kernel void map_exp_f32(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < count) {
        destination[index] = exp(source[index]);
    }
}

kernel void map_ln_f32(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < count) {
        destination[index] = log(source[index]);
    }
}

kernel void map_sqrt_f32(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < count) {
        destination[index] = sqrt(source[index]);
    }
}

kernel void map_tanh_f32(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < count) {
        destination[index] = tanh(source[index]);
    }
}

kernel void map_sin_f32(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < count) {
        destination[index] = sin(source[index]);
    }
}

kernel void map_cos_f32(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < count) {
        destination[index] = cos(source[index]);
    }
}

// MSL has no log1p/expm1 built-ins, so both kernels use Kahan's
// compensated forms: `u` absorbs the rounding of the naive step, and
// the exact argument is rescaled back through the `x / (u - 1)` (or
// `x / log(u)`) ratio, which cancels the rounding error analytically.
// At `u == 1` the limit is `x` itself.

kernel void map_log1p_f32(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < count) {
        float x = source[index];
        float u = 1.0f + x;
        destination[index] = (u == 1.0f) ? x : log(u) * (x / (u - 1.0f));
    }
}

kernel void map_expm1_f32(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < count) {
        float x = source[index];
        float u = exp(x);
        if (u == 1.0f) {
            destination[index] = x;
        } else if (fabs(x) > 0.693147f) {
            // Beyond `ln 2` the direct subtraction cancels less than
            // one bit, and it also covers the saturated extremes.
            destination[index] = u - 1.0f;
        } else {
            destination[index] = (u - 1.0f) * (x / log(u));
        }
    }
}
