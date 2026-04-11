"""Tests for MLX Metal fused PlanarQuant attention kernels.
Skip on non-Apple hardware."""
import pytest
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")

def test_givens_roundtrip():
    from turboquant.mlx_fused_planar_attention import _planar_rotate, _planar_unrotate
    x = mx.random.normal((1, 128))
    recovered = _planar_unrotate(_planar_rotate(x))
    mx.eval(recovered)
    assert mx.max(mx.abs(x.astype(mx.float32) - recovered.astype(mx.float32))).item() < 1e-5

def test_fused_qk_correctness():
    from turboquant.mlx_fused_planar_attention import (
        planar_fused_qk_scores, _compress, _decompress, _planar_rotate, _planar_unrotate, _CODEBOOKS)
    import math
    mx.random.seed(42); B,H,T,D,bits = 1,2,50,64,3
    k = mx.random.normal((B*H*T, D)).astype(mx.float32)
    kp, kn = _compress(k, bits, _planar_rotate)
    kd = _decompress(kp, kn, D, bits, _planar_unrotate, mx.float32).reshape(B,H,T,D)
    q = mx.random.normal((B,H,1,D)).astype(mx.float16)
    scale = 1.0/math.sqrt(D)
    ref = (q.astype(mx.float32) @ kd.swapaxes(-1,-2)) * scale
    fused = planar_fused_qk_scores(q, kp.reshape(B,H,T,-1), kn.reshape(B,H,T),
                                    mx.array(_CODEBOOKS[bits], dtype=mx.float32), scale, D, bits)
    mx.eval(ref, fused)
    assert mx.max(mx.abs(fused - ref)).item() < 0.001

def test_all_bit_widths():
    from turboquant.mlx_fused_planar_attention import (
        planar_fused_qk_scores, _compress, _planar_rotate, _CODEBOOKS)
    import math
    mx.random.seed(0); B,H,T,D = 1,2,30,64
    for bits in [2, 3, 4]:
        k = mx.random.normal((B*H*T, D)).astype(mx.float32)
        kp, kn = _compress(k, bits, _planar_rotate)
        q = mx.random.normal((B,H,1,D)).astype(mx.float16)
        scores = planar_fused_qk_scores(q, kp.reshape(B,H,T,-1), kn.reshape(B,H,T),
                                         mx.array(_CODEBOOKS[bits], dtype=mx.float32),
                                         1.0/math.sqrt(D), D, bits)
        mx.eval(scores)
        assert not mx.any(mx.isnan(scores)).item(), f"NaN at {bits}-bit"
