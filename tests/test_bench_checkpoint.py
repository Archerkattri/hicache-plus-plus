#!/usr/bin/env python3
"""CPU tests for bench_dit's per-1k-image checkpointing (CellCheckpoint).

Exercises the FID stat-accumulation logic with a tiny deterministic mock
"feature extractor" (no diffusion, no model, no GPU), mirroring main()'s loop
structure exactly: per-batch generator draws, per-batch accumulation, partial
checkpoint every ``every`` images, atomic final npz.

The kill is simulated by raising out of the loop and DISCARDING all in-memory
state; resume happens through a fresh CellCheckpoint reading only the partial
file plus the same generator fast-forward main() uses. Because the partial is
written atomically (temp + os.replace), this is exactly the state a SIGKILL
leaves behind.
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "benchmarks", "dit_imagenet"))
from bench_dit import CellCheckpoint, atomic_savez  # noqa: E402

N, BATCH, EVERY, DIM, SEED = 37, 5, 10, 6, 0
FP = f"mock|n{N}|b{BATCH}|seed{SEED}"


class KilledRun(Exception):
    pass


def _mock_feats(y, z, b):
    """Deterministic stand-in for the Inception features of one batch. Depends on
    BOTH generator draws (y, z), so any generator misalignment on resume changes
    the output and fails the bit-identical assertion."""
    return (z.reshape(b, -1)[:, :DIM] * 0.1 + y.float().unsqueeze(1) * 0.01).numpy()


def run_cell(out_npz, kill_after=None):
    """Mirror of bench_dit.main()'s accumulation loop with the mock extractor.
    If ``kill_after`` is set, raise KilledRun once that many images are done
    (in-memory state is then discarded by the caller, as a kill would)."""
    ckpt = CellCheckpoint(out_npz, FP, every=EVERY)
    done0, _ = ckpt.resume()
    g = torch.Generator().manual_seed(SEED)
    d = 0
    while d < done0:                                # main()'s fast-forward, verbatim shape
        b = min(BATCH, N - d)
        torch.randint(0, 1000, (b,), generator=g)
        torch.randn(b, 4, 2, 2, generator=g)
        d += b
    while ckpt.done < N:
        b = min(BATCH, N - ckpt.done)
        y = torch.randint(0, 1000, (b,), generator=g)
        z = torch.randn(b, 4, 2, 2, generator=g)
        ckpt.add(_mock_feats(y, z, b), b, 0.0125)
        if kill_after is not None and ckpt.done >= kill_after:
            raise KilledRun
    ckpt.finalize(N, latency_ms=ckpt.t_gen / N * 1000, compute_calls=7, steps=10,
                  method="mock", interval=3, cfg=1.5)
    return ckpt


def _load(path):
    with np.load(path) as z:
        return {k: np.array(z[k]) for k in z.files}


def test_uninterrupted_run_writes_final_and_cleans_partial():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "cell.npz")
        ck = run_cell(out)
        assert os.path.exists(out)
        assert not os.path.exists(ck.partial_path), "partial must be removed on finalize"
        assert not any(f.endswith(".tmp.npz") for f in os.listdir(td)), "no temp residue"
        z = _load(out)
        assert set(z) == {"mu", "sigma", "n", "latency_ms", "compute_calls", "steps",
                          "method", "interval", "cfg"}, "final npz format unchanged"
        assert z["mu"].shape == (DIM,) and z["sigma"].shape == (DIM, DIM)
        assert int(z["n"]) == N


def test_kill_at_checkpoint_then_resume_is_bit_identical():
    with tempfile.TemporaryDirectory() as td:
        ref = os.path.join(td, "ref.npz")
        run_cell(ref)
        out = os.path.join(td, "cell.npz")
        try:
            run_cell(out, kill_after=20)            # dies right after the 20-image checkpoint
        except KilledRun:
            pass
        partial = out[:-4] + ".partial.npz"
        assert os.path.exists(partial), "checkpoint must exist at the kill point"
        assert int(_load(partial)["done"]) == 20
        run_cell(out)                                # fresh objects; resume from disk only
        a, b = _load(ref), _load(out)
        assert np.array_equal(a["mu"], b["mu"]), "resumed mu must be bit-identical"
        assert np.array_equal(a["sigma"], b["sigma"]), "resumed sigma must be bit-identical"
        assert np.array_equal(a["latency_ms"], b["latency_ms"])
        assert not os.path.exists(partial)


def test_kill_between_checkpoints_resumes_from_last_checkpoint():
    """A kill mid-window (done=25, last checkpoint at 20) loses only the
    unsaved batch; the resumed run recomputes it and still matches exactly."""
    with tempfile.TemporaryDirectory() as td:
        ref = os.path.join(td, "ref.npz")
        run_cell(ref)
        out = os.path.join(td, "cell.npz")
        try:
            run_cell(out, kill_after=25)
        except KilledRun:
            pass
        partial = out[:-4] + ".partial.npz"
        assert int(_load(partial)["done"]) == 20, "partial must hold the last 1k-boundary state"
        run_cell(out)
        a, b = _load(ref), _load(out)
        assert np.array_equal(a["mu"], b["mu"])
        assert np.array_equal(a["sigma"], b["sigma"])


def test_double_kill_then_resume_is_bit_identical():
    with tempfile.TemporaryDirectory() as td:
        ref = os.path.join(td, "ref.npz")
        run_cell(ref)
        out = os.path.join(td, "cell.npz")
        for ka in (10, 30):
            try:
                run_cell(out, kill_after=ka)
            except KilledRun:
                pass
        run_cell(out)
        a, b = _load(ref), _load(out)
        assert np.array_equal(a["mu"], b["mu"])
        assert np.array_equal(a["sigma"], b["sigma"])


def test_fingerprint_mismatch_refuses_resume():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "cell.npz")
        try:
            run_cell(out, kill_after=20)
        except KilledRun:
            pass
        other = CellCheckpoint(out, "different|config", every=EVERY)
        try:
            other.resume()
        except RuntimeError:
            pass
        else:
            raise AssertionError("resume must refuse a partial from a different config")


def test_atomic_savez_leaves_no_temp_file():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.npz")
        atomic_savez(p, a=np.arange(3))
        assert os.listdir(td) == ["x.npz"]
        assert np.array_equal(_load(p)["a"], np.arange(3))


def main() -> int:
    ok = True
    for fn in (test_uninterrupted_run_writes_final_and_cleans_partial,
               test_kill_at_checkpoint_then_resume_is_bit_identical,
               test_kill_between_checkpoints_resumes_from_last_checkpoint,
               test_double_kill_then_resume_is_bit_identical,
               test_fingerprint_mismatch_refuses_resume,
               test_atomic_savez_leaves_no_temp_file):
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as e:
            ok = False
            print(f"[FAIL] {fn.__name__}: {e}")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
