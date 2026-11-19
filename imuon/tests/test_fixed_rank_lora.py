"""CPU regressions for the no-momentum fixed-rank LoRA optimizer."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "swift" / "trainers"))
from optimizers import Muon


def invsqrt(gram):
    values, vectors = torch.linalg.eigh(gram)
    return (vectors * values.rsqrt()) @ vectors.T


def partial_polar(matrix):
    left, values, right = torch.linalg.svd(matrix, full_matrices=False)
    cutoff = torch.finfo(values.dtype).eps * max(matrix.shape) * values.max()
    return (left * (values > cutoff)) @ right


def reference(a, b, ga, gb, ridge):
    identity = torch.eye(a.shape[0], dtype=a.dtype)
    right = invsqrt(a @ a.T + ridge * identity)
    left = invsqrt(b.T @ b + ridge * identity)
    return left @ partial_polar(left @ ga), partial_polar(gb @ right) @ right


def step(a, b, ga, gb, ridge=0, method="svd"):
    pa, pb = (torch.nn.Parameter(x.clone()) for x in (a, b))
    optimizer = Muon(muon_params=[pa, pb], adamw_params=[], lora_pairs=[(pa, pb)],
                     lr=0.01, wd=0, momentum=0, nesterov=False, ns_steps=5,
                     lora_riemannian_muon=True, lora_riemannian_variant="v5_compact",
                     lora_precond_eps=ridge, lora_riemannian_ortho_method=method,
                     lora_riemannian_adjust_lr=False)
    pa.grad, pb.grad = ga, gb
    optimizer.step()
    return (a - pa.detach()) / 0.01, (b - pb.detach()) / 0.01, optimizer


class FixedRankLoRA(unittest.TestCase):
    def setUp(self):
        rng = torch.Generator().manual_seed(110)
        self.a, self.b, self.g = (torch.randn(shape, generator=rng, dtype=torch.float64)
                                  for shape in ((3, 7), (9, 3), (9, 7)))
        self.ga, self.gb = self.b.T @ self.g, self.g @ self.a.T

    def test_symmetric_gram_reference(self):
        for ridge in (0, 1e-6):
            da, db, _ = step(self.a, self.b, self.ga, self.gb, ridge)
            expected = reference(self.a, self.b, self.ga, self.gb, ridge)
            torch.testing.assert_close((da, db), expected, rtol=1e-11, atol=1e-11)

    def test_gauge_equivariance(self):
        a, b = self.a, self.b
        gauge = a[:, :3]
        rotated_a, rotated_b = gauge @ a, b @ torch.linalg.inv(gauge)
        da, db, _ = step(a, b, self.ga, self.gb)
        dra, drb, _ = step(rotated_a, rotated_b, rotated_b.T @ self.g, self.g @ rotated_a.T)
        torch.testing.assert_close(db @ a + b @ da, drb @ rotated_a + rotated_b @ dra,
                                   rtol=1e-10, atol=1e-10)

    def test_zero_b_startup(self):
        for method in ("ns", "svd"):
            da, db, _ = step(self.a, torch.zeros_like(self.b), torch.zeros_like(self.ga),
                             self.gb, 1e-6, method)
            self.assertTrue(torch.isfinite(db).all() and db.norm() > 0)
            self.assertEqual(da.count_nonzero(), 0)

    def test_packed_qv(self):
        blocks = [(self.a, self.b, self.ga, self.gb),
                  (self.a * 1.3, self.b * 0.7, self.ga * 0.7, self.gb * 1.3)]
        da, db, optimizer = step(*(torch.cat(parts) for parts in zip(*blocks)))
        expected = [reference(*block, 0) for block in blocks]
        torch.testing.assert_close((da, db), tuple(torch.cat(parts) for parts in zip(*expected)))
        self.assertEqual(optimizer._diagnostic_updated_logical_pairs, 2)
        self.assertEqual(optimizer._diagnostic_skipped_pairs, 0)


if __name__ == "__main__":
    unittest.main()
