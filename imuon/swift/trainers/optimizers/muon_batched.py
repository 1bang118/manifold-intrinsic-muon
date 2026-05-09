# Copyright (c) Alibaba, Inc. and its affiliates.
#
# Kernel-efficient batched Muon optimizer for LoRA fine-tuning.
# Groups same-shape LoRA pairs and parameters, then uses batched CUDA
# operations (batched QR, SVD, bmm) to minimize kernel launches.
#
# This file is independent of muon.py and does not affect existing experiments.
# It is a drop-in replacement for Muon with the same constructor interface.

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from .muon import _maybe_bool, _maybe_float, _maybe_int


def _batched_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Batched Newton-Schulz iteration for orthogonalization.

    Batched version of zeropower_via_newtonschulz5.  Operates on a 3D tensor
    ``(batch, rows, cols)`` using batched matmul — one CUDA kernel per matmul
    for the entire batch instead of one per matrix.

    Args:
        G: ``(N, rows, cols)`` batch of 2-D matrices.
        steps: Number of Newton-Schulz iterations.

    Returns:
        ``(N, rows, cols)`` batch of approximately-orthogonalized matrices.
    """
    assert G.ndim == 3
    a, b, c = (3.4445, -4.7750, 2.0315)

    if G.device.type == 'cuda':
        X = G.bfloat16()
    elif G.device.type == 'mps':
        X = G.to(dtype=torch.float16)
    else:
        X = G.to(dtype=torch.float32)

    # Make every matrix "wide" (rows <= cols), same as scalar version.
    transposed = X.size(1) > X.size(2)
    if transposed:
        X = X.transpose(-2, -1)

    # Per-matrix spectral-norm approximation via Frobenius norm.
    norms = X.flatten(1).norm(dim=1).unsqueeze(-1).unsqueeze(-1) + 1e-7  # (N,1,1)
    X = X / norms

    for _ in range(steps):
        A = X @ X.transpose(-2, -1)          # (N, K, K)
        B = b * A + c * (A @ A)              # (N, K, K)
        X = a * X + B @ X                    # (N, K, cols_or_rows)

    if transposed:
        X = X.transpose(-2, -1)
    return X


class MuonBatched(torch.optim.Optimizer):
    """Kernel-efficient Muon with batched CUDA operations.

    Drop-in replacement for :class:`Muon` that groups same-shape LoRA pairs
    (and vanilla Muon parameters) at init time, then at every ``step()``
    stacks them into batch tensors and calls *one* batched QR / SVD / bmm
    per shape group instead of *N* individual calls.

    .. note::
        Packed (merged QKV) LoRA is **not** supported in this implementation.
        All LoRA pairs must have matching rank (``A.shape[0] == B.shape[1]``).

    Constructor signature is identical to :class:`Muon`.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps=1e-8,
        lora_precond=False,
        lora_precond_eps=1e-6,
        lora_pairs=None,
        lora_riemannian_muon=False,
        lora_riemannian_ortho_method='ns',
        lora_riemannian_adjust_lr=True,
        lora_riemannian_variant='full',
        **kwargs,
    ):
        lr = _maybe_float(lr)
        wd = _maybe_float(wd)
        momentum = _maybe_float(momentum)
        nesterov = _maybe_bool(nesterov)
        ns_steps = _maybe_int(ns_steps)
        adamw_eps = _maybe_float(adamw_eps)
        lora_precond_eps = _maybe_float(lora_precond_eps)
        lora_riemannian_muon = _maybe_bool(lora_riemannian_muon)
        lora_riemannian_adjust_lr = _maybe_bool(lora_riemannian_adjust_lr)
        lora_riemannian_variant = str(lora_riemannian_variant).lower()
        if lora_riemannian_variant not in ('full', 'v5', 'full_cholqr', 'v5_cholqr', 'v5_rank_aware'):
            raise ValueError(
                f"MuonBatched supports variants 'full', 'v5', 'full_cholqr', 'v5_cholqr', 'v5_rank_aware', "
                f"got: {lora_riemannian_variant}"
            )

        beta1, beta2 = adamw_betas
        adamw_betas = (_maybe_float(beta1), _maybe_float(beta2))

        defaults = dict(
            lr=lr, wd=wd, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, adamw_betas=adamw_betas, adamw_eps=adamw_eps,
        )

        muon_params = list(muon_params or [])
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params = list(muon_params) + list(adamw_params)
        super().__init__(params, defaults)

        self.lora_precond_eps = lora_precond_eps
        self.lora_pairs = list(lora_pairs or [])
        self.lora_riemannian_muon = lora_riemannian_muon
        self.lora_riemannian_ortho_method = str(lora_riemannian_ortho_method)
        self.lora_riemannian_adjust_lr = lora_riemannian_adjust_lr
        self.lora_riemannian_variant = lora_riemannian_variant

        # Diagnostic counters (same interface as Muon for the benchmark)
        self._diagnostic_total_pairs = 0
        self._diagnostic_updated_pairs = 0
        self._diagnostic_skipped_pairs = 0
        self._diagnostic_printed = False

        for p in muon_params:
            assert p.ndim >= 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

        # Reject packed LoRA — not supported in batched mode.
        for A, B in self.lora_pairs:
            if A.ndim == 2 and B.ndim == 2 and A.shape[0] != B.shape[1]:
                raise NotImplementedError(
                    "Packed (merged QKV) LoRA is not supported in MuonBatched. "
                    f"Got A.shape={tuple(A.shape)}, B.shape={tuple(B.shape)}"
                )

        # Pre-compute shape groups.
        self._lora_shape_groups = self._build_lora_shape_groups()
        self._vanilla_shape_groups = self._build_vanilla_shape_groups(muon_params)

        # Identity-matrix cache: {(size, device, dtype): Tensor}
        self._eye_cache: Dict = {}

        # CUDA streams for overlapping independent ops in V1 CholQR.
        self._stream1: Optional[torch.cuda.Stream] = None
        self._stream2: Optional[torch.cuda.Stream] = None

    # ------------------------------------------------------------------
    # Shape grouping
    # ------------------------------------------------------------------

    def _build_lora_shape_groups(self):
        """Group LoRA pairs by ``(A.shape, B.shape)`` for batched ops."""
        groups: Dict[tuple, List[Tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
        for A, B in self.lora_pairs:
            if A.ndim != 2 or B.ndim != 2:
                continue
            if not (self.state.get(A, {}).get("use_muon", False)
                    and self.state.get(B, {}).get("use_muon", False)):
                continue
            key = (A.shape[0], A.shape[1], B.shape[0], B.shape[1])
            groups[key].append((A, B))
        return dict(groups)

    def _build_vanilla_shape_groups(self, muon_params):
        """Group vanilla Muon parameters by 2-D shape for batched NS5."""
        groups: Dict[tuple, List[torch.Tensor]] = defaultdict(list)
        for p in muon_params:
            shape_2d = (p.shape[0], math.prod(p.shape[1:]))
            groups[shape_2d].append(p)
        return dict(groups)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_streams(self) -> Tuple[torch.cuda.Stream, torch.cuda.Stream]:
        """Lazily create two CUDA streams for overlapping independent ops."""
        if self._stream1 is None:
            self._stream1 = torch.cuda.Stream()
            self._stream2 = torch.cuda.Stream()
        return self._stream1, self._stream2

    def _get_eye(self, size: int, device, dtype) -> torch.Tensor:
        key = (size, device, dtype)
        if key not in self._eye_cache:
            self._eye_cache[key] = torch.eye(size, device=device, dtype=dtype)
        return self._eye_cache[key]

    def adjust_lr_for_muon(self, lr: float, param_shape) -> float:
        A = param_shape[0]
        B = math.prod(param_shape[1:])
        return lr * 0.2 * math.sqrt(max(A, B))

    def _batched_ortho(self, mat_batch: torch.Tensor, ns_steps: int) -> torch.Tensor:
        """Batched orthogonalization (SVD or NS5) on ``(N, rows, cols)``."""
        method = (self.lora_riemannian_ortho_method or 'ns').lower()
        if method == 'svd':
            m32 = mat_batch.to(dtype=torch.float32)
            U, _, Vh = torch.linalg.svd(m32, full_matrices=False)
            return (U @ Vh).to(dtype=mat_batch.dtype)
        return _batched_newtonschulz5(mat_batch, steps=ns_steps).to(dtype=mat_batch.dtype)

    def _batched_cholqr(
        self, M: torch.Tensor, eps: float = 1e-6,
        return_gram: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, ...]:
        """Cholesky-QR: thin QR via Gram matrix Cholesky.

        Given M of shape (N, rows, cols) with rows >= cols, compute Q, R such
        that M ≈ Q @ R, Q has orthonormal columns, R is upper-triangular.

        The Gram matmul and Q=M@R_inv use M's dtype (bf16 on CUDA, matching
        vanilla Muon's NS5).  Only the tiny (cols×cols) Cholesky and
        triangular solve run in f32.

        Args:
            M: (N, rows, cols) with rows >= cols.
            eps: regularization added to Gram diagonal.
            return_gram: if True, also return the f32 Gram matrix (before eps).

        Returns:
            Q: (N, rows, cols) with orthonormal columns, same dtype as M.
            R: (N, cols, cols) upper-triangular, f32.
            G32: (N, cols, cols) Gram matrix, f32 — only if return_gram=True.
        """
        # Gram matrix in f32 for numerical stability (tiny c×c, no perf cost).
        M32 = M.to(torch.float32)
        G32 = M32.transpose(-2, -1) @ M32                    # (N, c, c) f32
        G32_raw = G32.clone() if return_gram else None
        G32.diagonal(dim1=-2, dim2=-1).add_(eps)
        L = torch.linalg.cholesky(G32)                       # (N, c, c) f32
        R = L.transpose(-2, -1).contiguous()                 # (N, c, c) f32
        eye_c = self._get_eye(R.shape[-1], R.device, torch.float32)
        eye_c_N = eye_c.unsqueeze(0).expand(R.shape[0], -1, -1)
        R_inv = torch.linalg.solve_triangular(R, eye_c_N, upper=True)  # (N, c, c) f32
        # Q = M @ R_inv: large matmul in M's dtype.
        Q = M @ R_inv.to(M.dtype)                            # (N, rows, cols)
        if return_gram:
            return Q, R, G32_raw
        return Q, R

    # ------------------------------------------------------------------
    # Batched Vanilla Muon
    # ------------------------------------------------------------------

    def _batched_vanilla_muon_step(self, params: List[torch.Tensor], group):
        """Batched vanilla Muon: stack same-shape params → batched NS5 → scatter."""
        lr = group["lr"]
        wd = group["wd"]
        momentum = group["momentum"]
        ns_steps = group["ns_steps"]

        # Re-group params that actually have gradients (shape is already matched).
        shape_groups: Dict[tuple, List[torch.Tensor]] = defaultdict(list)
        for p in params:
            if p.grad is None:
                continue
            shape_2d = (p.shape[0], math.prod(p.shape[1:]))
            shape_groups[shape_2d].append(p)

        for (rows, cols), param_list in shape_groups.items():
            grads = []
            valid = []
            for p in param_list:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(rows, cols)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf.clone()

                grads.append(g)
                valid.append(p)

            if not grads:
                continue

            # (N, rows, cols) — one batched NS5 call.
            G_batch = torch.stack(grads)
            U_batch = _batched_newtonschulz5(G_batch, steps=ns_steps)

            for i, p in enumerate(valid):
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * wd)
                p.data.add_(U_batch[i].reshape(p.shape).to(dtype=p.dtype), alpha=-adjusted_lr)

    # ------------------------------------------------------------------
    # Batched Riemannian V1 (full)
    # ------------------------------------------------------------------

    def _batched_lora_v1(self, group) -> set:
        """Batched Riemannian V1 update across all shape groups."""
        momentum_coeff = group["momentum"]
        ns_steps = group["ns_steps"]
        lr = group["lr"]
        wd = group["wd"]
        eps = self.lora_precond_eps
        updated: set = set()

        for shape_key, pairs in self._lora_shape_groups.items():
            r_A, n_A, m_B, r_B = shape_key  # A: (r, n), B: (m, r)
            r = r_A
            m = m_B
            n = n_A

            # Collect valid pairs with momentum.
            As, Bs, MAs, MBs = [], [], [], []
            valid_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

            for A, B in pairs:
                self._diagnostic_total_pairs += 1
                if A.grad is None or B.grad is None:
                    self._diagnostic_skipped_pairs += 1
                    continue

                gA, gB = A.grad, B.grad
                st_A, st_B = self.state[A], self.state[B]
                if "momentum_buffer" not in st_A:
                    st_A["momentum_buffer"] = torch.zeros_like(gA)
                if "momentum_buffer" not in st_B:
                    st_B["momentum_buffer"] = torch.zeros_like(gB)
                buf_A = st_A["momentum_buffer"]
                buf_B = st_B["momentum_buffer"]
                buf_A.mul_(momentum_coeff).add_(gA)
                buf_B.mul_(momentum_coeff).add_(gB)

                if group["nesterov"]:
                    MA = gA.add(buf_A, alpha=momentum_coeff)
                    MB = gB.add(buf_B, alpha=momentum_coeff)
                else:
                    MA = buf_A.clone()
                    MB = buf_B.clone()

                As.append(A.data)
                Bs.append(B.data)
                MAs.append(MA)
                MBs.append(MB)
                valid_pairs.append((A, B))

            if not valid_pairs:
                continue

            N = len(valid_pairs)

            # Stack into batched tensors — all float32 for linalg stability.
            A_bat = torch.stack(As).to(torch.float32)     # (N, r, n)
            B_bat = torch.stack(Bs).to(torch.float32)     # (N, m, r)
            MA_bat = torch.stack(MAs).to(torch.float32)   # (N, r, n)
            MB_bat = torch.stack(MBs).to(torch.float32)   # (N, m, r)

            # X = [MB | B] → (N, m, 2r)
            X_bat = torch.cat([MB_bat, B_bat], dim=2)
            # Y = [A ; MA] → (N, 2r, n)
            Y_bat = torch.cat([A_bat, MA_bat], dim=1)

            # Batched QR of X (N, m, 2r) and Y^T (N, n, 2r).
            Qx, Rx = torch.linalg.qr(X_bat, mode='reduced')        # (N,m,2r), (N,2r,2r)
            YT_bat = Y_bat.transpose(-2, -1)                        # (N, n, 2r)
            Qy, Ry = torch.linalg.qr(YT_bat, mode='reduced')       # (N,n,2r), (N,2r,2r)

            # Core = Rx @ Ry^T → (N, 2r, 2r).
            core = Rx @ Ry.transpose(-2, -1)

            # Batched ortho of the small (N, 2r, 2r) cores.
            Ocore = self._batched_ortho(core, ns_steps=ns_steps)

            # Gram inverses via batched solve: (A A^T + eps I)^{-1} and (B^T B + eps I)^{-1}.
            eye_r = self._get_eye(r, A_bat.device, torch.float32)
            eye_r_N = eye_r.unsqueeze(0).expand(N, -1, -1)

            AAT = A_bat @ A_bat.transpose(-2, -1) + eps * eye_r     # (N, r, r)
            BTB = B_bat.transpose(-2, -1) @ B_bat + eps * eye_r     # (N, r, r)
            inv_AAT = torch.linalg.solve(AAT, eye_r_N)              # (N, r, r)
            inv_BTB = torch.linalg.solve(BTB, eye_r_N)              # (N, r, r)

            # dB = Qx @ Ocore @ (Qy^T @ A^T) @ inv_AAT
            A_proj = Qy.transpose(-2, -1) @ A_bat.transpose(-2, -1) # (N, 2r, r)
            tmp = Ocore @ A_proj                                     # (N, 2r, r)
            OB = Qx @ tmp                                            # (N, m, r)
            dB_bat = OB @ inv_AAT                                    # (N, m, r)

            # dA = inv_BTB @ (B^T @ Qx) @ Ocore @ Qy^T
            B_proj = B_bat.transpose(-2, -1) @ Qx                   # (N, r, 2r)
            tmp2 = B_proj @ Ocore                                    # (N, r, 2r)
            OA = tmp2 @ Qy.transpose(-2, -1)                        # (N, r, n)
            dA_bat = inv_BTB @ OA                                    # (N, r, n)

            # Scatter updates back.
            self._scatter_lora_updates(valid_pairs, dA_bat, dB_bat, lr, wd)
            for A, B in valid_pairs:
                updated.add(A)
                updated.add(B)
                self._diagnostic_updated_pairs += 1

        return updated

    # ------------------------------------------------------------------
    # Batched Riemannian V1 (Cholesky-QR, all-matmul)
    # ------------------------------------------------------------------

    def _batched_lora_v1_cholqr(self, group) -> set:
        """Batched Riemannian V1 with Cholesky-QR — no LAPACK QR kernels.

        Same algorithm as ``_batched_lora_v1`` but replaces:
          - ``torch.linalg.qr`` → Cholesky-QR (matmul + tiny Cholesky)
          - ``torch.linalg.solve`` for Gram inverses → Cholesky + triangular solve
        All heavy computation is batched matmuls (cuBLAS), matching vanilla
        Muon's kernel profile for a fair wall-clock comparison.
        """
        momentum_coeff = group["momentum"]
        ns_steps = group["ns_steps"]
        lr = group["lr"]
        wd = group["wd"]
        eps = self.lora_precond_eps
        updated: set = set()

        for shape_key, pairs in self._lora_shape_groups.items():
            r_A, n_A, m_B, r_B = shape_key
            r = r_A
            m = m_B
            n = n_A

            As, Bs, MAs, MBs = [], [], [], []
            valid_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

            for A, B in pairs:
                self._diagnostic_total_pairs += 1
                if A.grad is None or B.grad is None:
                    self._diagnostic_skipped_pairs += 1
                    continue

                gA, gB = A.grad, B.grad
                st_A, st_B = self.state[A], self.state[B]
                if "momentum_buffer" not in st_A:
                    st_A["momentum_buffer"] = torch.zeros_like(gA)
                if "momentum_buffer" not in st_B:
                    st_B["momentum_buffer"] = torch.zeros_like(gB)
                buf_A = st_A["momentum_buffer"]
                buf_B = st_B["momentum_buffer"]
                buf_A.mul_(momentum_coeff).add_(gA)
                buf_B.mul_(momentum_coeff).add_(gB)

                if group["nesterov"]:
                    MA = gA.add(buf_A, alpha=momentum_coeff)
                    MB = gB.add(buf_B, alpha=momentum_coeff)
                else:
                    MA = buf_A.clone()
                    MB = buf_B.clone()

                As.append(A.data)
                Bs.append(B.data)
                MAs.append(MA)
                MBs.append(MB)
                valid_pairs.append((A, B))

            if not valid_pairs:
                continue

            N = len(valid_pairs)

            # Use bf16 for large matmuls (matching vanilla Muon's NS5),
            # f32 only for tiny r×r Cholesky/solve ops.
            compute_dtype = torch.bfloat16 if As[0].device.type == 'cuda' else torch.float32
            A_bat = torch.stack(As).to(compute_dtype)     # (N, r, n)
            B_bat = torch.stack(Bs).to(compute_dtype)     # (N, m, r)
            MA_bat = torch.stack(MAs).to(compute_dtype)   # (N, r, n)
            MB_bat = torch.stack(MBs).to(compute_dtype)   # (N, m, r)

            # X = [MB | B] → (N, m, 2r),  Y = [A ; MA] → (N, 2r, n)
            X_bat = torch.cat([MB_bat, B_bat], dim=2)
            Y_bat = torch.cat([A_bat, MA_bat], dim=1)

            # ---- Cholesky-QR with CUDA stream parallelism ----
            # Phase 1: CholQR(X) and CholQR(Y^T) are independent → overlap.
            use_streams = A_bat.device.type == 'cuda'
            if use_streams:
                s1, s2 = self._get_streams()
                # Record current stream so s1/s2 wait for stack/cat to finish.
                main_stream = torch.cuda.current_stream()
                s1.wait_stream(main_stream)
                s2.wait_stream(main_stream)

                with torch.cuda.stream(s1):
                    Qx, Rx, Gx = self._batched_cholqr(X_bat, eps=eps, return_gram=True)
                with torch.cuda.stream(s2):
                    YT_bat = Y_bat.transpose(-2, -1)
                    Qy, Ry = self._batched_cholqr(YT_bat, eps=eps)

                main_stream.wait_stream(s1)
                main_stream.wait_stream(s2)
            else:
                Qx, Rx, Gx = self._batched_cholqr(X_bat, eps=eps, return_gram=True)
                YT_bat = Y_bat.transpose(-2, -1)
                Qy, Ry = self._batched_cholqr(YT_bat, eps=eps)

            # Phase 2: Core ortho (sequential — needs both Rx and Ry).
            core = Rx @ Ry.transpose(-2, -1)
            Ocore = _batched_newtonschulz5(core, steps=ns_steps).to(torch.float32)
            Ocore_cd = Ocore.to(compute_dtype)

            # Phase 3: Gram inverses + projections — dB and dA are independent → overlap.
            # AAT: reuse Cholesky from CholQR of Y^T (block Cholesky extraction).
            # BTB: extract subblock from Gx, separate Cholesky.
            eye_r = self._get_eye(r, A_bat.device, torch.float32)
            eye_r_N = eye_r.unsqueeze(0).expand(N, -1, -1)

            if use_streams:
                s1.wait_stream(main_stream)
                s2.wait_stream(main_stream)

                with torch.cuda.stream(s1):
                    # dB: inv_AAT + projections
                    L_AAT = Ry[:, :r, :r].transpose(-2, -1).contiguous()
                    inv_AAT = torch.cholesky_solve(eye_r_N, L_AAT)
                    A_proj = Qy.transpose(-2, -1) @ A_bat.transpose(-2, -1)
                    tmp = Ocore_cd @ A_proj
                    OB = Qx @ tmp
                    dB_bat = OB.float() @ inv_AAT

                with torch.cuda.stream(s2):
                    # dA: inv_BTB + projections
                    BTB = Gx[:, r:, r:].clone()
                    BTB.diagonal(dim1=-2, dim2=-1).add_(eps)
                    L_BTB = torch.linalg.cholesky(BTB)
                    inv_BTB = torch.cholesky_solve(eye_r_N, L_BTB)
                    B_proj = B_bat.transpose(-2, -1) @ Qx
                    tmp2 = B_proj @ Ocore_cd
                    OA = tmp2 @ Qy.transpose(-2, -1)
                    dA_bat = inv_BTB @ OA.float()

                main_stream.wait_stream(s1)
                main_stream.wait_stream(s2)
            else:
                L_AAT = Ry[:, :r, :r].transpose(-2, -1).contiguous()
                inv_AAT = torch.cholesky_solve(eye_r_N, L_AAT)
                BTB = Gx[:, r:, r:].clone()
                BTB.diagonal(dim1=-2, dim2=-1).add_(eps)
                L_BTB = torch.linalg.cholesky(BTB)
                inv_BTB = torch.cholesky_solve(eye_r_N, L_BTB)

                A_proj = Qy.transpose(-2, -1) @ A_bat.transpose(-2, -1)
                tmp = Ocore_cd @ A_proj
                OB = Qx @ tmp
                dB_bat = OB.float() @ inv_AAT

                B_proj = B_bat.transpose(-2, -1) @ Qx
                tmp2 = B_proj @ Ocore_cd
                OA = tmp2 @ Qy.transpose(-2, -1)
                dA_bat = inv_BTB @ OA.float()

            self._scatter_lora_updates(valid_pairs, dA_bat, dB_bat, lr, wd)
            for A, B in valid_pairs:
                updated.add(A)
                updated.add(B)
                self._diagnostic_updated_pairs += 1

        return updated

    # ------------------------------------------------------------------
    # Batched Riemannian V5
    # ------------------------------------------------------------------

    def _batched_lora_v5(self, group) -> set:
        """Batched Riemannian V5 update (original form, numerically stable).

        From Section 3.5 of the paper:
            dB = Ortho(M_t P_A) A^T (AA^T)^{-1}
            dA = (B^T B)^{-1} B^T Ortho(P_B M_t)

        Uses gram inverses (same as V1) instead of QR R^{-1} to avoid
        blow-up when B=0 at LoRA initialization.
        """
        momentum_coeff = group["momentum"]
        ns_steps = group["ns_steps"]
        lr = group["lr"]
        wd = group["wd"]
        eps = self.lora_precond_eps
        updated: set = set()

        for shape_key, pairs in self._lora_shape_groups.items():
            r_A, n_A, m_B, r_B = shape_key
            r = r_A
            m = m_B
            n = n_A

            As, Bs, MAs, MBs = [], [], [], []
            valid_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

            for A, B in pairs:
                self._diagnostic_total_pairs += 1
                if A.grad is None or B.grad is None:
                    self._diagnostic_skipped_pairs += 1
                    continue

                gA, gB = A.grad, B.grad
                st_A, st_B = self.state[A], self.state[B]
                if "momentum_buffer" not in st_A:
                    st_A["momentum_buffer"] = torch.zeros_like(gA)
                if "momentum_buffer" not in st_B:
                    st_B["momentum_buffer"] = torch.zeros_like(gB)
                buf_A = st_A["momentum_buffer"]
                buf_B = st_B["momentum_buffer"]
                buf_A.mul_(momentum_coeff).add_(gA)
                buf_B.mul_(momentum_coeff).add_(gB)

                if group["nesterov"]:
                    MA = gA.add(buf_A, alpha=momentum_coeff)
                    MB = gB.add(buf_B, alpha=momentum_coeff)
                else:
                    MA = buf_A.clone()
                    MB = buf_B.clone()

                As.append(A.data)
                Bs.append(B.data)
                MAs.append(MA)
                MBs.append(MB)
                valid_pairs.append((A, B))

            if not valid_pairs:
                continue

            N = len(valid_pairs)

            A_bat = torch.stack(As).to(torch.float32)
            B_bat = torch.stack(Bs).to(torch.float32)
            MA_bat = torch.stack(MAs).to(torch.float32)
            MB_bat = torch.stack(MBs).to(torch.float32)

            # X = [MB | B] → (N, m, 2r),  Y = [A ; MA] → (N, 2r, n)
            X_bat = torch.cat([MB_bat, B_bat], dim=2)
            Y_bat = torch.cat([A_bat, MA_bat], dim=1)

            # Gram inverses (same as V1)
            eye_r = self._get_eye(r, A_bat.device, torch.float32)
            eye_r_N = eye_r.unsqueeze(0).expand(N, -1, -1)
            AAT = A_bat @ A_bat.transpose(-2, -1) + eps * eye_r     # (N, r, r)
            BTB = B_bat.transpose(-2, -1) @ B_bat + eps * eye_r     # (N, r, r)
            inv_AAT = torch.linalg.solve(AAT, eye_r_N)              # (N, r, r)
            inv_BTB = torch.linalg.solve(BTB, eye_r_N)              # (N, r, r)

            # --- dB = Ortho(M_t P_A) A^T (AA^T)^{-1} ---
            # P_A = A^T inv_AAT A, so M_t P_A = X @ Y_1 where Y_1 = Y A^T inv_AAT A
            YAT = Y_bat @ A_bat.transpose(-2, -1)                   # (N, 2r, r)
            Y_1 = (YAT @ inv_AAT) @ A_bat                           # (N, 2r, n)

            # Low-rank Ortho(X @ Y_1) via QR
            Qx_B, Rx_B = torch.linalg.qr(X_bat, mode='reduced')    # (N,m,2r), (N,2r,2r)
            Y1T = Y_1.transpose(-2, -1)                             # (N, n, 2r)
            Qy_B, Ry_B = torch.linalg.qr(Y1T, mode='reduced')      # (N,n,2r), (N,2r,2r)
            core_B = Rx_B @ Ry_B.transpose(-2, -1)                  # (N, 2r, 2r)
            Ocore_B = self._batched_ortho(core_B, ns_steps=ns_steps)

            A_proj = Qy_B.transpose(-2, -1) @ A_bat.transpose(-2, -1)  # (N, 2r, r)
            tmp = Ocore_B @ A_proj                                      # (N, 2r, r)
            OB = Qx_B @ tmp                                             # (N, m, r)
            dB_bat = OB @ inv_AAT                                       # (N, m, r)

            # --- dA = (B^TB)^{-1} B^T Ortho(P_B M_t) ---
            # P_B = B inv_BTB B^T, so P_B M_t = X_2 @ Y where X_2 = B inv_BTB B^T X
            BTX = B_bat.transpose(-2, -1) @ X_bat                   # (N, r, 2r)
            X_2 = B_bat @ (inv_BTB @ BTX)                           # (N, m, 2r)

            # Low-rank Ortho(X_2 @ Y) via QR
            Qx_A, Rx_A = torch.linalg.qr(X_2, mode='reduced')      # (N,m,2r), (N,2r,2r)
            YT_bat = Y_bat.transpose(-2, -1)                        # (N, n, 2r)
            Qy_A, Ry_A = torch.linalg.qr(YT_bat, mode='reduced')   # (N,n,2r), (N,2r,2r)
            core_A = Rx_A @ Ry_A.transpose(-2, -1)                  # (N, 2r, 2r)
            Ocore_A = self._batched_ortho(core_A, ns_steps=ns_steps)

            B_proj = B_bat.transpose(-2, -1) @ Qx_A                 # (N, r, 2r)
            tmp2 = B_proj @ Ocore_A                                  # (N, r, 2r)
            OA = tmp2 @ Qy_A.transpose(-2, -1)                      # (N, r, n)
            dA_bat = inv_BTB @ OA                                    # (N, r, n)

            # Scatter updates back.
            self._scatter_lora_updates(valid_pairs, dA_bat, dB_bat, lr, wd)
            for A, B in valid_pairs:
                updated.add(A)
                updated.add(B)
                self._diagnostic_updated_pairs += 1

        return updated

    # ------------------------------------------------------------------
    # Batched Riemannian V5 (Cholesky-QR, all-matmul)
    # ------------------------------------------------------------------

    def _batched_lora_v5_cholqr(self, group) -> set:
        """Batched Riemannian V5 with Cholesky-QR (original form, numerically stable).

        Same as ``_batched_lora_v5`` but uses Cholesky-QR for the low-rank
        Ortho decomposition instead of LAPACK QR.

        From Section 3.5:
            dB = Ortho(M_t P_A) A^T (AA^T)^{-1}
            dA = (B^T B)^{-1} B^T Ortho(P_B M_t)
        """
        momentum_coeff = group["momentum"]
        ns_steps = group["ns_steps"]
        lr = group["lr"]
        wd = group["wd"]
        eps = self.lora_precond_eps
        updated: set = set()

        for shape_key, pairs in self._lora_shape_groups.items():
            r_A, n_A, m_B, r_B = shape_key
            r = r_A
            m = m_B
            n = n_A

            As, Bs, MAs, MBs = [], [], [], []
            valid_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

            for A, B in pairs:
                self._diagnostic_total_pairs += 1
                if A.grad is None or B.grad is None:
                    self._diagnostic_skipped_pairs += 1
                    continue

                gA, gB = A.grad, B.grad
                st_A, st_B = self.state[A], self.state[B]
                if "momentum_buffer" not in st_A:
                    st_A["momentum_buffer"] = torch.zeros_like(gA)
                if "momentum_buffer" not in st_B:
                    st_B["momentum_buffer"] = torch.zeros_like(gB)
                buf_A = st_A["momentum_buffer"]
                buf_B = st_B["momentum_buffer"]
                buf_A.mul_(momentum_coeff).add_(gA)
                buf_B.mul_(momentum_coeff).add_(gB)

                if group["nesterov"]:
                    MA = gA.add(buf_A, alpha=momentum_coeff)
                    MB = gB.add(buf_B, alpha=momentum_coeff)
                else:
                    MA = buf_A.clone()
                    MB = buf_B.clone()

                As.append(A.data)
                Bs.append(B.data)
                MAs.append(MA)
                MBs.append(MB)
                valid_pairs.append((A, B))

            if not valid_pairs:
                continue

            N = len(valid_pairs)

            # Use bf16 for large matmuls, f32 for tiny r×r ops.
            compute_dtype = torch.bfloat16 if As[0].device.type == 'cuda' else torch.float32
            A_bat = torch.stack(As).to(compute_dtype)
            B_bat = torch.stack(Bs).to(compute_dtype)
            MA_bat = torch.stack(MAs).to(compute_dtype)
            MB_bat = torch.stack(MBs).to(compute_dtype)

            # X = [MB | B] → (N, m, 2r),  Y = [A ; MA] → (N, 2r, n)
            X_bat = torch.cat([MB_bat, B_bat], dim=2)
            Y_bat = torch.cat([A_bat, MA_bat], dim=1)

            # Gram inverses in f32
            A_f32 = A_bat.float()
            B_f32 = B_bat.float()
            eye_r = self._get_eye(r, A_bat.device, torch.float32)
            eye_r_N = eye_r.unsqueeze(0).expand(N, -1, -1)
            AAT = A_f32 @ A_f32.transpose(-2, -1) + eps * eye_r     # (N, r, r)
            BTB = B_f32.transpose(-2, -1) @ B_f32 + eps * eye_r     # (N, r, r)
            inv_AAT = torch.linalg.solve(AAT, eye_r_N)              # (N, r, r)
            inv_BTB = torch.linalg.solve(BTB, eye_r_N)              # (N, r, r)

            # --- dB = Ortho(M_t P_A) A^T (AA^T)^{-1} ---
            # Y_1 = Y A^T inv_AAT A  (projected momentum right factor)
            YAT = Y_bat @ A_bat.transpose(-2, -1)                   # (N, 2r, r) bf16
            Y_1 = (YAT.float() @ inv_AAT) @ A_bat.float()          # (N, 2r, n) f32
            Y_1 = Y_1.to(compute_dtype)

            # Low-rank Ortho(X @ Y_1) via Cholesky-QR
            Qx_B, Rx_B = self._batched_cholqr(X_bat, eps=eps)
            Y1T = Y_1.transpose(-2, -1)
            Qy_B, Ry_B = self._batched_cholqr(Y1T, eps=eps)
            core_B = Rx_B @ Ry_B.transpose(-2, -1)                  # (N, 2r, 2r) f32
            Ocore_B = _batched_newtonschulz5(core_B.to(compute_dtype), steps=ns_steps)

            A_proj = Qy_B.transpose(-2, -1) @ A_bat.transpose(-2, -1)  # (N, 2r, r)
            tmp = Ocore_B @ A_proj                                      # (N, 2r, r)
            OB = Qx_B @ tmp                                             # (N, m, r)
            dB_bat = OB.float() @ inv_AAT                               # (N, m, r)

            # --- dA = (B^TB)^{-1} B^T Ortho(P_B M_t) ---
            # X_2 = B inv_BTB B^T X  (projected momentum left factor)
            BTX = B_bat.transpose(-2, -1) @ X_bat                   # (N, r, 2r) bf16
            X_2 = B_bat.float() @ (inv_BTB @ BTX.float())           # (N, m, 2r) f32
            X_2 = X_2.to(compute_dtype)

            # Low-rank Ortho(X_2 @ Y) via Cholesky-QR
            Qx_A, Rx_A = self._batched_cholqr(X_2, eps=eps)
            YT_bat = Y_bat.transpose(-2, -1)
            Qy_A, Ry_A = self._batched_cholqr(YT_bat, eps=eps)
            core_A = Rx_A @ Ry_A.transpose(-2, -1)                  # (N, 2r, 2r) f32
            Ocore_A = _batched_newtonschulz5(core_A.to(compute_dtype), steps=ns_steps)

            B_proj = B_bat.transpose(-2, -1) @ Qx_A                 # (N, r, 2r)
            tmp2 = B_proj @ Ocore_A                                  # (N, r, 2r)
            OA = tmp2 @ Qy_A.transpose(-2, -1)                      # (N, r, n)
            dA_bat = inv_BTB @ OA.float()                            # (N, r, n)

            self._scatter_lora_updates(valid_pairs, dA_bat, dB_bat, lr, wd)
            for A, B in valid_pairs:
                updated.add(A)
                updated.add(B)
                self._diagnostic_updated_pairs += 1

        return updated

    # ------------------------------------------------------------------
    # Batched Riemannian V5 — rank-aware compact form
    # ------------------------------------------------------------------

    def _batched_lora_v5_rank_aware(self, group) -> set:
        """Batched Riemannian V5, rank-aware compact form.

        The PDF derivation gives V5 as
            dB = Ortho(M_t P_A) A^T (A A^T)^{-1}
            dA = (B^T B)^{-1} B^T Ortho(P_B M_t)

        Although M_t = X Y has rank up to 2r, the projected objects satisfy
            rank(P_B M_t) <= r,    rank(M_t P_A) <= r.

        The existing V5 / V5_cholqr paths factor through (P_B X) Y, which
        carries the rank-r object inside a 2r-column matrix and forces a
        rank-deficient 2r x 2r core. LAPACK QR fills the null space with
        arbitrary directions; Cholesky-QR fails with a non-PD Gram error.

        This variant uses the tighter factorization
            P_B M_t = B alpha,    M_t P_A = beta A,
        and after thin QR
            B = Q_B R_B,    A^T = Q_A R_A
        rewrites the updates as
            dA = R_B^{-1} Ortho(Q_B^T M_t)
            dB = Ortho(M_t Q_A) R_A^{-T}

        Q_B^T M_t and M_t Q_A both live at the intrinsic rank r of the
        problem, so NS5/SVD operates on full-rank matrices and Cholesky-QR
        on (B) and (A^T) is well-conditioned.

        IMPORTANT: this computes the *exact* polar factor of P_B M_t, whereas
        v5 / v5_cholqr produce an approximation contaminated by null-space
        arbitrariness. It is not numerically equivalent to those paths and
        must NOT be silently substituted for the V5 used in training.
        """
        momentum_coeff = group["momentum"]
        ns_steps = group["ns_steps"]
        lr = group["lr"]
        wd = group["wd"]
        eps = self.lora_precond_eps
        updated: set = set()

        for shape_key, pairs in self._lora_shape_groups.items():
            r_A, n_A, m_B, r_B = shape_key
            r = r_A
            m = m_B
            n = n_A

            As, Bs, MAs, MBs = [], [], [], []
            valid_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

            for A, B in pairs:
                self._diagnostic_total_pairs += 1
                if A.grad is None or B.grad is None:
                    self._diagnostic_skipped_pairs += 1
                    continue

                gA, gB = A.grad, B.grad
                st_A, st_B = self.state[A], self.state[B]
                if "momentum_buffer" not in st_A:
                    st_A["momentum_buffer"] = torch.zeros_like(gA)
                if "momentum_buffer" not in st_B:
                    st_B["momentum_buffer"] = torch.zeros_like(gB)
                buf_A = st_A["momentum_buffer"]
                buf_B = st_B["momentum_buffer"]
                buf_A.mul_(momentum_coeff).add_(gA)
                buf_B.mul_(momentum_coeff).add_(gB)

                if group["nesterov"]:
                    MA = gA.add(buf_A, alpha=momentum_coeff)
                    MB = gB.add(buf_B, alpha=momentum_coeff)
                else:
                    MA = buf_A.clone()
                    MB = buf_B.clone()

                As.append(A.data)
                Bs.append(B.data)
                MAs.append(MA)
                MBs.append(MB)
                valid_pairs.append((A, B))

            if not valid_pairs:
                continue

            N = len(valid_pairs)

            compute_dtype = torch.bfloat16 if As[0].device.type == 'cuda' else torch.float32
            A_bat = torch.stack(As).to(compute_dtype)
            B_bat = torch.stack(Bs).to(compute_dtype)
            MA_bat = torch.stack(MAs).to(compute_dtype)
            MB_bat = torch.stack(MBs).to(compute_dtype)

            # CholQR on rank-r factors only — both generically full rank.
            # No 2r-wide rank-deficient intermediate.
            Q_B, R_B = self._batched_cholqr(B_bat, eps=eps)               # (N,m,r), (N,r,r) f32
            AT_bat = A_bat.transpose(-2, -1).contiguous()
            Q_A, R_A = self._batched_cholqr(AT_bat, eps=eps)              # (N,n,r), (N,r,r) f32

            eye_r = self._get_eye(r, B_bat.device, torch.float32)
            eye_r_N = eye_r.unsqueeze(0).expand(N, -1, -1)
            R_B_inv = torch.linalg.solve_triangular(R_B, eye_r_N, upper=True)  # (N,r,r) f32
            R_A_inv = torch.linalg.solve_triangular(R_A, eye_r_N, upper=True)  # (N,r,r) f32

            # === dA = R_B^{-1} Ortho(Q_B^T M_t) ===
            # Q_B^T M_t = Q_B^T (M_B A + B M_A)
            #          = (Q_B^T M_B) A + (Q_B^T B) M_A
            #          = (Q_B^T M_B) A + R_B M_A           since Q_B^T B = R_B
            QBT_MB = Q_B.transpose(-2, -1) @ MB_bat                       # (N, r, r)  bf16
            R_B_cd = R_B.to(compute_dtype)
            QBT_Mt = QBT_MB @ A_bat + R_B_cd @ MA_bat                     # (N, r, n)
            # Polar factor of (r, n) full-rank matrix. NS5 transposes wide
            # matrices internally; X X^T is r x r (small).
            ortho_QBT_Mt = self._batched_ortho(
                QBT_Mt.float(), ns_steps=ns_steps
            ).to(compute_dtype)                                            # (N, r, n)
            dA_bat = R_B_inv.to(compute_dtype) @ ortho_QBT_Mt              # (N, r, n)

            # === dB = Ortho(M_t Q_A) R_A^{-T} ===
            # M_t Q_A = (M_B A + B M_A) Q_A
            #        = M_B (A Q_A) + B (M_A Q_A)
            #        = M_B R_A^T   + B (M_A Q_A)           since A Q_A = R_A^T
            MA_QA = MA_bat @ Q_A.to(compute_dtype)                         # (N, r, r)
            R_A_T_cd = R_A.transpose(-2, -1).to(compute_dtype)
            Mt_QA = MB_bat @ R_A_T_cd + B_bat @ MA_QA                      # (N, m, r)
            # Polar factor of (m, r) full-rank matrix. NS5 transposes tall
            # matrices internally; X X^T is r x r (small).
            ortho_Mt_QA = self._batched_ortho(
                Mt_QA.float(), ns_steps=ns_steps
            ).to(compute_dtype)                                            # (N, m, r)
            dB_bat = ortho_Mt_QA @ R_A_inv.transpose(-2, -1).to(compute_dtype)  # (N, m, r)

            self._scatter_lora_updates(valid_pairs, dA_bat.float(), dB_bat.float(), lr, wd)
            for A, B in valid_pairs:
                updated.add(A)
                updated.add(B)
                self._diagnostic_updated_pairs += 1

        return updated

    # ------------------------------------------------------------------
    # Common update scatter
    # ------------------------------------------------------------------

    def _scatter_lora_updates(
        self,
        pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        dA_bat: torch.Tensor,
        dB_bat: torch.Tensor,
        lr: float,
        wd: float,
    ):
        """Write batched dA/dB updates back to individual parameter tensors."""
        for i, (A, B) in enumerate(pairs):
            dA = dA_bat[i]
            dB = dB_bat[i]

            if wd:
                A.data.mul_(1 - lr * wd)
                B.data.mul_(1 - lr * wd)

            if self.lora_riemannian_adjust_lr:
                lrA = self.adjust_lr_for_muon(lr, A.shape)
                lrB = self.adjust_lr_for_muon(lr, B.shape)
            else:
                lrA = lrB = lr

            A.data.add_(dA.to(dtype=A.dtype), alpha=-lrA)
            B.data.add_(dB.to(dtype=B.dtype), alpha=-lrB)

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self._diagnostic_printed:
            self._diagnostic_total_pairs = 0
            self._diagnostic_updated_pairs = 0
            self._diagnostic_skipped_pairs = 0

        for group in self.param_groups:
            # ── Muon parameters ──
            muon_params = [p for p in group["params"] if self.state[p]["use_muon"]]

            # 1. Batched Riemannian LoRA update (if enabled).
            updated_lora_params: set = set()
            if self.lora_riemannian_muon and self.lora_pairs:
                if self.lora_riemannian_variant == 'full':
                    updated_lora_params = self._batched_lora_v1(group)
                elif self.lora_riemannian_variant == 'v5':
                    updated_lora_params = self._batched_lora_v5(group)
                elif self.lora_riemannian_variant == 'full_cholqr':
                    updated_lora_params = self._batched_lora_v1_cholqr(group)
                elif self.lora_riemannian_variant == 'v5_cholqr':
                    updated_lora_params = self._batched_lora_v5_cholqr(group)
                elif self.lora_riemannian_variant == 'v5_rank_aware':
                    updated_lora_params = self._batched_lora_v5_rank_aware(group)

            # 2. Batched vanilla Muon for remaining params.
            remaining = [p for p in muon_params if p not in updated_lora_params]
            if remaining:
                self._batched_vanilla_muon_step(remaining, group)

            # ── AdamW fallback (identical to Muon) ──
            adamw_params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in adamw_params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step_count = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g_hat = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1 ** step_count
                bias_correction2 = 1 - beta2 ** step_count
                scale = bias_correction1 / bias_correction2 ** 0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g_hat, alpha=-lr / scale)

        # Diagnostic print on first step.
        if not self._diagnostic_printed and self.lora_riemannian_muon and self.lora_pairs:
            print("=" * 80)
            print("MuonBatched Optimizer - LoRA Riemannian Update Diagnostics:")
            print(f"  Variant: {self.lora_riemannian_variant}")
            print(f"  Total LoRA pairs found: {len(self.lora_pairs)}")
            print(f"  Shape groups: {len(self._lora_shape_groups)}")
            for key, pairs in self._lora_shape_groups.items():
                print(f"    A({key[0]},{key[1]}) B({key[2]},{key[3]}): {len(pairs)} pairs")
            print(f"  Pairs updated this step: {self._diagnostic_updated_pairs}")
            print(f"  Pairs skipped this step: {self._diagnostic_skipped_pairs}")
            if self._diagnostic_updated_pairs == 0 and len(self.lora_pairs) > 0:
                print("  WARNING: No LoRA pairs were updated! Check shapes.")
            print("=" * 80)
            self._diagnostic_printed = True

        return loss
