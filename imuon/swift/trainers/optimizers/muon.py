# Copyright (c) Alibaba, Inc. and its affiliates.
#
# Vendored from MoonshotAI/Moonlight `examples/toy_train.py` (Muon optimizer),
# which itself adapts the Newton–Schulz orthogonalization from KellerJordan/Muon.
import math
from typing import Dict, Iterable, Optional, Tuple

import torch


def _maybe_int(x) -> int:
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        return int(float(x))
    return int(x)


def _maybe_float(x) -> float:
    if isinstance(x, float):
        return x
    if isinstance(x, str):
        return float(x)
    return float(x)


def _maybe_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        v = x.strip().lower()
        if v in {'1', 'true', 'yes', 'y', 'on'}:
            return True
        if v in {'0', 'false', 'no', 'n', 'off'}:
            return False
    return bool(x)


def _compile_if_available(fn):
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            return torch.compile(fn)
        except Exception:
            return fn
    return fn


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@_compile_if_available
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    steps = _maybe_int(steps)
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Moonlight runs this in bf16 on CUDA. MPS doesn't reliably support bf16 matmul kernels.
    if G.device.type == 'cuda':
        X = G.bfloat16()
    elif G.device.type == 'mps':
        X = G.to(dtype=torch.float16)
    else:
        X = G.to(dtype=torch.float32)

    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
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
        # Riemannian LoRA preconditioner (2402.02347-style):
        # precondition LoRA A/B gradients using the other factor's r×r Gram matrix.
        lora_precond=False,
        lora_precond_eps=1e-6,
        lora_pairs=None,
        # LoRA-specific "Riemannian Muon" update (for low-rank factors).
        # When enabled, LoRA A/B pairs are updated jointly using a low-rank approximation
        # of Ortho(M_t) computed via thin-QR + a small 2r×2r orthogonalization.
        lora_riemannian_muon=False,
        lora_riemannian_ortho_method='ns',  # 'ns' or 'svd' for the 2r×2r core
        lora_riemannian_adjust_lr=True,
        # Riemannian Muon variant selection:
        # 'full' (default): Original - uses full momentum M = MB@A + B@MA, then Ortho(M)
        # 'v2': Variant 2 - dB = Ortho(M_B) @ (AA^T)^{-1}, simpler per-factor orthogonalization
        # 'v3': Variant 3 - dB = Ortho(M_B @ (AA^T)^{-1}), orthogonalize after preconditioning
        # 'v4': Variant 4 - Ortho(M_B), Ortho(M_A), then project onto the horizontal space
        # 'v5': Variant 5 - Ortho(M_t Q_A) R_A^{-T}, R_B^{-1} Ortho(Q_B^T M_t)
        lora_riemannian_variant='full',
        **kwargs,
    ):

        lr = _maybe_float(lr)
        wd = _maybe_float(wd)
        momentum = _maybe_float(momentum)
        nesterov = _maybe_bool(nesterov)
        ns_steps = _maybe_int(ns_steps)
        adamw_eps = _maybe_float(adamw_eps)
        lora_precond = _maybe_bool(lora_precond)
        lora_precond_eps = _maybe_float(lora_precond_eps)
        lora_riemannian_muon = _maybe_bool(lora_riemannian_muon)
        lora_riemannian_adjust_lr = _maybe_bool(lora_riemannian_adjust_lr)
        lora_riemannian_variant = str(lora_riemannian_variant).lower()
        if lora_riemannian_variant not in ('full', 'v2', 'v3', 'v4', 'v5', 'v5_warmup', 'v5_compact', 'v5_compact_warmup'):
            raise ValueError(
                "lora_riemannian_variant must be 'full', 'v2', 'v3', 'v4', 'v5', 'v5_warmup', 'v5_compact', "
                f"or 'v5_compact_warmup', got: {lora_riemannian_variant}"
            )
        beta1, beta2 = adamw_betas
        adamw_betas = (_maybe_float(beta1), _maybe_float(beta2))

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        muon_params = list(muon_params or [])
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params = list(muon_params)
        params.extend(adamw_params)
        super().__init__(params, defaults)
        self.lora_precond = lora_precond
        self.lora_precond_eps = lora_precond_eps
        self.lora_pairs = list(lora_pairs or [])
        self.lora_riemannian_muon = lora_riemannian_muon
        self.lora_riemannian_ortho_method = str(lora_riemannian_ortho_method)
        self.lora_riemannian_adjust_lr = lora_riemannian_adjust_lr
        self.lora_riemannian_variant = lora_riemannian_variant
        self._riemannian_step = 0  # global step counter for v5_warmup
        self._v5_warmup_steps = 50  # number of V1 warmup steps before switching to V5

        # Diagnostic counters for tracking LoRA updates
        self._diagnostic_total_pairs = 0
        self._diagnostic_updated_pairs = 0
        self._diagnostic_skipped_pairs = 0
        self._diagnostic_printed = False
        
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim >= 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A = param_shape[0]
        B = math.prod(param_shape[1:])
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def _inv_gram(self, gram):
        """Invert a tiny r×r Gram matrix efficiently.

        - CUDA/CPU: compute on-device to avoid GPU↔CPU sync/copies.
        - MPS: fall back to CPU (MPS linalg coverage can be incomplete).
        """
        gram = gram.detach()
        if gram.device.type == 'mps':
            gram_cpu = gram.cpu()
            eye_cpu = torch.eye(gram_cpu.shape[0], device=gram_cpu.device, dtype=gram_cpu.dtype)
            try:
                L = torch.linalg.cholesky(gram_cpu)
                inv_cpu = torch.cholesky_solve(eye_cpu, L)
            except Exception:
                inv_cpu = torch.linalg.solve(gram_cpu, eye_cpu)
            return inv_cpu.to(device=gram.device, dtype=gram.dtype)

        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        try:
            L = torch.linalg.cholesky(gram)
            return torch.cholesky_solve(eye, L)
        except Exception:
            return torch.linalg.solve(gram, eye)

    def _detect_packed_lora(self, A, B, r):
        """Detect if A, B are packed (e.g., merged QKV attention).
        
        Packed LoRA tensors have shape:
        - A: (k*r, d_in) where k is the number of packed components
        - B: (k*d_out, r) where k is the number of packed components
        
        Returns:
            (is_packed, k, d_block) where:
            - is_packed: True if tensors are packed
            - k: number of packed components
            - d_block: dimension of each output block (d_out)
        """
        if A.shape[0] % r == 0 and B.shape[1] == r:
            k = A.shape[0] // r
            if k > 1 and B.shape[0] % k == 0:
                d_block = B.shape[0] // k
                return True, k, d_block
        return False, 1, B.shape[0]

    def _ortho_core(self, mat: torch.Tensor, *, ns_steps: int) -> torch.Tensor:
        """Compute the polar/orthogonal factor of a small square matrix."""
        method = (self.lora_riemannian_ortho_method or 'ns').lower()
        if method == 'svd':
            # Use float32 for numerical stability; mat is tiny (2r×2r).
            m32 = mat.to(dtype=torch.float32)
            U, _, Vh = torch.linalg.svd(m32, full_matrices=False)
            return (U @ Vh).to(dtype=mat.dtype)
        # Default: Newton–Schulz (Muon-style)
        return zeropower_via_newtonschulz5(mat, steps=ns_steps).to(dtype=mat.dtype)

    def _ortho_low_rank_xy(self, X: torch.Tensor, Y: torch.Tensor, *, ns_steps: int):
        """Approximate Ortho(XY) without forming XY.

        XY has shape (m×n) but rank <= 2r. We compute thin QR factorizations of X and Y^T:
          X  = Qx Rx
          Y^T = Qy Ry
        Then XY = Qx (Rx Ry^T) Qy^T and Ortho(XY) ≈ Qx Ortho(Rx Ry^T) Qy^T.
        """
        X32 = X.to(dtype=torch.float32)
        Y32 = Y.to(dtype=torch.float32)
        Qx, Rx = torch.linalg.qr(X32, mode='reduced')
        Qy, Ry = torch.linalg.qr(Y32.T, mode='reduced')
        core = Rx @ Ry.T
        Ocore = self._ortho_core(core, ns_steps=ns_steps)
        return Qx.to(dtype=X.dtype), Ocore.to(dtype=X.dtype), Qy.to(dtype=X.dtype)

    def _apply_lora_riemannian_muon(self, precond_grads: Dict[torch.Tensor, torch.Tensor], group) -> set:
        """Update LoRA A/B pairs jointly using a low-rank Riemannian Muon step.

        Returns the set of parameters updated here so the vanilla per-parameter Muon update can skip them.
        """
        if not (self.lora_riemannian_muon and self.lora_pairs):
            return set()
        momentum = group["momentum"]
        ns_steps = group["ns_steps"]
        lr = group["lr"]
        wd = group["wd"]
        updated = set()

        for A, B in self.lora_pairs:
            self._diagnostic_total_pairs += 1
            
            # Expect LoRA convention: A is (r, n), B is (m, r) so W = B @ A.
            if A.grad is None or B.grad is None:
                self._diagnostic_skipped_pairs += 1
                continue
            if not (self.state.get(A, {}).get("use_muon", False) and self.state.get(B, {}).get("use_muon", False)):
                self._diagnostic_skipped_pairs += 1
                continue
            if A.ndim != 2 or B.ndim != 2:
                self._diagnostic_skipped_pairs += 1
                continue

            gA = precond_grads.get(A, A.grad)
            gB = precond_grads.get(B, B.grad)
            if gA is None or gB is None:
                self._diagnostic_skipped_pairs += 1
                continue

            # Apply momentum to LoRA gradients (same as vanilla Muon)
            state_A = self.state[A]
            state_B = self.state[B]
            
            if "momentum_buffer" not in state_A:
                state_A["momentum_buffer"] = torch.zeros_like(gA)
            if "momentum_buffer" not in state_B:
                state_B["momentum_buffer"] = torch.zeros_like(gB)
            
            buf_A = state_A["momentum_buffer"]
            buf_B = state_B["momentum_buffer"]
            
            buf_A.mul_(momentum).add_(gA)
            buf_B.mul_(momentum).add_(gB)
            
            if group["nesterov"]:
                gA = gA.add(buf_A, alpha=momentum)
                gB = gB.add(buf_B, alpha=momentum)
            else:
                gA = buf_A
                gB = buf_B

            # Check for packed LoRA (e.g., QKV merged attention)
            # Standard: A is (r, d_in), B is (d_out, r)
            # Packed: A is (k*r, d_in), B is (k*d_out, r)
            if A.shape[0] != B.shape[1]:
                # Check if this is packed LoRA
                is_packed, k, d_block = self._detect_packed_lora(A, B, B.shape[1])
                if not is_packed:
                    # Not a valid LoRA pair, skip
                    self._diagnostic_skipped_pairs += 1
                    continue
                
                # Handle packed LoRA by processing each sub-pair
                r = B.shape[1]
                d_in = A.shape[1]
                
                # Reshape A: (k*r, d_in) -> (k, r, d_in)
                A_reshaped = A.data.view(k, r, d_in)
                gA_reshaped = gA.view(k, r, d_in)
                
                # Reshape B: (k*d_block, r) -> (k, d_block, r)
                B_reshaped = B.data.view(k, d_block, r)
                gB_reshaped = gB.view(k, d_block, r)
                
                # Process each sub-pair independently
                for i in range(k):
                    A_i = A_reshaped[i]  # (r, d_in)
                    B_i = B_reshaped[i]  # (d_block, r)
                    gA_i = gA_reshaped[i]
                    gB_i = gB_reshaped[i]
                    
                    # Apply update to this sub-pair
                    self._apply_single_lora_pair_update(
                        A_i, B_i, gA_i, gB_i, r, momentum, ns_steps, lr, wd, group
                    )
                
                updated.add(A)
                updated.add(B)
                self._diagnostic_updated_pairs += 1
                continue
            
            # Standard non-packed LoRA pair
            r = A.shape[0]
            self._apply_single_lora_pair_update(
                A.data, B.data, gA, gB, r, momentum, ns_steps, lr, wd, group
            )
            
            updated.add(A)
            updated.add(B)
            self._diagnostic_updated_pairs += 1

        return updated

    def _apply_single_lora_pair_update(self, A, B, gA, gB, r, momentum, ns_steps, lr, wd, group):
        """Apply Riemannian Muon update to a single LoRA A/B pair.
        
        Routes to the appropriate variant based on self.lora_riemannian_variant.
        
        Args:
            A: LoRA A tensor (r, d_in)
            B: LoRA B tensor (d_out, r)
            gA: gradient for A
            gB: gradient for B
            r: rank
            momentum: momentum coefficient
            ns_steps: Newton-Schulz steps
            lr: learning rate
            wd: weight decay
            group: optimizer group
        """
        variant = self.lora_riemannian_variant

        if variant == 'v5_warmup':
            # Run V1 for the first few steps to grow B away from zero, then switch to V5
            if self._riemannian_step <= self._v5_warmup_steps:
                variant = 'full'
            else:
                variant = 'v5'
        elif variant == 'v5_compact_warmup':
            # V1 for the first few steps to grow B away from zero, then switch to V5 compact
            if self._riemannian_step <= self._v5_warmup_steps:
                variant = 'full'
            else:
                variant = 'v5_compact'

        if variant == 'v2':
            self._apply_single_lora_pair_update_v2(A, B, gA, gB, r, momentum, ns_steps, lr, wd, group)
        elif variant == 'v3':
            self._apply_single_lora_pair_update_v3(A, B, gA, gB, r, momentum, ns_steps, lr, wd, group)
        elif variant == 'v4':
            self._apply_single_lora_pair_update_v4(A, B, gA, gB, r, momentum, ns_steps, lr, wd, group)
        elif variant == 'v5':
            self._apply_single_lora_pair_update_v5(A, B, gA, gB, r, momentum, ns_steps, lr, wd, group)
        elif variant == 'v5_compact':
            self._apply_single_lora_pair_update_v5_compact(A, B, gA, gB, r, momentum, ns_steps, lr, wd, group)
        else:
            # Default: 'full' variant - original implementation
            self._apply_single_lora_pair_update_full(A, B, gA, gB, r, momentum, ns_steps, lr, wd, group)

    def _apply_single_lora_pair_update_full(self, A, B, gA, gB, r, momentum, ns_steps, lr, wd, group):
        """Original Riemannian Muon: Ortho(M) where M = MB@A + B@MA (full momentum).
        
        Update directions:
            dB = Ortho(M) @ A^T @ (A A^T)^{-1}
            dA = (B^T B)^{-1} @ B^T @ Ortho(M)
        
        Note: gA and gB are already momentum-accumulated gradients from the caller.
        """
        # gA and gB are momentum-accumulated (MA, MB in the math)
        MA = gA
        MB = gB

        # Low-rank approximation: M ≈ MB @ A + B @ MA = [MB, B] @ [A; MA]
        X = torch.cat([MB, B], dim=1)          # (m, 2r)
        Y = torch.cat([A, MA], dim=0)          # (2r, n)
        Qx, Ocore, Qy = self._ortho_low_rank_xy(X, Y, ns_steps=ns_steps)

        # Gram inverses (with damping) for stability.
        eye = torch.eye(r, device=A.device, dtype=torch.float32)
        eps = self.lora_precond_eps
        inv_AAT = self._inv_gram((A.to(torch.float32) @ A.to(torch.float32).T) + self.lora_precond_eps * eye)
        inv_BTB = self._inv_gram((B.to(torch.float32).T @ B.to(torch.float32)) + self.lora_precond_eps * eye)

        # dB = Ortho(M) @ A^T @ (A A^T)^-1
        A_proj = Qy.T.to(torch.float32) @ A.to(torch.float32).T         # (2r, r)
        tmp = Ocore.to(torch.float32) @ A_proj                           # (2r, r)
        OB = Qx.to(torch.float32) @ tmp                                  # (m, r)
        dB = OB @ inv_AAT.to(torch.float32)                              # (m, r)

        # dA = (B^T B)^-1 @ B^T @ Ortho(M)
        B_proj = B.to(torch.float32).T @ Qx.to(torch.float32)            # (r, 2r)
        tmp2 = B_proj @ Ocore.to(torch.float32)                          # (r, 2r)
        OA = tmp2 @ Qy.to(torch.float32).T                               # (r, n)
        dA = inv_BTB.to(torch.float32) @ OA                              # (r, n)

        # Apply weight decay and update
        self._apply_lora_update(A, B, dA, dB, lr, wd)

    def _apply_single_lora_pair_update_v2(self, A, B, gA, gB, r, momentum, ns_steps, lr, wd, group):
        """Variant 2: Per-factor orthogonalization before preconditioning.
        
        Update directions:
            dB = Ortho(M_B) @ (A A^T)^{-1}
            dA = (B^T B)^{-1} @ Ortho(M_A)
        
        Simpler than full variant - orthogonalizes each factor's momentum independently.
        Note: gA and gB are already momentum-accumulated gradients from the caller.
        Note: Ortho() method is determined by lora_riemannian_ortho_method flag.
        """
        MA = gA
        MB = gB

        # Gram inverses (with damping) for stability.
        eye = torch.eye(r, device=A.device, dtype=torch.float32)
        eps = self.lora_precond_eps
        inv_AAT = self._inv_gram((A.to(torch.float32) @ A.to(torch.float32).T) + self.lora_precond_eps * eye)
        inv_BTB = self._inv_gram((B.to(torch.float32).T @ B.to(torch.float32)) + self.lora_precond_eps * eye)

        # Orthogonalize momentum terms - method determined by flag
        method = (self.lora_riemannian_ortho_method or 'ns').lower()
        
        if method == 'svd':
            # Use SVD for exact orthogonalization
            MB32 = MB.to(torch.float32)
            U_MB, _, Vh_MB = torch.linalg.svd(MB32, full_matrices=False)
            ortho_MB = (U_MB @ Vh_MB).to(torch.float32)
            
            MA32 = MA.to(torch.float32)
            U_MA, _, Vh_MA = torch.linalg.svd(MA32, full_matrices=False)
            ortho_MA = (U_MA @ Vh_MA).to(torch.float32)
        else:
            # Default: Newton-Schulz
            ortho_MB = zeropower_via_newtonschulz5(MB.to(torch.float32), steps=ns_steps).to(torch.float32)
            ortho_MA = zeropower_via_newtonschulz5(MA.to(torch.float32), steps=ns_steps).to(torch.float32)

        # dB = Ortho(M_B) @ (A A^T)^{-1}
        dB = ortho_MB @ inv_AAT                                          # (m, r)

        # dA = (B^T B)^{-1} @ Ortho(M_A)
        dA = inv_BTB @ ortho_MA                                          # (r, n)

        # Apply weight decay and update
        self._apply_lora_update(A, B, dA, dB, lr, wd)

    def _apply_single_lora_pair_update_v3(self, A, B, gA, gB, r, momentum, ns_steps, lr, wd, group):
        """Variant 3: Orthogonalize after preconditioning.
        
        Update directions:
            dB = Ortho(M_B @ (A A^T)^{-1})
            dA = Ortho((B^T B)^{-1} @ M_A)
        
        Preconditions momentum first, then orthogonalizes the result.
        Note: gA and gB are already momentum-accumulated gradients from the caller.
        Note: Ortho() method is determined by lora_riemannian_ortho_method flag.
        """
        MA = gA
        MB = gB

        # Gram inverses (with damping) for stability.
        eye = torch.eye(r, device=A.device, dtype=torch.float32)
        inv_AAT = self._inv_gram((A.to(torch.float32) @ A.to(torch.float32).T) + self.lora_precond_eps * eye)
        inv_BTB = self._inv_gram((B.to(torch.float32).T @ B.to(torch.float32)) + self.lora_precond_eps * eye)

        # Precondition first
        precond_MB = MB.to(torch.float32) @ inv_AAT
        precond_MA = inv_BTB @ MA.to(torch.float32)
        
        # Then orthogonalize - method determined by flag
        method = (self.lora_riemannian_ortho_method or 'ns').lower()
        
        if method == 'svd':
            # Use SVD for exact orthogonalization
            U_MB, _, Vh_MB = torch.linalg.svd(precond_MB, full_matrices=False)
            dB = (U_MB @ Vh_MB).to(torch.float32)      # (m, r)
            
            U_MA, _, Vh_MA = torch.linalg.svd(precond_MA, full_matrices=False)
            dA = (U_MA @ Vh_MA).to(torch.float32)      # (r, n)
        else:
            # Default: Newton-Schulz
            dB = zeropower_via_newtonschulz5(precond_MB, steps=ns_steps).to(torch.float32)      # (m, r)
            dA = zeropower_via_newtonschulz5(precond_MA, steps=ns_steps).to(torch.float32)      # (r, n)

        # Apply weight decay and update
        self._apply_lora_update(A, B, dA, dB, lr, wd)

    def _apply_single_lora_pair_update_v4(self, A, B, gA, gB, r, momentum, ns_steps, lr, wd, group):
        """Variant 4: Ortho(M_A), Ortho(M_B), then project onto horizontal space.

        Update directions:
            B~ = Ortho(M_B)
            A~ = Ortho(M_A)  
            Lambda = 0.5 * (A~ A^T (A A^T)^{-1} - (B^T B)^{-1} B^T B~)
            dB = B~ + B Lambda
            dA = A~ - Lambda A

        Note: gA and gB are already momentum-accumulated gradients from the caller.
        Note: Ortho() method is determined by lora_riemannian_ortho_method flag:
              'svd' uses exact SVD orthogonalization, 'ns' uses Newton-Schulz (vanilla Muon).
        """
        MA = gA
        MB = gB

        # Gram inverses (with damping) for stability.
        eye = torch.eye(r, device=A.device, dtype=torch.float32)
        A32 = A.to(torch.float32)
        B32 = B.to(torch.float32)
        inv_AAT = self._inv_gram((A32 @ A32.T) + self.lora_precond_eps * eye)
        inv_BTB = self._inv_gram((B32.T @ B32) + self.lora_precond_eps * eye)

        # Orthonormalize momentum terms - method determined by lora_riemannian_ortho_method flag
        method = (self.lora_riemannian_ortho_method or 'ns').lower()
        
        if method == 'svd':
            # Use SVD for exact orthogonalization
            MB32 = MB.to(torch.float32)
            U_MB, _, Vh_MB = torch.linalg.svd(MB32, full_matrices=False)
            ortho_MB = (U_MB @ Vh_MB).to(torch.float32)
            
            MA32 = MA.to(torch.float32)
            U_MA, _, Vh_MA = torch.linalg.svd(MA32, full_matrices=False)
            ortho_MA = (U_MA @ Vh_MA).to(torch.float32)
        else:
            # Default: Newton-Schulz (vanilla Muon behavior)
            # NS handles transposition internally for tall matrices
            ortho_MB = zeropower_via_newtonschulz5(MB.to(torch.float32), steps=ns_steps).to(torch.float32)
            ortho_MA = zeropower_via_newtonschulz5(MA.to(torch.float32), steps=ns_steps).to(torch.float32)

        # Horizontal-space projection.
        left_term = (ortho_MA @ A32.T) @ inv_AAT
        right_term = inv_BTB @ (B32.T @ ortho_MB)
        lam = 0.5 * (left_term - right_term)

        dB = ortho_MB + (B32 @ lam)
        dA = ortho_MA - (lam @ A32)

        # Apply weight decay and update
        self._apply_lora_update(A, B, dA, dB, lr, wd)

    def _apply_single_lora_pair_update_v5(self, A, B, gA, gB, r, momentum, ns_steps, lr, wd, group):
        """Variant 5: Project-then-orthogonalize (original form, numerically stable).

        From Section 3.5 of the paper:
            dB = Ortho(M_t P_A) A^T (AA^T)^{-1}
            dA = (B^T B)^{-1} B^T Ortho(P_B M_t)

        where P_A = A^T(AA^T)^{-1}A projects onto row(A),
              P_B = B(B^TB)^{-1}B^T projects onto col(B).

        Uses low-rank M_t = XY for efficient computation, same pattern as V1.
        Numerically stable: when B=0, B^T kills the large (B^TB)^{-1}.

        Note: gA and gB are already momentum-accumulated gradients from the caller.
        """
        MA = gA
        MB = gB

        A32 = A.to(torch.float32)
        B32 = B.to(torch.float32)

        # Low-rank approximation: M_t ≈ XY
        X = torch.cat([MB.to(torch.float32), B32], dim=1)   # (m, 2r)
        Y = torch.cat([A32, MA.to(torch.float32)], dim=0)   # (2r, n)

        # Gram inverses (with damping), same as V1
        eye = torch.eye(r, device=A.device, dtype=torch.float32)
        inv_AAT = self._inv_gram(A32 @ A32.T + self.lora_precond_eps * eye)
        inv_BTB = self._inv_gram(B32.T @ B32 + self.lora_precond_eps * eye)

        # --- dB = Ortho(M_t P_A) A^T (AA^T)^{-1} ---
        # P_A = A^T(AA^T)^{-1}A, so M_t P_A = X @ Y_1 where Y_1 = Y A^T inv_AAT A
        Y_1 = (Y @ A32.T @ inv_AAT) @ A32                    # (2r, n)
        Qx_B, Ocore_B, Qy_B = self._ortho_low_rank_xy(X, Y_1, ns_steps=ns_steps)

        A_proj = Qy_B.T.to(torch.float32) @ A32.T            # (2r, r)
        tmp = Ocore_B.to(torch.float32) @ A_proj              # (2r, r)
        OB = Qx_B.to(torch.float32) @ tmp                    # (m, r)
        dB = OB @ inv_AAT                                     # (m, r)

        # --- dA = (B^TB)^{-1} B^T Ortho(P_B M_t) ---
        # P_B = B(B^TB)^{-1}B^T, so P_B M_t = X_2 @ Y where X_2 = B inv_BTB B^T X
        X_2 = B32 @ (inv_BTB @ (B32.T @ X))                  # (m, 2r)
        Qx_A, Ocore_A, Qy_A = self._ortho_low_rank_xy(X_2, Y, ns_steps=ns_steps)

        B_proj = B32.T @ Qx_A.to(torch.float32)              # (r, 2r)
        tmp2 = B_proj @ Ocore_A.to(torch.float32)            # (r, 2r)
        OA = tmp2 @ Qy_A.to(torch.float32).T                 # (r, n)
        dA = inv_BTB @ OA                                     # (r, n)

        # Apply weight decay and update
        self._apply_lora_update(A, B, dA, dB, lr, wd)

    def _apply_single_lora_pair_update_v5_compact(self, A, B, gA, gB, r, momentum, ns_steps, lr, wd, group):
        """Variant v5_compact: compact-QR form of V5 (Section 3.3 of the paper draft).

        Algorithm (exact match to scripts/test_lora_stability_exact.py compute_v5_compact):
            Q_A, R_A = QR(A^T)                                    # Q_A: (n,r), R_A: (r,r)
            Q_B, R_B = QR(B)                                      # Q_B: (m,r), R_B: (r,r)
            dB = Ortho(M_B R_A^T + B (M_A Q_A)) @ (R_A^T R_A + eps I)^{-1} R_A^T
            dA = (R_B^T R_B + eps I)^{-1} R_B^T @ Ortho(Q_B^T M_B A + R_B M_A)

        Orthogonalization targets are r×? / ?×r full-rank matrices, so Newton-Schulz
        operates at the intrinsic rank r of the problem without the 2r-column null
        space present in the projection form.

        Note: gA and gB are already momentum-accumulated gradients from the caller.
        """
        MA = gA
        MB = gB

        A32 = A.to(torch.float32)
        B32 = B.to(torch.float32)
        MA32 = MA.to(torch.float32)
        MB32 = MB.to(torch.float32)

        # Thin QR on rank-r factors.
        Q_A, R_A = torch.linalg.qr(A32.T, mode='reduced')   # Q_A: (n, r), R_A: (r, r)
        Q_B, R_B = torch.linalg.qr(B32,   mode='reduced')   # Q_B: (m, r), R_B: (r, r)

        eye = torch.eye(r, device=A.device, dtype=torch.float32)
        inv_RBT_RB = self._inv_gram(R_B.T @ R_B + self.lora_precond_eps * eye)
        inv_RAT_RA = self._inv_gram(R_A.T @ R_A + self.lora_precond_eps * eye)

        # --- dB = Ortho(M_t Q_A) @ (R_A^T R_A + eps I)^{-1} R_A^T ---
        # M_t Q_A = M_B (A Q_A) + B (M_A Q_A) = M_B R_A^T + B (M_A Q_A)   since A Q_A = R_A^T
        Mt_QA = MB32 @ R_A.T + B32 @ (MA32 @ Q_A)                         # (m, r)
        ortho_Mt_QA = self._ortho_core(Mt_QA, ns_steps=ns_steps).to(torch.float32)
        dB = ortho_Mt_QA @ (inv_RAT_RA @ R_A).T                            # (m, r)

        # --- dA = (R_B^T R_B + eps I)^{-1} R_B^T @ Ortho(Q_B^T M_t) ---
        # Q_B^T M_t = (Q_B^T M_B) A + (Q_B^T B) M_A = (Q_B^T M_B) A + R_B M_A
        QB_Mt = Q_B.T @ MB32 @ A32 + R_B @ MA32                            # (r, n)
        ortho_QB_Mt = self._ortho_core(QB_Mt, ns_steps=ns_steps).to(torch.float32)
        dA = inv_RBT_RB @ (R_B.T @ ortho_QB_Mt)                            # (r, n)

        # Apply weight decay and update
        self._apply_lora_update(A, B, dA, dB, lr, wd)

    def _apply_lora_update(self, A, B, dA, dB, lr, wd):
        """Apply the computed update directions to LoRA A and B tensors.
        
        Handles weight decay and optional LR adjustment.
        """
        # Apply weight decay (same convention as vanilla Muon).
        if wd:
            B.mul_(1 - lr * wd)
            A.mul_(1 - lr * wd)

        # Apply update with the same Muon LR adjustment (optional).
        B_shape = (B.shape[0], B.shape[1])
        A_shape = (A.shape[0], A.shape[1])
        
        if self.lora_riemannian_adjust_lr:
            lrB = self.adjust_lr_for_muon(lr, B_shape)
            lrA = self.adjust_lr_for_muon(lr, A_shape)
        else:
            lrB = lrA = lr
        B.add_(dB.to(dtype=B.dtype), alpha=-lrB)
        A.add_(dA.to(dtype=A.dtype), alpha=-lrA)

    def _riemannian_precond_pair(self, A, B, gA, gB):
        # Supports two common conventions by detecting which axis is the shared rank r.
        # Case 1: A is (r, d_in), B is (d_out, r).
        if A.ndim == 2 and B.ndim == 2 and A.shape[0] == B.shape[1]:
            r = A.shape[0]
            A32 = A.detach().to(dtype=torch.float32)
            B32 = B.detach().to(dtype=torch.float32)
            gA32 = gA.to(dtype=torch.float32)
            gB32 = gB.to(dtype=torch.float32)

            gram_B = B32.T @ B32
            gram_A = A32 @ A32.T
            eye = torch.eye(r, device=gram_A.device, dtype=gram_A.dtype)
            gram_B = gram_B + self.lora_precond_eps * eye
            gram_A = gram_A + self.lora_precond_eps * eye

            inv_gram_B = self._inv_gram(gram_B)
            inv_gram_A = self._inv_gram(gram_A)
            gA_pre = inv_gram_B @ gA32
            gB_pre = gB32 @ inv_gram_A
            return gA_pre.to(dtype=gA.dtype), gB_pre.to(dtype=gB.dtype)

        # Case 2: A is (d_in, r), B is (r, d_out).
        if A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[0]:
            r = A.shape[1]
            A32 = A.detach().to(dtype=torch.float32)
            B32 = B.detach().to(dtype=torch.float32)
            gA32 = gA.to(dtype=torch.float32)
            gB32 = gB.to(dtype=torch.float32)

            gram_B = B32 @ B32.T
            gram_A = A32.T @ A32
            eye = torch.eye(r, device=gram_A.device, dtype=gram_A.dtype)
            gram_B = gram_B + self.lora_precond_eps * eye
            gram_A = gram_A + self.lora_precond_eps * eye

            inv_gram_B = self._inv_gram(gram_B)
            inv_gram_A = self._inv_gram(gram_A)
            gA_pre = gA32 @ inv_gram_B
            gB_pre = inv_gram_A @ gB32
            return gA_pre.to(dtype=gA.dtype), gB_pre.to(dtype=gB.dtype)

        return gA, gB

    def step(self, closure: Optional[callable] = None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Reset diagnostic counters at start of step
        if not self._diagnostic_printed:
            self._diagnostic_total_pairs = 0
            self._diagnostic_updated_pairs = 0
            self._diagnostic_skipped_pairs = 0

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            precond_grads = {}
            if self.lora_precond and self.lora_pairs and not self.lora_riemannian_muon:
                for A, B in self.lora_pairs:
                    if A.grad is None or B.grad is None:
                        continue
                    if not (self.state.get(A, {}).get("use_muon", False) and self.state.get(B, {}).get("use_muon", False)):
                        continue
                    if A.ndim != 2 or B.ndim != 2:
                        continue
                    gA_pre, gB_pre = self._riemannian_precond_pair(A.data, B.data, A.grad, B.grad)
                    precond_grads[A] = gA_pre
                    precond_grads[B] = gB_pre

            updated_lora_params = self._apply_lora_riemannian_muon(precond_grads, group)
            if self.lora_riemannian_muon and self.lora_pairs:
                self._riemannian_step += 1

            # generate weight updates
            for p in params:
                # sanity check
                if p in updated_lora_params:
                    continue
                g = precond_grads.get(p, p.grad)
                if g is None:
                    continue
                orig_shape = p.data.shape
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u.reshape(orig_shape).to(dtype=p.dtype), alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        # Print diagnostic information on first step
        if not self._diagnostic_printed and self.lora_riemannian_muon and self.lora_pairs:
            print("=" * 80)
            print("Muon Optimizer - LoRA Riemannian Update Diagnostics:")
            print(f"  Variant: {self.lora_riemannian_variant}")
            print(f"  Total LoRA pairs found: {len(self.lora_pairs)}")
            print(f"  Pairs updated this step: {self._diagnostic_updated_pairs}")
            print(f"  Pairs skipped this step: {self._diagnostic_skipped_pairs}")
            if self._diagnostic_updated_pairs == 0 and len(self.lora_pairs) > 0:
                print("  ⚠️  WARNING: No LoRA pairs were updated! Check shapes.")
            print("=" * 80)
            self._diagnostic_printed = True

        return loss
