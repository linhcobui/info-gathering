"""Batched DTW on GPU, anti-diagonal DP. Verified against a naive reference in dtw_verify.py."""
import torch

def dtw_batch(A, B, na, nb, inf=1e9):
    """A,B: (P,T,D) zero-padded.  na,nb: (P,) true lengths.  -> (P,) path-length-normalised DTW."""
    P, T, _ = A.shape
    dev = A.device
    C = torch.cdist(A, B)                                        # (P,T,T)
    ar = torch.arange(T, device=dev)
    C = C.masked_fill((ar[None, :, None] >= na[:, None, None]) |
                      (ar[None, None, :] >= nb[:, None, None]), inf)
    R = torch.full((P, T + 1, T + 1), inf, device=dev, dtype=torch.float32)
    R[:, 0, 0] = 0.0
    for d in range(2, 2 * T + 1):                                # anti-diagonals: 2T steps, not T^2
        i = torch.arange(max(1, d - T), min(T, d - 1) + 1, device=dev)
        if not len(i):
            continue
        j = d - i
        R[:, i, j] = C[:, i - 1, j - 1] + torch.minimum(
            torch.minimum(R[:, i - 1, j], R[:, i, j - 1]), R[:, i - 1, j - 1])
    out = R[torch.arange(P, device=dev), na, nb]
    return out / (na + nb).float()
