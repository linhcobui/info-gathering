"""CLS-token error of the DINO-WM. The WM predicts patch tokens only, so a predicted CLS is obtained by decoding the
predicted patches to pixels with the WM's own token decoder and re-encoding them with the same frozen DINOv2:
    cls_hat = DINOv2_CLS(decoder(ê_{t+1}))
To cancel the decoder's artefacts, the reference is the real tokens through the same path,
    cls_ref = DINOv2_CLS(decoder(e_{t+1}))
and err_cls = ||cls_hat - cls_ref|| (L2 over the 384 numbers), err_cls_cos = 1 - cosine. copy_cls is the
copy-last baseline in the same space, ||DINOv2_CLS(decoder(e_t)) - cls_ref||.
"""
import sys

import torch

from .config import ROOT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLS_METRICS = ("err_cls", "err_cls_cos", "copy_cls")


class ClsScorer:
    def __init__(self, wm):
        sys.path.insert(0, str(ROOT / "scripts"))
        from cache_dinov2_ogbar import load_dinov2
        if getattr(wm.model, "decoder", None) is None:
            raise ValueError("the WM checkpoint has no token decoder")
        self.wm, self.dino = wm, load_dinov2(wm.device)
        self.mean = torch.tensor(IMAGENET_MEAN, device=wm.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=wm.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def cls(self, tokens):
        """(b, 256, 384) patch tokens -> (b, 384) CLS of their decoded image (decoder output in [-1, 1])."""
        with torch.autocast(self.wm.device.type, dtype=torch.bfloat16, enabled=self.wm.bf16):
            img = self.wm.model.decoder(tokens.float())
        x = (img.float().clamp(-1, 1) + 1) / 2
        return self.dino.forward_features((x - self.mean) / self.std)["x_norm_clstoken"].float()

    @torch.no_grad()
    def metrics(self, pred, real, prev):
        b = len(pred)
        c = self.cls(torch.cat([pred, real.float(), prev.float()]))
        ch, cr, cp = c[:b], c[b:2 * b], c[2 * b:]
        return dict(err_cls=(ch - cr).norm(dim=-1).cpu().numpy(),
                    err_cls_cos=(1 - torch.nn.functional.cosine_similarity(ch, cr, dim=-1)).cpu().numpy(),
                    copy_cls=(cp - cr).norm(dim=-1).cpu().numpy())
