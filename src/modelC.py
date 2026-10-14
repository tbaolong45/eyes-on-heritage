"""Model C: siamese DINOv2 with LoRA (rank 16, qkv/proj of the last 8 blocks) + Model-B head, image-space augs."""
import math, io
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from common import *
from pairs import label_pairs, valid_dates, point_weights, aggregate, infer_pairs
from embed import BACKBONES, dihedral, inv_dihedral
from modelB import Head, loss_fn

CROP = 128      # 64 m (config B2)
SIDE = 224      # 16x16 tokens of 14 px


def _std(img, valid):
    f = img.astype(np.float32) / 255.0
    if valid.sum() > 50:
        mu = f[valid].mean(0); sd = f[valid].std(0) + 1e-3; f = (f - mu) / sd
    f[~valid] = 0
    return f


def augment_photometric(rgb, rng):
    x = rgb.astype(np.float32) / 255.0
    x = x * (1 + rng.uniform(-0.3, 0.3))                                      # brightness
    m = x.mean(); x = (x - m) * (1 + rng.uniform(-0.3, 0.3)) + m              # contrast
    g = x.mean(-1, keepdims=True); x = g + (x - g) * (1 + rng.uniform(-0.3, 0.3))  # saturation
    x = np.clip(x, 0, 1)
    hsv = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(rng.uniform(-0.05, 0.05) * 180)) % 180  # hue
    x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    x = x ** rng.uniform(0.7, 1.4)                                            # gamma
    x = x * rng.uniform(0.85, 1.15, size=3)[None, None]                      # per-channel sensor gains
    if rng.rand() < 0.3:
        a = rng.uniform(0, 0.4); x = (1 - a) * x + a * rng.uniform(0.75, 0.95)  # synthetic haze
    x = np.clip(x, 0, 1)
    if rng.rand() < 0.3:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.2))
    out = (x * 255).astype(np.uint8)
    if rng.rand() < 0.5:
        ok, enc = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.randint(40, 96))])
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
    return out


def crop_pair(tb, ta, rng, train, side=SIDE, centre_mask=0.0, scale=1.0):
    """tb/ta: (384,384,4). Returns standardised (3,SIDE,SIDE) tensors and validity maps."""
    H = tb.shape[0]; c = H // 2
    s = rng.uniform(0.85, 1.15) if train else scale
    cs = int(round(CROP * s))
    outs = []
    for k, t in enumerate((tb, ta)):
        dx, dy = (rng.randint(-10, 11, 2) if (train and k == 1) else (0, 0))
        y0, x0 = c - cs // 2 + dy, c - cs // 2 + dx
        cr = t[y0:y0 + cs, x0:x0 + cs]
        rgb = cr[..., :3]; v = cr[..., 3] > 0
        if train:
            rgb = augment_photometric(rgb, rng)
            if rng.rand() < 0.5:   # periphery cutout (never the central 16 m = 32 px)
                ch = rng.randint(10, cs // 3); cy, cx = rng.randint(0, cs - ch, 2)
                if not (abs(cy + ch / 2 - cs / 2) < 16 + ch / 2 and abs(cx + ch / 2 - cs / 2) < 16 + ch / 2):
                    v = v.copy(); v[cy:cy + ch, cx:cx + ch] = False
            if centre_mask > 0 and rng.rand() < centre_mask:
                # point-centred square: keep only a random 16-48 m half-width square around the point
                r = rng.randint(32, 97); o = cs // 2
                v = v.copy(); keep = np.zeros_like(v); keep[max(0, o - r):o + r, max(0, o - r):o + r] = True; v &= keep
        f = _std(rgb, v)
        f = cv2.resize(f, (side, side), interpolation=cv2.INTER_CUBIC)
        vv = cv2.resize(v.astype(np.float32), (side, side), interpolation=cv2.INTER_AREA)
        outs += [torch.from_numpy(f.transpose(2, 0, 1).copy()), torch.from_numpy(vv)]
    return outs


class PairImgDS(torch.utils.data.Dataset):
    def __init__(self, items, tiles, train, seed=0, side=SIDE, centre_mask=0.0):
        self.items, self.tiles, self.train, self.seed = items, tiles, train, seed
        self.side, self.centre_mask = side, centre_mask

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        pid, i, j, lab, w = self.items[n]
        rng = np.random.RandomState((self.seed * 100003 + n * 7919 + np.random.randint(1 << 30)) % (2 ** 31))
        T = self.tiles[pid]
        xb, vb, xa, va = crop_pair(T[i], T[j], rng, self.train, self.side, self.centre_mask)
        k = rng.randint(8) if self.train else 0
        return xb, vb, xa, va, k, lab, np.float32(w)


class Siamese(nn.Module):
    def __init__(self, bb="dinov2-l", rank=16, last_k=8, drop=0.1):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        self.timm = "timm" in BACKBONES[bb]
        if self.timm:
            import timm
            m = timm.create_model(BACKBONES[bb]["timm"], pretrained=True, num_classes=0)
            L = len(m.blocks); self.mid = int(round(2 * L / 3)); self.D = m.embed_dim; self.L = L
            tm = [f"blocks.{i}.attn.{n}" for i in range(L - last_k, L) for n in ("qkv", "proj")]
            self.bb = get_peft_model(m, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=tm))
            self.head = Head(2 * self.D, 16, drop=drop, sigma_tokens=4.0)
            return
        from transformers import AutoModel
        m = AutoModel.from_pretrained(BACKBONES[bb]["hf"], dtype=torch.float32)
        L = m.config.num_hidden_layers; self.mid = int(round(2 * L / 3)); self.D = m.config.hidden_size
        tm = []
        for i in range(L - last_k, L):
            p = f"encoder.layer.{i}.attention"
            tm += [f"{p}.q_proj", f"{p}.k_proj", f"{p}.v_proj", f"{p}.o_proj"]
        self.bb = get_peft_model(m, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=tm))
        self.head = Head(2 * self.D, 16, drop=drop, sigma_tokens=4.0)

    def tokens(self, x, k):
        x = torch.stack([dihedral(x[n], int(k[n])) for n in range(len(x))]) if k is not None else x
        if self.timm:
            gm, gl = self.bb.base_model.model.forward_intermediates(x, indices=[self.mid - 1, self.L - 1], norm=True,
                                                                    output_fmt="NCHW", intermediates_only=True)
            t = torch.cat([gl, gm], 1)
            if k is not None:
                t = torch.stack([inv_dihedral(t[n], int(k[n])) for n in range(len(t))])
            return t.flatten(2).transpose(1, 2)
        out = self.bb(pixel_values=x, output_hidden_states=True)
        nreg = 0
        def g(h):
            t = h[:, 1 + nreg:]; B, N, D = t.shape
            return t.transpose(1, 2).reshape(B, D, 16, 16)
        t = torch.cat([g(out.last_hidden_state), g(out.hidden_states[self.mid])], 1)
        if k is not None:
            t = torch.stack([inv_dihedral(t[n], int(k[n])) for n in range(len(t))])
        return t.flatten(2).transpose(1, 2)

    def forward(self, xb, xa, m, k=None):
        return self.head(self.tokens(xb, k), self.tokens(xa, k), m)


class ModelC:
    @property
    def side(self):
        return 16 * BACKBONES[self.cfg.get("bb", "dinov2-l")]["patch"]

    def __init__(self, pts, cfg, device="cuda"):
        self.pts, self.cfg, self.dev = pts.reset_index(drop=True), cfg, device
        self.meta, self.tiles = {}, {}
        for pid, site in zip(self.pts.point_id, self.pts.site):
            z = np.load(tile_path(site, pid), allow_pickle=True)
            self.meta[pid] = (np.array([int(d[:4]) for d in z["dates"]]), z["bad"].astype(bool), list(z["quality"]))
            self.tiles[pid] = z["tiles"]

    def make_net(self):
        cfg = self.cfg
        return Siamese(bb=cfg.get("bb", "dinov2-l"), rank=cfg.get("rank", 16), last_k=cfg.get("last_k", 8))

    def items(self, idx, wp):
        out = []
        for k in idx:
            r = self.pts.iloc[k]; years, bad, q = self.meta[r.point_id]
            vi, _ = valid_dates(bad, q)
            lp = label_pairs(r.y, r.year, years, list(vi), reverse=self.cfg.get("reverse", True))
            tot = sum(l[3] * (2.0 if l[4] else 1.0) for l in lp) or 1
            out += [(r.point_id, int(i), int(j), int(lab), float(wp[k] * pw * (2.0 if core else 1.0) / tot), k)
                    for i, j, lab, pw, core in lp]
        return out

    def _epoch_items(self, by_pt, pt_w, mw):
        ep = []
        for k, v in by_pt.items():
            pw = np.array([i[4] for i in v]); c = v[np.random.choice(len(v), p=pw / pw.sum())]
            ep.append((c[0], c[1], c[2], c[3], pt_w[k] / mw))
        return ep

    def _train(self, it_tr, it_va, epochs, seed, stop_at=None):
        cfg = self.cfg; torch.manual_seed(seed); np.random.seed(seed)
        net = self.make_net().to(self.dev)
        by_pt = {}
        for it in it_tr: by_pt.setdefault(it[5], []).append(it)
        pt_w = {k: sum(i[4] for i in v) for k, v in by_pt.items()}; mw = np.mean(list(pt_w.values()))
        bs = cfg.get("bs", 32); spe = math.ceil(len(by_pt) / bs); total = epochs * spe
        lora = [p for n, p in net.bb.named_parameters() if p.requires_grad]
        lora_ids = {id(p) for p in lora}
        opt = torch.optim.AdamW([{"params": lora, "lr": cfg.get("lr_lora", 1e-4)},
                                 {"params": [p for p in net.parameters() if id(p) not in lora_ids], "lr": cfg.get("lr", 3e-4)}],
                                weight_decay=0.05)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / (0.05 * total)) *
                                                  0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
        cnt = np.bincount([i[3] for i in it_tr], minlength=4) + 1
        prior = torch.tensor(cnt / cnt.sum(), dtype=torch.float32, device=self.dev)
        best = (1e9, None, 0)
        for ep in range(stop_at or epochs):
            net.train()
            dl = torch.utils.data.DataLoader(PairImgDS(self._epoch_items(by_pt, pt_w, mw), self.tiles, True, seed + ep, self.side, cfg.get("centre_mask", 0.0)),
                                             batch_size=bs, shuffle=True, num_workers=cfg.get("workers", 5))
            for xb, vb, xa, va, k, y, w in dl:
                xb, xa = xb.to(self.dev), xa.to(self.dev)
                m = F.adaptive_avg_pool2d(torch.minimum(vb, va)[:, None], 16).flatten(1).to(self.dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = net(xb, xa, m, k)
                loss = loss_fn(lg.float(), y.to(self.dev), w.to(self.dev).float(), cfg, prior)
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            if it_va is not None and ((ep + 1) % 5 == 0 or ep == epochs - 1):
                vl = self._val_loss(net, it_va)
                if vl < best[0]:
                    best = (vl, {n: p.detach().clone() for n, p in net.state_dict().items()
                                 if getattr(net, "full_ckpt", False) or "lora" in n or n.startswith("head")}, ep + 1)
        if best[1] is not None:
            net.load_state_dict(best[1], strict=False)
        return net, best[2]

    @torch.no_grad()
    def _val_loss(self, net, it_va):
        net.eval()
        dl = torch.utils.data.DataLoader(PairImgDS([v[:5] for v in it_va], self.tiles, False, 0, self.side), batch_size=64,
                                         num_workers=self.cfg.get("workers", 5))
        L = W = 0.0
        for xb, vb, xa, va, k, y, w in dl:
            m = F.adaptive_avg_pool2d(torch.minimum(vb, va)[:, None], 16).flatten(1).to(self.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = net(xb.to(self.dev), xa.to(self.dev), m).float()
            w = w.to(self.dev).float()
            L += float((F.cross_entropy(lg, y.to(self.dev), reduction="none") * w).sum()); W += float(w.sum())
        return L / max(W, 1e-9)

    def fit(self, tr, va, seed=0, all_tr=None):
        wp = point_weights(self.pts, self.cfg.get("label_w", True), self.cfg.get("balance", True))
        self.net, be = self._train(self.items(tr, wp), self.items(va, wp), self.cfg.get("epochs", 40), seed)
        return dict(best_ep=be)

    def fit_final(self, tr, seed, epochs):
        wp = point_weights(self.pts, self.cfg.get("label_w", True), self.cfg.get("balance", True))
        net, _ = self._train(self.items(tr, wp), None, self.cfg.get("epochs", 40), seed, stop_at=int(epochs))
        self.nets = getattr(self, "nets", []) + [net]

    @torch.no_grad()
    def predict(self, idx):
        nets = getattr(self, "nets", None) or [self.net]
        out = 0
        for n in nets:
            self.net = n; out = out + self._predict1(idx)
        return out / len(nets)

    @torch.no_grad()
    def _predict1(self, idx):
        net = self.net; net.eval(); tta = self.cfg.get("tta", 8)
        Ps, owner = [], []
        rng = np.random.RandomState(0)
        for n, kk in enumerate(idx):
            pid = self.pts.point_id.iloc[kk]; years, bad, q = self.meta[pid]
            vi, _ = valid_dates(bad, q); pr = infer_pairs(pid, vi, years)
            T = self.tiles[pid]; ts = sorted({t for p in pr for t in p})
            acc = 0
            scales = self.cfg.get("ms_tta", [1.0])
            for sc in scales:
              X, V = {}, {}
              for t in ts:
                  xb, vb, _, _ = crop_pair(T[t], T[t], rng, False, self.side, 0.0, sc); X[t] = xb; V[t] = vb
              xs = torch.stack([X[t] for t in ts]).to(self.dev)
              for v in range(tta):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tok = net.tokens(xs, torch.full((len(ts),), v))
                pos = {t: a for a, t in enumerate(ts)}
                fb = tok[[pos[i] for i, _ in pr]]; fa = tok[[pos[j] for _, j in pr]]
                m = torch.stack([F.adaptive_avg_pool2d(torch.minimum(V[i], V[j])[None, None], 16).flatten() for i, j in pr]).to(self.dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    acc = acc + torch.softmax(net.head(fb, fa, m).float(), -1)
            Ps.append((acc / (tta * len(scales))).cpu().numpy()); owner += [n] * len(pr)
        return aggregate(np.concatenate(Ps), np.array(owner), len(idx), mode=self.cfg.get("agg", "top2"), k=2)
