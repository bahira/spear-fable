"""SPEAR core partagé : kernels, features, embedding (mu/sigma figés), tiny policy.

Une seule définition pour l'entraînement ET l'inférence — le bug
d'embedding constant (normalisation batch de 1) vient de la duplication.
"""
import re
from pathlib import Path

import numpy as np

# ---------- Kernels closed-form (SPEAR / SpearVM) ----------
def spear_gelu(x):
    x = np.asarray(x, dtype=float)
    u = np.clip(0.306923 * x + 0.501, 0.0, 1.002)
    return 0.997729 * (x * u) - 0.004004

def spear_lorentz(x):
    b = np.tanh(0.5 * np.asarray(x, dtype=float))
    return 1.0 / np.sqrt(np.maximum(1e-12, 1.0 - 0.8 * b * b))

def spear_gauss(x):
    return np.tanh(0.6 * np.asarray(x, dtype=float))

def spear_soft(x):
    return np.log1p(np.exp(np.clip(np.asarray(x, dtype=float), -6, 6)))

def spear_kepler(x):
    return np.sqrt(np.abs(np.asarray(x, dtype=float)) ** 3)

KERNELS = dict(gelu=spear_gelu, lorentz=spear_lorentz, gauss=spear_gauss,
               soft=spear_soft, kepler=spear_kepler)
EMB_KERNELS = [spear_gelu, spear_lorentz, spear_gauss, spear_soft]

# ---------- Features structurelles (10 dims, identiques train/inférence) ----------
TOOL_RE = r"(tool|bash|edit|exec|lance|exécute)"
CODE_WORDS = ("écris", "write", "code", "fonction", "script")
MATH_WORDS = ("gelu", "lorentz", "kepler", "math", "kernel")

def features_from_text(text: str) -> np.ndarray:
    t = (text or "").lower()
    return np.array([
        np.log1p(len(t)) / 5,
        len(re.findall(TOOL_RE, t)) / 3,
        (t.count("```") + t.count("def ")) / 3,
        1.0 if any(k in t for k in MATH_WORDS) else 0.0,
        1.0 if any(k in t for k in CODE_WORDS) else 0.0,
        t.count(" ") / 30,
        len(re.findall(r"\d+", t)) / 5,
        1.0 if "?" in t else 0.0,
        1.0 if ("merci" in t or "thanks" in t) else 0.0,
        0.0,  # slot "tour" — rempli par l'agent ; sigma=0 au training => neutre
    ], dtype=np.float32)

# ---------- Embedding SPEAR multi-kernels ----------
class SpearEmb:
    """Expansion résiduelle raw + kernels(raw @ W), L2-normalisée.

    mu/sigma sont FIGÉS au fit (stats du corpus d'entraînement) et réutilisés
    à l'inférence : un batch de 1 ne s'annule plus.
    """
    def __init__(self, W, mu=None, sigma=None, out_mu=None, out_sigma=None):
        self.W = np.asarray(W, np.float32)
        self.d_in = self.W.shape[0]
        self.d_out = self.d_in + self.W.shape[1] * len(EMB_KERNELS)
        self.mu = np.zeros(self.d_in, np.float32) if mu is None else np.asarray(mu, np.float32)
        self.sigma = np.ones(self.d_in, np.float32) if sigma is None else np.asarray(sigma, np.float32)
        # Stats de la sortie d'expansion (pour le policy) — figées au fit.
        self.out_mu = np.zeros(self.d_out, np.float32) if out_mu is None else np.asarray(out_mu, np.float32)
        self.out_sigma = np.ones(self.d_out, np.float32) if out_sigma is None else np.asarray(out_sigma, np.float32)

    @classmethod
    def fit(cls, X, extra=8, seed=42):
        X = np.atleast_2d(np.asarray(X, np.float32))
        rng = np.random.RandomState(seed)
        W = (rng.randn(X.shape[1], extra) * 0.35).astype(np.float32)
        emb = cls(W, X.mean(0), X.std(0))
        e = emb.expand(X)
        emb.out_mu, emb.out_sigma = e.mean(0).astype(np.float32), e.std(0).astype(np.float32)
        return emb

    def _prep(self, X):
        X = np.asarray(X, np.float32)
        if X.ndim == 1:
            X = X[None, :]
        d = self.d_in
        X = X[:, :d] if X.shape[1] >= d else np.pad(X, ((0, 0), (0, d - X.shape[1])))
        # Dims constantes au training (sigma~0) : aucune info -> forcées à 0.
        # (sinon /1e-5 explose sur toute valeur != mu — bug mémoire + policy)
        alive = self.sigma > 1e-5
        Xn = np.where(alive, (X - self.mu) / np.where(alive, self.sigma, 1.0), 0.0)
        z = Xn @ self.W
        parts = [Xn] + [np.asarray(k(z), np.float32) for k in EMB_KERNELS]
        return np.concatenate(parts, axis=1)

    def expand(self, X):
        """Expansion brute multi-kernels (sans L2)."""
        return self._prep(X)

    def transform(self, X):
        """Expansion standardisée par les stats du training — entrée du policy."""
        e = self._prep(X)
        # Même garde que _prep : dims d'expansion constantes au training -> 0
        # (sinon /1e-5 explose dès que l'inférence sort du domaine vu au fit).
        alive_o = self.out_sigma > 1e-5
        return np.where(alive_o, (e - self.out_mu) / np.where(alive_o, self.out_sigma, 1.0), 0.0)

    def raw_code(self, X):
        """Features standardisées par mu/sigma du training, L2-normalisées.

        ponytail: utilisé pour la RÉCALL mémoire — la projection aléatoire +
        kernels diluent les diffs de features 10-dim dans 42 dims de bruit,
        ce qui rendait le top-hit arbitraire. L'expansion reste pour le policy.
        """
        X = np.asarray(X, np.float32)
        if X.ndim == 1:
            X = X[None, :]
        d = self.d_in
        X = X[:, :d] if X.shape[1] >= d else np.pad(X, ((0, 0), (0, d - X.shape[1])))
        alive = self.sigma > 1e-5
        Xn = np.where(alive, (X - self.mu) / np.where(alive, self.sigma, 1.0), 0.0)
        return Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-6)

    def __call__(self, X):
        """Version L2-normalisée — similarité cosinus (mémoire vectorielle)."""
        t = self.transform(X)
        return t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-6)

# ---------- Tiny policy (MLP gelu, ~qqs centaines de params) ----------
class TinyPolicy:
    def __init__(self, W1, b1, W2, b2):
        self.W1, self.b1 = np.asarray(W1, np.float32), np.asarray(b1, np.float32)
        self.W2, self.b2 = np.asarray(W2, np.float32), np.asarray(b2, np.float32)

    @classmethod
    def init(cls, d_in, h=32, seed=7):
        rng = np.random.RandomState(seed)
        return cls(rng.randn(d_in, h) * (1.2 / np.sqrt(d_in)),
                   np.zeros(h), rng.randn(h, 1) * 0.25, np.zeros(1))

    def logits(self, X):
        return spear_gelu(X @ self.W1 + self.b1) @ self.W2 + self.b2

    def predict_proba(self, X):
        X = np.asarray(X, np.float32)
        if X.ndim == 1:
            X = X[None, :]
        return 1 / (1 + np.exp(-np.clip(self.logits(X), -20, 20)))

    def fit(self, X, y, epochs=150, lr=0.03, batch=64, seed=0, val=None):
        n = len(X)
        rng = np.random.RandomState(seed)
        y = y.reshape(-1, 1)
        best_acc, best_w = -1.0, None
        for ep in range(epochs):
            lr_t = lr * (0.4 + 0.6 * np.cos(np.pi * ep / epochs))
            idx = rng.permutation(n)
            for i in range(0, n, batch):
                b = idx[i:i + batch]
                xb, yb = X[b], y[b]
                z = xb @ self.W1 + self.b1
                a = spear_gelu(z)
                pred = 1 / (1 + np.exp(-np.clip(a @ self.W2 + self.b2, -12, 12)))
                # ponytail: clip de gradient — sans ça, lr>=0.05 diverge en dim 42.
                dL = np.clip((pred - yb) / len(b), -0.01, 0.01)
                self.W2 -= lr_t * (a.T @ dL)
                self.b2 -= lr_t * dL.sum(0)
                da = dL @ self.W2.T
                u = 0.306923 * z + 0.501
                inside = ((u > 0) & (u < 1.002)).astype(np.float32)
                dz = da * 0.997729 * (np.clip(u, 0, 1.002) + z * 0.306923 * inside)
                self.W1 -= lr_t * (xb.T @ dz)
                self.b1 -= lr_t * dz.sum(0)
            if val is not None:
                acc = self.accuracy(val[0], val[1])
                if acc >= best_acc:
                    best_acc = acc
                    best_w = tuple(w.copy() for w in
                                   (self.W1, self.b1, self.W2, self.b2))
        if best_w is not None:  # early stopping : meilleurs poids de validation
            self.W1, self.b1, self.W2, self.b2 = best_w
        return best_acc

    def accuracy(self, X, y):
        p = (self.predict_proba(X).ravel() > 0.5).astype(np.float32)
        return float((p == y.ravel()).mean())

# ---------- Sauvegarde / chargement ----------
MODEL_KEYS = ("emb_W", "mu", "sigma", "out_mu", "out_sigma", "W1", "b1", "W2", "b2")

def save_model(path: Path, emb: SpearEmb, pol: TinyPolicy, meta: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, emb_W=emb.W, mu=emb.mu, sigma=emb.sigma,
             out_mu=emb.out_mu, out_sigma=emb.out_sigma,
             W1=pol.W1, b1=pol.b1, W2=pol.W2, b2=pol.b2)
    if meta is not None:
        import json
        path.with_suffix(".json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

def load_model(path: Path):
    # Context manager : np.load laisse sinon le .npz ouvert (blocage suppression Windows).
    with np.load(Path(path)) as data:
        opt = lambda k: data[k] if k in data.files else None
        emb = SpearEmb(data["emb_W"], data["mu"], data["sigma"],
                       opt("out_mu"), opt("out_sigma"))
        pol = TinyPolicy(data["W1"], data["b1"], data["W2"], data["b2"])
        snapshot = {k: data[k] for k in data.files}
    return emb, pol, snapshot
