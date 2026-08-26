# SPEAR × Fable-5 — Ultra-Small Agentic Experiment

Interactive agent that **talks, writes code, executes it, calls SPEAR math kernels, and uses vector memory**.

Built on top of:
- **SPEAR** closed-form algebraic kernels (GELU, Lorentz, …)
- **SPEAR Ultra Embeddings** (multi-kernel residual expansion)
- **TinyPolicy** pre-trained on Fable-5-style structural signals (tool-use / complexity)
- Optional **AVX2** kernels for the hot path

## Quick start

```bash
# requires only numpy
pip install numpy

cd spear_fable_project
python src/agent.py        # agent interactif (routeur synthétique)
python tests/test_all.py   # suite de tests (8 checks)

# avec la policy entraînée sur données réelles :
python src/agent.py --policy models/agentinstruct_policy.npz
```

Installation dev optionnelle (`spear-agent` en commande) : `pip install -e .`

La mémoire vectorielle persiste entre les sessions dans `.spear_memory.json`
(items appris via « écris … », ré-embeddés avec le modèle courant).

Le modèle `models/fable_policy.npz` (routeur de l'agent) est fourni pré-entraîné.
Deux modèles sont inclus :

| Modèle | Données | Cible | Test acc |
|---|---|---|---|
| `fable_policy.npz` | synthétique, 4 000 traj. (labels génératifs) | « nécessite outil/code ? » | 98.4% |
| `agentinstruct_policy.npz` | **réel** : [THUDM/AgentInstruct](https://huggingface.co/datasets/THUDM/AgentInstruct), 1 866 trajectoires d'agents (os/db/webshop/alfworld…) | trajectoire long-horizon | **95.7%** (baseline brut : 78.6%) |

Ré-entraînement :

```bash
python examples/train_policy.py                                  # routeur (synthétique)
python examples/train_policy.py --hf THUDM/AgentInstruct \
       --out models/agentinstruct_policy.npz                     # données réelles
python tests/test_all.py                                         # 7 checks
```

`--hf` nécessite `pip install datasets`. Chaque modèle a un sidecar `.json`
(méta : dataset, cible, accuracy).

## What you can say

- `Salut` / `qui es-tu ?`
- `aide`
- `Écris gelu` → generates SPEAR GELU code
- `Exécute` → runs last generated code
- `lorentz(0.9)` / `gelu(1.5)` → direct kernel call
- `Écris fibonacci` then `Exécute`
- `mémoire` / `status` → complexity score + memory hits
- `list kernels`

## Project layout

```
spear_fable_project/
├── README.md
├── requirements.txt
├── models/
│   ├── fable_policy.npz      # routeur agent (emb mu/sigma figés + policy)
│   ├── fable_policy.json     # méta du routeur
│   ├── agentinstruct_policy.npz  # policy entraînée sur AgentInstruct réel
│   └── agentinstruct_policy.json
├── src/
│   ├── spear_core.py        # noyau partagé : kernels, features, embedding, policy
│   └── agent.py             # agent interactif (exécution en subprocess isolé)
├── kernels/
│   ├── spear_avx_emb.c      # AVX2 source
│   └── libspear_emb.dll     # précompilée (Windows) — sinon voir ci-dessous
├── examples/
│   ├── train_policy.py      # entraînement (synthétique ou --hf)
│   └── bench_spear_avx.py   # bench NumPy vs AVX2 (skip si lib absente)
└── tests/
    └── test_all.py          # python tests/test_all.py (ou pytest)
```

## Notes

- No GPU / no large LLM required.
- Everything is pure NumPy + a few closed-form formulas.
- L'embedding utilise des stats mu/sigma **figées au training** : l'inférence
  single-row est correcte (les dims constantes au training sont neutralisées).
- Early stopping sur split de validation dans `TinyPolicy.fit` — sans lui,
  l'entraînement diverge au-delà du point optimal (constaté sur données réelles).
- Le code généré s'exécute dans un subprocess isolé (`python -I`, timeout 10 s) —
  isolation process, pas une sandbox de sécurité : usage local uniquement.
- Le routeur de l'agent est entraîné sur des labels **génératifs** (non
  circulaires) ; le modèle AgentInstruct utilise un comptage d'actions réel.

## Kernels AVX2 (optionnel)

`libspear_emb.dll` est fournie précompilée (Windows x64, MinGW). Sinon :

```bash
gcc -O3 -mavx2 -mfma -shared -o kernels/libspear_emb.dll kernels/spear_avx_emb.c   # Windows
gcc -O3 -mavx2 -mfma -shared -fPIC -o kernels/libspear_emb.so kernels/spear_avx_emb.c -lm  # Linux
python examples/bench_spear_avx.py
```

Mesuré (1M éléments, laptop CPU) : GELU **×6.15 bit-exact** (err 1.8e-15),
Gauss ×6.9 / Lorentz ×4.9 (tanh rapide approx., erreur ~1e-1 attendue).
Sans la lib, tout fonctionne en NumPy pur et le bench skip proprement.

## CI

`.github/workflows/ci.yml` : tests (standalone + pytest) sur
ubuntu/windows × Python 3.10/3.12 + smoke entraînement + boot agent.

## License

MIT (kernels inspired by bahira/SPEAR & SpearVM work).
