#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PASSO 5 — COMPARAÇÃO DE SILHUETAS (facas) para vários K
Gera UMA figura com o silhouette plot "das facas" de cada K lado a lado,
para inspecionar visualmente se os dados se separam bem em algum K.
Só silhueta — não refaz perfis nem validação.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

FORMA_CSV = "./saida_passo3/matriz_forma.csv"
DUR_CSV   = "./saida_passo4_2/features_durabilidade.csv"
OUT_DIR   = "./saida_passo5_k4"
K_LIST    = [2, 3, 4, 5, 6, 7]   # quais K comparar
N_FACAS   = 6000                 # amostra p/ a silhueta por ponto
os.makedirs(OUT_DIR, exist_ok=True)

# --- matriz (igual ao passo 5)
forma = pd.read_csv(FORMA_CSV, index_col="aluno")
dur   = pd.read_csv(DUR_CSV,   index_col="aluno")
forma_cols = [c for c in forma.columns if c.startswith("forma_w")]
base = forma.join(dur, how="inner")
feat_cols = forma_cols + ["semanas_mortas", "p_comeco", "p_meio", "p_fim"]
Xz = StandardScaler().fit_transform(base[feat_cols].values)

rng = np.random.RandomState(0)
amostra = rng.choice(len(Xz), size=min(N_FACAS, len(Xz)), replace=False)
Xs = Xz[amostra]

# --- grade de subplots (2 linhas)
ncol = 3
nrow = int(np.ceil(len(K_LIST) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow))
axes = np.atleast_1d(axes).ravel()

for ax, K in zip(axes, K_LIST):
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Xz)
    ls = km.labels_[amostra]
    sil_media = silhouette_score(Xz, km.labels_, sample_size=6000, random_state=0)
    sil_pontos = silhouette_samples(Xs, ls)

    ax.set_xlim([-0.2, 1.0]); ax.set_ylim([0, len(Xs) + (K + 1) * 10])
    y_lower = 10
    for c in range(K):
        vals = np.sort(sil_pontos[ls == c])
        y_upper = y_lower + len(vals)
        cor = cm.nipy_spectral(float(c) / K)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                         facecolor=cor, edgecolor=cor, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * len(vals), str(c), fontsize=8)
        y_lower = y_upper + 10
    ax.axvline(x=sil_media, color="red", ls="--")
    ax.set_title(f"K={K}  (silhueta média = {sil_media:.3f})")
    ax.set_xlabel("coef. de silhueta"); ax.set_yticks([])

# esconde subplots sobrando
for ax in axes[len(K_LIST):]:
    ax.axis("off")

plt.suptitle("Comparação de silhueta (facas) por K — quanto mais à direita do zero, melhor",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_comparacao_silhuetas.png"), dpi=130, bbox_inches="tight")
plt.close()

print(f"Gravado: {os.path.join(os.path.abspath(OUT_DIR), '05_comparacao_silhuetas.png')}")
print("\nLeitura: se NENHUM K produz facas largas e bem à direita do zero,")
print("os dados não têm grupos naturalmente separáveis — são um gradiente contínuo.")