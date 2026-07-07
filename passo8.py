#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PASSO 8 — AGRUPAMENTO HIERÁRQUICO (Lista Q3)
Sobre as MESMAS 5 features da opção A (passo 6), varia a MEDIDA DE DISTÂNCIA
(euclidiana, manhattan, cosseno) e o LINKAGE (ward, average, complete) e
compara com o K-means.
- Hierárquico é O(n^2): roda numa AMOSTRA representativa (padrão e necessário).
- Compara cada combinação com o K-means via Adjusted Rand Index (ARI):
  ARI alto = os dois métodos concordam -> reforça a existência dos perfis (H1).

Requer: pandas, numpy, matplotlib, scikit-learn, scipy
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

FEAT_CSV = "./saida_passo6/features_avaliacao.csv"
OUT_DIR  = "./saida_passo8"
K        = 3          # nº de grupos (igual ao K-means escolhido)
N_SAMPLE = 6000       # amostra p/ o hierárquico (memória O(n^2))
os.makedirs(OUT_DIR, exist_ok=True)

class Tee:
    def __init__(self, *s): self.s = s
    def write(self, m):
        for x in self.s: x.write(m)
    def flush(self):
        for x in self.s: x.flush()
_orig = sys.stdout
_log = open(os.path.join(OUT_DIR, "saida_terminal.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, _log)

print("="*70); print("PASSO 8 — HIERÁRQUICO (varia distância e linkage)"); print("="*70)

df = pd.read_csv(FEAT_CSV, index_col=0)
feat_cols = ["tendencia","antecedencia","durabilidade","concentracao","cobertura_final"]
Xz_full = StandardScaler().fit_transform(df[feat_cols].values)

# K-means na base toda (referência p/ comparação)
km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Xz_full)
df["kmeans"] = km.labels_

# amostra p/ o hierárquico
rng = np.random.RandomState(0)
idx = rng.choice(len(df), size=min(N_SAMPLE, len(df)), replace=False)
Xs = Xz_full[idx]
km_s = df["kmeans"].values[idx]
res_s = df["final_result"].values[idx]
print(f"\nAmostra para o hierárquico: {len(Xs):,} alunos")

# combinações: (linkage, métrica)  -- ward só aceita euclidiana
combos = [
    ("ward",     "euclidean"),
    ("average",  "euclidean"),
    ("complete", "euclidean"),
    ("average",  "cityblock"),   # manhattan
    ("complete", "cityblock"),
    ("average",  "cosine"),
    ("complete", "cosine"),
]

resultados = []
linkages = {}
for metodo, metrica in combos:
    Z = linkage(Xs, method=metodo, metric=metrica)
    labels = fcluster(Z, t=K, criterion="maxclust")
    # silhueta na mesma métrica usada
    sil = silhouette_score(Xs, labels, metric=metrica) if len(set(labels)) > 1 else np.nan
    ari = adjusted_rand_score(km_s, labels)            # concordância com K-means
    tamanhos = np.bincount(labels)[1:]                 # fcluster rotula a partir de 1
    aprov = [round(100*np.mean(np.isin(res_s[labels == c], ["Pass","Distinction"])), 1)
             for c in range(1, K+1)]
    resultados.append({"linkage": metodo, "metrica": metrica,
                       "silhueta": round(sil, 3), "ARI_vs_kmeans": round(ari, 3),
                       "tamanhos": tamanhos.tolist(), "aprovacao_por_grupo": aprov})
    linkages[(metodo, metrica)] = Z

tab = pd.DataFrame(resultados)
print("\n>>> COMPARAÇÃO DAS COMBINAÇÕES (hierárquico):")
print(tab.to_string(index=False))
print("\nLeitura:")
print("  silhueta  : separação dos grupos (maior = melhor)")
print("  ARI_vs_kmeans: concordância com o K-means (1=idêntico, 0=aleatório)")

# --- dendrogramas (grade)
n = len(combos); ncol = 3; nrow = int(np.ceil(n/ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow))
axes = np.atleast_1d(axes).ravel()
for ax, (metodo, metrica) in zip(axes, combos):
    dendrogram(linkages[(metodo, metrica)], truncate_mode="lastp", p=15,
               ax=ax, no_labels=True, color_threshold=0)
    ax.set_title(f"{metodo} + {metrica}", fontsize=10)
    ax.set_xlabel("grupos"); ax.set_ylabel("distância")
for ax in axes[n:]: ax.axis("off")
plt.suptitle("Dendrogramas por (linkage + métrica)", fontweight="bold", y=1.01)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "01_dendrogramas.png"), dpi=130, bbox_inches="tight")
plt.close()

# --- comparação: silhueta e ARI por combinação
tab["combo"] = tab["linkage"] + "+" + tab["metrica"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.barh(tab["combo"], tab["silhueta"], color="#4C72B0"); a1.set_title("Silhueta por combinação")
a1.set_xlabel("silhueta"); a1.invert_yaxis()
a2.barh(tab["combo"], tab["ARI_vs_kmeans"], color="#2E7D32"); a2.set_title("Concordância com K-means (ARI)")
a2.set_xlabel("ARI"); a2.invert_yaxis()
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "02_comparacao_metricas.png"), dpi=130)
plt.close()

# --- melhor combinação (por silhueta) -> perfil + comparação com K-means
melhor = tab.loc[tab["silhueta"].idxmax()]
mb = (melhor["linkage"], melhor["metrica"])
print(f"\nMelhor combinação por silhueta: {mb[0]} + {mb[1]}  (silhueta={melhor['silhueta']}, ARI={melhor['ARI_vs_kmeans']})")
lab_best = fcluster(linkages[mb], t=K, criterion="maxclust")

# perfil das 5 features por grupo (do melhor hierárquico)
dfb = pd.DataFrame(Xs, columns=feat_cols); dfb["grupo"] = lab_best
cen = dfb.groupby("grupo")[feat_cols].mean()
cen.T.plot(kind="bar", figsize=(10,5))
plt.title(f"Perfil das features — hierárquico ({mb[0]}+{mb[1]}, K={K})")
plt.ylabel("valor médio (padronizado)"); plt.xticks(rotation=20)
plt.legend(title="grupo", bbox_to_anchor=(1.02,1)); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_perfil_melhor_hierarquico.png"), dpi=130); plt.close()

# tabela cruzada hierárquico x kmeans (concordância visual)
cross = pd.crosstab(lab_best, km_s)
print(f"\nCruzamento hierárquico ({mb[0]}+{mb[1]}) x K-means (linhas=hier, colunas=kmeans):")
print(cross.to_string())

tab.drop(columns="combo").to_csv(os.path.join(OUT_DIR, "comparacao_hierarquico.csv"), index=False)

print("\n" + "="*70)
print(f"Gravado em {os.path.abspath(OUT_DIR)}:")
print("  - 01_dendrogramas.png            (um dendrograma por combinação)")
print("  - 02_comparacao_metricas.png     (silhueta e ARI por combinação)")
print("  - 03_perfil_melhor_hierarquico.png")
print("  - comparacao_hierarquico.csv     saida_terminal.txt")
print(f"\nPasso 8 OK. Amostra={len(Xs)}, K={K}.")
sys.stdout = _orig; _log.close()