#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PASSO 13 — AGRUPAMENTO HIERÁRQUICO SOBRE A CURVA (validação do passo 12)
Mesma lógica do passo 8 (varia distância/linkage, compara com K-means via ARI),
só que agora em cima da curva de esforço do passo 12 (só alunos persistentes,
já que "desistente" é regra, não clustering) em vez dos 5 indicadores do passo 6.

Diferença importante em relação ao passo 8: lá, escolher "melhor combinação
por silhueta" caiu numa armadilha (average+cityblock tinha silhueta alta só
por isolar 2 outliers, 9 e 72 alunos, e jogar 98% da amostra num cluster só).
Para não repetir isso, aqui comparamos cada combinação hierárquica TAMBÉM
contra o rótulo independente do passo 3 (timing/subtipo — regra, não
clustering), não só contra o K-means. Isso é o teste real de qualidade: um
cluster degenerado (poucos outliers isolados) não bate com o passo 3 mesmo
tendo silhueta alta.

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

CURVA_CSV = "./saida_passo12/curvas_e_perfis.csv"
DUR_CSV   = "./saida_passo3/features_durabilidade.csv"
CP_CSV    = "./saida_passo3/comeco_pesado_detalhado.csv"
OUT_DIR   = "./saida_passo13"
K         = 3          # mesmo K do estágio 2 do passo 12
N_SAMPLE  = 6000        # amostra p/ o hierárquico (memória O(n^2))
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

print("="*70); print("PASSO 13 — HIERÁRQUICO SOBRE A CURVA (validação do passo 12)"); print("="*70)

feat_cols = [f"c{i}" for i in range(6)]

# 1. Carregar a curva do passo 12 + rótulo independente do passo 3
C = pd.read_csv(CURVA_CSV, index_col=0)
dur = pd.read_csv(DUR_CSV, index_col=0)
cp = pd.read_csv(CP_CSV, index_col=0)

Cv = C.join(dur[["timing"]], how="left").join(cp[["subtipo"]], how="left")
Cv["perfil_passo3"] = Cv["subtipo"]
Cv.loc[Cv["timing"] == "equilibrado", "perfil_passo3"] = "equilibrado"
Cv.loc[Cv["timing"] == "fim-pesado", "perfil_passo3"] = "fim-pesado"

# só persistentes: "desistente" é regra (ativo_no_fim), não faz sentido comparar
# clustering contra uma regra que os dois lados já usam igual (seria circular)
df = Cv[Cv["perfil"] != "desistente"].copy()
print(f"\nPersistentes (fora do K-means/hierárquico do desistente): {len(df):,}")

Xz_full = StandardScaler().fit_transform(df[feat_cols].values)

# K-means do passo 12 já está salvo em df["perfil"] (referência p/ ARI)
rng = np.random.RandomState(0)
idx = rng.choice(len(df), size=min(N_SAMPLE, len(df)), replace=False)
Xs = Xz_full[idx]
km_s = df["perfil"].values[idx]
ref_s = df["perfil_passo3"].values[idx]
res_s = df["final_result"].values[idx]
print(f"Amostra para o hierárquico: {len(Xs):,} alunos")

combos = [
    ("ward",     "euclidean"),
    ("average",  "euclidean"),
    ("complete", "euclidean"),
    ("average",  "cityblock"),
    ("complete", "cityblock"),
    ("average",  "cosine"),
    ("complete", "cosine"),
]

resultados = []
linkages = {}
for metodo, metrica in combos:
    Z = linkage(Xs, method=metodo, metric=metrica)
    labels = fcluster(Z, t=K, criterion="maxclust")
    sil = silhouette_score(Xs, labels, metric=metrica) if len(set(labels)) > 1 else np.nan
    ari_km = adjusted_rand_score(km_s, labels)
    ari_ref = adjusted_rand_score(ref_s, labels)
    tamanhos = np.bincount(labels)[1:]
    aprov = [round(100*np.mean(np.isin(res_s[labels == c], ["Pass","Distinction"])), 1)
             for c in range(1, K+1)]
    resultados.append({"linkage": metodo, "metrica": metrica,
                       "silhueta": round(sil, 3), "ARI_vs_kmeans": round(ari_km, 3),
                       "ARI_vs_passo3": round(ari_ref, 3),
                       "tamanhos": tamanhos.tolist(), "aprovacao_por_grupo": aprov})
    linkages[(metodo, metrica)] = Z

tab = pd.DataFrame(resultados)
print("\n>>> COMPARAÇÃO DAS COMBINAÇÕES (hierárquico):")
print(tab.to_string(index=False))
print("\nLeitura:")
print("  silhueta      : separação interna dos grupos (maior = melhor, mas cuidado com clusters degenerados)")
print("  ARI_vs_kmeans : concordância com o K-means do passo 12 (1=idêntico, 0=aleatório)")
print("  ARI_vs_passo3 : concordância com o rótulo INDEPENDENTE do passo 3 (o teste que importa de verdade)")

# --- dendrogramas
n = len(combos); ncol = 3; nrow = int(np.ceil(n/ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow))
axes = np.atleast_1d(axes).ravel()
for ax, (metodo, metrica) in zip(axes, combos):
    dendrogram(linkages[(metodo, metrica)], truncate_mode="lastp", p=15,
               ax=ax, no_labels=True, color_threshold=0)
    ax.set_title(f"{metodo} + {metrica}", fontsize=10)
    ax.set_xlabel("grupos"); ax.set_ylabel("distância")
for ax in axes[n:]: ax.axis("off")
plt.suptitle("Dendrogramas por (linkage + métrica) — curva do passo 12", fontweight="bold", y=1.01)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "01_dendrogramas.png"), dpi=130, bbox_inches="tight")
plt.close()

# --- comparação: silhueta, ARI vs kmeans, ARI vs passo3
tab["combo"] = tab["linkage"] + "+" + tab["metrica"]
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5))
a1.barh(tab["combo"], tab["silhueta"], color="#4C72B0"); a1.set_title("Silhueta por combinação")
a1.set_xlabel("silhueta"); a1.invert_yaxis()
a2.barh(tab["combo"], tab["ARI_vs_kmeans"], color="#2E7D32"); a2.set_title("Concordância com K-means (passo12)")
a2.set_xlabel("ARI"); a2.invert_yaxis()
a3.barh(tab["combo"], tab["ARI_vs_passo3"], color="#C62828"); a3.set_title("Concordância com passo3 (referência real)")
a3.set_xlabel("ARI"); a3.invert_yaxis()
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "02_comparacao_metricas.png"), dpi=130)
plt.close()

# --- melhor combinação: por ARI vs passo3 (não por silhueta -- lição do passo 8)
melhor = tab.loc[tab["ARI_vs_passo3"].idxmax()]
mb = (melhor["linkage"], melhor["metrica"])
print(f"\nMelhor combinação por ARI_vs_passo3 (critério real, não silhueta): {mb[0]} + {mb[1]}")
print(f"  silhueta={melhor['silhueta']}  ARI_vs_kmeans={melhor['ARI_vs_kmeans']}  ARI_vs_passo3={melhor['ARI_vs_passo3']}")
melhor_sil = tab.loc[tab["silhueta"].idxmax()]
print(f"\n(Para comparação: melhor por silhueta pura seria {melhor_sil['linkage']}+{melhor_sil['metrica']}, "
      f"silhueta={melhor_sil['silhueta']}, mas ARI_vs_passo3={melhor_sil['ARI_vs_passo3']} -- "
      f"olhar só a silhueta pode enganar, como já vimos no passo 8.)")

lab_best = fcluster(linkages[mb], t=K, criterion="maxclust")

# perfil das 6 features da curva por grupo do melhor hierárquico
dfb = pd.DataFrame(Xs, columns=feat_cols); dfb["grupo"] = lab_best
cen = dfb.groupby("grupo")[feat_cols].mean()
cen.T.plot(kind="bar", figsize=(10,5))
plt.title(f"Perfil da curva — hierárquico ({mb[0]}+{mb[1]}, K={K})")
plt.ylabel("valor médio (padronizado)"); plt.xticks(rotation=0)
plt.legend(title="grupo", bbox_to_anchor=(1.02,1)); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_perfil_melhor_hierarquico.png"), dpi=130); plt.close()

# tabela cruzada hierárquico x k-means x passo3 (concordância visual)
cross_km = pd.crosstab(lab_best, km_s)
cross_ref = pd.crosstab(lab_best, ref_s)
print(f"\nCruzamento hierárquico ({mb[0]}+{mb[1]}) x K-means (linhas=hier, colunas=kmeans):")
print(cross_km.to_string())
print(f"\nCruzamento hierárquico ({mb[0]}+{mb[1]}) x passo3 (linhas=hier, colunas=passo3):")
print(cross_ref.to_string())

tab.drop(columns="combo").to_csv(os.path.join(OUT_DIR, "comparacao_hierarquico.csv"), index=False)
cross_ref.to_csv(os.path.join(OUT_DIR, "crosstab_melhor_vs_passo3.csv"))

print("\n" + "="*70)
print(f"Gravado em {os.path.abspath(OUT_DIR)}:")
print("  - 01_dendrogramas.png            (um dendrograma por combinação)")
print("  - 02_comparacao_metricas.png     (silhueta, ARI vs kmeans, ARI vs passo3)")
print("  - 03_perfil_melhor_hierarquico.png")
print("  - comparacao_hierarquico.csv     crosstab_melhor_vs_passo3.csv   saida_terminal.txt")
print(f"\nPasso 13 OK. Amostra={len(Xs)}, K={K}.")
sys.stdout = _orig; _log.close()
