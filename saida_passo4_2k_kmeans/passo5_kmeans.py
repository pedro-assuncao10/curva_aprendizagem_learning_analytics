#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PASSO 3 — K-MEANS (achar os perfis)
Lê os dados limpos e consolidados gerados pelo Passo 1.
Entrada: FORMA (30 fatias) + FEATURES (semanas mortas, centro de massa, etc).
Faz: padronização -> cotovelo + silhueta (escolher K) -> K-means -> perfis.
Saída principal: a curva média de cada cluster (os perfis) e a validação contra
o resultado final (sem usar o resultado para clusterizar).

Requer: pandas, numpy, matplotlib, scikit-learn
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

# --- ARQUIVOS DE ENTRADA DO PASSO 1 ---
DIR_PASSO1 = "./saida_passo1"
FORMA_CSV  = os.path.join(DIR_PASSO1, "matriz_forma.csv")
FEAT_CSV   = os.path.join(DIR_PASSO1, "features_interpretaveis.csv")

OUT_DIR   = "./saida_passo4_2k_kmeans"
K_RANGE   = range(2, 9)        # testa K de 2 a 8
DEFAULT_K = 2                  # forçando K=2 baseado na análise anterior
os.makedirs(OUT_DIR, exist_ok=True)

class Tee:
    def __init__(self, *s): self.s = s
    def write(self, m):
        for x in self.s: x.write(m)
    def flush(self):
        for x in self.s: x.flush()
_log = open(os.path.join(OUT_DIR, "saida_terminal.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, _log)

print("="*70)
print("PASSO 3 — K-MEANS (Descobrir Perfis de Engajamento)")
print("="*70)

# --- 1. carregar e juntar forma + features ---
forma = pd.read_csv(FORMA_CSV, index_col="aluno")
feat  = pd.read_csv(FEAT_CSV, index_col="aluno")

forma_cols = [c for c in forma.columns if c.startswith("forma_w")]
base = forma.join(feat, how="inner")

print(f"\nAlunos válidos para clusterização: {len(base):,}")

# Features que entram na clusterização (forma + features temporais selecionadas)
# NOTA: final_result e volume ficam de fora para evitar data leakage
feat_temporais = ["semanas_mortas", "centro_massa", "inclinacao"]
feat_cols = forma_cols + feat_temporais

X = base[feat_cols].values
Xz = StandardScaler().fit_transform(X)   # padroniza (média 0, desvio 1)
print(f"Features na clusterização: {len(feat_cols)}  ({len(forma_cols)} colunas de forma + {len(feat_temporais)} interpretáveis)")

# Guardados para validação (NÃO entram no modelo de aprendizado)
resultado = base["final_result"]
faixa_vol = base["faixa_volume"]

# --- 2. cotovelo + silhueta para escolher K ---
print("\nTestando K =", list(K_RANGE))
inercia, silh = [], []
rng = np.random.RandomState(0)
# Silhueta é cara de calcular, pegamos uma amostra
amostra = rng.choice(len(Xz), size=min(5000, len(Xz)), replace=False)

for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xz)
    inercia.append(km.inertia_)
    silh.append(silhouette_score(Xz[amostra], km.labels_[amostra]))
    print(f"  K={k}:  inércia={km.inertia_:,.0f}   silhueta={silh[-1]:.3f}")

melhor_k = list(K_RANGE)[int(np.argmax(silh))]
K = DEFAULT_K or melhor_k
print(f"\nMelhor K pela silhueta: {melhor_k}    | Usando K = {K}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
a1.plot(list(K_RANGE), inercia, "o-"); a1.set_title("Cotovelo (inércia)")
a1.set_xlabel("K"); a1.set_ylabel("inércia")
a2.plot(list(K_RANGE), silh, "o-", color="green"); a2.set_title("Silhueta média")
a2.set_xlabel("K"); a2.set_ylabel("silhueta"); a2.axvline(melhor_k, ls="--", color="gray")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_cotovelo_silhueta.png"), dpi=130)
plt.close()

# --- 3. K-means final ---
km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Xz)
base["cluster"] = km.labels_

# --- 4. PERFIS: curva média de forma por cluster ---
eixo = np.arange(len(forma_cols)) / (len(forma_cols) - 1)
plt.figure(figsize=(9, 5))
for c in range(K):
    media = base.loc[base["cluster"] == c, forma_cols].mean().values
    plt.plot(eixo, media, lw=2, marker="o", ms=3, label=f"cluster {c} (n={int((base['cluster']==c).sum())})")
plt.xlabel("fração do curso (0=início, 1=fim)")
plt.ylabel("proporção média do esforço")
plt.title(f"Perfis encontrados pelo K-means (K={K}) — curva média por cluster")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_perfis_curva_media.png"), dpi=130)
plt.close()

# --- 5. validação: cluster x resultado final (H2) e cluster x volume ---
print("\n--- PERFIL DE CADA CLUSTER ---")
resumo = []
for c in range(K):
    m = base["cluster"] == c
    aprov = resultado[m].isin(["Pass", "Distinction"]).mean() * 100
    resumo.append({
        "cluster": c, 
        "n": int(m.sum()), 
        "%base": round(100*m.mean(), 1),
        "semanas_mortas": round(base.loc[m, "semanas_mortas"].mean(), 1),
        "centro_massa": round(base.loc[m, "centro_massa"].mean(), 2),
        "aprovacao_%": round(aprov, 1),
    })
resumo = pd.DataFrame(resumo)
print(resumo.to_string(index=False))

comp = pd.crosstab(base["cluster"], resultado, normalize="index").mul(100).round(1)
print("\nComposição do resultado final por cluster (%):")
print(comp.to_string())

volx = pd.crosstab(base["cluster"], faixa_vol, normalize="index").mul(100).round(1)
print("\nDistribuição de faixa de volume por cluster (%):")
print(volx.to_string())

# gráfico de validação: resultado por cluster
ordem = [c for c in ["Distinction", "Pass", "Fail"] if c in comp.columns]
comp[ordem].plot(kind="bar", stacked=True, figsize=(8.5, 5),
                 color={"Distinction": "#2E7D32", "Pass": "#8BC34A", "Fail": "#FF7043"})
plt.ylabel("% dos alunos do cluster"); plt.xlabel("cluster")
plt.title("Resultado final por cluster (validação H2)")
plt.xticks(rotation=0); plt.legend(title="", bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_resultado_por_cluster.png"), dpi=130)
plt.close()

# --- 6. DIAGNÓSTICO no padrão acadêmico: facas da silhueta + mapa PCA ---
sil_pts_K = silhouette_score(Xz, base["cluster"].values, sample_size=6000, random_state=0)
amostra_f = rng.choice(len(Xz), size=min(6000, len(Xz)), replace=False)
Xs, ls = Xz[amostra_f], base["cluster"].values[amostra_f]
sil_pontos = silhouette_samples(Xs, ls)

pca = PCA(n_components=2, random_state=0).fit(Xz)
proj = pca.transform(Xz)
centros = pca.transform(km.cluster_centers_)
var = pca.explained_variance_ratio_ * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

# ESQUERDA — silhouette plot "das facas"
ax1.set_xlim([-0.2, 1.0]); ax1.set_ylim([0, len(Xs) + (K + 1) * 10])
y_lower = 10
for c in range(K):
    vals = np.sort(sil_pontos[ls == c])
    y_upper = y_lower + len(vals)
    cor = cm.nipy_spectral(float(c) / K)
    ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                      facecolor=cor, edgecolor=cor, alpha=0.7)
    ax1.text(-0.05, y_lower + 0.5 * len(vals), str(c))
    y_lower = y_upper + 10
ax1.axvline(x=sil_pts_K, color="red", ls="--", label=f"média = {sil_pts_K:.3f}")
ax1.set_title(f"Silhueta por aluno, por cluster (K={K})")
ax1.set_xlabel("coeficiente de silhueta"); ax1.set_yticks([])
ax1.legend(loc="lower right")

# DIREITA — mapa PCA com os centros marcados
cores = cm.nipy_spectral(base["cluster"].values.astype(float) / K)
ax2.scatter(proj[:, 0], proj[:, 1], c=cores, s=4, alpha=0.35)
ax2.scatter(centros[:, 0], centros[:, 1], marker="o", c="white", s=260,
            edgecolor="k", zorder=3)
for c in range(K):
    ax2.scatter(centros[c, 0], centros[c, 1], marker=f"${c}$", c="k", s=60, zorder=4)
ax2.set_title("Alunos em 2D (PCA) — cor = cluster, números = centros")
ax2.set_xlabel(f"PCA 1 ({var[0]:.0f}% da variação)")
ax2.set_ylabel(f"PCA 2 ({var[1]:.0f}% da variação)")

plt.suptitle(f"Diagnóstico de silhueta — K-means com K={K}", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_silhueta_facas_e_mapa.png"), dpi=130)
plt.close()

# --- saídas CSV ---
base[["cluster", "final_result", "faixa_volume", "semanas_mortas", "centro_massa"]].to_csv(os.path.join(OUT_DIR, "alunos_clusters.csv"))
resumo.to_csv(os.path.join(OUT_DIR, "perfil_dos_clusters.csv"), index=False)

print("\n" + "="*70)
print(f"Gravado em {os.path.abspath(OUT_DIR)}:")
print("  - 01_cotovelo_silhueta.png      (como escolhemos K)")
print("  - 02_perfis_curva_media.png     (OS PERFIS: curva média de cada cluster)")
print("  - 03_resultado_por_cluster.png  (cada perfil passa mais ou menos?)")
print("  - 04_silhueta_facas_e_mapa.png  (diagnóstico PCA e Facas)")
print("  - alunos_clusters.csv           (A tabela-chave: Aluno -> Cluster)")
print("  - perfil_dos_clusters.csv       (O resumo executivo)")
print("  - saida_terminal.txt")
print(f"\nPasso 3 OK. K={K}. Os clusters foram encontrados com sucesso e estão limpos!")
_log.close()