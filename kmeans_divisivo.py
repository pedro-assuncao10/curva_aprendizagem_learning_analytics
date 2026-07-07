# -*- coding: utf-8 -*-
"""
Teste divisivo: k=2 separa os tardios; re-aplica k-means so no cluster restante.

Etapas:
  1. k=2 sobre todas as curvas de ciclo -> identifica o cluster "tardio"
     (maior centro de massa temporal) e o cluster "resto".
  2. Varredura de k (2..6) so dentro do "resto" (cotovelo + silhueta).
  3. Solucao combinada = tardio congelado + sub-clusters do resto.
  4. Comparacao com o k=3 direto: silhueta global, ARI e tabela de cruzamento.

Saidas em resultados_kmeans/divisivo/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

from kmeans_ciclos import (carregar_vle_diario, montar_ciclos, construir_matriz,
                           OUT_DIR, N_BINS, BIN_LABELS, SEED,
                           SURF, INK, SEC, MUT, SERIES)

OUT = os.path.join(OUT_DIR, "divisivo")
SUB_K_RANGE = range(2, 7)

# mesma cor por perfil nas duas soluções (segue a figura do k=3 direto)
CORES_PERFIL = {"Adiantado": SERIES[0], "Tardio": SERIES[1],
                "Equilibrado": SERIES[2]}


def centro_de_massa(centroides):
    centros_bin = (np.arange(N_BINS) + 0.5) / N_BINS
    return centroides @ centros_bin


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Carregando base de ciclos...")
    diario = carregar_vle_diario()
    tmas, submissores = montar_ciclos()
    X, meta, _ = construir_matriz(diario, tmas, submissores)
    print(f"  {X.shape[0]:,} ciclos x {X.shape[1]} bins")

    # ---- etapa 1: k=2 global
    km2 = KMeans(n_clusters=2, n_init=50, random_state=SEED).fit(X)
    com2 = centro_de_massa(km2.cluster_centers_)
    cl_tardio = int(np.argmax(com2))
    mask_resto = km2.labels_ != cl_tardio
    n_tardio = int((~mask_resto).sum())
    print(f"  k=2: tardio n={n_tardio}, resto n={mask_resto.sum()}")

    # ---- etapa 2: varredura de k dentro do resto
    X0 = X[mask_resto]
    linhas = []
    for k in SUB_K_RANGE:
        km = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(X0)
        linhas.append({"k": k, "inercia": km.inertia_,
                       "silhueta_media": silhouette_score(X0, km.labels_)})
    met = pd.DataFrame(linhas)
    met.to_csv(os.path.join(OUT, "metricas_sub_k.csv"), index=False)
    melhor_sub_k = int(met.loc[met["silhueta_media"].idxmax(), "k"])
    print(met.to_string(index=False))
    print(f"  melhor sub-k dentro do resto: {melhor_sub_k}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(met["k"], met["inercia"], color=SERIES[0], linewidth=2,
                 marker="o", markersize=7)
    axes[0].set_xlabel("sub-k (dentro do cluster restante)")
    axes[0].set_ylabel("inércia (WCSS)")
    axes[0].set_title("Cotovelo — sub-clusterização do resto")
    axes[1].plot(met["k"], met["silhueta_media"], color=SERIES[0], linewidth=2,
                 marker="o", markersize=7)
    axes[1].axvline(melhor_sub_k, color=SERIES[2], linewidth=2, linestyle="--")
    axes[1].annotate(f"melhor sub-k = {melhor_sub_k}",
                     (melhor_sub_k, met["silhueta_media"].max()),
                     textcoords="offset points", xytext=(10, -4),
                     color=SEC, fontsize=9)
    axes[1].set_xlabel("sub-k (dentro do cluster restante)")
    axes[1].set_ylabel("silhueta média")
    axes[1].set_title("Silhueta — sub-clusterização do resto")
    fig.savefig(os.path.join(OUT, "selecao_sub_k.png"))
    plt.close(fig)

    # ---- etapa 3: solucao combinada (tardio congelado + sub-k=2 no resto)
    km_sub = KMeans(n_clusters=2, n_init=50, random_state=SEED).fit(X0)
    rot_comb = np.full(len(X), -1, dtype=int)
    rot_comb[~mask_resto] = 2                     # tardio congelado
    rot_comb[mask_resto] = km_sub.labels_

    centroides = np.vstack([km_sub.cluster_centers_,
                            km2.cluster_centers_[cl_tardio]])
    com = centro_de_massa(centroides)
    ordem = np.argsort(com)
    nomes_por_pos = ["Adiantado", "Equilibrado", "Tardio"]
    nomes = {int(cl): nomes_por_pos[pos] for pos, cl in enumerate(ordem)}

    sil_comb = silhouette_score(X, rot_comb)
    print(f"  silhueta global da solução combinada: {sil_comb:.4f}")

    # centroides da solucao combinada
    x_pos = np.arange(N_BINS)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cl in range(3):
        cor = CORES_PERFIL[nomes[cl]]
        n = int((rot_comb == cl).sum())
        ax.plot(x_pos, centroides[cl], color=cor, linewidth=2,
                marker="o", markersize=7)
        ax.annotate(f"{nomes[cl]}  (n={n})", (x_pos[-1], centroides[cl][-1]),
                    textcoords="offset points", xytext=(10, 0),
                    color=cor, fontsize=9, fontweight="bold", va="center")
    ax.set_xticks(x_pos, BIN_LABELS)
    ax.set_xlim(-0.3, N_BINS - 0.2)
    ax.set_ylabel("proporção do esforço no ciclo")
    ax.set_xlabel("momento do ciclo de avaliação (tempo relativo)")
    ax.set_title("Curvas-tipo — solução divisiva (k=2 → sub-k=2)")
    fig.subplots_adjust(right=0.78)
    fig.savefig(os.path.join(OUT, "centroides_divisivo.png"))
    plt.close(fig)

    # ---- etapa 4: comparacao com o k=3 direto
    km3 = KMeans(n_clusters=3, n_init=50, random_state=SEED).fit(X)
    com3 = centro_de_massa(km3.cluster_centers_)
    ordem3 = np.argsort(com3)
    nomes3 = {int(cl): nomes_por_pos[pos] for pos, cl in enumerate(ordem3)}
    sil_k3 = silhouette_score(X, km3.labels_)

    ari = adjusted_rand_score(km3.labels_, rot_comb)
    perfil_direto = pd.Series(km3.labels_).map(nomes3)
    perfil_divisivo = pd.Series(rot_comb).map(nomes)
    cruz = pd.crosstab(perfil_direto.rename("k3 direto"),
                       perfil_divisivo.rename("divisivo"))
    cruz.to_csv(os.path.join(OUT, "cruzamento_direto_x_divisivo.csv"))
    concord = (perfil_direto.values == perfil_divisivo.values).mean()

    atrib = meta.copy()
    atrib["perfil_divisivo"] = perfil_divisivo.values
    atrib["perfil_k3_direto"] = perfil_direto.values
    atrib.to_csv(os.path.join(OUT, "atribuicoes_ciclos.csv"), index=False)

    with open(os.path.join(OUT, "resumo.txt"), "w", encoding="utf-8") as f:
        f.write("SOLUÇÃO DIVISIVA (k=2 global -> k-means no cluster restante)\n\n")
        f.write(f"Etapa 1 (k=2): tardio n={n_tardio}, resto n={int(mask_resto.sum())}\n")
        f.write(f"Etapa 2: varredura dentro do resto -> melhor sub-k = {melhor_sub_k}\n")
        f.write(met.to_string(index=False) + "\n\n")
        f.write("Centróides da solução combinada:\n")
        for cl in range(3):
            v = ", ".join(f"{p:.3f}" for p in centroides[cl])
            f.write(f"  {nomes[cl]}: [{v}]  n={int((rot_comb == cl).sum())}\n")
        f.write(f"\nSilhueta global: divisiva = {sil_comb:.4f} | "
                f"k=3 direto = {sil_k3:.4f}\n")
        f.write(f"ARI (divisiva vs k=3 direto) = {ari:.4f} | "
                f"concordância de rótulos = {concord:.1%}\n\n")
        f.write("Cruzamento k=3 direto x divisiva (nº de ciclos):\n")
        f.write(cruz.to_string())

    print(f"  ARI vs k=3 direto: {ari:.4f} | concordância: {concord:.1%}")
    print(f"Concluído. Resultados em: {OUT}")


if __name__ == "__main__":
    main()
