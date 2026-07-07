# -*- coding: utf-8 -*-
"""
Agrupamento hierarquico sobre as mesmas curvas de ciclo do k-means (Lista 2, Q3).

Experimentos:
  1. Quatro linkages (Ward, completo, medio, simples) sobre a mesma matriz.
  2. Correlacao cofenetica por linkage; silhueta para cortes k=2..8.
  3. Dendrogramas (truncados) com corte em 3 grupos destacado.
  4. Detalhe do Ward k=3: curvas medias dos grupos + comparacao com o
     k-means k=3 (ARI, concordancia de rotulos, tabela de cruzamento).

Uso:
  python hierarquico.py                 # 4 bins -> resultados_hierarquico/
  python hierarquico.py --bins 8        # -> resultados_hierarquico_8bins/
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score

import kmeans_ciclos as kc
from kmeans_ciclos import (BASE_DIR, SEED, SURF, INK, SEC, MUT, GRID, BASE,
                           SERIES, CORES_PERFIL)

LINKAGES = ["ward", "complete", "average", "single"]
NOME_LINKAGE = {"ward": "Ward", "complete": "Completo",
                "average": "Médio", "single": "Simples"}
K_RANGE = range(2, 9)
K_FOCO = 3


def nomear_por_com(medias):
    """Mesma regra do k-means: nomeia grupos pelo centro de massa temporal."""
    return kc.nomear_perfis(medias)


def main():
    parser = argparse.ArgumentParser(description="Hierárquico sobre curvas de ciclo")
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    kc.N_BINS = args.bins
    kc.BIN_LABELS = kc.rotulos_bins(args.bins)
    sufixo = "" if args.bins == 4 else f"_{args.bins}bins"
    out = args.out or f"resultados_hierarquico{sufixo}"
    out = out if os.path.isabs(out) else os.path.join(BASE_DIR, out)
    os.makedirs(out, exist_ok=True)

    print(f"Configuração: {args.bins} bins -> {out}")
    print("Construindo a mesma matriz de curvas do k-means...")
    diario = kc.carregar_vle_diario()
    tmas, submissores = kc.montar_ciclos()
    X, meta, _ = kc.construir_matriz(diario, tmas, submissores)
    print(f"  matriz {X.shape[0]:,} ciclos x {X.shape[1]} bins")

    D = pdist(X)   # distancia euclidiana condensada (mesma metrica do k-means)

    # ---- experimento 1: linkages x k
    linhas, Zs = [], {}
    for met in LINKAGES:
        Z = hierarchy.linkage(D, method=met)
        Zs[met] = Z
        coph = hierarchy.cophenet(Z, D)[0]
        for k in K_RANGE:
            rot = hierarchy.fcluster(Z, k, criterion="maxclust")
            if len(np.unique(rot)) < 2:
                sil = np.nan
            else:
                sil = silhouette_score(X, rot)
            tamanhos = np.sort(np.bincount(rot)[1:])[::-1]
            linhas.append({"linkage": met, "k": k, "silhueta_media": sil,
                           "cofenetica": coph,
                           "maior_grupo": int(tamanhos[0]),
                           "menor_grupo": int(tamanhos[-1])})
        print(f"  {NOME_LINKAGE[met]}: cofenética = {coph:.3f}")
    tab = pd.DataFrame(linhas)
    tab.to_csv(os.path.join(out, "metricas_linkage_k.csv"), index=False)

    # silhueta por k, uma linha por linkage
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, met in enumerate(LINKAGES):
        sel = tab[tab["linkage"] == met]
        ax.plot(sel["k"], sel["silhueta_media"], color=SERIES[i], linewidth=2,
                marker="o", markersize=6)
        ax.annotate(NOME_LINKAGE[met],
                    (sel["k"].iloc[-1], sel["silhueta_media"].iloc[-1]),
                    textcoords="offset points", xytext=(10, 0),
                    color=SERIES[i], fontsize=9, fontweight="bold", va="center")
    ax.set_xlabel("número de grupos (corte do dendrograma)")
    ax.set_ylabel("silhueta média")
    ax.set_title(f"Silhueta por linkage — curvas de ciclo ({args.bins} bins)")
    fig.subplots_adjust(right=0.82)
    fig.savefig(os.path.join(out, "silhueta_por_linkage.png"))
    plt.close(fig)

    # ---- experimento 2: dendrogramas truncados, corte em 3 destacado
    hierarchy.set_link_color_palette([SERIES[0], SERIES[1], SERIES[2],
                                      SERIES[4], SERIES[5]])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, met in zip(axes.flat, LINKAGES):
        Z = Zs[met]
        limiar = Z[-(K_FOCO - 1), 2]          # abaixo disso = K_FOCO subarvores
        hierarchy.dendrogram(Z, ax=ax, truncate_mode="lastp", p=40,
                             color_threshold=limiar,
                             above_threshold_color=BASE,
                             no_labels=True)
        ax.axhline(limiar, color=MUT, linewidth=1, linestyle="--")
        coph = tab.loc[tab["linkage"] == met, "cofenetica"].iloc[0]
        ax.set_title(f"{NOME_LINKAGE[met]}  (cofenética = {coph:.3f})", fontsize=11)
        ax.set_ylabel("distância de fusão", fontsize=9)
        ax.grid(visible=False)
        ax.tick_params(labelsize=8)
    fig.suptitle(f"Dendrogramas por linkage — corte em {K_FOCO} grupos tracejado "
                 f"({args.bins} bins)", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "dendrogramas.png"))
    plt.close(fig)
    hierarchy.set_link_color_palette(None)

    # ---- experimento 3: detalhe do Ward k=3 e comparacao com o k-means
    rot_ward = hierarchy.fcluster(Zs["ward"], K_FOCO, criterion="maxclust") - 1
    medias = np.vstack([X[rot_ward == c].mean(axis=0) for c in range(K_FOCO)])
    nomes, com = nomear_por_com(medias)
    sil_ward = silhouette_score(X, rot_ward)
    sam_ward = silhouette_samples(X, rot_ward)

    def cor_grupo(cl):
        return CORES_PERFIL.get(nomes[cl], SERIES[cl % len(SERIES)])

    # curvas medias dos grupos do Ward
    x_pos = np.arange(args.bins)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cl in range(K_FOCO):
        n = int((rot_ward == cl).sum())
        ax.plot(x_pos, medias[cl], color=cor_grupo(cl), linewidth=2,
                marker="o", markersize=6 if args.bins <= 6 else 4)
        ax.annotate(f"{nomes[cl]}  (n={n:,})", (x_pos[-1], medias[cl][-1]),
                    textcoords="offset points", xytext=(10, 0),
                    color=cor_grupo(cl), fontsize=9, fontweight="bold",
                    va="center")
    ax.set_xticks(x_pos, kc.BIN_LABELS,
                  fontsize=9 if args.bins <= 6 else 8,
                  rotation=0 if args.bins <= 6 else 45)
    ax.set_xlim(-0.3, args.bins - 0.2)
    ax.set_ylabel("proporção do esforço no ciclo")
    ax.set_xlabel("momento do ciclo de avaliação (tempo relativo)")
    ax.set_title(f"Curvas médias dos grupos — Ward, corte em {K_FOCO}")
    fig.subplots_adjust(right=0.78)
    fig.savefig(os.path.join(out, "curvas_ward_k3.png"))
    plt.close(fig)

    # facas do Ward k=3
    fig, ax = plt.subplots(figsize=(6.5, 5))
    y0 = 10
    for cl in range(K_FOCO):
        vals = np.sort(sam_ward[rot_ward == cl])
        ax.fill_betweenx(np.arange(y0, y0 + len(vals)), 0, vals,
                         facecolor=cor_grupo(cl), edgecolor=cor_grupo(cl))
        ax.text(-0.03, y0 + len(vals) / 2, f"{nomes[cl]}\n(n={len(vals):,})",
                ha="right", va="center", fontsize=9, color=SEC)
        y0 += len(vals) + 40
    ax.axvline(sil_ward, color=INK, linewidth=1.2, linestyle="--")
    ax.annotate(f"média = {sil_ward:.3f}", (sil_ward, y0),
                textcoords="offset points", xytext=(6, -12),
                fontsize=9, color=SEC)
    ax.set_yticks([])
    ax.set_xlabel("coeficiente de silhueta")
    ax.set_title(f"Diagrama de silhueta — Ward, corte em {K_FOCO}")
    ax.grid(axis="y", visible=False)
    fig.savefig(os.path.join(out, "silhueta_facas_ward_k3.png"))
    plt.close(fig)

    # comparacao com o k-means k=3 (mesma matriz, mesma semente do pipeline)
    km3 = KMeans(n_clusters=K_FOCO, n_init=50, random_state=SEED).fit(X)
    nomes_km, _ = kc.nomear_perfis(km3.cluster_centers_)
    sil_km = silhouette_score(X, km3.labels_)
    ari = adjusted_rand_score(km3.labels_, rot_ward)
    perfil_km = pd.Series(km3.labels_).map(nomes_km)
    perfil_ward = pd.Series(rot_ward).map(nomes)
    concord = (perfil_km.values == perfil_ward.values).mean()
    cruz = pd.crosstab(perfil_km.rename("k-means k=3"),
                       perfil_ward.rename("Ward k=3"))
    cruz.to_csv(os.path.join(out, "cruzamento_kmeans_x_ward.csv"))

    atrib = meta.copy()
    atrib["grupo_ward"] = perfil_ward.values
    atrib["grupo_kmeans"] = perfil_km.values
    atrib.to_csv(os.path.join(out, "atribuicoes_ciclos.csv"), index=False)

    with open(os.path.join(out, "resumo.txt"), "w", encoding="utf-8") as f:
        f.write(f"HIERÁRQUICO sobre as curvas de ciclo ({args.bins} bins)\n\n")
        f.write("Correlação cofenética por linkage:\n")
        for met in LINKAGES:
            coph = tab.loc[tab["linkage"] == met, "cofenetica"].iloc[0]
            f.write(f"  {NOME_LINKAGE[met]:<9}: {coph:.4f}\n")
        f.write("\nSilhueta média por linkage e k:\n")
        piv = tab.pivot(index="k", columns="linkage", values="silhueta_media")
        f.write(piv.round(4).to_string())
        f.write("\n\nTamanho do menor grupo (k=3) por linkage:\n")
        for met in LINKAGES:
            sel = tab[(tab["linkage"] == met) & (tab["k"] == K_FOCO)]
            f.write(f"  {NOME_LINKAGE[met]:<9}: menor={int(sel['menor_grupo'].iloc[0]):,} "
                    f"maior={int(sel['maior_grupo'].iloc[0]):,}\n")
        f.write(f"\nWard k={K_FOCO}: silhueta = {sil_ward:.4f} "
                f"(k-means k={K_FOCO}: {sil_km:.4f})\n")
        f.write("Curvas médias (Ward):\n")
        for cl in range(K_FOCO):
            v = ", ".join(f"{p:.3f}" for p in medias[cl])
            f.write(f"  {nomes[cl]}: [{v}]  n={int((rot_ward == cl).sum())}\n")
        f.write(f"\nARI (k-means x Ward) = {ari:.4f} | "
                f"concordância de rótulos = {concord:.1%}\n\n")
        f.write("Cruzamento k-means x Ward (nº de ciclos):\n")
        f.write(cruz.to_string())

    print(f"  Ward k=3: silhueta {sil_ward:.4f} | k-means k=3: {sil_km:.4f}")
    print(f"  ARI k-means x Ward: {ari:.4f} | concordância: {concord:.1%}")
    print(f"Concluído. Resultados em: {out}")


if __name__ == "__main__":
    main()
