# -*- coding: utf-8 -*-
"""
Hierarquico variando a MEDIDA DE DISTANCIA entre pontos (complemento da Q3 e
espelho do comparativo k-medoids).

Para cada metrica (euclidiana, Manhattan, cosseno, Chebyshev):
  - linkages completo, medio e simples (nativos em qualquer metrica);
  - Ward apenas onde e valido: euclidiana pura e, para o cosseno, Ward sobre
    os vetores normalizados em L2 (distancia de corda, monotonica ao cosseno);
  - cofenetica, silhueta (na propria metrica) para cortes k=2..8, dendrogramas;
  - detalhe do corte k=3 do melhor linkage UTILIZAVEL (grupos nao degenerados)
    com curvas medias, facas e ARI contra o k-means euclidiano k=3 (baseline).

Saida: resultados_hierarquico_distancias/
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, adjusted_rand_score

import sys
_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_RAIZ, "05_kmeans_curvas_ciclo"))
sys.path.insert(0, os.path.join(_RAIZ, "07_kmedoids_distancias"))
import kmeans_ciclos as kc
from kmeans_ciclos import BASE_DIR, SEED, SURF, INK, SEC, MUT, BASE, SERIES
from distancias import (METRICAS, NOME_METRICA, PASTA_METRICA, COR_METRICA,
                        plot_curvas, plot_facas)

K_RANGE = range(2, 9)
K_FOCO = 3
NOME_LINKAGE = {"ward": "Ward", "ward_corda": "Ward (corda)",
                "complete": "Completo", "average": "Médio", "single": "Simples"}
COR_LINKAGE = {"ward": SERIES[0], "ward_corda": SERIES[1],
               "complete": SERIES[5], "average": SERIES[2],
               "single": SERIES[4]}


def linkages_da_metrica(met):
    lk = ["complete", "average", "single"]
    if met == "euclidean":
        lk = ["ward"] + lk
    if met == "cosine":
        lk = ["ward_corda"] + lk
    return lk


def main():
    parser = argparse.ArgumentParser(description="Hierárquico x métrica de distância")
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    kc.N_BINS = args.bins
    kc.BIN_LABELS = kc.rotulos_bins(args.bins)
    out_raiz = args.out or ("4bins" if args.bins == 4 else f"{args.bins}bins")
    out_raiz = (out_raiz if os.path.isabs(out_raiz)
                else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  out_raiz))
    os.makedirs(out_raiz, exist_ok=True)

    print(f"Configuração: {args.bins} bins -> {out_raiz}")
    diario = kc.carregar_vle_diario()
    tmas, submissores = kc.montar_ciclos()
    X, meta, _ = kc.construir_matriz(diario, tmas, submissores)
    n = len(X)
    limiar_util = max(50, int(0.05 * n))   # menor grupo aceitavel no k=3
    print(f"  matriz {n:,} ciclos x {X.shape[1]} bins "
          f"(grupo útil no k=3: >= {limiar_util})")

    km3 = KMeans(n_clusters=K_FOCO, n_init=50, random_state=SEED).fit(X)

    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)   # p/ Ward (corda)

    todas, destaque = [], []
    for met in METRICAS:
        pasta = os.path.join(out_raiz, PASTA_METRICA[met])
        os.makedirs(pasta, exist_ok=True)
        print(f"  {NOME_METRICA[met]}...")
        D = pdist(X, metric=met)
        Dq = squareform(D)
        lks = linkages_da_metrica(met)

        Zs, registros = {}, []
        for lk in lks:
            if lk == "ward":
                Z = hierarchy.linkage(D, method="ward")
            elif lk == "ward_corda":
                Z = hierarchy.linkage(pdist(Xn), method="ward")
            else:
                Z = hierarchy.linkage(D, method=lk)
            Zs[lk] = Z
            coph = hierarchy.cophenet(Z, D)[0]
            for k in K_RANGE:
                rot = hierarchy.fcluster(Z, k, criterion="maxclust")
                n_grupos = len(np.unique(rot))
                if n_grupos < 2:
                    sil = np.nan
                else:
                    sil = silhouette_samples(Dq, rot,
                                             metric="precomputed").mean()
                tam = np.sort(np.bincount(rot)[1:])[::-1]
                registros.append({"metrica": met, "linkage": lk, "k": k,
                                  "silhueta_media": sil, "cofenetica": coph,
                                  "n_grupos_obtidos": n_grupos,
                                  "maior_grupo": int(tam[0]),
                                  "menor_grupo": int(tam[-1])})
        tab = pd.DataFrame(registros)
        tab.to_csv(os.path.join(pasta, "metricas_linkage_k.csv"), index=False)
        todas.append(tab)

        # silhueta por k, uma linha por linkage
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for lk in lks:
            sel = tab[tab["linkage"] == lk]
            ax.plot(sel["k"], sel["silhueta_media"], color=COR_LINKAGE[lk],
                    linewidth=2, marker="o", markersize=6,
                    label=NOME_LINKAGE[lk])
        ax.set_xlabel("número de grupos (corte)")
        ax.set_ylabel("silhueta média (na própria métrica)")
        ax.set_title(f"Silhueta por linkage — {NOME_METRICA[met]}")
        leg = ax.legend(frameon=False, fontsize=9)
        for t in leg.get_texts():
            t.set_color(SEC)
        fig.savefig(os.path.join(pasta, "silhueta_por_linkage.png"))
        plt.close(fig)

        # dendrogramas
        hierarchy.set_link_color_palette([SERIES[0], SERIES[1], SERIES[2],
                                          SERIES[4], SERIES[5]])
        ncols = 2
        nrows = -(-len(lks) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
        for ax, lk in zip(np.atleast_1d(axes).flat, lks):
            Z = Zs[lk]
            limiar = Z[-(K_FOCO - 1), 2]
            hierarchy.dendrogram(Z, ax=ax, truncate_mode="lastp", p=40,
                                 color_threshold=limiar,
                                 above_threshold_color=BASE, no_labels=True)
            ax.axhline(limiar, color=MUT, linewidth=1, linestyle="--")
            coph = tab.loc[tab["linkage"] == lk, "cofenetica"].iloc[0]
            ax.set_title(f"{NOME_LINKAGE[lk]}  (cofenética = {coph:.3f})",
                         fontsize=11)
            ax.set_ylabel("distância de fusão", fontsize=9)
            ax.grid(visible=False)
            ax.tick_params(labelsize=8)
        for ax in np.atleast_1d(axes).flat[len(lks):]:
            ax.set_visible(False)
        fig.suptitle(f"Dendrogramas — {NOME_METRICA[met]} "
                     f"(corte em {K_FOCO} tracejado)",
                     fontweight="bold", color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(pasta, "dendrogramas.png"))
        plt.close(fig)
        hierarchy.set_link_color_palette(None)

        # melhor linkage UTILIZAVEL no k=3 (sem grupos degenerados)
        k3 = tab[(tab["k"] == K_FOCO) &
                 (tab["n_grupos_obtidos"] == K_FOCO) &
                 (tab["menor_grupo"] >= limiar_util)]
        with open(os.path.join(pasta, "resumo.txt"), "w",
                  encoding="utf-8") as f:
            f.write(f"HIERÁRQUICO — {NOME_METRICA[met]} ({args.bins} bins)\n\n")
            f.write("Silhueta por linkage e k:\n")
            f.write(tab.pivot(index="k", columns="linkage",
                              values="silhueta_media").round(4).to_string())
            f.write("\n\nk=3 por linkage (tamanhos):\n")
            for lk in lks:
                sel = tab[(tab["linkage"] == lk) & (tab["k"] == K_FOCO)]
                f.write(f"  {NOME_LINKAGE[lk]:<12}: grupos obtidos="
                        f"{int(sel['n_grupos_obtidos'].iloc[0])} "
                        f"menor={int(sel['menor_grupo'].iloc[0]):,} "
                        f"maior={int(sel['maior_grupo'].iloc[0]):,}\n")
            if len(k3) == 0:
                f.write(f"\nNenhum linkage produziu 3 grupos utilizáveis "
                        f"(menor >= {limiar_util}).\n")
            else:
                lk_best = k3.sort_values("silhueta_media",
                                         ascending=False)["linkage"].iloc[0]
                rot = hierarchy.fcluster(Zs[lk_best], K_FOCO,
                                         criterion="maxclust") - 1
                sam = silhouette_samples(Dq, rot, metric="precomputed")
                medias, nomes = plot_curvas(
                    X, rot, K_FOCO, os.path.join(pasta, "curvas_k3.png"),
                    f"Curvas médias — {NOME_LINKAGE[lk_best]}, "
                    f"{NOME_METRICA[met]}, k=3")
                plot_facas(sam, rot, K_FOCO, nomes,
                           os.path.join(pasta, "silhueta_facas_k3.png"),
                           f"Silhueta — {NOME_LINKAGE[lk_best]}, "
                           f"{NOME_METRICA[met]}, k=3")
                ari = adjusted_rand_score(km3.labels_, rot)
                f.write(f"\nMelhor linkage utilizável no k=3: "
                        f"{NOME_LINKAGE[lk_best]} "
                        f"(sil={sam.mean():.4f}, ARI vs baseline={ari:.4f})\n")
                f.write("Curvas médias:\n")
                for cl in range(K_FOCO):
                    v = ", ".join(f"{p:.3f}" for p in medias[cl])
                    f.write(f"  {nomes[cl]}: [{v}]  "
                            f"n={int((rot == cl).sum()):,}\n")
                destaque.append({"metrica": NOME_METRICA[met],
                                 "linkage": NOME_LINKAGE[lk_best],
                                 "silhueta_k3": sam.mean(),
                                 "ari_vs_baseline": ari,
                                 "menor_grupo": int(np.bincount(rot).min())})
        del D, Dq, Zs

    tudo = pd.concat(todas, ignore_index=True)
    tudo.to_csv(os.path.join(out_raiz, "metricas_hierarquico.csv"), index=False)

    # ---- comparativo raiz: silhueta k=3 por metrica x linkage
    k3 = tudo[tudo["k"] == K_FOCO].copy()
    k3["util"] = ((k3["n_grupos_obtidos"] == K_FOCO) &
                  (k3["menor_grupo"] >= limiar_util))
    fig, ax = plt.subplots(figsize=(11, 5))
    largura = 0.16
    xs = np.arange(len(METRICAS))
    ordem_lk = ["ward", "ward_corda", "complete", "average", "single"]
    for j, lk in enumerate(ordem_lk):
        vals, pos, utils = [], [], []
        for i, met in enumerate(METRICAS):
            sel = k3[(k3["metrica"] == met) & (k3["linkage"] == lk)]
            if len(sel):
                vals.append(sel["silhueta_media"].iloc[0])
                pos.append(i + (j - 2) * largura)
                utils.append(bool(sel["util"].iloc[0]))
        if not vals:
            continue
        for p, v, u in zip(pos, vals, utils):
            ax.bar(p, 0 if np.isnan(v) else v, width=largura * 0.92,
                   color=COR_LINKAGE[lk], alpha=1.0 if u else 0.30,
                   edgecolor=SURF, linewidth=1)
        ax.bar(np.nan, np.nan, color=COR_LINKAGE[lk],
               label=NOME_LINKAGE[lk])
    ax.set_xticks(xs, [NOME_METRICA[m] for m in METRICAS])
    ax.set_ylabel("silhueta média no k=3 (na própria métrica)")
    ax.set_title("Hierárquico k=3 — silhueta por métrica e linkage "
                 "(barras apagadas = partição degenerada)")
    ax.grid(axis="x", visible=False)
    leg = ax.legend(frameon=False, fontsize=9, ncols=5)
    for t in leg.get_texts():
        t.set_color(SEC)
    fig.savefig(os.path.join(out_raiz, "comparativo_silhueta_k3.png"))
    plt.close(fig)

    resumo = pd.DataFrame(destaque)
    with open(os.path.join(out_raiz, "resumo_geral.txt"), "w",
              encoding="utf-8") as f:
        f.write(f"HIERÁRQUICO x MÉTRICA ({args.bins} bins)\n")
        f.write(f"Grupo utilizável no k=3: menor grupo >= {limiar_util}\n")
        f.write("Baseline: k-means euclidiano k=3.\n\n")
        f.write("Melhor linkage utilizável por métrica (k=3):\n")
        f.write(resumo.round(4).to_string(index=False) if len(resumo)
                else "  (nenhum)")
    if len(resumo):
        print(resumo.round(3).to_string(index=False))
    print(f"Concluído. Resultados em: {out_raiz}")


if __name__ == "__main__":
    main()
