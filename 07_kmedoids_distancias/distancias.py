# -*- coding: utf-8 -*-
"""
Variacao da medida de distancia com K-MEDOIDS (PAM), no mesmo padrao do
pipeline k-means (Lista 2, Q2: variar a medida de distancia/similaridade).

O k-means classico e euclidiano por construcao (a media so minimiza soma de
quadrados euclidiana). Para variar a metrica corretamente usamos o K-MEDOIDS,
implementado aqui do zero sobre a matriz de distancias pre-computada: o
representante de cada grupo e um ciclo real (medoide).

Para CADA metrica (euclidiana, Manhattan, cosseno, Chebyshev), curvas de
4 bins, replica-se o desenho do k-means:
  <metrica>/selecao_k/   cotovelo, silhueta media (melhor k marcado) e
                         grade de facas por k
  <metrica>/k3/          solucao confirmatoria com 3 grupos
  <metrica>/divisivo/    divide o k duas vezes (k=2 -> sub-k=2)
  <metrica>/resumo.txt

Na raiz: comparativo entre metricas + ARI de cada solucao k=3 contra o
k-means euclidiano k=3 (baseline do pipeline principal).

Uso:
  python distancias.py            # 4 bins -> resultados_distancias/
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, adjusted_rand_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "05_kmeans_curvas_ciclo"))
import kmeans_ciclos as kc
from kmeans_ciclos import BASE_DIR, SEED, SURF, INK, SEC, MUT, SERIES, CORES_PERFIL

METRICAS = ["euclidean", "cityblock", "cosine", "chebyshev"]
NOME_METRICA = {"euclidean": "Euclidiana", "cityblock": "Manhattan",
                "cosine": "Cosseno", "chebyshev": "Chebyshev"}
PASTA_METRICA = {"euclidean": "euclidiana", "cityblock": "manhattan",
                 "cosine": "cosseno", "chebyshev": "chebyshev"}
COR_METRICA = {"euclidean": SERIES[0], "cityblock": SERIES[1],
               "cosine": SERIES[2], "chebyshev": SERIES[4]}
K_RANGE = range(2, 9)
SUB_K_RANGE = range(2, 7)
K_FOCO = 3
N_INIT = 5
MAX_ITER = 30


# ---------------------------------------------------------------- k-medoids
def kmedoids(Dm, k, seed, n_init=N_INIT, max_iter=MAX_ITER):
    """PAM alternado sobre matriz de distancias pre-computada."""
    n = Dm.shape[0]
    melhor = None
    for init in range(n_init):
        rng = np.random.default_rng(seed + init)
        med = [int(rng.integers(n))]
        for _ in range(k - 1):          # init estilo k-means++
            dmin = Dm[:, med].min(axis=1).astype(np.float64)
            p = dmin ** 2
            s = p.sum()
            med.append(int(rng.choice(n, p=p / s)) if s > 0
                       else int(rng.integers(n)))
        med = np.array(med)
        for _ in range(max_iter):
            rot = np.argmin(Dm[:, med], axis=1)
            novo = med.copy()
            for c in range(k):
                idx = np.where(rot == c)[0]
                if len(idx):
                    sub = Dm[np.ix_(idx, idx)]
                    novo[c] = idx[np.argmin(sub.sum(axis=1))]
            if np.array_equal(np.sort(novo), np.sort(med)):
                med = novo
                break
            med = novo
        rot = np.argmin(Dm[:, med], axis=1)
        custo = float(Dm[np.arange(n), med[rot]].sum())
        if melhor is None or custo < melhor[0]:
            melhor = (custo, med.copy(), rot.copy())
    return melhor


# ---------------------------------------------------------------- graficos
def plot_curvas(X, rot, k, out_png, titulo):
    """Curvas medias dos grupos, cores fixas por perfil. Devolve nomes."""
    medias = np.vstack([X[rot == c].mean(axis=0) for c in range(k)])
    nomes, com = kc.nomear_perfis(medias)
    x_pos = np.arange(X.shape[1])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cl in range(k):
        cor = CORES_PERFIL.get(nomes[cl], SERIES[cl % len(SERIES)])
        n = int((rot == cl).sum())
        ax.plot(x_pos, medias[cl], color=cor, linewidth=2, marker="o",
                markersize=6)
        ax.annotate(f"{nomes[cl]}  (n={n:,})", (x_pos[-1], medias[cl][-1]),
                    textcoords="offset points", xytext=(10, 0), color=cor,
                    fontsize=9, fontweight="bold", va="center")
    ax.set_xticks(x_pos, kc.BIN_LABELS)
    ax.set_xlim(-0.3, X.shape[1] - 0.2)
    ax.set_ylabel("proporção do esforço no ciclo")
    ax.set_xlabel("momento do ciclo de avaliação (tempo relativo)")
    ax.set_title(titulo)
    fig.subplots_adjust(right=0.76)
    fig.savefig(out_png)
    plt.close(fig)
    return medias, nomes


def plot_facas(sam, rot, k, nomes, out_png, titulo):
    sil = sam.mean()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    y0 = 10
    for cl in range(k):
        vals = np.sort(sam[rot == cl])
        cor = CORES_PERFIL.get(nomes.get(cl, ""), SERIES[cl % len(SERIES)])
        ax.fill_betweenx(np.arange(y0, y0 + len(vals)), 0, vals,
                         facecolor=cor, edgecolor=cor)
        ax.text(-0.03, y0 + len(vals) / 2,
                f"{nomes.get(cl, f'C{cl}')}\n(n={len(vals):,})",
                ha="right", va="center", fontsize=9, color=SEC)
        y0 += len(vals) + 40
    ax.axvline(sil, color=INK, linewidth=1.2, linestyle="--")
    ax.annotate(f"média = {sil:.3f}", (sil, y0), textcoords="offset points",
                xytext=(6, -12), fontsize=9, color=SEC)
    ax.set_yticks([])
    ax.set_xlabel("coeficiente de silhueta (na própria métrica)")
    ax.set_title(titulo)
    ax.grid(axis="y", visible=False)
    fig.savefig(out_png)
    plt.close(fig)


# ---------------------------------------------------------------- etapas
def varredura(Dm, out_sel, rotulo):
    """Varredura de k: cotovelo, silhueta media e grade de facas por k."""
    os.makedirs(out_sel, exist_ok=True)
    linhas, ajustes = [], {}
    for k in K_RANGE:
        custo, med, rot = kmedoids(Dm, k, SEED)
        sam = silhouette_samples(Dm, rot, metric="precomputed")
        linhas.append({"k": k, "custo": custo, "silhueta_media": sam.mean()})
        ajustes[k] = (rot, med, sam)
    met = pd.DataFrame(linhas)
    met.to_csv(os.path.join(out_sel, "metricas_k.csv"), index=False)
    melhor_k = int(met.loc[met["silhueta_media"].idxmax(), "k"])

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(met["k"], met["custo"], color=SERIES[0], linewidth=2,
            marker="o", markersize=7)
    ax.set_xlabel("número de grupos (k)")
    ax.set_ylabel("custo total (soma das distâncias ao medoide)")
    ax.set_title(f"Método do cotovelo — k-medoids {rotulo}")
    fig.savefig(os.path.join(out_sel, "cotovelo.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(met["k"], met["silhueta_media"], color=SERIES[0], linewidth=2,
            marker="o", markersize=7)
    ax.axvline(melhor_k, color=SERIES[2], linewidth=2, linestyle="--")
    ax.annotate(f"melhor k = {melhor_k}",
                (melhor_k, met["silhueta_media"].max()),
                textcoords="offset points", xytext=(10, -2), color=SEC,
                fontsize=9)
    if melhor_k != K_FOCO:
        y3 = met.loc[met["k"] == K_FOCO, "silhueta_media"].iloc[0]
        ax.scatter([K_FOCO], [y3], s=90, facecolor="none",
                   edgecolor=SERIES[5], linewidth=2, zorder=5)
        ax.annotate("k = 3 (confirmatório)", (K_FOCO, y3),
                    textcoords="offset points", xytext=(10, 8), color=SEC,
                    fontsize=9)
    ax.set_xlabel("número de grupos (k)")
    ax.set_ylabel("silhueta média (na própria métrica)")
    ax.set_title(f"Análise de silhueta — k-medoids {rotulo}")
    fig.savefig(os.path.join(out_sel, "silhueta_media.png"))
    plt.close(fig)

    # grade de facas por k
    ncols = 4
    nrows = -(-len(K_RANGE) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows))
    for ax, k in zip(axes.flat, K_RANGE):
        rot, _, sam = ajustes[k]
        y0 = 10
        for cl in range(k):
            vals = np.sort(sam[rot == cl])
            cor = SERIES[cl % len(SERIES)]
            ax.fill_betweenx(np.arange(y0, y0 + len(vals)), 0, vals,
                             facecolor=cor, edgecolor=cor)
            ax.text(-0.04, y0 + len(vals) / 2, f"{len(vals):,}", ha="right",
                    va="center", fontsize=7, color=SEC)
            y0 += len(vals) + 60
        ax.axvline(sam.mean(), color=INK, linewidth=1, linestyle="--")
        ax.set_title(f"k={k}  (média={sam.mean():.3f})", fontsize=10)
        ax.set_yticks([])
        ax.set_xlim(-0.25, 0.85)
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(K_RANGE):]:
        ax.set_visible(False)
    fig.supxlabel("coeficiente de silhueta (na própria métrica)",
                  fontsize=10, color=SEC)
    fig.suptitle(f"Diagramas de silhueta por k — k-medoids {rotulo}",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out_sel, "silhueta_facas_todos_k.png"))
    plt.close(fig)

    return melhor_k, met, ajustes


def divisivo(X, Dm, ajustes, rot_k3, out_div, rotulo):
    """Divide o k duas vezes: k=2, depois sub-k dentro do grupo restante."""
    os.makedirs(out_div, exist_ok=True)
    rot2 = ajustes[2][0]
    medias2 = np.vstack([X[rot2 == c].mean(axis=0) for c in range(2)])
    _, com2 = kc.nomear_perfis(medias2)
    cl_sep = int(np.argmax(com2))               # grupo "tardio" separado
    mask = rot2 != cl_sep
    Dsub = Dm[np.ix_(mask, mask)]

    linhas = []
    sub_ajustes = {}
    for k in SUB_K_RANGE:
        custo, med, rot = kmedoids(Dsub, k, SEED)
        sam = silhouette_samples(Dsub, rot, metric="precomputed")
        linhas.append({"k": k, "custo": custo, "silhueta_media": sam.mean()})
        sub_ajustes[k] = rot
    met = pd.DataFrame(linhas)
    met.to_csv(os.path.join(out_div, "metricas_sub_k.csv"), index=False)
    melhor_sub = int(met.loc[met["silhueta_media"].idxmax(), "k"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(met["k"], met["custo"], color=SERIES[0], linewidth=2,
                 marker="o", markersize=7)
    axes[0].set_xlabel("sub-k (dentro do grupo restante)")
    axes[0].set_ylabel("custo total")
    axes[0].set_title("Cotovelo — sub-divisão do resto")
    axes[1].plot(met["k"], met["silhueta_media"], color=SERIES[0],
                 linewidth=2, marker="o", markersize=7)
    axes[1].axvline(melhor_sub, color=SERIES[2], linewidth=2, linestyle="--")
    axes[1].annotate(f"melhor sub-k = {melhor_sub}",
                     (melhor_sub, met["silhueta_media"].max()),
                     textcoords="offset points", xytext=(10, -4), color=SEC,
                     fontsize=9)
    axes[1].set_xlabel("sub-k (dentro do grupo restante)")
    axes[1].set_ylabel("silhueta média")
    axes[1].set_title("Silhueta — sub-divisão do resto")
    fig.suptitle(f"Divisivo (k=2 → sub-k) — k-medoids {rotulo}",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out_div, "selecao_sub_k.png"))
    plt.close(fig)

    rot_comb = np.full(len(X), 2, dtype=int)
    rot_comb[mask] = sub_ajustes[2]
    sam_comb = silhouette_samples(Dm, rot_comb, metric="precomputed")
    _, nomes_comb = plot_curvas(X, rot_comb, 3,
                                os.path.join(out_div, "curvas_divisivo.png"),
                                f"Curvas-tipo — divisivo (k=2 → sub-k=2), {rotulo}")
    ari = adjusted_rand_score(rot_k3, rot_comb)
    return melhor_sub, sam_comb.mean(), ari


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Variação de distância (k-medoids)")
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
    print(f"  matriz {X.shape[0]:,} ciclos x {X.shape[1]} bins")

    # baseline do pipeline principal: k-means euclidiano k=3
    km3 = KMeans(n_clusters=K_FOCO, n_init=50, random_state=SEED).fit(X)

    geral, curvas_k3, todas_met = [], {}, []
    for met in METRICAS:
        rotulo = f"[{NOME_METRICA[met]}]"
        print(f"  {NOME_METRICA[met]}: matriz de distâncias + varredura...")
        Dm = squareform(pdist(X, metric=met)).astype(np.float64)
        pasta = os.path.join(out_raiz, PASTA_METRICA[met])

        melhor_k, tab, ajustes = varredura(Dm, os.path.join(pasta, "selecao_k"),
                                           rotulo)
        tab["metrica"] = met
        todas_met.append(tab)

        # melhor k matematico
        rot_b, _, sam_b = ajustes[melhor_k]
        out_b = os.path.join(pasta, f"melhor_k_{melhor_k}")
        os.makedirs(out_b, exist_ok=True)
        _, nomes_b = plot_curvas(X, rot_b, melhor_k,
                                 os.path.join(out_b, "curvas.png"),
                                 f"Curvas-tipo (melhor k={melhor_k}) {rotulo}")
        plot_facas(sam_b, rot_b, melhor_k, nomes_b,
                   os.path.join(out_b, "silhueta_facas.png"),
                   f"Silhueta (k={melhor_k}) {rotulo}")

        # k=3 confirmatorio
        rot3, _, sam3 = ajustes[K_FOCO]
        out3 = os.path.join(pasta, "k3")
        os.makedirs(out3, exist_ok=True)
        medias3, nomes3 = plot_curvas(X, rot3, K_FOCO,
                                      os.path.join(out3, "curvas.png"),
                                      f"Curvas-tipo (k=3) {rotulo}")
        plot_facas(sam3, rot3, K_FOCO, nomes3,
                   os.path.join(out3, "silhueta_facas.png"),
                   f"Silhueta (k=3) {rotulo}")
        curvas_k3[met] = (medias3, nomes3, rot3, sam3.mean())

        # divisivo: divide o k duas vezes
        melhor_sub, sil_div, ari_div = divisivo(
            X, Dm, ajustes, rot3, os.path.join(pasta, "divisivo"), rotulo)

        ari_base = adjusted_rand_score(km3.labels_, rot3)
        tam3 = sorted(np.bincount(rot3), reverse=True)
        geral.append({"metrica": NOME_METRICA[met], "melhor_k": melhor_k,
                      "sil_melhor_k": ajustes[melhor_k][2].mean(),
                      "sil_k3": sam3.mean(), "melhor_sub_k": melhor_sub,
                      "sil_divisivo": sil_div, "ari_divisivo_vs_k3": ari_div,
                      "ari_k3_vs_baseline": ari_base,
                      "grupos_k3": str(tam3)})

        with open(os.path.join(pasta, "resumo.txt"), "w", encoding="utf-8") as f:
            f.write(f"K-MEDOIDS {NOME_METRICA[met]} ({args.bins} bins)\n\n")
            f.write(f"Melhor k pela silhueta: {melhor_k} "
                    f"(sil={ajustes[melhor_k][2].mean():.4f})\n")
            f.write(tab.drop(columns='metrica').to_string(index=False))
            f.write(f"\n\nk=3: grupos={tam3}, sil={sam3.mean():.4f}\n")
            f.write("Curvas médias (k=3):\n")
            for cl in range(K_FOCO):
                v = ", ".join(f"{p:.3f}" for p in medias3[cl])
                f.write(f"  {nomes3[cl]}: [{v}]\n")
            f.write(f"\nDivisivo: melhor sub-k={melhor_sub}, "
                    f"sil global={sil_div:.4f}, ARI vs k=3={ari_div:.4f}\n")
            f.write(f"ARI k=3 vs k-means euclidiano (baseline): {ari_base:.4f}\n")
        del Dm

    pd.concat(todas_met, ignore_index=True).to_csv(
        os.path.join(out_raiz, "metricas_kmedoids.csv"), index=False)
    resumo_geral = pd.DataFrame(geral)
    resumo_geral.to_csv(os.path.join(out_raiz, "comparativo_metricas.csv"),
                        index=False)

    # ---- comparativo raiz: silhueta por k e curvas k=3, por metrica
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    tudo = pd.concat(todas_met, ignore_index=True)
    for met in METRICAS:
        sel = tudo[tudo["metrica"] == met]
        base_c = sel.loc[sel["k"] == 2, "custo"].iloc[0]
        axes[0].plot(sel["k"], sel["custo"] / base_c, color=COR_METRICA[met],
                     linewidth=2, marker="o", markersize=6)
        axes[1].plot(sel["k"], sel["silhueta_media"], color=COR_METRICA[met],
                     linewidth=2, marker="o", markersize=6,
                     label=NOME_METRICA[met])
    axes[0].set_xlabel("número de grupos (k)")
    axes[0].set_ylabel("custo relativo ao k=2")
    axes[0].set_title("Cotovelo por métrica")
    axes[1].set_xlabel("número de grupos (k)")
    axes[1].set_ylabel("silhueta média (na própria métrica)")
    axes[1].set_title("Silhueta por métrica")
    leg = axes[1].legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(SEC)
    fig.suptitle("K-medoids — comparativo entre medidas de distância",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out_raiz, "comparativo_cotovelo_silhueta.png"))
    plt.close(fig)

    x_pos = np.arange(args.bins)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, met in zip(axes.flat, METRICAS):
        medias3, nomes3, rot3, sil3 = curvas_k3[met]
        ari = resumo_geral.loc[resumo_geral["metrica"] == NOME_METRICA[met],
                               "ari_k3_vs_baseline"].iloc[0]
        for cl in range(K_FOCO):
            cor = CORES_PERFIL.get(nomes3[cl], SERIES[cl])
            n = int((rot3 == cl).sum())
            ax.plot(x_pos, medias3[cl], color=cor, linewidth=2, marker="o",
                    markersize=5)
            ax.annotate(f"{nomes3[cl]} ({n:,})", (x_pos[-1], medias3[cl][-1]),
                        textcoords="offset points", xytext=(8, 0), color=cor,
                        fontsize=8, fontweight="bold", va="center")
        ax.set_xticks(x_pos, kc.BIN_LABELS, fontsize=8)
        ax.set_xlim(-0.3, args.bins + 1.0)
        ax.set_title(f"{NOME_METRICA[met]}  (sil={sil3:.3f}, "
                     f"ARI vs baseline={ari:.2f})", fontsize=11)
        ax.set_ylabel("proporção do esforço", fontsize=9)
    fig.suptitle("Curvas médias dos 3 grupos por métrica de distância — k-medoids",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out_raiz, "comparativo_curvas_k3.png"))
    plt.close(fig)

    with open(os.path.join(out_raiz, "resumo_geral.txt"), "w",
              encoding="utf-8") as f:
        f.write(f"K-MEDOIDS — VARIAÇÃO DA MEDIDA DE DISTÂNCIA ({args.bins} bins)\n")
        f.write("Baseline: k-means euclidiano k=3 do pipeline principal.\n\n")
        f.write(resumo_geral.round(4).to_string(index=False))

    print(resumo_geral.round(3).to_string(index=False))
    print(f"Concluído. Resultados em: {out_raiz}")


if __name__ == "__main__":
    main()
