# -*- coding: utf-8 -*-
"""
Clusterizacao K-means de curvas de esforco por ciclo de avaliacao (OULAD, FFF 2014J).

Pipeline:
  1. Ciclos = intervalos entre deadlines dos 5 TMAs do FFF 2014J.
  2. Curva de cada ciclo = distribuicao do esforco (log1p dos cliques diarios)
     em N bins de tempo relativo do ciclo, normalizada pela soma (soma = 1).
  3. So entram ciclos de alunos que submeteram o TMA daquele ciclo (remove o
     vies de desistentes) e com atividade minima no ciclo.
  4. Varredura k=2..8 (cotovelo + silhueta) -> <saida>/selecao_k
  5. Resultados completos para o melhor k matematico -> <saida>/melhor_k_<K>
     e para k=3 (escolha confirmatoria: adiantado/equilibrado/tardio) -> <saida>/k3

Uso:
  python kmeans_ciclos.py                       # 4 bins -> resultados_kmeans/
  python kmeans_ciclos.py --bins 8 --out resultados_kmeans_8bins
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

# ---------------------------------------------------------------- configuracao
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
OUT_DIR = os.path.join(BASE_DIR, "resultados_kmeans")
CACHE = os.path.join(BASE_DIR, "cache_vle_fff2014j.csv")

MODULE, PRES = "FFF", "2014J"
N_BINS = 4                      # bins de tempo relativo dentro do ciclo
K_RANGE = range(2, 9)           # varredura de k
K_CONFIRMATORIO = 3             # adiantado / equilibrado / tardio
MIN_CLICKS_CICLO = 5            # atividade minima para o ciclo ter "forma"
MIN_DIAS_ATIVOS = 2
SEED = 42

# paleta de referencia validada (modo claro)
SURF, INK, SEC, MUT = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
          "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
# cores de status para resultado final (fixas, nunca reusadas como serie)
COR_RESULTADO = {"Distinction": "#006300", "Pass": "#0ca30c",
                 "Fail": "#ec835a", "Withdrawn": "#898781"}
# a cor segue o perfil em todas as figuras (k=3), independente do indice do cluster
CORES_PERFIL = {"Adiantado": SERIES[0], "Tardio": SERIES[1],
                "Equilibrado": SERIES[2]}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUT, "ytick.color": MUT,
    "axes.labelcolor": SEC, "text.color": INK,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def rotulos_bins(n):
    if n == 4:
        return ["início", "meio 1", "meio 2", "véspera"]
    bordas = np.linspace(0, 100, n + 1)
    return [f"{bordas[i]:.0f}–{bordas[i + 1]:.0f}%" for i in range(n)]


BIN_LABELS = rotulos_bins(N_BINS)


# ---------------------------------------------------------------- preparacao
def carregar_vle_diario():
    """Cliques diarios por aluno no FFF 2014J (cache local para reruns)."""
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE)
    partes = []
    caminho = os.path.join(DATA_DIR, "studentVle.csv")
    for chunk in pd.read_csv(
        caminho, chunksize=1_000_000,
        usecols=["code_module", "code_presentation", "id_student", "date", "sum_click"],
        dtype={"code_module": "category", "code_presentation": "category",
               "id_student": "int32", "date": "int16", "sum_click": "int32"},
    ):
        sel = chunk[(chunk["code_module"] == MODULE) &
                    (chunk["code_presentation"] == PRES)]
        if len(sel):
            partes.append(sel[["id_student", "date", "sum_click"]])
    vle = pd.concat(partes, ignore_index=True)
    diario = (vle.groupby(["id_student", "date"], as_index=False)["sum_click"]
                 .sum())
    diario.to_csv(CACHE, index=False)
    return diario


def montar_ciclos():
    """Deadlines dos TMAs e submissoes correspondentes."""
    ass = pd.read_csv(os.path.join(DATA_DIR, "assessments.csv"))
    tmas = (ass[(ass["code_module"] == MODULE) &
                (ass["code_presentation"] == PRES) &
                (ass["assessment_type"] == "TMA")]
            .sort_values("date").reset_index(drop=True))
    sub = pd.read_csv(os.path.join(DATA_DIR, "studentAssessment.csv"),
                      usecols=["id_assessment", "id_student"])
    submissores = {row.id_assessment: set(sub.loc[sub["id_assessment"] == row.id_assessment,
                                                  "id_student"])
                   for row in tmas.itertuples()}
    return tmas, submissores


def construir_matriz(diario, tmas, submissores):
    """Uma linha por (aluno, ciclo): proporcao de esforco em N_BINS bins."""
    limites = [0] + tmas["date"].astype(int).tolist()   # [0, 24, 52, 94, 136, 199]
    registros, vetores = [], []
    descartados_atividade = 0

    diario = diario.copy()
    diario["log_click"] = np.log1p(diario["sum_click"])

    for i in range(len(tmas)):
        ini, fim = limites[i], limites[i + 1]
        id_tma = int(tmas.loc[i, "id_assessment"])
        if i == 0:
            janela = diario[(diario["date"] >= ini) & (diario["date"] <= fim)]
        else:
            janela = diario[(diario["date"] > ini) & (diario["date"] <= fim)]
        janela = janela[janela["id_student"].isin(submissores[id_tma])].copy()

        rel = (janela["date"] - ini) / (fim - ini)
        janela["bin"] = np.clip((rel * N_BINS).astype(int), 0, N_BINS - 1)

        agg = janela.groupby("id_student").agg(
            total_clicks=("sum_click", "sum"),
            dias_ativos=("date", "nunique"),
        )
        pesos = (janela.pivot_table(index="id_student", columns="bin",
                                    values="log_click", aggfunc="sum",
                                    fill_value=0.0)
                 .reindex(columns=range(N_BINS), fill_value=0.0))

        validos = agg[(agg["total_clicks"] >= MIN_CLICKS_CICLO) &
                      (agg["dias_ativos"] >= MIN_DIAS_ATIVOS)].index
        descartados_atividade += len(agg) - len(validos)
        pesos = pesos.loc[pesos.index.intersection(validos)]

        prop = pesos.div(pesos.sum(axis=1), axis=0)
        for sid, linha in prop.iterrows():
            registros.append({"id_student": int(sid), "ciclo": i + 1,
                              "id_assessment": id_tma,
                              "total_clicks": int(agg.loc[sid, "total_clicks"])})
            vetores.append(linha.to_numpy())

    X = np.vstack(vetores)
    meta = pd.DataFrame(registros)
    return X, meta, descartados_atividade


# ---------------------------------------------------------------- k-means
def varredura_k(X, out_sel):
    os.makedirs(out_sel, exist_ok=True)
    linhas, ajustes = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(X)
        sil = silhouette_score(X, km.labels_)
        linhas.append({"k": k, "inercia": km.inertia_, "silhueta_media": sil})
        ajustes.append((k, km.labels_, silhouette_samples(X, km.labels_), sil))
    met = pd.DataFrame(linhas)

    # grade de facas: um diagrama de silhueta por k, para comparação direta
    ncols = 4
    nrows = -(-len(ajustes) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows))
    for ax, (k, rot, sam, sil) in zip(axes.flat, ajustes):
        y0 = 10
        for cl in range(k):
            vals = np.sort(sam[rot == cl])
            cor = SERIES[cl % len(SERIES)]
            ax.fill_betweenx(np.arange(y0, y0 + len(vals)), 0, vals,
                             facecolor=cor, edgecolor=cor)
            ax.text(-0.04, y0 + len(vals) / 2, f"{len(vals):,}",
                    ha="right", va="center", fontsize=7, color=SEC)
            y0 += len(vals) + 60
        ax.axvline(sil, color=INK, linewidth=1, linestyle="--")
        ax.set_title(f"k={k}  (média={sil:.3f})", fontsize=10)
        ax.set_yticks([])
        ax.set_xlim(-0.25, 0.85)
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(ajustes):]:
        ax.set_visible(False)
    fig.supxlabel("coeficiente de silhueta", fontsize=10, color=SEC)
    fig.suptitle(f"Diagramas de silhueta por k ({N_BINS} bins) — n de ciclos à esquerda de cada faca",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out_sel, "silhueta_facas_todos_k.png"))
    plt.close(fig)
    met.to_csv(os.path.join(out_sel, "metricas_k.csv"), index=False)

    melhor_k = int(met.loc[met["silhueta_media"].idxmax(), "k"])

    # cotovelo
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(met["k"], met["inercia"], color=SERIES[0], linewidth=2,
            marker="o", markersize=7)
    ax.set_xlabel("número de clusters (k)")
    ax.set_ylabel("inércia (WCSS)")
    ax.set_title("Método do cotovelo — curvas de ciclo FFF 2014J")
    for _, r in met.iterrows():
        ax.annotate(f"{r['inercia']:.0f}", (r["k"], r["inercia"]),
                    textcoords="offset points", xytext=(8, 6),
                    fontsize=8, color=SEC)
    fig.savefig(os.path.join(out_sel, "cotovelo.png"))
    plt.close(fig)

    # silhueta media
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(met["k"], met["silhueta_media"], color=SERIES[0], linewidth=2,
            marker="o", markersize=7)
    ax.axvline(melhor_k, color=SERIES[2], linewidth=2, linestyle="--")
    ax.annotate(f"melhor k = {melhor_k}", (melhor_k, met["silhueta_media"].max()),
                textcoords="offset points", xytext=(10, -2),
                color=SEC, fontsize=9)
    if K_CONFIRMATORIO != melhor_k:
        y3 = met.loc[met["k"] == K_CONFIRMATORIO, "silhueta_media"].iloc[0]
        ax.scatter([K_CONFIRMATORIO], [y3], s=90, facecolor="none",
                   edgecolor=SERIES[5], linewidth=2, zorder=5)
        ax.annotate("k = 3 (confirmatório)", (K_CONFIRMATORIO, y3),
                    textcoords="offset points", xytext=(10, 8),
                    color=SEC, fontsize=9)
    ax.set_xlabel("número de clusters (k)")
    ax.set_ylabel("silhueta média")
    ax.set_title("Análise de silhueta — curvas de ciclo FFF 2014J")
    fig.savefig(os.path.join(out_sel, "silhueta_media.png"))
    plt.close(fig)

    return melhor_k, met


def nomear_perfis(centroides):
    """Nomeia clusters pelo centro de massa temporal (k=3: adiantado/equilibrado/tardio)."""
    centros_bin = (np.arange(N_BINS) + 0.5) / N_BINS
    com = centroides @ centros_bin            # centro de massa em [0,1]
    ordem = np.argsort(com)
    nomes = {}
    if len(centroides) == 3:
        rotulos = ["Adiantado", "Equilibrado", "Tardio"]
        for pos, cl in enumerate(ordem):
            nomes[cl] = rotulos[pos]
    else:
        for pos, cl in enumerate(ordem):
            nomes[cl] = f"Cluster {cl} (com={com[cl]:.2f})"
    return nomes, com


def relatorio_k(X, meta, k, out_dir, info_alunos):
    os.makedirs(out_dir, exist_ok=True)
    km = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(X)
    rot = km.labels_
    sil_med = silhouette_score(X, rot)
    sil_amostras = silhouette_samples(X, rot)
    nomes, com = nomear_perfis(km.cluster_centers_)

    def cor_cluster(cl):
        return CORES_PERFIL.get(nomes[cl], SERIES[cl % len(SERIES)])

    atrib = meta.copy()
    atrib["cluster"] = rot
    atrib["perfil"] = atrib["cluster"].map(nomes)
    atrib["silhueta"] = sil_amostras
    atrib.to_csv(os.path.join(out_dir, "atribuicoes_ciclos.csv"), index=False)

    tam = (atrib.groupby(["cluster", "perfil"])
           .agg(n_ciclos=("cluster", "size"),
                silhueta_media=("silhueta", "mean"),
                mediana_clicks=("total_clicks", "median"))
           .reset_index())
    tam["centro_de_massa"] = tam["cluster"].map(dict(enumerate(com)))
    tam.to_csv(os.path.join(out_dir, "tamanhos_clusters.csv"), index=False)

    # --- centroides como curvas
    x_pos = np.arange(N_BINS)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cl in range(k):
        cor = cor_cluster(cl)
        ax.plot(x_pos, km.cluster_centers_[cl], color=cor, linewidth=2,
                marker="o", markersize=6 if N_BINS <= 6 else 4)
        n = int((rot == cl).sum())
        ax.annotate(f"{nomes[cl]}  (n={n})",
                    (x_pos[-1], km.cluster_centers_[cl][-1]),
                    textcoords="offset points", xytext=(10, 0),
                    color=cor, fontsize=9, fontweight="bold", va="center")
    ax.set_xticks(x_pos, BIN_LABELS,
                  fontsize=9 if N_BINS <= 6 else 8,
                  rotation=0 if N_BINS <= 6 else 45)
    ax.set_xlim(-0.3, N_BINS - 0.2)
    ax.set_ylabel("proporção do esforço no ciclo")
    ax.set_xlabel("momento do ciclo de avaliação (tempo relativo)")
    ax.set_title(f"Curvas-tipo dos perfis (centróides, k={k}, {N_BINS} bins)")
    fig.subplots_adjust(right=0.78)
    fig.savefig(os.path.join(out_dir, "centroides.png"))
    plt.close(fig)

    # --- desenhadores reutilizados (figuras isoladas + diagnostico combinado)
    def desenhar_facas(ax):
        y0 = 10
        for cl in range(k):
            vals = np.sort(sil_amostras[rot == cl])
            cor = cor_cluster(cl)
            ax.fill_betweenx(np.arange(y0, y0 + len(vals)), 0, vals,
                             facecolor=cor, edgecolor=cor)
            ax.text(-0.03, y0 + len(vals) / 2, f"{nomes[cl]}\n(n={len(vals):,})",
                    ha="right", va="center", fontsize=9, color=SEC)
            y0 += len(vals) + 40
        ax.axvline(sil_med, color=INK, linewidth=1.2, linestyle="--")
        ax.annotate(f"média = {sil_med:.3f}", (sil_med, y0),
                    textcoords="offset points", xytext=(6, -12),
                    fontsize=9, color=SEC)
        ax.set_yticks([])
        ax.set_xlabel("coeficiente de silhueta")
        ax.grid(axis="y", visible=False)

    pca = PCA(n_components=2, random_state=SEED)
    P = pca.fit_transform(X)
    C = pca.transform(km.cluster_centers_)
    var = pca.explained_variance_ratio_ * 100

    def desenhar_pca(ax):
        for cl in range(k):
            m = rot == cl
            cor = cor_cluster(cl)
            ax.scatter(P[m, 0], P[m, 1], s=8, color=cor, alpha=0.35,
                       linewidths=0, label=f"{nomes[cl]} (n={m.sum():,})")
        ax.scatter(C[:, 0], C[:, 1], s=140, marker="X", color=INK,
                   edgecolor=SURF, linewidth=1.5, zorder=5)
        ax.set_xlabel(f"CP1 ({var[0]:.0f}% da variância)")
        ax.set_ylabel(f"CP2 ({var[1]:.0f}% da variância)")
        leg = ax.legend(frameon=False, fontsize=9, markerscale=2.2)
        for t in leg.get_texts():
            t.set_color(SEC)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    desenhar_facas(ax)
    ax.set_title(f"Diagrama de silhueta (k={k})")
    fig.savefig(os.path.join(out_dir, "silhueta_amostras.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    desenhar_pca(ax)
    ax.set_title(f"Ciclos no plano das 2 primeiras componentes (k={k})")
    fig.savefig(os.path.join(out_dir, "dispersao_pca.png"))
    plt.close(fig)

    # --- diagnostico combinado: facas + mapa PCA
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    desenhar_facas(axes[0])
    axes[0].set_title("Silhueta por ciclo")
    desenhar_pca(axes[1])
    axes[1].set_title("Ciclos em 2D (PCA)")
    fig.suptitle(f"Diagnóstico de silhueta — curvas de ciclo, k={k} ({N_BINS} bins)",
                 fontweight="bold", color=INK)
    fig.savefig(os.path.join(out_dir, "silhueta_facas_e_mapa.png"))
    plt.close(fig)

    # --- perfil por aluno: consistencia entre ciclos
    def perfil_aluno(grupo):
        contagem = grupo["perfil"].value_counts()
        moda, freq = contagem.index[0], contagem.iloc[0]
        n = len(grupo)
        if n >= 3 and freq / n >= 0.6:
            return pd.Series({"perfil_aluno": moda, "n_ciclos": n,
                              "consistencia": freq / n})
        return pd.Series({"perfil_aluno": "Misto" if n >= 3 else "Insuficiente",
                          "n_ciclos": n, "consistencia": freq / n})

    alunos = atrib.groupby("id_student").apply(perfil_aluno,
                                               include_groups=False).reset_index()
    alunos = alunos.merge(info_alunos, on="id_student", how="left")
    alunos.to_csv(os.path.join(out_dir, "perfil_alunos.csv"), index=False)

    cruz = pd.crosstab(alunos["perfil_aluno"], alunos["final_result"])
    cruz.to_csv(os.path.join(out_dir, "perfil_x_resultado.csv"))

    # --- resultado final por perfil (barras 100% empilhadas)
    ordem_perfis = ([nomes[cl] for cl in np.argsort(com)] +
                    ["Misto", "Insuficiente"])
    ordem_perfis = [p for p in ordem_perfis
                    if p in alunos["perfil_aluno"].unique()]
    ordem_resultado = [r for r in ["Distinction", "Pass", "Fail", "Withdrawn"]
                       if r in cruz.columns]
    pct = (cruz.div(cruz.sum(axis=1), axis=0) * 100).reindex(ordem_perfis)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(ordem_perfis))
    base_y = np.zeros(len(ordem_perfis))
    for res in ordem_resultado:
        vals = pct[res].to_numpy()
        ax.bar(x, vals, bottom=base_y, width=0.62,
               color=COR_RESULTADO[res], label=res,
               edgecolor=SURF, linewidth=2)
        for xi, (v, b) in enumerate(zip(vals, base_y)):
            if v >= 7:
                ax.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="#ffffff")
        base_y += vals
    n_por_perfil = cruz.sum(axis=1).reindex(ordem_perfis)
    ax.set_xticks(x, [f"{p}\n(n={int(n_por_perfil[p]):,})"
                      for p in ordem_perfis], fontsize=9)
    ax.set_ylabel("% dos alunos")
    ax.set_ylim(0, 100)
    ax.set_title(f"Resultado final por perfil de preparação (k={k})")
    ax.grid(axis="x", visible=False)
    leg = ax.legend(frameon=False, fontsize=9, loc="center left",
                    bbox_to_anchor=(1.01, 0.5))
    for t in leg.get_texts():
        t.set_color(SEC)
    fig.savefig(os.path.join(out_dir, "resultado_por_perfil.png"))
    plt.close(fig)

    with open(os.path.join(out_dir, "resumo.txt"), "w", encoding="utf-8") as f:
        f.write(f"k = {k} | bins = {N_BINS} | silhueta média = {sil_med:.4f} | "
                f"inércia = {km.inertia_:.1f}\n\n")
        f.write("Centróides (proporção do esforço por bin do ciclo):\n")
        for cl in range(k):
            v = ", ".join(f"{p:.3f}" for p in km.cluster_centers_[cl])
            f.write(f"  {nomes[cl]}: [{v}]  n={int((rot == cl).sum())}\n")
        f.write("\nPerfil dos alunos (consistência entre ciclos):\n")
        f.write(alunos["perfil_aluno"].value_counts().to_string())
        f.write("\n\nPerfil x resultado final:\n")
        f.write(cruz.to_string())
    return sil_med


# ---------------------------------------------------------------- main
def main():
    global N_BINS, BIN_LABELS, OUT_DIR

    parser = argparse.ArgumentParser(description="K-means de curvas de ciclo (OULAD FFF 2014J)")
    parser.add_argument("--bins", type=int, default=N_BINS,
                        help="número de bins de tempo relativo por ciclo (padrão: 4)")
    parser.add_argument("--out", type=str, default=None,
                        help="pasta de saída (padrão: resultados_kmeans)")
    args = parser.parse_args()

    N_BINS = args.bins
    BIN_LABELS = rotulos_bins(N_BINS)
    if args.out:
        OUT_DIR = args.out if os.path.isabs(args.out) else os.path.join(BASE_DIR, args.out)

    print(f"Configuração: {N_BINS} bins por ciclo -> {OUT_DIR}")
    print("1/4 Carregando cliques diários FFF 2014J...")
    diario = carregar_vle_diario()
    print(f"    {len(diario):,} pares aluno x dia, "
          f"{diario['id_student'].nunique():,} alunos com atividade")

    tmas, submissores = montar_ciclos()
    print("    TMAs (deadlines):", tmas["date"].astype(int).tolist())

    print("2/4 Construindo curvas por ciclo...")
    X, meta, descartados = construir_matriz(diario, tmas, submissores)
    print(f"    matriz {X.shape[0]:,} ciclos x {X.shape[1]} bins "
          f"({meta['id_student'].nunique():,} alunos; "
          f"{descartados} ciclos descartados por baixa atividade)")

    info = pd.read_csv(os.path.join(DATA_DIR, "studentInfo.csv"),
                       usecols=["code_module", "code_presentation",
                                "id_student", "final_result"])
    info = info[(info["code_module"] == MODULE) &
                (info["code_presentation"] == PRES)][["id_student", "final_result"]]

    print("3/4 Varredura de k (cotovelo + silhueta)...")
    melhor_k, met = varredura_k(X, os.path.join(OUT_DIR, "selecao_k"))
    print(met.to_string(index=False))
    print(f"    melhor k pela silhueta: {melhor_k}")

    print("4/4 Relatórios finais...")
    sil_best = relatorio_k(X, meta, melhor_k,
                           os.path.join(OUT_DIR, f"melhor_k_{melhor_k}"), info)
    print(f"    melhor_k_{melhor_k}: silhueta {sil_best:.4f}")
    sil_3 = relatorio_k(X, meta, K_CONFIRMATORIO,
                        os.path.join(OUT_DIR, "k3"), info)
    print(f"    k3: silhueta {sil_3:.4f}")
    print(f"\nConcluído. Resultados em: {OUT_DIR}")


if __name__ == "__main__":
    main()
