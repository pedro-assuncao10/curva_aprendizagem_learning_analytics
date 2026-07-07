# -*- coding: utf-8 -*-
"""
Modelo preditivo (H3): a FORMA da curva adiciona poder preditivo alem do
VOLUME para prever aprovacao (Pass+Distinction) vs reprovacao (Fail)?

Desenho:
  - Unidade: aluno do FFF 2014J com >=1 ciclo valido; Withdrawn excluido.
  - Cobertura (quais TMAs submeteu) presente em TODOS os conjuntos, para a
    forma nao ganhar credito por apenas codificar buracos.
  - Conjuntos de features:
      volume       cobertura + cliques (total e por ciclo, log1p)
      forma        cobertura + 20 proporcoes (5 ciclos x 4 bins)
      volume+forma uniao dos dois
      +nota TMA1   uniao + nota da primeira avaliacao (modelo operacional)
  - Modelos: Regressao Logistica, Random Forest, MLP.
  - Avaliacao: AUC em validacao cruzada estratificada 5-fold x 4 repeticoes
    (mesmas particoes para todos -> comparacao PAREADA por fold).
  - Teste da H3: t pareado + Wilcoxon dos AUCs (volume+forma vs volume).
  - Cenario extra "alerta precoce": apenas ciclos 1-2 + nota TMA1.

Saida: resultados_preditivo/
"""

import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, \
    cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score

import kmeans_ciclos as kc
from kmeans_ciclos import (BASE_DIR, DATA_DIR, MODULE, PRES, SEED,
                           SURF, INK, SEC, MUT, GRID, BASE, SERIES)

OUT = os.path.join(BASE_DIR, "resultados_preditivo")
BIN_NOMES = ["início", "meio1", "meio2", "véspera"]
COR_MODELO = {"Logística": SERIES[0], "Random Forest": SERIES[1],
              "MLP": SERIES[2]}


# ---------------------------------------------------------------- tabela aluno
def montar_tabela():
    diario = kc.carregar_vle_diario()
    tmas, submissores = kc.montar_ciclos()
    X, meta, _ = kc.construir_matriz(diario, tmas, submissores)

    df = meta.copy()
    for b in range(4):
        df[f"b{b + 1}"] = X[:, b]

    wide = df.pivot_table(index="id_student", columns="ciclo",
                          values=["b1", "b2", "b3", "b4", "total_clicks"],
                          aggfunc="first")
    wide.columns = [f"{v}_c{c}" for v, c in wide.columns]

    tab = wide.reset_index()
    for c in range(1, 6):
        tab[f"sub_c{c}"] = tab[f"total_clicks_c{c}"].notna().astype(int)
        tab[f"vol_log_c{c}"] = np.log1p(tab[f"total_clicks_c{c}"].fillna(0))
        for b in range(1, 5):
            tab[f"b{b}_c{c}"] = tab[f"b{b}_c{c}"].fillna(0.0)
    tab["n_ciclos"] = tab[[f"sub_c{c}" for c in range(1, 6)]].sum(axis=1)
    tab["vol_total_log"] = np.log1p(
        tab[[f"total_clicks_c{c}" for c in range(1, 6)]].fillna(0).sum(axis=1))

    # nota do primeiro TMA
    id_tma1 = int(tmas.sort_values("date")["id_assessment"].iloc[0])
    sub = pd.read_csv(os.path.join(DATA_DIR, "studentAssessment.csv"),
                      usecols=["id_assessment", "id_student", "score"])
    n1 = (sub[sub["id_assessment"] == id_tma1]
          .dropna(subset=["score"])[["id_student", "score"]]
          .rename(columns={"score": "nota_tma1"}))
    tab = tab.merge(n1, on="id_student", how="left")
    tab["tem_nota1"] = tab["nota_tma1"].notna().astype(int)
    tab["nota_tma1"] = tab["nota_tma1"].fillna(tab["nota_tma1"].median())

    info = pd.read_csv(os.path.join(DATA_DIR, "studentInfo.csv"),
                       usecols=["code_module", "code_presentation",
                                "id_student", "final_result"])
    info = info[(info["code_module"] == MODULE) &
                (info["code_presentation"] == PRES)][["id_student",
                                                      "final_result"]]
    tab = tab.merge(info, on="id_student", how="left")
    tab = tab[tab["final_result"].isin(["Pass", "Distinction", "Fail"])].copy()
    tab["aprovado"] = tab["final_result"].isin(["Pass", "Distinction"]).astype(int)
    return tab


def conjuntos_features(ciclos):
    """Conjuntos de features para uma lista de ciclos (ex.: 1..5 ou 1..2)."""
    cob = [f"sub_c{c}" for c in ciclos] + (["n_ciclos"] if len(ciclos) == 5
                                           else [])
    vol = cob + [f"vol_log_c{c}" for c in ciclos] + (
        ["vol_total_log"] if len(ciclos) == 5 else [])
    forma = cob + [f"b{b}_c{c}" for c in ciclos for b in range(1, 5)]
    ambos = vol + [f for f in forma if f not in vol]
    ambos_nota = ambos + ["nota_tma1", "tem_nota1"]
    return {"volume": vol, "forma": forma, "volume+forma": ambos,
            "volume+forma+nota TMA1": ambos_nota}


def fazer_modelos():
    return {
        "Logística": Pipeline([("sc", StandardScaler()),
                               ("m", LogisticRegression(
                                   max_iter=3000, class_weight="balanced",
                                   random_state=SEED))]),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced",
            random_state=SEED, n_jobs=-1),
        "MLP": Pipeline([("sc", StandardScaler()),
                         ("m", MLPClassifier(
                             hidden_layer_sizes=(32, 16), alpha=1e-3,
                             max_iter=1000, early_stopping=True,
                             n_iter_no_change=20, random_state=SEED))]),
    }


# ---------------------------------------------------------------- avaliacao
def avaliar_cenario(tab, conjuntos, rotulo):
    y = tab["aprovado"].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEED)
    aucs = {}   # (conjunto, modelo) -> vetor de AUCs por fold
    for nome_cj, feats in conjuntos.items():
        Xf = tab[feats].to_numpy(dtype=float)
        for nome_md, mod in fazer_modelos().items():
            sc = cross_val_score(mod, Xf, y, cv=cv, scoring="roc_auc",
                                 n_jobs=-1)
            aucs[(nome_cj, nome_md)] = sc
            print(f"    [{rotulo}] {nome_cj:<24} {nome_md:<14} "
                  f"AUC={sc.mean():.4f} ±{sc.std():.4f}")
    return aucs


def testes_h3(aucs, conjuntos_nomes):
    """t pareado + Wilcoxon: volume+forma vs volume, por modelo."""
    linhas = []
    for nome_md in COR_MODELO:
        a_vol = aucs[("volume", nome_md)]
        a_amb = aucs[("volume+forma", nome_md)]
        delta = a_amb - a_vol
        t, p_t = stats.ttest_rel(a_amb, a_vol)
        try:
            w, p_w = stats.wilcoxon(a_amb, a_vol)
        except ValueError:
            p_w = np.nan
        linhas.append({"modelo": nome_md,
                       "auc_volume": a_vol.mean(),
                       "auc_ambos": a_amb.mean(),
                       "delta_auc": delta.mean(),
                       "p_t_pareado": p_t, "p_wilcoxon": p_w,
                       "folds_melhora": int((delta > 0).sum()),
                       "n_folds": len(delta)})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------- graficos
def grafico_auc(aucs_full, aucs_early, conjuntos_nomes):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, aucs, titulo in [(axes[0], aucs_full, "Curso completo (5 ciclos)"),
                             (axes[1], aucs_early,
                              "Alerta precoce (ciclos 1–2)")]:
        xs = np.arange(len(conjuntos_nomes))
        larg = 0.24
        for j, nome_md in enumerate(COR_MODELO):
            medias = [aucs[(cj, nome_md)].mean() for cj in conjuntos_nomes]
            erros = [aucs[(cj, nome_md)].std() for cj in conjuntos_nomes]
            pos = xs + (j - 1) * larg
            ax.bar(pos, medias, width=larg * 0.9, color=COR_MODELO[nome_md],
                   edgecolor=SURF, linewidth=1.5,
                   label=nome_md if ax is axes[0] else None)
            ax.errorbar(pos, medias, yerr=erros, fmt="none", ecolor=SEC,
                        elinewidth=1, capsize=2)
            for x, m in zip(pos, medias):
                ax.text(x, m + 0.012, f"{m:.3f}", ha="center", fontsize=7.5,
                        color=SEC, rotation=90)
        ax.set_xticks(xs, [c.replace("+", "\n+") for c in conjuntos_nomes],
                      fontsize=9)
        ax.axhline(0.5, color=BASE, linewidth=1, linestyle="--")
        ax.set_title(titulo)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0.45, 0.92)
    axes[0].set_ylabel("AUC (média de 20 folds, barra = desvio)")
    leg = axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(SEC)
    fig.suptitle("H3 — A forma adiciona poder preditivo além do volume?",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "auc_comparativo.png"))
    plt.close(fig)


def grafico_roc(tab, conjuntos):
    """ROC (5-fold, empilhado) do melhor modelo por conjunto — curso completo."""
    y = tab["aprovado"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fig, ax = plt.subplots(figsize=(6.8, 6))
    cores = [SERIES[0], SERIES[2], SERIES[1], SERIES[4]]
    for cor, (nome_cj, feats) in zip(cores, conjuntos.items()):
        Xf = tab[feats].to_numpy(dtype=float)
        mod = fazer_modelos()["Random Forest"]
        prob = cross_val_predict(mod, Xf, y, cv=cv, method="predict_proba",
                                 n_jobs=-1)[:, 1]
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        ax.plot(fpr, tpr, color=cor, linewidth=2,
                label=f"{nome_cj}  (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color=BASE, linewidth=1, linestyle="--")
    ax.set_xlabel("taxa de falsos positivos")
    ax.set_ylabel("taxa de verdadeiros positivos")
    ax.set_title("Curvas ROC — Random Forest, validação cruzada 5-fold")
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(SEC)
    fig.savefig(os.path.join(OUT, "roc_curvas.png"))
    plt.close(fig)


def rotulo_feature(f):
    if f.startswith("b"):
        b, c = f[1], f[-1]
        return f"forma: ciclo {c} · {BIN_NOMES[int(b) - 1]}"
    if f.startswith("vol_log_c"):
        return f"volume: ciclo {f[-1]}"
    if f == "vol_total_log":
        return "volume: total do curso"
    if f.startswith("sub_c"):
        return f"cobertura: submeteu TMA{f[-1]}"
    if f == "n_ciclos":
        return "cobertura: nº de ciclos"
    if f == "nota_tma1":
        return "nota do TMA1"
    if f == "tem_nota1":
        return "cobertura: tem nota TMA1"
    return f


def grafico_importancias(tab, conjuntos):
    feats = conjuntos["volume+forma+nota TMA1"]
    Xf = tab[feats].to_numpy(dtype=float)
    y = tab["aprovado"].to_numpy()
    rf = fazer_modelos()["Random Forest"].fit(Xf, y)
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values()[-12:]
    cores = []
    for f in imp.index:
        if f.startswith("b"):
            cores.append(SERIES[2])
        elif "vol" in f:
            cores.append(SERIES[0])
        elif f == "nota_tma1":
            cores.append(SERIES[4])
        else:
            cores.append(MUT)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh([rotulo_feature(f) for f in imp.index], imp.values, color=cores,
            edgecolor=SURF, linewidth=1)
    ax.set_xlabel("importância (Random Forest, modelo completo)")
    ax.set_title("Quais features o modelo mais usa?")
    ax.grid(axis="y", visible=False)
    fig.savefig(os.path.join(OUT, "importancias.png"))
    plt.close(fig)


def tabela_sintese(h3_full, h3_early, melhor):
    def quebrar(txt, largura):
        linhas = []
        for l in str(txt).split("\n"):
            linhas.extend(textwrap.wrap(l, largura) or [""])
        return linhas

    linhas_tab = []
    for cen, h3 in [("Curso completo", h3_full), ("Alerta precoce", h3_early)]:
        for _, r in h3.iterrows():
            ok = r["p_t_pareado"] < 0.05 and r["delta_auc"] > 0
            linhas_tab.append((
                [quebrar(f"{cen} — {r['modelo']}", 22),
                 quebrar("volume+forma supera só volume?", 26),
                 quebrar(f"AUC {r['auc_volume']:.3f} → {r['auc_ambos']:.3f} "
                         f"(Δ = {r['delta_auc']:+.4f})\n"
                         f"t pareado p={r['p_t_pareado']:.4f} | "
                         f"Wilcoxon p={r['p_wilcoxon']:.4f}\n"
                         f"forma venceu em {int(r['folds_melhora'])}/"
                         f"{int(r['n_folds'])} folds", 46),
                 quebrar("SIM: ganho significativo — H3 suportada"
                         if ok else
                         ("Ganho não significativo neste modelo"
                          if r["delta_auc"] > 0 else
                          "SEM ganho — volume já basta neste modelo"), 40)],
                max(4, 0), ok))
    alturas = [max(len(c) for c in cels) + 0.9 for cels, _, _ in linhas_tab]
    total = sum(alturas) + 1.5
    fig, ax = plt.subplots(figsize=(15, 0.26 * total + 1.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(total, 0)
    ax.axis("off")
    col_x = [0.01, 0.19, 0.40, 0.74]
    for x, h in zip(col_x, ["Cenário — modelo", "Pergunta",
                            "Resultado numérico", "Veredito"]):
        ax.text(x, 0.65, h, fontweight="bold", color=INK, va="center",
                fontsize=10.5)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1.3, facecolor="#f0efec",
                               edgecolor="none", zorder=0))
    ax.plot([0, 1], [1.3, 1.3], color=BASE, linewidth=1)
    y = 1.5
    for (cels, _, ok), h in zip(linhas_tab, alturas):
        for j, (x, cel) in enumerate(zip(col_x, cels)):
            if j == 3:
                cor, peso = ("#006300" if ok else "#b3261e"), "bold"
            elif j == 0:
                cor, peso = INK, "bold"
            else:
                cor, peso = SEC, "normal"
            ax.text(x, y + 0.15, "\n".join(cel), va="top", fontsize=9.5,
                    color=cor, fontweight=peso, linespacing=1.45)
        y += h
        ax.plot([0, 1], [y, y], color=GRID, linewidth=0.8)
    ax.set_title(f"Teste da H3 — {melhor}", fontweight="bold", color=INK,
                 pad=14)
    fig.savefig(os.path.join(OUT, "tabela_h3.png"))
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    print("Montando tabela por aluno...")
    tab = montar_tabela()
    n_apr = int(tab["aprovado"].sum())
    print(f"  {len(tab):,} alunos (aprovados={n_apr}, "
          f"reprovados={len(tab) - n_apr})")
    tab.to_csv(os.path.join(OUT, "tabela_treino.csv"), index=False)

    cj_full = conjuntos_features([1, 2, 3, 4, 5])
    cj_early = conjuntos_features([1, 2])

    print("Cenário curso completo...")
    aucs_full = avaliar_cenario(tab, cj_full, "completo")
    print("Cenário alerta precoce...")
    aucs_early = avaliar_cenario(tab, cj_early, "precoce")

    nomes_cj = list(cj_full.keys())
    grafico_auc(aucs_full, aucs_early, nomes_cj)
    grafico_roc(tab, cj_full)
    grafico_importancias(tab, cj_full)

    h3_full = testes_h3(aucs_full, nomes_cj)
    h3_early = testes_h3(aucs_early, nomes_cj)
    h3_full.to_csv(os.path.join(OUT, "h3_curso_completo.csv"), index=False)
    h3_early.to_csv(os.path.join(OUT, "h3_alerta_precoce.csv"), index=False)

    melhor_auc = max(aucs_full[(cj, md)].mean()
                     for cj in nomes_cj for md in COR_MODELO)
    tabela_sintese(h3_full, h3_early,
                   f"melhor AUC do estudo: {melhor_auc:.3f}")

    linhas_md = []
    for rot, aucs in [("completo", aucs_full), ("precoce", aucs_early)]:
        for (cj, md), sc in aucs.items():
            linhas_md.append({"cenario": rot, "conjunto": cj, "modelo": md,
                              "auc_media": sc.mean(), "auc_dp": sc.std()})
    pd.DataFrame(linhas_md).to_csv(os.path.join(OUT, "auc_todos.csv"),
                                   index=False)

    with open(os.path.join(OUT, "resumo.txt"), "w", encoding="utf-8") as f:
        f.write("MODELO PREDITIVO — TESTE DA H3 (FFF 2014J)\n")
        f.write(f"n={len(tab)} alunos | aprovados={n_apr} | "
                f"reprovados={len(tab) - n_apr} | Withdrawn excluído\n")
        f.write("CV: 5-fold x 4 repetições, folds pareados entre conjuntos\n\n")
        for rot, aucs, h3 in [("CURSO COMPLETO", aucs_full, h3_full),
                              ("ALERTA PRECOCE (ciclos 1-2)", aucs_early,
                               h3_early)]:
            f.write(f"== {rot} ==\nAUC médio por conjunto x modelo:\n")
            for cj in nomes_cj:
                vals = "  ".join(f"{md}={aucs[(cj, md)].mean():.4f}"
                                 for md in COR_MODELO)
                f.write(f"  {cj:<24} {vals}\n")
            f.write("Teste H3 (volume+forma vs volume):\n")
            f.write(h3.round(4).to_string(index=False))
            f.write("\n\n")
    print(f"Concluído. Resultados em: {OUT}")


if __name__ == "__main__":
    main()
