# -*- coding: utf-8 -*-
"""
Quem o aluno E (demografia, disponivel no dia 0) vs o que o aluno FAZ
(comportamento: volume+forma+cobertura+nota TMA1).

Blocos de features:
  demografia    escolaridade previa, faixa etaria, IMD (privacao socio-
                economica), tentativas anteriores, creditos, deficiencia,
                genero - tudo conhecido na matricula (dia 0)
  comportamento volume+forma+cobertura+nota TMA1 (dos cenarios anteriores)
  ambos         uniao

Cenarios: curso completo e alerta precoce (ciclos 1-2). Mesmo protocolo:
RF/Logistica/MLP, CV 5-fold x 4 repeticoes, folds pareados, t pareado.

Saida: resultados_preditivo/demografia_*.png / .csv / resumo_demografia.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "05_kmeans_curvas_ciclo"))
from kmeans_ciclos import (BASE_DIR, DATA_DIR, MODULE, PRES, SEED,
                           SURF, INK, SEC, MUT, GRID, BASE, SERIES)
from modelo_preditivo import (montar_tabela, conjuntos_features,
                              fazer_modelos, OUT, COR_MODELO)

MAPA_EDUCACAO = {"No Formal quals": 0, "Lower Than A Level": 1,
                 "A Level or Equivalent": 2, "HE Qualification": 3,
                 "Post Graduate Qualification": 4}
MAPA_IDADE = {"0-35": 0, "35-55": 1, "55<=": 2}


def montar_demografia():
    info = pd.read_csv(os.path.join(DATA_DIR, "studentInfo.csv"))
    info = info[(info["code_module"] == MODULE) &
                (info["code_presentation"] == PRES)].copy()
    demo = pd.DataFrame({"id_student": info["id_student"]})
    demo["educacao"] = info["highest_education"].map(MAPA_EDUCACAO)
    demo["faixa_etaria"] = info["age_band"].map(MAPA_IDADE)
    imd = (info["imd_band"].astype(str).str.replace("%", "", regex=False)
           .str.split("-").str[0])
    demo["imd"] = pd.to_numeric(imd, errors="coerce")
    demo["tem_imd"] = demo["imd"].notna().astype(int)
    demo["imd"] = demo["imd"].fillna(demo["imd"].median())
    demo["tentativas_previas"] = info["num_of_prev_attempts"]
    demo["creditos"] = info["studied_credits"]
    demo["deficiencia"] = (info["disability"] == "Y").astype(int)
    demo["genero_m"] = (info["gender"] == "M").astype(int)
    return demo


def main():
    tab = montar_tabela()
    demo = montar_demografia()
    tab = tab.merge(demo, on="id_student", how="left")
    feats_demo = ["educacao", "faixa_etaria", "imd", "tem_imd",
                  "tentativas_previas", "creditos", "deficiencia", "genero_m"]
    y = tab["aprovado"].to_numpy()
    print(f"n={len(tab)} | demografia: {len(feats_demo)} features")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEED)
    cenarios = {"Curso completo": [1, 2, 3, 4, 5],
                "Alerta precoce": [1, 2]}

    aucs, linhas = {}, []
    for cen, ciclos in cenarios.items():
        comp = conjuntos_features(ciclos)["volume+forma+nota TMA1"]
        blocos = {"demografia (dia 0)": feats_demo,
                  "comportamento": comp,
                  "demografia+comportamento": feats_demo + comp}
        for nome_bl, feats in blocos.items():
            Xf = tab[feats].to_numpy(dtype=float)
            for nome_md, mod in fazer_modelos().items():
                sc = cross_val_score(mod, Xf, y, cv=cv, scoring="roc_auc",
                                     n_jobs=-1)
                aucs[(cen, nome_bl, nome_md)] = sc
                linhas.append({"cenario": cen, "bloco": nome_bl,
                               "modelo": nome_md, "auc": sc.mean(),
                               "dp": sc.std()})
                print(f"  [{cen}] {nome_bl:<26} {nome_md:<14} "
                      f"AUC={sc.mean():.4f} ±{sc.std():.4f}")
    df = pd.DataFrame(linhas)
    df.to_csv(os.path.join(OUT, "demografia_aucs.csv"), index=False)

    # testes pareados: comportamento vs demografia; ambos vs comportamento
    testes = []
    for cen in cenarios:
        for md in COR_MODELO:
            a_d = aucs[(cen, "demografia (dia 0)", md)]
            a_c = aucs[(cen, "comportamento", md)]
            a_a = aucs[(cen, "demografia+comportamento", md)]
            _, p_cd = stats.ttest_rel(a_c, a_d)
            _, p_ac = stats.ttest_rel(a_a, a_c)
            testes.append({"cenario": cen, "modelo": md,
                           "auc_demo": a_d.mean(), "auc_comp": a_c.mean(),
                           "auc_ambos": a_a.mean(),
                           "delta_comp_vs_demo": (a_c - a_d).mean(),
                           "p_comp_vs_demo": p_cd,
                           "delta_ambos_vs_comp": (a_a - a_c).mean(),
                           "p_ambos_vs_comp": p_ac})
    tdf = pd.DataFrame(testes)
    tdf.to_csv(os.path.join(OUT, "demografia_testes.csv"), index=False)

    # grafico
    blocos_ordem = ["demografia (dia 0)", "comportamento",
                    "demografia+comportamento"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), sharey=True)
    for ax, cen in zip(axes, cenarios):
        xs = np.arange(len(blocos_ordem))
        larg = 0.24
        for j, md in enumerate(COR_MODELO):
            medias = [aucs[(cen, bl, md)].mean() for bl in blocos_ordem]
            erros = [aucs[(cen, bl, md)].std() for bl in blocos_ordem]
            pos = xs + (j - 1) * larg
            ax.bar(pos, medias, width=larg * 0.9, color=COR_MODELO[md],
                   edgecolor=SURF, linewidth=1.5,
                   label=md if ax is axes[0] else None)
            ax.errorbar(pos, medias, yerr=erros, fmt="none", ecolor=SEC,
                        elinewidth=1, capsize=2)
            for x, m in zip(pos, medias):
                ax.text(x, m + 0.012, f"{m:.3f}", ha="center", fontsize=7.5,
                        color=SEC, rotation=90)
        ax.set_xticks(xs, [b.replace("+", "\n+").replace(" (", "\n(")
                           for b in blocos_ordem], fontsize=9)
        ax.axhline(0.5, color=BASE, linewidth=1, linestyle="--")
        ax.set_title(cen)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0.45, 0.97)
    axes[0].set_ylabel("AUC (média de 20 folds)")
    leg = axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(SEC)
    fig.suptitle("Quem o aluno É (demografia) vs o que o aluno FAZ "
                 "(comportamento)", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "demografia_comparativo.png"))
    plt.close(fig)

    with open(os.path.join(OUT, "resumo_demografia.txt"), "w",
              encoding="utf-8") as f:
        f.write("DEMOGRAFIA vs COMPORTAMENTO — AUC (CV 5x4, folds pareados)\n\n")
        f.write(df.round(4).to_string(index=False))
        f.write("\n\nTestes pareados:\n")
        f.write(tdf.round(4).to_string(index=False))
    print("Concluído:", OUT)


if __name__ == "__main__":
    main()
