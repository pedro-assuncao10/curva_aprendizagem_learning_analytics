# -*- coding: utf-8 -*-
"""
Curva precision-recall e otimizacao do limiar de alerta.

Classe-alvo = REPROVADO (e quem o sistema de alerta quer capturar).
Probabilidades fora-do-fold (RF, volume+forma+nota TMA1), 5-fold.

Para cada cenario (completo, precoce):
  - curva precision-recall com pontos de operacao marcados
    (limiar padrao 0,5; melhor F1; melhor F2 - que pesa recall 2x);
  - tabela de limiares para metas de recall 70/80/90%;
  - matriz de confusao re-calculada no limiar recomendado (melhor F2).

Saida: resultados_preditivo/curva_pr.png, confusao_limiar_otimizado.png,
       limiares.csv, resumo_limiar.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_curve, confusion_matrix

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "05_kmeans_curvas_ciclo"))
from kmeans_ciclos import SEED, SURF, INK, SEC, MUT, GRID, BASE, SERIES
from modelo_preditivo import montar_tabela, conjuntos_features, fazer_modelos, OUT

TONS = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]


def fbeta(prec, rec, beta):
    b2 = beta * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (1 + b2) * prec * rec / (b2 * prec + rec)
    return np.nan_to_num(f)


def analisar(tab, ciclos, titulo):
    y_rep = (tab["aprovado"] == 0).to_numpy().astype(int)   # 1 = reprovado
    feats = conjuntos_features(ciclos)["volume+forma+nota TMA1"]
    Xf = tab[feats].to_numpy(dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    prob_rep = cross_val_predict(fazer_modelos()["Random Forest"], Xf,
                                 tab["aprovado"].to_numpy(), cv=cv,
                                 method="predict_proba", n_jobs=-1)[:, 0]

    prec, rec, thr = precision_recall_curve(y_rep, prob_rep)
    # precision_recall_curve devolve thr com len-1; alinhar descartando o ultimo pt
    prec_t, rec_t = prec[:-1], rec[:-1]
    f1 = fbeta(prec_t, rec_t, 1.0)
    f2 = fbeta(prec_t, rec_t, 2.0)
    i_f1, i_f2 = int(np.argmax(f1)), int(np.argmax(f2))

    def ponto_limiar(t):
        p = (prob_rep >= t).astype(int)
        cm = confusion_matrix(y_rep, p, labels=[1, 0])
        recall = cm[0, 0] / cm[0].sum()
        precisao = cm[0, 0] / max(cm[:, 0].sum(), 1)
        return recall, precisao

    metas = []
    for alvo in (0.70, 0.80, 0.90):
        ok = rec_t >= alvo
        if ok.any():
            j = int(np.argmax(prec_t * ok))
            metas.append({"meta_recall": alvo, "limiar": thr[j],
                          "recall": rec_t[j], "precisao": prec_t[j],
                          "alarmes": int((prob_rep >= thr[j]).sum())})
    return {"titulo": titulo, "prob": prob_rep, "y": y_rep,
            "prec": prec, "rec": rec, "thr": thr,
            "f1_idx": i_f1, "f2_idx": i_f2, "f1": f1, "f2": f2,
            "prec_t": prec_t, "rec_t": rec_t,
            "p05": ponto_limiar(0.5), "metas": metas,
            "base_rate": y_rep.mean()}


def main():
    tab = montar_tabela()
    res = [analisar(tab, [1, 2, 3, 4, 5], "Curso completo (5 ciclos)"),
           analisar(tab, [1, 2], "Alerta precoce (ciclos 1–2)")]

    # ---- curvas PR
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, r in zip(axes, res):
        ax.plot(r["rec"], r["prec"], color=SERIES[0], linewidth=2)
        ax.axhline(r["base_rate"], color=BASE, linewidth=1, linestyle="--")
        ax.annotate(f"taxa-base = {r['base_rate']:.0%}",
                    (0.02, r["base_rate"]), textcoords="offset points",
                    xytext=(2, 5), fontsize=8, color=MUT)
        # pontos de operacao
        pontos = [("limiar 0,5", r["p05"][0], r["p05"][1], SERIES[4]),
                  (f"melhor F1 (limiar {r['thr'][r['f1_idx']]:.2f})",
                   r["rec_t"][r["f1_idx"]], r["prec_t"][r["f1_idx"]],
                   SERIES[2]),
                  (f"melhor F2 (limiar {r['thr'][r['f2_idx']]:.2f})",
                   r["rec_t"][r["f2_idx"]], r["prec_t"][r["f2_idx"]],
                   SERIES[5])]
        for nome, rc, pc, cor in pontos:
            ax.scatter([rc], [pc], s=70, color=cor, zorder=5,
                       edgecolor=SURF, linewidth=1.5)
            ax.annotate(nome, (rc, pc), textcoords="offset points",
                        xytext=(8, 6), fontsize=8.5, color=cor,
                        fontweight="bold")
        ax.set_xlabel("recall do Reprovado (fração dos em risco capturada)")
        ax.set_ylabel("precisão do alerta")
        ax.set_title(r["titulo"])
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
    fig.suptitle("Curva precision-recall — classe-alvo: Reprovado "
                 "(RF, volume+forma+nota TMA1)", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "curva_pr.png"))
    plt.close(fig)

    # ---- matriz de confusao no limiar recomendado (melhor F2)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, r in zip(axes, res):
        t = float(r["thr"][r["f2_idx"]])
        pred = (r["prob"] >= t).astype(int)
        cm = confusion_matrix(r["y"], pred, labels=[1, 0])
        rotulos = ["Reprovado", "Aprovado"]
        norm = cm / cm.sum(axis=1, keepdims=True)
        for i in range(2):
            for j in range(2):
                frac = norm[i, j]
                cor = TONS[min(int(frac * len(TONS)), len(TONS) - 1)]
                ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, facecolor=cor,
                                           edgecolor=SURF, linewidth=3))
                cor_txt = INK if frac < 0.5 else "#ffffff"
                ax.text(j + 0.5, 1 - i + 0.58, f"{cm[i, j]:,}", ha="center",
                        va="center", fontsize=17, fontweight="bold",
                        color=cor_txt)
                ax.text(j + 0.5, 1 - i + 0.30,
                        f"{frac * 100:.0f}% da classe real", ha="center",
                        va="center", fontsize=8.5, color=cor_txt)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5], [f"previsto\n{x}" for x in rotulos],
                      fontsize=9.5)
        ax.set_yticks([1.5, 0.5], [f"real {x}" for x in rotulos],
                      fontsize=9.5)
        ax.set_aspect("equal")
        ax.grid(visible=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        recall = cm[0, 0] / cm[0].sum()
        precisao = cm[0, 0] / max(cm[:, 0].sum(), 1)
        ax.set_title(f"{r['titulo']} — limiar {t:.2f} (melhor F2)\n"
                     f"recall Reprovado={recall:.1%} · "
                     f"precisão={precisao:.1%}", fontsize=10.5)
    fig.suptitle("Matrizes de confusão no limiar otimizado para alerta "
                 "(F2: recall vale 2×)", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "confusao_limiar_otimizado.png"))
    plt.close(fig)

    # ---- tabela de limiares
    linhas = []
    for r in res:
        linhas.append({"cenario": r["titulo"], "ponto": "limiar 0,50",
                       "limiar": 0.5, "recall": r["p05"][0],
                       "precisao": r["p05"][1]})
        for nome, idx in [("melhor F1", r["f1_idx"]),
                          ("melhor F2", r["f2_idx"])]:
            linhas.append({"cenario": r["titulo"], "ponto": nome,
                           "limiar": float(r["thr"][idx]),
                           "recall": float(r["rec_t"][idx]),
                           "precisao": float(r["prec_t"][idx])})
        for m in r["metas"]:
            linhas.append({"cenario": r["titulo"],
                           "ponto": f"meta recall {m['meta_recall']:.0%}",
                           "limiar": float(m["limiar"]),
                           "recall": float(m["recall"]),
                           "precisao": float(m["precisao"])})
    df = pd.DataFrame(linhas)
    df.to_csv(os.path.join(OUT, "limiares.csv"), index=False)

    with open(os.path.join(OUT, "resumo_limiar.txt"), "w",
              encoding="utf-8") as f:
        f.write("OTIMIZAÇÃO DO LIMIAR DE ALERTA (classe-alvo: Reprovado)\n")
        f.write("Probabilidades fora-do-fold; F2 pesa recall 2x a precisão\n\n")
        f.write(df.round(3).to_string(index=False))
    print(df.round(3).to_string(index=False))
    print("Concluído:", OUT)


if __name__ == "__main__":
    main()
