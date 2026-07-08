# -*- coding: utf-8 -*-
"""
Matrizes de confusao do modelo preditivo (RF, volume+forma+nota TMA1),
com predicoes fora-do-fold (cross_val_predict, 5-fold estratificado).

Dois cenarios: curso completo (5 ciclos) e alerta precoce (ciclos 1-2).
Saida: resultados_preditivo/matriz_confusao.png + resumo_confusao.txt
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "05_kmeans_curvas_ciclo"))
from kmeans_ciclos import SEED, SURF, INK, SEC, MUT, GRID
from modelo_preditivo import montar_tabela, conjuntos_features, fazer_modelos, OUT

# tons sequenciais da paleta (azul claro -> escuro)
TONS = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]


def celula_cor(frac):
    """Cor pela fracao da classe verdadeira (0..1)."""
    idx = min(int(frac * len(TONS)), len(TONS) - 1)
    return TONS[idx]


def main():
    tab = montar_tabela()
    y = tab["aprovado"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    cenarios = [
        ("Curso completo (5 ciclos)", conjuntos_features([1, 2, 3, 4, 5])),
        ("Alerta precoce (ciclos 1–2)", conjuntos_features([1, 2])),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    relatorios = []
    for ax, (titulo, cjs) in zip(axes, cenarios):
        feats = cjs["volume+forma+nota TMA1"]
        Xf = tab[feats].to_numpy(dtype=float)
        mod = fazer_modelos()["Random Forest"]
        pred = cross_val_predict(mod, Xf, y, cv=cv, n_jobs=-1)

        # linhas = verdade, colunas = predicao; classe 0 = Reprovado
        cm = confusion_matrix(y, pred, labels=[0, 1])
        rotulos = ["Reprovado", "Aprovado"]
        norm = cm / cm.sum(axis=1, keepdims=True)

        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1,
                                           facecolor=celula_cor(norm[i, j]),
                                           edgecolor=SURF, linewidth=3))
                cor_txt = INK if norm[i, j] < 0.5 else "#ffffff"
                ax.text(j + 0.5, 1 - i + 0.58, f"{cm[i, j]:,}",
                        ha="center", va="center", fontsize=17,
                        fontweight="bold", color=cor_txt)
                ax.text(j + 0.5, 1 - i + 0.30,
                        f"{norm[i, j] * 100:.0f}% da classe real",
                        ha="center", va="center", fontsize=8.5, color=cor_txt)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5], [f"previsto\n{r}" for r in rotulos],
                      fontsize=9.5)
        ax.set_yticks([1.5, 0.5], [f"real {r}" for r in rotulos],
                      fontsize=9.5)
        ax.set_aspect("equal")
        ax.grid(visible=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        acc = (pred == y).mean()
        rec_rep = cm[0, 0] / cm[0].sum()      # recall do Reprovado
        prec_rep = cm[0, 0] / max(cm[:, 0].sum(), 1)
        rec_apr = cm[1, 1] / cm[1].sum()
        ax.set_title(f"{titulo}\nacurácia={acc:.1%} · "
                     f"recall Reprovado={rec_rep:.1%} · "
                     f"precisão Reprovado={prec_rep:.1%}",
                     fontsize=10.5)
        relatorios.append((titulo, classification_report(
            y, pred, target_names=rotulos, digits=3)))

    fig.suptitle("Matrizes de confusão — Random Forest, "
                 "volume+forma+nota TMA1 (predições fora-do-fold)",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "matriz_confusao.png"))
    plt.close(fig)

    with open(os.path.join(OUT, "resumo_confusao.txt"), "w",
              encoding="utf-8") as f:
        f.write("MATRIZES DE CONFUSÃO — RF volume+forma+nota TMA1\n")
        f.write("Predições fora-do-fold (5-fold estratificado), "
                "limiar padrão 0,5\n\n")
        for titulo, rel in relatorios:
            f.write(f"== {titulo} ==\n{rel}\n")
    print("Concluído:", os.path.join(OUT, "matriz_confusao.png"))


if __name__ == "__main__":
    main()
