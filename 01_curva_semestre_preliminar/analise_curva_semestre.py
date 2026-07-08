#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
ANÁLISE PRELIMINAR — A CURVA DO SEMESTRE INTEIRO (todo o OULAD)
Fusão dos antigos passos 1 (forma), 2 (volume) e 3 (durabilidade).
=============================================================================

Este estudo precede o pipeline final (curvas por ciclo de avaliação) e
responde três perguntas sobre a curva de engajamento do SEMESTRE INTEIRO:

  PARTE A (volume) ..... quanto cada aluno clicou no total? quem clica mais
                         passa mais? (baseline do projeto)
  PARTE B (forma) ...... normalizando o tempo pela duração do curso (30
                         fatias) e a curva pela soma do aluno, que "formas"
                         de estudo existem? (mediana, heatmap, forma x volume)
  PARTE C (durabilidade) o grupo "começo-pesado" mistura dois opostos:
                         quem ABANDONOU cedo e quem ANTECIPOU e sustentou.
                         Semanas mortas + atividade no último terço os separam.

Saídas: 6 gráficos + saida_terminal.txt + features_durabilidade.csv
        (este CSV é INSUMO de saida_passo6/passo6.py e saida_passo20/passo20.py)

COMO RODAR (da raiz do projeto):
  python 01_curva_semestre_preliminar/analise_curva_semestre.py
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "./dataset"
OUT_DIR = "./01_curva_semestre_preliminar"
N_FATIAS = 30            # resolução temporal da curva do semestre
MIN_CLIQUES = 30         # piso de atividade (abaixo disso a "forma" é ruído)
MIN_FATIAS_ATIVAS = 3
LIMIAR_TIMING = 0.10     # |p_comeco - p_fim| acima disso define o timing
PLOT_AMOSTRA = 200
os.makedirs(OUT_DIR, exist_ok=True)


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, msg):
        for s in self.streams: s.write(msg)
    def flush(self):
        for s in self.streams: s.flush()


_log = open(os.path.join(OUT_DIR, "saida_terminal.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, _log)


def chave(df):
    return (df["code_module"] + "_" + df["code_presentation"]
            + "_" + df["id_student"].astype(str))


# ===========================================================================
# CARGA (uma leitura só do studentVle serve às três partes)
# ===========================================================================
print("=" * 70)
print("ANÁLISE PRELIMINAR — CURVA DO SEMESTRE (todo o OULAD)")
print("=" * 70)

courses = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
courses["curso"] = courses["code_module"] + "_" + courses["code_presentation"]
dur = dict(zip(courses["curso"], courses["module_presentation_length"]))

info = pd.read_csv(os.path.join(DATA_DIR, "studentInfo.csv"),
                   usecols=["code_module", "code_presentation",
                            "id_student", "final_result"])
info["aluno"] = chave(info)
resultado = info.set_index("aluno")["final_result"]

print("lendo studentVle.csv (~10,6M linhas)...")
vle = pd.read_csv(os.path.join(DATA_DIR, "studentVle.csv"),
                  usecols=["code_module", "code_presentation",
                           "id_student", "date", "sum_click"],
                  dtype={"id_student": np.int32, "date": np.int16,
                         "sum_click": np.int32})
vle["curso"] = vle["code_module"] + "_" + vle["code_presentation"]
vle["aluno"] = chave(vle)
vle["length"] = vle["curso"].map(dur)

# ===========================================================================
# PARTE A — VOLUME: quem clica mais passa mais? (baseline)
# ===========================================================================
print("\n" + "-" * 70)
print("PARTE A — VOLUME")
print("-" * 70)

volume = vle.groupby("aluno")["sum_click"].sum().rename("total_cliques")
faixa = pd.qcut(volume, 3, labels=["baixo", "medio", "alto"])
cortes = volume.quantile([1 / 3, 2 / 3]).round().astype(int).tolist()
print(f"Alunos com >=1 clique: {len(volume):,}")
print(f"Tercis de volume: baixo < {cortes[0]} <= medio <= {cortes[1]} < alto")

tabA = pd.DataFrame({"total_cliques": volume, "faixa_volume": faixa,
                     "final_result": resultado.reindex(volume.index)})
tabA["aprovado"] = tabA["final_result"].isin(["Pass", "Distinction"])
taxa = tabA.groupby("faixa_volume", observed=True)["aprovado"].mean().mul(100)
print("\n>>> BASELINE — aprovação (Pass+Distinction) por faixa de volume:")
print(taxa.round(1).to_string())

plt.figure(figsize=(8, 4.5))
plt.hist(np.log10(volume.clip(lower=1)), bins=50, color="#4C72B0")
for q, lab in zip([1 / 3, 2 / 3], ["corte baixo|medio", "corte medio|alto"]):
    v = np.log10(volume.quantile(q))
    plt.axvline(v, color="crimson", ls="--")
    plt.text(v, plt.ylim()[1] * 0.9, lab, rotation=90, va="top", fontsize=8)
plt.xlabel("log10(total de cliques)")
plt.ylabel("nº de alunos")
plt.title("Distribuição do VOLUME total (linhas = tercis)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_distribuicao_volume.png"), dpi=130)
plt.close()

comp = pd.crosstab(tabA["faixa_volume"], tabA["final_result"],
                   normalize="index") * 100
comp = comp[[c for c in ["Distinction", "Pass", "Fail", "Withdrawn"]
             if c in comp.columns]]
plt.figure(figsize=(8, 5))
ax = comp.plot(kind="bar", stacked=True, ax=plt.gca(),
               color={"Distinction": "#2E7D32", "Pass": "#8BC34A",
                      "Fail": "#FF7043", "Withdrawn": "#9E9E9E"})
for c in ax.containers:
    labels = [f"{v.get_height():.1f}%" if v.get_height() > 3 else "" for v in c]
    ax.bar_label(c, labels=labels, label_type="center", color="white",
                 fontsize=9, fontweight="bold")
plt.ylabel("% dos alunos da faixa")
plt.xlabel("faixa de volume")
plt.title("Resultado final por faixa de volume (baseline do projeto)")
plt.xticks(rotation=0)
plt.legend(title="", bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_resultado_por_volume.png"), dpi=130)
plt.close()

# ===========================================================================
# PARTE B — FORMA: a curva do semestre em 30 fatias de tempo relativo
# ===========================================================================
print("\n" + "-" * 70)
print("PARTE B — FORMA (curva do semestre, sem desistentes)")
print("-" * 70)

dentro = vle[(vle["date"] >= 0) & (vle["date"] <= vle["length"])].copy()
frac = dentro["date"] / dentro["length"].clip(lower=1)
dentro["fatia"] = np.minimum((frac * N_FATIAS).astype(int), N_FATIAS - 1)
dentro["terco"] = np.where(frac < 1 / 3, "comeco",
                           np.where(frac < 2 / 3, "meio", "fim"))

serie = (dentro.groupby(["aluno", "fatia"])["sum_click"].sum()
         .unstack(fill_value=0).reindex(columns=range(N_FATIAS), fill_value=0))

total = serie.sum(axis=1)
ativas = (serie > 0).sum(axis=1)
withdrawn = set(info.loc[info["final_result"] == "Withdrawn", "aluno"])
mask = ((total >= MIN_CLIQUES) & (ativas >= MIN_FATIAS_ATIVAS)
        & ~serie.index.isin(withdrawn))
print(f"Alunos com clique no curso: {len(serie):,}")
print(f"  removidos por esparsidade (<{MIN_CLIQUES} cliques ou "
      f"<{MIN_FATIAS_ATIVAS} fatias ativas) ou desistência: {(~mask).sum():,}")
serie = serie[mask]
total = total[mask]
print(f"Base da forma: {len(serie):,} alunos x {N_FATIAS} fatias")

forma = serie.div(serie.sum(axis=1), axis=0)
P = forma.values
eixo = np.arange(N_FATIAS) / (N_FATIAS - 1)
centro_massa = (P * np.arange(N_FATIAS)).sum(axis=1) / (N_FATIAS - 1)
faixa_B = pd.qcut(total, 3, labels=["baixo", "medio", "alto"])

plt.figure(figsize=(9, 5))
plt.plot(eixo, np.median(P, axis=0), lw=2, label="forma mediana")
plt.fill_between(eixo, np.percentile(P, 25, axis=0),
                 np.percentile(P, 75, axis=0), alpha=0.2,
                 label="intervalo interquartil")
plt.xlabel("fração do curso (0=início, 1=fim)")
plt.ylabel("proporção do esforço")
plt.title("Curva representativa do semestre (mediana e IQR)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_curva_mediana_forma.png"), dpi=130)
plt.close()

amostra = pd.Series(centro_massa, index=forma.index).sample(
    min(PLOT_AMOSTRA, len(forma)), random_state=0).sort_values().index
plt.figure(figsize=(9, 6))
plt.imshow(forma.loc[amostra].values, aspect="auto", cmap="magma",
           extent=[0, 1, 0, len(amostra)])
plt.colorbar(label="proporção do esforço")
plt.xlabel("fração do curso")
plt.ylabel("alunos (ordenados por centro de massa)")
plt.title(f"Heatmap da forma — {len(amostra)} alunos amostrados")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_heatmap_forma.png"), dpi=130)
plt.close()

plt.figure(figsize=(9, 5))
for fx in ["baixo", "medio", "alto"]:
    idx = faixa_B.index[faixa_B == fx]
    plt.plot(eixo, forma.loc[idx].values.mean(axis=0), lw=2,
             label=f"volume {fx}")
plt.xlabel("fração do curso")
plt.ylabel("proporção do esforço")
plt.title("Forma média por faixa de volume — a forma quase não muda com o volume")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_forma_por_faixa_volume.png"), dpi=130)
plt.close()

# ===========================================================================
# PARTE C — DURABILIDADE: abandonador x antecipador (e o CSV-insumo)
# ===========================================================================
print("\n" + "-" * 70)
print("PARTE C — DURABILIDADE (abandonou cedo vs antecipou e sustentou)")
print("-" * 70)

ter = (dentro.groupby(["aluno", "terco"])["sum_click"].sum()
       .unstack(fill_value=0))
for c in ["comeco", "meio", "fim"]:
    if c not in ter:
        ter[c] = 0
ter = ter.reindex(serie.index).fillna(0)[["comeco", "meio", "fim"]]

res = resultado.reindex(serie.index)
manter = res.isin(["Pass", "Fail", "Distinction"]).values
serieC, terC, resC = serie[manter], ter[manter], res[manter]
totalC = serieC.sum(axis=1)
prop = terC.div(totalC, axis=0)

dfC = pd.DataFrame({
    "total": totalC.values,
    "p_comeco": prop["comeco"].values,
    "p_meio": prop["meio"].values,
    "p_fim": prop["fim"].values,
    "semanas_mortas": (serieC.values == 0).sum(axis=1),
    "cliques_ultimo_terco": terC["fim"].values,
    "ativo_no_fim": (terC["fim"] > 0).values,
    "final_result": resC.values,
}, index=serieC.index)
dfC["aprovado"] = dfC["final_result"].isin(["Pass", "Distinction"])
dfC["faixa_volume"] = pd.qcut(dfC["total"], 3, labels=["baixo", "medio", "alto"])
dif = dfC["p_comeco"] - dfC["p_fim"]
dfC["timing"] = np.where(dif > LIMIAR_TIMING, "comeco-pesado",
                np.where(dif < -LIMIAR_TIMING, "fim-pesado", "equilibrado"))

cp = dfC[dfC["timing"] == "comeco-pesado"].copy()
cp["subtipo"] = np.where(cp["ativo_no_fim"], "antecipou_sustentou",
                         "abandonou_cedo")
print(f"Base (sem desistentes): {len(dfC):,} | começo-pesado: {len(cp):,}")
print("\nSubtipos dentro de começo-pesado:")
print(cp["subtipo"].value_counts().to_string())
taxa_cp = (cp.groupby(["faixa_volume", "subtipo"], observed=True)["aprovado"]
           .mean().mul(100).round(1).unstack()
           .reindex(index=["baixo", "medio", "alto"]))
print("\n>>> APROVAÇÃO (%) — abandonou cedo vs antecipou e sustentou:")
print(taxa_cp.to_string())

ax = taxa_cp.plot(kind="bar", figsize=(8.5, 5),
                  color={"abandonou_cedo": "#C62828",
                         "antecipou_sustentou": "#2E7D32"})
plt.ylabel("taxa de aprovação (%)")
plt.xlabel("faixa de volume")
plt.title("Começo-pesado dividido: abandonou cedo vs antecipou e sustentou")
plt.xticks(rotation=0)
plt.ylim(0, 100)
plt.legend(title="", bbox_to_anchor=(1.02, 1))
for cont in ax.containers:
    ax.bar_label(cont, fmt="%.0f", fontsize=8, padding=2)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "06_abandonou_vs_sustentou.png"), dpi=130)
plt.close()

# CSV-insumo dos estudos seguintes (passo6 e passo20 leem este arquivo)
dfC.drop(columns="aprovado").to_csv(
    os.path.join(OUT_DIR, "features_durabilidade.csv"))

print("\n" + "=" * 70)
print(f"Gravado em {os.path.abspath(OUT_DIR)}:")
print("  01_distribuicao_volume.png      02_resultado_por_volume.png")
print("  03_curva_mediana_forma.png      04_heatmap_forma.png")
print("  05_forma_por_faixa_volume.png   06_abandonou_vs_sustentou.png")
print("  features_durabilidade.csv       (insumo do passo6 e do passo20)")
print("  saida_terminal.txt")
sys.stdout = sys.__stdout__
_log.close()
