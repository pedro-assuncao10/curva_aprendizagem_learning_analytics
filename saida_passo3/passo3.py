#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PASSO 4.2 — SEPARAR ABANDONADOR DE ANTECIPADOR
O grupo "começo-pesado" do Passo 4 era suspeito: misturava dois alunos opostos
com o mesmo centro de massa no início:
  - ABANDONOU CEDO    -> clicou no começo e zerou o resto do curso
  - ANTECIPOU/SUSTENTOU-> pico no começo, mas manteve atividade até o fim
O que os distingue é a DURABILIDADE do engajamento. Aqui medimos isso com:
  - semanas mortas (fatias do curso com zero clique)
  - atividade no último terço do curso
e mostramos que os dois têm aprovação MUITO diferente.
Também salvamos essas features de durabilidade para o K-means (passo 5).
Desistentes removidos.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR  = "./dataset"
OUT_DIR   = "./saida_passo3"
N_FATIAS  = 30
MIN_CLIQUES = 30
LIMIAR_TIMING = 0.10
os.makedirs(OUT_DIR, exist_ok=True)

class Tee:
    def __init__(self, *s): self.s = s
    def write(self, m):
        for x in self.s: x.write(m)
    def flush(self):
        for x in self.s: x.flush()
_log = open(os.path.join(OUT_DIR, "saida_terminal.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, _log)

print("="*70)
print("PASSO 4.2 — ABANDONADOR vs ANTECIPADOR")
print("="*70)

def chave(df):
    return df["code_module"] + "_" + df["code_presentation"] + "_" + df["id_student"].astype(str)

courses = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
courses["curso"] = courses["code_module"] + "_" + courses["code_presentation"]
dur = dict(zip(courses["curso"], courses["module_presentation_length"]))

vle = pd.read_csv(os.path.join(DATA_DIR, "studentVle.csv"),
                  usecols=["code_module", "code_presentation", "id_student", "date", "sum_click"])
vle["curso"] = vle["code_module"] + "_" + vle["code_presentation"]
vle["aluno"] = chave(vle)
vle["length"] = vle["curso"].map(dur)

dentro = vle[(vle["date"] >= 0) & (vle["date"] <= vle["length"])].copy()
frac = dentro["date"] / dentro["length"].clip(lower=1)
dentro["fatia"] = np.minimum((frac * N_FATIAS).astype(int), N_FATIAS - 1)
dentro["terco"] = np.where(frac < 1/3, "comeco", np.where(frac < 2/3, "meio", "fim"))

# série fina (30 fatias) -> p/ semanas mortas
serie = (dentro.groupby(["aluno", "fatia"])["sum_click"].sum()
                .unstack(fill_value=0).reindex(columns=range(N_FATIAS), fill_value=0))
serie = serie[serie.sum(axis=1) >= MIN_CLIQUES]

# cliques por terço -> p/ proporções e atividade no fim
ter = dentro.groupby(["aluno", "terco"])["sum_click"].sum().unstack(fill_value=0)
for c in ["comeco", "meio", "fim"]:
    if c not in ter: ter[c] = 0
ter = ter.reindex(serie.index)[["comeco", "meio", "fim"]]

# resultado + remover desistentes
info = pd.read_csv(os.path.join(DATA_DIR, "studentInfo.csv"),
                   usecols=["code_module", "code_presentation", "id_student", "final_result"])
info["aluno"] = chave(info)
res = info.set_index("aluno")["final_result"].reindex(serie.index)
manter = res.isin(["Pass", "Fail", "Distinction"]).values
serie, ter, res = serie[manter], ter[manter], res[manter]

total = serie.sum(axis=1)
prop = ter.div(total, axis=0)
prop.columns = ["p_comeco", "p_meio", "p_fim"]

# ---- features de DURABILIDADE
semanas_mortas    = (serie.values == 0).sum(axis=1)
cliques_ult_terco = ter["fim"].values
ativo_no_fim      = cliques_ult_terco > 0

df = pd.DataFrame({
    "total": total.values,
    "p_comeco": prop["p_comeco"].values,
    "p_meio":   prop["p_meio"].values,
    "p_fim":    prop["p_fim"].values,
    "semanas_mortas": semanas_mortas,
    "cliques_ultimo_terco": cliques_ult_terco,
    "ativo_no_fim": ativo_no_fim,
    "final_result": res.values,
}, index=serie.index)
df["aprovado"] = df["final_result"].isin(["Pass", "Distinction"])
df["faixa_volume"] = pd.qcut(df["total"], 3, labels=["baixo", "medio", "alto"])

# timing (igual passo 4)
dif = df["p_comeco"] - df["p_fim"]
df["timing"] = np.where(dif > LIMIAR_TIMING, "comeco-pesado",
               np.where(dif < -LIMIAR_TIMING, "fim-pesado", "equilibrado"))

# ---- DENTRO de começo-pesado: separar os dois subtipos
cp = df[df["timing"] == "comeco-pesado"].copy()
cp["subtipo"] = np.where(cp["ativo_no_fim"], "antecipou_sustentou", "abandonou_cedo")

print(f"\nBase total (sem desistentes): {len(df):,}")
print(f"Grupo 'começo-pesado':        {len(cp):,}")
print("\nDentro de começo-pesado, os dois subtipos:")
print(cp["subtipo"].value_counts().to_string())
print(f"\nDurabilidade média por subtipo:")
print(cp.groupby("subtipo")[["semanas_mortas", "p_fim"]].mean().round(3).to_string())

# aprovação por subtipo x faixa de volume
ordem_v = ["baixo", "medio", "alto"]
taxa = (cp.groupby(["faixa_volume", "subtipo"], observed=True)["aprovado"]
          .mean().mul(100).round(1).unstack().reindex(index=ordem_v))
qtd = (cp.groupby(["faixa_volume", "subtipo"], observed=True).size()
         .unstack().reindex(index=ordem_v))
print("\n>>> APROVAÇÃO (%) dentro de começo-pesado: abandonou vs sustentou")
print(taxa.to_string())
print("\n    (nº de alunos:)")
print(qtd.to_string())

# ---- gráfico
ax = taxa.plot(kind="bar", figsize=(8.5, 5),
               color={"abandonou_cedo": "#C62828", "antecipou_sustentou": "#2E7D32"})
plt.ylabel("taxa de aprovação (%)"); plt.xlabel("faixa de volume")
plt.title("Começo-pesado dividido: abandonou cedo vs antecipou e sustentou")
plt.xticks(rotation=0); plt.ylim(0, 100)
plt.legend(title="", bbox_to_anchor=(1.02, 1))
for cont in ax.containers:
    ax.bar_label(cont, fmt="%.0f", fontsize=8, padding=2)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_abandonou_vs_sustentou.png"), dpi=130)
plt.close()

# ---- saídas (features prontas p/ o K-means do passo 5)
feats = df.drop(columns="aprovado")
feats.to_csv(os.path.join(OUT_DIR, "features_durabilidade.csv"))
cp.drop(columns="aprovado").to_csv(os.path.join(OUT_DIR, "comeco_pesado_detalhado.csv"))

print("\n" + "="*70)
print(f"Gravado em {os.path.abspath(OUT_DIR)}:")
print("  - features_durabilidade.csv     (todos os alunos + semanas_mortas, p_fim, ativo_no_fim... p/ o K-means)")
print("  - comeco_pesado_detalhado.csv   (só começo-pesado, com o subtipo)")
print("  - 01_abandonou_vs_sustentou.png (a prova de que os dois subtipos diferem)")
print("  - saida_terminal.txt")
print("\nPasso 4.2 OK. A estrutura de durabilidade está pronta para o K-means (passo 5).")
_log.close()