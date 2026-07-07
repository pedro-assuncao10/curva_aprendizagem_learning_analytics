#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PASSO 2 — VOLUME
Uma responsabilidade: quanto cada aluno clicou no total.
Soma os cliques por aluno, separa em faixas (pouco/médio/muito) e cruza
com o resultado final para um primeiro insight: clicar mais => passar mais?
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "./dataset"
OUT_DIR  = "./saida_passo2"
os.makedirs(OUT_DIR, exist_ok=True)

# espelha tudo que vai pro terminal também num arquivo .txt
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, msg):
        for s in self.streams: s.write(msg)
    def flush(self):
        for s in self.streams: s.flush()

_log = open(os.path.join(OUT_DIR, "saida_terminal.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, _log)

print("="*70)
print("PASSO 2 — VOLUME")
print("="*70)

# chave de aluno = módulo + apresentação + id (mesmo id em cursos != é aluno diferente)
def chave(df):
    return df["code_module"] + "_" + df["code_presentation"] + "_" + df["id_student"].astype(str)

# --- 1. somar os cliques de cada aluno (a definição de "volume")
vle = pd.read_csv(os.path.join(DATA_DIR, "studentVle.csv"),
                  usecols=["code_module", "code_presentation", "id_student", "sum_click"])
vle["aluno"] = chave(vle)
volume = vle.groupby("aluno")["sum_click"].sum().rename("total_cliques")
print(f"\nAlunos com pelo menos 1 clique: {len(volume):,}")
print(f"Cliques por aluno -> min={volume.min()}, mediana={int(volume.median())}, "
      f"média={volume.mean():.0f}, max={volume.max()}")

# --- 2. separar em faixas pouco/médio/muito (tercis: cada faixa = 1/3 dos alunos)
faixa = pd.qcut(volume, 3, labels=["baixo", "medio", "alto"])
cortes = volume.quantile([1/3, 2/3]).round().astype(int).tolist()
print(f"\nCortes dos tercis: baixo < {cortes[0]} cliques <= medio <= {cortes[1]} < alto")
print(faixa.value_counts().sort_index().to_string())

# --- 3. cruzar com o resultado final (de studentInfo) -> o insight
info = pd.read_csv(os.path.join(DATA_DIR, "studentInfo.csv"),
                   usecols=["code_module", "code_presentation", "id_student", "final_result"])
info["aluno"] = chave(info)
res = info.set_index("aluno")["final_result"]

tab = pd.DataFrame({"total_cliques": volume, "faixa_volume": faixa, "final_result": res})

# taxa de aprovação (Pass + Distinction) por faixa de volume
tab["aprovado"] = tab["final_result"].isin(["Pass", "Distinction"])
taxa = tab.groupby("faixa_volume", observed=True)["aprovado"].mean().mul(100).round(1)
print("\n>>> INSIGHT — taxa de aprovação (Pass+Distinction) por faixa de volume:")
print(taxa.to_string())

# --- gráfico 1: distribuição do volume com os cortes
plt.figure(figsize=(8, 4.5))
# ... existing code ...
plt.title("Distribuição do VOLUME total (linhas = cortes pouco/médio/muito)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_distribuicao_volume.png"), dpi=130)
plt.close()

# --- gráfico 2: o insight — composição de resultado por faixa de volume
# CORREÇÃO: simplificando o agrupamento para evitar os rótulos duplos no eixo X
comp = pd.crosstab(tab["faixa_volume"], tab["final_result"], normalize='index') * 100

ordem = ["Distinction", "Pass", "Fail", "Withdrawn"]
comp = comp[[c for c in ordem if c in comp.columns]]

plt.figure(figsize=(8, 5))
ax = comp.plot(kind="bar", stacked=True, ax=plt.gca(),
               color={"Distinction": "#2E7D32", "Pass": "#8BC34A",
                      "Fail": "#FF7043", "Withdrawn": "#9E9E9E"})

plt.ylabel("% dos alunos da faixa"); plt.xlabel("Faixa de Volume")
plt.title("Resultado final por faixa de volume")
plt.xticks(rotation=0)

# ADIÇÃO: Escrever as porcentagens dentro das barras
for c in ax.containers:
    # Adiciona o label apenas se a barra tiver um tamanho relevante (ex: > 3%) para não encavalar texto
    labels = [f"{v.get_height():.1f}%" if v.get_height() > 3 else "" for v in c]
    ax.bar_label(c, labels=labels, label_type='center', color='white', fontsize=9, fontweight='bold')

plt.legend(title="", bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_resultado_por_volume.png"), dpi=130)
plt.close()

# --- saída
tab.drop(columns="aprovado").to_csv(os.path.join(OUT_DIR, "volume_por_aluno.csv"))

print("\n" + "="*70)
print(f"Gravado em {os.path.abspath(OUT_DIR)}:")
print("  - volume_por_aluno.csv          (aluno, total_cliques, faixa_volume, final_result)")
print("  - 01_distribuicao_volume.png    (quantos alunos em cada nível de volume)")
print("  - 02_resultado_por_volume.png   (o insight: passa mais quem clica mais?)")
print("  - saida_terminal.txt            (tudo que apareceu no terminal)")
print("\nPasso 2 OK. Próximo: Passo 3 — forma.")

_log.close()