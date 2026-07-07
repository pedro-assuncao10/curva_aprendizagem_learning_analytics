#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPLORAÇÃO PRÉ-PASSO 7 — QUANTAS AVALIAÇÕES EXISTEM POR CURSO?
Objetivo: Mapear o calendário do OULAD para descobrir qual é o número mais
frequente de provas nos cursos, ajudando a definir o `N_PROVAS` empiricamente.
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "./dataset"
OUT_DIR  = "./saida_analise_avaliacoes"
os.makedirs(OUT_DIR, exist_ok=True)

class Tee:
    def __init__(self, *s): self.s = s
    def write(self, m):
        for x in self.s: x.write(m)
    def flush(self):
        for x in self.s: x.flush()

_orig = sys.stdout
_log = open(os.path.join(OUT_DIR, "relatorio_avaliacoes.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, _log)

print("="*70)
print("ANÁLISE DE CALENDÁRIO: NÚMERO DE AVALIAÇÕES POR CURSO")
print("="*70)

# Carregamos o arquivo de avaliações
asm = pd.read_csv(os.path.join(DATA_DIR, "assessments.csv"))
asm["curso"] = asm["code_module"] + "_" + asm["code_presentation"]

# Removemos avaliações sem data (geralmente Exames Finais que não constam no log padrão)
n_total = len(asm)
asm_com_data = asm.dropna(subset=["date"])
n_com_data = len(asm_com_data)

print(f"Total de avaliações listadas: {n_total}")
print(f"Avaliações com data válida (utilizáveis): {n_com_data}")
print("-" * 70)

# Contamos quantas avaliações válidas sobraram para cada curso
contagem_provas = asm_com_data.groupby("curso").size()

# Agrupamos os cursos pelo número de provas que eles possuem
distribuicao = contagem_provas.value_counts().sort_index()

print("\n--- DISTRIBUIÇÃO GERAL ---")
print("Nº de Provas  |  Qtd de Cursos")
print("-" * 30)
for n_provas, qtd_cursos in distribuicao.items():
    print(f"  {n_provas:2d} provas   |  {qtd_cursos:2d} cursos")

print("\n--- DETALHAMENTO: QUAIS CURSOS TÊM QUANTAS PROVAS? ---")
for n_provas, qtd_cursos in distribuicao.items():
    cursos_com_n = contagem_provas[contagem_provas == n_provas].index.tolist()
    print(f"\n[{n_provas} Provas] -> Presente em {qtd_cursos} cursos:")
    print(f"  {', '.join(cursos_com_n)}")

plt.figure(figsize=(9, 5))
ax = distribuicao.plot(kind="bar", color="#4C72B0", edgecolor="black")

plt.title("Distribuição do Número de Avaliações por Curso", fontweight="bold")
plt.xlabel("Número Exato de Avaliações (Provas com Data)")
plt.ylabel("Quantidade de Cursos")
plt.xticks(rotation=0)

# Adicionando os rótulos de dados em cima das barras
for container in ax.containers:
    ax.bar_label(container, padding=3, fontweight="bold", color="#333333")

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "distribuicao_avaliacoes.png"), dpi=130)
plt.close()

print("\n" + "="*70)
print(f"Análise concluída. Resultados salvos em: {os.path.abspath(OUT_DIR)}")
print("  - relatorio_avaliacoes.txt (Este texto completo)")
print("  - distribuicao_avaliacoes.png (Gráfico de barras)")

sys.stdout = _orig
_log.close()