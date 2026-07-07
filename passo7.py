#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PASSO 7 — CURVA REAL POR AVALIAÇÃO (Intervalo Completo)
Ajuste Metodológico: Em vez de usar uma janela fixa de véspera (W dias), 
o script agora atribui TODO o volume de cliques de um aluno ao intervalo 
entre uma avaliação e a próxima.
Filtro: APENAS cursos que possuam um número FIXO de avaliações (TARGET=13).

Requer: pandas, numpy, matplotlib, scikit-learn
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

DATA_DIR = "./dataset"
DUR_CSV  = "./saida_passo3/features_durabilidade.csv"
OUT_DIR  = "./saida_passo7"
# Várias linhas de assessments.csv compartilham a MESMA data (ex.: 7 CMAs + Exam
# no mesmo dia). O que importa para os intervalos é o nº de DATAS ÚNICAS, não de
# linhas — por isso o filtro usa datas únicas em vez de TARGET_N_PROVAS de linhas.
TARGET_N_PROVAS = 6 # <-- nº de datas de avaliação distintas (turmas FFF/DDD)
K_RANGE  = range(2, 9)
DEFAULT_K = 4       # Comparável aos outros passos
os.makedirs(OUT_DIR, exist_ok=True)

class Tee:
    def __init__(self, *s): self.s = s
    def write(self, m):
        for x in self.s: x.write(m)
    def flush(self):
        for x in self.s: x.flush()
_orig = sys.stdout
_log = open(os.path.join(OUT_DIR, "saida_terminal.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, _log)

print("="*70); print("PASSO 7 — CURVAS REAIS (Intervalos Completos)"); print("="*70)

def chave(df):
    return df["code_module"] + "_" + df["code_presentation"] + "_" + df["id_student"].astype(str)
def curso_de(a):
    p = a.split("_"); return p[0] + "_" + p[1]

# 1. Carregar Alunos Válidos
val = pd.read_csv(DUR_CSV, index_col="aluno")
validos = set(val.index)

# 2. Carregar Calendário e aplicar Filtro de Datas de Prova Únicas
asm = pd.read_csv(os.path.join(DATA_DIR, "assessments.csv"))
asm["curso"] = asm["code_module"] + "_" + asm["code_presentation"]
asm = asm.dropna(subset=["date"]); asm["date"] = asm["date"].astype(int)

datas_curso_todas = {c: np.unique(g["date"].values) for c, g in asm.groupby("curso")}
datas_curso = {c: dts for c, dts in datas_curso_todas.items() if len(dts) == TARGET_N_PROVAS}

# Duração (em dias) de cada intervalo por curso: do dia 0 até a 1ª data, depois
# entre datas consecutivas. Necessário porque os intervalos NÃO têm o mesmo
# tamanho (ex.: 19 dias no 1º vs 56 dias no último) — isso será usado para
# converter cliques em TAXA (cliques/dia) e não deixar o intervalo mais longo
# dominar a curva só por acumular mais cliques brutos.
duracoes_curso = {c: np.diff(np.concatenate(([0], dts))).astype(float)
                  for c, dts in datas_curso.items()}

print(f"\nAlunos originais válidos: {len(validos):,}")
print(f"Cursos filtrados (exatamente {TARGET_N_PROVAS} datas de prova distintas): {list(datas_curso.keys())}")

# 3. Filtrar base de Logs
vle = pd.read_csv(os.path.join(DATA_DIR, "studentVle.csv"),
                  usecols=["code_module","code_presentation","id_student","date","sum_click"])
vle["aluno"] = chave(vle)
vle["curso"] = vle["code_module"] + "_" + vle["code_presentation"]
vle = vle[vle["curso"].isin(datas_curso.keys())]
vle = vle[vle["aluno"].isin(validos)].copy()

# 4. Atribuir cada clique ao INTERVALO DA PRÓXIMA PROVA (Removido o limite de 7 dias)
partes = []
for curso, sub in vle.groupby("curso"):
    dts = datas_curso[curso]
    day = sub["date"].values
    
    # idx será a ordem da prova. Se o clique for <= a data da prova 1, cai na prova 1.
    idx = np.searchsorted(dts, day, side="left")
    
    # Mantemos todos os cliques que ocorreram até o dia da última prova
    ok = idx < len(dts)
    
    if ok.sum() == 0: continue
    partes.append(pd.DataFrame({"aluno": sub["aluno"].values[ok],
                                "prova": idx[ok],
                                "clicks": sub["sum_click"].values[ok]}))
pre = pd.concat(partes, ignore_index=True)
pre = pre.groupby(["aluno","prova"])["clicks"].sum().reset_index()
piv = pre.pivot_table(index="aluno", columns="prova", values="clicks", fill_value=0)

# Garantir todas as colunas de intervalo
for i in range(TARGET_N_PROVAS):
    if i not in piv.columns:
        piv[i] = 0
piv = piv[range(TARGET_N_PROVAS)]

# 5. Cliques Brutos -> TAXA (cliques/dia) -> Proporção da taxa
# Dividir pela duração do intervalo antes de normalizar evita que o intervalo
# mais longo do calendário "vença" a curva só por ter mais dias — a proporção
# passa a refletir intensidade de esforço, não duração da janela.
dur_por_aluno = np.vstack([duracoes_curso[curso_de(a)] for a in piv.index])
taxa = piv.values / dur_por_aluno

somas = piv.sum(axis=1)                 # total de cliques (para filtro e intensidade)
soma_taxa = taxa.sum(axis=1)            # total de taxa (para normalizar o formato da curva)
mask = soma_taxa > 0
piv, taxa, somas, soma_taxa = piv[mask], taxa[mask], somas[mask], soma_taxa[mask]

prop = pd.DataFrame(taxa / soma_taxa[:, None], index=piv.index)

feat_cols = [f"prova_{i+1}" for i in range(TARGET_N_PROVAS)]
prop.columns = feat_cols
C = prop.join(val[["final_result","faixa_volume"]], how="inner")
duracao_total = dur_por_aluno[mask].sum(axis=1)
C["intensidade_media"] = somas / duracao_total   # cliques médios por dia no curso

print(f"Alunos retidos para a Clusterização ({TARGET_N_PROVAS} Intervalos): {len(C):,}")

# 6. Clusterização
Xz = StandardScaler().fit_transform(C[feat_cols].values)

inercia, silh = [], []
rng = np.random.RandomState(0)
am = rng.choice(len(Xz), size=min(5000, len(Xz)), replace=False)
print("\nEscolha de K:")
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xz)
    inercia.append(km.inertia_); silh.append(silhouette_score(Xz[am], km.labels_[am]))
    print(f"  K={k}:  inércia={km.inertia_:,.0f}   silhueta={silh[-1]:.3f}")
melhor_k = list(K_RANGE)[int(np.argmax(silh))]
K = DEFAULT_K or melhor_k
print(f"\nMelhor K (silhueta): {melhor_k}  | usando K = {K}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
a1.plot(list(K_RANGE), inercia, "o-"); a1.set_title("Cotovelo (inércia)"); a1.set_xlabel("K"); a1.set_ylabel("inércia")
a2.plot(list(K_RANGE), silh, "o-", color="green"); a2.set_title("Silhueta média"); a2.set_xlabel("K")
a2.axvline(melhor_k, ls="--", color="gray")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "01_cotovelo_silhueta.png"), dpi=130); plt.close()

km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Xz)
C["cluster"] = km.labels_

# 7. Gráficos
eixo = np.arange(1, TARGET_N_PROVAS + 1)
plt.figure(figsize=(10, 5))
for c in range(K):
    al = C.index[C["cluster"] == c]
    plt.plot(eixo, C.loc[al, feat_cols].mean().values, lw=2, marker="o",
             label=f"cluster {c} (n={len(al)})")
plt.xlabel(f"Intervalos entre Provas (1 a {TARGET_N_PROVAS})")
plt.ylabel("Proporção média da taxa de esforço (cliques/dia)")
plt.title(f"Perfis Geométricos (Taxa por Dia) - {TARGET_N_PROVAS} Intervalos (K={K})")
plt.xticks(eixo)
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_perfis_curva_real.png"), dpi=130); plt.close()

print("\n--- PERFIL DE CADA CLUSTER ---")
resumo = []
for c in range(K):
    m = C["cluster"] == c
    aprov = C.loc[m, "final_result"].isin(["Pass","Distinction"]).mean()*100
    resumo.append({"cluster": c, "n": int(m.sum()), "%base": round(100*m.mean(),1),
                   "aprovacao_%": round(aprov,1),
                   "intensidade_media": round(C.loc[m,"intensidade_media"].mean(),1)})
resumo = pd.DataFrame(resumo); print(resumo.to_string(index=False))

comp = pd.crosstab(C["cluster"], C["final_result"], normalize="index").mul(100).round(1)
print("\nResultado final por cluster (%):"); print(comp.to_string())
volx = pd.crosstab(C["cluster"], C["faixa_volume"], normalize="index").mul(100).round(1)
print("\nFaixa de volume por cluster (%):"); print(volx.to_string())

ordem = [c for c in ["Distinction","Pass","Fail"] if c in comp.columns]
comp[ordem].plot(kind="bar", stacked=True, figsize=(8.5,5),
                 color={"Distinction":"#2E7D32","Pass":"#8BC34A","Fail":"#FF7043"})
plt.ylabel("% dos alunos"); plt.xlabel("cluster"); plt.title("Resultado final por cluster")
plt.xticks(rotation=0); plt.legend(title="", bbox_to_anchor=(1.02,1))
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "03_resultado_por_cluster.png"), dpi=130); plt.close()

sm = silhouette_score(Xz, C["cluster"].values, sample_size=6000, random_state=0)
amf = rng.choice(len(Xz), size=min(6000, len(Xz)), replace=False)
ls = C["cluster"].values[amf]; sp = silhouette_samples(Xz[amf], ls)
pca = PCA(n_components=2, random_state=0).fit(Xz); proj = pca.transform(Xz)
cen2 = pca.transform(km.cluster_centers_); var = pca.explained_variance_ratio_*100
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6)); yl = 10
for c in range(K):
    vals = np.sort(sp[ls == c]); yu = yl + len(vals); cor = cm.nipy_spectral(float(c)/K)
    ax1.fill_betweenx(np.arange(yl, yu), 0, vals, facecolor=cor, edgecolor=cor, alpha=0.7)
    ax1.text(-0.05, yl + 0.5*len(vals), str(c)); yl = yu + 10
ax1.axvline(x=sm, color="red", ls="--", label=f"média={sm:.3f}")
ax1.set_xlim([-0.2,1.0]); ax1.set_title(f"Silhueta por aluno (K={K})"); ax1.set_xlabel("coef. silhueta")
ax1.set_yticks([]); ax1.legend(loc="lower right")
cores = cm.nipy_spectral(C["cluster"].values.astype(float)/K)
ax2.scatter(proj[:,0], proj[:,1], c=cores, s=4, alpha=0.35)
ax2.scatter(cen2[:,0], cen2[:,1], marker="o", c="white", s=260, edgecolor="k", zorder=3)
for c in range(K): ax2.scatter(cen2[c,0], cen2[c,1], marker=f"${c}$", c="k", s=60, zorder=4)
ax2.set_title("Alunos em 2D (PCA)"); ax2.set_xlabel(f"PCA1 ({var[0]:.0f}%)"); ax2.set_ylabel(f"PCA2 ({var[1]:.0f}%)")
plt.suptitle(f"Diagnóstico de silhueta — K={K}", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "04_silhueta_facas_e_mapa.png"), dpi=130); plt.close()

C.to_csv(os.path.join(OUT_DIR, "curvas_reais_intervalo.csv"))
resumo.to_csv(os.path.join(OUT_DIR, "perfil_dos_clusters.csv"), index=False)

print("\n" + "="*70)
print(f"Gravado em {os.path.abspath(OUT_DIR)}:")
print(f"\nPasso 7 OK. Intervalos por data única de prova + curva normalizada por taxa (cliques/dia).")
sys.stdout = _orig; _log.close()