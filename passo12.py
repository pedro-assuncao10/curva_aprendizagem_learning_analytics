#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PASSO 12 — CURVA DE APRENDIZAGEM EM 2 ESTÁGIOS
Objetivo: separar 4 perfis de aluno pela FORMA da curva de esforço ao longo do
curso inteiro (não só em torno da prova):
  - desistente   : some de vez, não tem mais nenhuma atividade no último terço
  - antecipado    : pico de esforço no início, depois cai
  - equilibrado   : esforço distribuído de forma parecida do início ao fim
  - fim-pesado    : esforço cresce e se concentra perto do fim

ESTÁGIO 1 (regra, não é clustering): usa "ativo_no_fim" (já validado no passo 3
— separa desistente de persistente com ~1% vs ~85% de aprovação, quase
determinístico) em vez de "durabilidade", que confundia desistente com
fim-pesado (ver passo12.md).

ESTÁGIO 2 (K-means): só entre quem persiste, clusteriza a CURVA de esforço
(não indicadores resumidos) em K=3 grupos. A curva é construída assim:
  - 100% dos cliques de cada aluno são atribuídos a um intervalo do curso
    (entre duas provas, ou entre a última prova e o fim oficial do curso —
    ninguém é descartado).
  - cada intervalo vira uma TAXA (cliques / dias do intervalo), pra um
    intervalo comprido não vencer só por ser comprido.
  - a taxa é normalizada em proporção (soma 1) e reamostrada em Q pontos
    fixos, usando a POSIÇÃO REAL NO CALENDÁRIO do curso (fração de dias),
    não o índice da prova — isso permite comparar cursos com números de
    prova e espaçamentos diferentes na mesma régua.

Ver passo12.md para o histórico de decisões (por que abandonamos janela fixa
W=7/28 e adaptativa em favor desse cálculo).

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
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from sklearn.decomposition import PCA

DATA_DIR = "./dataset"
DUR_CSV  = "./saida_passo3/features_durabilidade.csv"
CP_CSV   = "./saida_passo3/comeco_pesado_detalhado.csv"
OUT_DIR  = "./saida_passo12"
Q        = 6     # pontos fixos de reamostragem da curva (posição real no calendário, 0 a 1)
K_RANGE  = range(2, 9)
DEFAULT_K = 3     # perfis dentro de quem persiste: antecipado / equilibrado / fim-pesado
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

print("="*70); print("PASSO 12 — CURVA EM 2 ESTÁGIOS (desistente/antecipado/equilibrado/fim-pesado)"); print("="*70)

def chave(df):
    return df["code_module"] + "_" + df["code_presentation"] + "_" + df["id_student"].astype(str)

# 1. Alunos válidos + sinal de persistência (já calculado e validado no passo 3)
val = pd.read_csv(DUR_CSV, index_col="aluno")
validos = set(val.index)
cp = pd.read_csv(CP_CSV, index_col=0)
print(f"\nAlunos válidos: {len(validos):,}")

# 2. Calendário: datas de prova únicas por curso + intervalo final até o fim oficial
courses = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
courses["curso"] = courses["code_module"] + "_" + courses["code_presentation"]
comprimento = dict(zip(courses["curso"], courses["module_presentation_length"]))

asm = pd.read_csv(os.path.join(DATA_DIR, "assessments.csv"))
asm["curso"] = asm["code_module"] + "_" + asm["code_presentation"]
asm = asm.dropna(subset=["date"]); asm["date"] = asm["date"].astype(int)
datas_curso_prova = {c: np.unique(g["date"].values) for c, g in asm.groupby("curso")}

datas_curso = {}
for c, dts in datas_curso_prova.items():
    fim = comprimento[c]
    datas_curso[c] = np.append(dts, fim) if fim > dts[-1] else dts
duracoes_curso = {c: np.diff(np.concatenate(([0], dts))).astype(float) for c, dts in datas_curso.items()}
xpos_curso = {c: dts / comprimento[c] for c, dts in datas_curso.items()}   # posição no calendário (0 a 1)
print(f"Cursos com calendário de provas: {len(datas_curso)}")

# 3. Atribuir 100% dos cliques de cada aluno válido ao intervalo em que ocorreram
vle = pd.read_csv(os.path.join(DATA_DIR, "studentVle.csv"),
                  usecols=["code_module","code_presentation","id_student","date","sum_click"])
vle["aluno"] = chave(vle)
vle["curso"] = vle["code_module"] + "_" + vle["code_presentation"]
vle = vle[vle["aluno"].isin(validos)].copy()

partes = []
for curso, sub in vle.groupby("curso"):
    if curso not in datas_curso: continue
    dts = datas_curso[curso]
    day = sub["date"].values
    idx = np.searchsorted(dts, day, side="left")
    ok = idx < len(dts)
    if ok.sum() == 0: continue
    partes.append(pd.DataFrame({"aluno": sub["aluno"].values[ok], "curso": curso,
                                "intervalo": idx[ok], "clicks": sub["sum_click"].values[ok]}))
pre = pd.concat(partes, ignore_index=True)
pre = pre.groupby(["aluno","curso","intervalo"])["clicks"].sum().reset_index()

capturado, disponivel = pre["clicks"].sum(), vle["sum_click"].sum()
print(f"Cliques capturados: {capturado:,} de {disponivel:,} disponíveis ({100*capturado/disponivel:.2f}%)")

# 4. Construir a curva por aluno (taxa/dia -> proporção -> reamostrada em Q pontos)
curvas = {}
for aluno, g in pre.groupby("aluno"):
    curso = g["curso"].iloc[0]
    n = len(datas_curso[curso])
    v = np.zeros(n)
    v[g["intervalo"].values] = g["clicks"].values
    taxa = v / duracoes_curso[curso]
    s = taxa.sum()
    if s <= 0: continue
    shape = taxa / s
    x = xpos_curso[curso]
    xg = np.linspace(1/(2*Q), 1 - 1/(2*Q), Q)
    curvas[aluno] = np.interp(xg, x, shape)

feat_cols = [f"c{i}" for i in range(Q)]
C = pd.DataFrame.from_dict(curvas, orient="index", columns=feat_cols)
C = C.join(val[["final_result","faixa_volume","ativo_no_fim"]], how="inner")
C["aprovado"] = C["final_result"].isin(["Pass","Distinction"])
print(f"Alunos com curva calculada: {len(C):,}")

# 5. ESTÁGIO 1 (regra): separar desistente de persistente
persist = C[C["ativo_no_fim"]].copy()
n_desist = (~C["ativo_no_fim"]).sum()
print(f"\nEstágio 1 (regra ativo_no_fim): desistentes={n_desist:,}  persistentes={len(persist):,}")
print(f"Aprovação dos desistentes: {100*C.loc[~C['ativo_no_fim'],'aprovado'].mean():.1f}%")
print(f"Aprovação dos persistentes: {100*persist['aprovado'].mean():.1f}%")

# 6. ESTÁGIO 2 (K-means): clusteriza a curva só entre quem persiste
Xz = StandardScaler().fit_transform(persist[feat_cols].values)

inercia, silh = [], []
rng = np.random.RandomState(0)
amostra = rng.choice(len(Xz), size=min(5000, len(Xz)), replace=False)
print("\nEscolha de K (estágio 2, só persistentes):")
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xz)
    inercia.append(km.inertia_); silh.append(silhouette_score(Xz[amostra], km.labels_[amostra]))
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
persist["cluster"] = km.labels_

# 7. Nomear cada cluster do estágio 2 pela forma da curva (centro de massa temporal):
# menor centro de massa = esforço concentrado cedo = "antecipado"; maior = "fim-pesado".
centro_massa = {}
for c in range(K):
    media = persist.loc[persist["cluster"] == c, feat_cols].mean().values
    centro_massa[c] = (media * np.arange(Q)).sum() / media.sum()
ordem = sorted(centro_massa, key=centro_massa.get)
nomes = {ordem[0]: "antecipado", ordem[-1]: "fim-pesado"}
for c in ordem[1:-1]:
    nomes[c] = "equilibrado"
persist["perfil"] = persist["cluster"].map(nomes)

C["perfil"] = "desistente"
C.loc[persist.index, "perfil"] = persist["perfil"]
ordem_perfis = ["desistente", "antecipado", "equilibrado", "fim-pesado"]

# 8. Gráfico do PADRÃO DE CURVAS — os 4 perfis, mesmo estilo dos passos 6/7
eixo = np.linspace(1/(2*Q), 1 - 1/(2*Q), Q)
plt.figure(figsize=(10, 5))
cores = {"desistente": "#C62828", "antecipado": "#1976D2", "equilibrado": "#F9A825", "fim-pesado": "#2E7D32"}
for p in ordem_perfis:
    al = C.index[C["perfil"] == p]
    plt.plot(eixo, C.loc[al, feat_cols].mean().values, lw=2, marker="o",
             color=cores[p], label=f"{p} (n={len(al):,})")
plt.xlabel("posição no curso (0 = início, 1 = fim oficial)")
plt.ylabel("proporção média da taxa de esforço (cliques/dia)")
plt.title(f"Perfis de curva de aprendizagem (estágio 1 regra + estágio 2 K={K})")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_perfis_curvas.png"), dpi=130); plt.close()

# 9. Validação: cruzamento com os perfis "verdadeiros" do passo 3 (timing/subtipo)
dur = pd.read_csv(DUR_CSV, index_col=0)
Cv = C.join(dur[["timing"]], how="left").join(cp[["subtipo"]], how="left")
Cv["perfil_passo3"] = Cv["subtipo"]
Cv.loc[Cv["timing"] == "equilibrado", "perfil_passo3"] = "equilibrado"
Cv.loc[Cv["timing"] == "fim-pesado", "perfil_passo3"] = "fim-pesado"
Cv.loc[~Cv["ativo_no_fim"], "perfil_passo3"] = "desistente"

cross = pd.crosstab(Cv["perfil"], Cv["perfil_passo3"])
cross_pct = pd.crosstab(Cv["perfil"], Cv["perfil_passo3"], normalize="index").mul(100).round(1)
ari_estagio2 = adjusted_rand_score(
    persist.join(Cv[["perfil_passo3"]])["perfil_passo3"].astype("category").cat.codes,
    persist["cluster"])
print("\n--- VALIDAÇÃO: cruzamento perfil (passo12) x perfil (passo3, referência) ---")
print(cross.to_string())
print("\n--- mesmo cruzamento em % por linha ---")
print(cross_pct.to_string())
print(f"\nARI do estágio 2 (K-means na curva) vs perfil de referência do passo3: {ari_estagio2:.3f}")
cross.to_csv(os.path.join(OUT_DIR, "validacao_crosstab.csv"))

# 10. Tabela-resumo por perfil (mesmo padrão de perfil_dos_clusters.csv dos outros passos)
resumo = []
for p in ordem_perfis:
    m = C["perfil"] == p
    resumo.append({"perfil": p, "n": int(m.sum()), "%base": round(100*m.mean(), 1),
                   "aprovacao_%": round(100*C.loc[m, "aprovado"].mean(), 1)})
resumo = pd.DataFrame(resumo)
print("\n--- PERFIL DE CADA GRUPO (final, estágio 1 + estágio 2) ---")
print(resumo.to_string(index=False))
resumo.to_csv(os.path.join(OUT_DIR, "perfil_dos_grupos.csv"), index=False)

comp = pd.crosstab(C["perfil"], C["final_result"], normalize="index").mul(100).round(1)
ordem_fr = [c for c in ["Distinction","Pass","Fail"] if c in comp.columns]
comp = comp.reindex(ordem_perfis)[ordem_fr]
print("\nResultado final por perfil (%):"); print(comp.to_string())
comp.plot(kind="bar", stacked=True, figsize=(8.5, 5),
         color={"Distinction":"#2E7D32","Pass":"#8BC34A","Fail":"#FF7043"})
plt.ylabel("% dos alunos"); plt.xlabel("perfil"); plt.title("Resultado final por perfil de curva")
plt.xticks(rotation=0); plt.legend(title="", bbox_to_anchor=(1.02,1))
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "03_resultado_por_cluster.png"), dpi=130); plt.close()

# 11. Silhueta (facas) + mapa PCA — diagnóstico do estágio 2 (mesmo estilo dos outros passos)
sm = silhouette_score(Xz, persist["cluster"].values, sample_size=6000, random_state=0)
amf = rng.choice(len(Xz), size=min(6000, len(Xz)), replace=False)
ls = persist["cluster"].values[amf]; sp = silhouette_samples(Xz[amf], ls)
pca = PCA(n_components=2, random_state=0).fit(Xz); proj = pca.transform(Xz)
cen2 = pca.transform(km.cluster_centers_); var = pca.explained_variance_ratio_ * 100
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6)); yl = 10
for c in range(K):
    vals = np.sort(sp[ls == c]); yu = yl + len(vals); cor = cm.nipy_spectral(float(c)/K)
    ax1.fill_betweenx(np.arange(yl, yu), 0, vals, facecolor=cor, edgecolor=cor, alpha=0.7)
    ax1.text(-0.05, yl + 0.5*len(vals), nomes[c]); yl = yu + 10
ax1.axvline(x=sm, color="red", ls="--", label=f"média={sm:.3f}")
ax1.set_xlim([-0.2, 1.0]); ax1.set_title(f"Silhueta por aluno — estágio 2 (K={K})"); ax1.set_xlabel("coef. silhueta")
ax1.set_yticks([]); ax1.legend(loc="lower right")
cores_pca = cm.nipy_spectral(persist["cluster"].values.astype(float) / K)
ax2.scatter(proj[:,0], proj[:,1], c=cores_pca, s=4, alpha=0.35)
ax2.scatter(cen2[:,0], cen2[:,1], marker="o", c="white", s=260, edgecolor="k", zorder=3)
for c in range(K): ax2.scatter(cen2[c,0], cen2[c,1], marker=f"${nomes[c][0].upper()}$", c="k", s=60, zorder=4)
ax2.set_title("Alunos persistentes em 2D (PCA)"); ax2.set_xlabel(f"PCA1 ({var[0]:.0f}%)"); ax2.set_ylabel(f"PCA2 ({var[1]:.0f}%)")
plt.suptitle(f"Diagnóstico de silhueta — estágio 2, K={K}", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "04_silhueta_facas_e_mapa.png"), dpi=130); plt.close()

# 12. Tabela comparativa dos métodos de W testados até chegar aqui (pra registro no MD)
comparacao_w = pd.DataFrame([
    {"metodo": "W=7 fixo (antecedencia escalar)",        "corr_ativo_fim": 0.112, "corr_aprovado": 0.181},
    {"metodo": "W=14 fixo (antecedencia escalar)",       "corr_ativo_fim": 0.080, "corr_aprovado": 0.160},
    {"metodo": "W=28 fixo (antecedencia escalar)",       "corr_ativo_fim": -0.023, "corr_aprovado": 0.078},
    {"metodo": "W adaptativo max(28,gap) (antecedencia)", "corr_ativo_fim": -0.062, "corr_aprovado": 0.032},
])
comparacao_w.to_csv(os.path.join(OUT_DIR, "comparacao_metodos_w.csv"), index=False)
print("\n--- Comparação dos métodos de janela testados (ver passo12.md) ---")
print(comparacao_w.to_string(index=False))

C.to_csv(os.path.join(OUT_DIR, "curvas_e_perfis.csv"))

print("\n" + "="*70)
print(f"Gravado em {os.path.abspath(OUT_DIR)}:")
print("  - 01_cotovelo_silhueta.png       02_perfis_curvas.png")
print("  - 03_resultado_por_cluster.png   04_silhueta_facas_e_mapa.png")
print("  - curvas_e_perfis.csv            perfil_dos_grupos.csv")
print("  - validacao_crosstab.csv         comparacao_metodos_w.csv")
print("  - passo12.md                     saida_terminal.txt")
print(f"\nPasso 12 OK. Estágio1 (regra) + Estágio2 (K={K} na curva). ARI vs passo3={ari_estagio2:.3f}")
sys.stdout = _orig; _log.close()
