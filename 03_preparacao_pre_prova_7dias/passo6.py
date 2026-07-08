#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PASSO 6 — FEATURES ANCORADAS NAS AVALIAÇÕES (opção A: indicadores resumo)
Em vez de 30 fatias arbitrárias, descrevemos o aluno pelo COMPORTAMENTO em
torno de cada avaliação do curso. Como cada curso tem um nº diferente de
provas, resumimos a "lista de provas" em INDICADORES de tamanho fixo, iguais
para todo aluno:
  - tendencia     : ele se prepara cada vez MAIS ou MENOS ao longo das provas
  - antecedencia  : estuda dias antes (alto) ou só na véspera (baixo)
  - durabilidade  : em que fração das provas teve algum pré-estudo
  - concentracao  : o pré-estudo é espalhado ou empilhado numa prova só
  - cobertura_final: quanto do pré-estudo caiu na ÚLTIMA prova (engajou até o fim?)
Volume (intensidade média) fica de fora da clusterização, só p/ validar.
Withdrawn já estão removidos (vêm filtrados de features_durabilidade.csv).

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
DUR_CSV  = "./01_curva_semestre_preliminar/features_durabilidade.csv"   # define os alunos válidos + resultado
OUT_DIR  = "./03_preparacao_pre_prova_7dias"
W        = 7        # janela de pré-estudo (dias antes da prova)
Q        = 5        # pontos p/ a curva de preparação reamostrada
K_RANGE  = range(2, 9)
DEFAULT_K = 3       # comparável ao passo 5; troque se quiser
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

print("="*70); print("PASSO 6 — FEATURES POR AVALIAÇÃO (opção A)"); print("="*70)

def chave(df):
    return df["code_module"] + "_" + df["code_presentation"] + "_" + df["id_student"].astype(str)
def curso_de(aluno):  # "AAA_2013J_123" -> "AAA_2013J"
    p = aluno.split("_"); return p[0] + "_" + p[1]

# --- alunos válidos (não-desistentes, >=30 cliques) + resultado/volume
val = pd.read_csv(DUR_CSV, index_col="aluno")
validos = set(val.index)
print(f"\nAlunos válidos (de passo 4.2): {len(validos):,}")

# --- datas das avaliações por curso
asm = pd.read_csv(os.path.join(DATA_DIR, "assessments.csv"))
asm["curso"] = asm["code_module"] + "_" + asm["code_presentation"]
asm = asm.dropna(subset=["date"])
asm["date"] = asm["date"].astype(int)
datas_curso = {c: np.sort(g["date"].values) for c, g in asm.groupby("curso")}
n_provas = {c: len(d) for c, d in datas_curso.items()}
print(f"Cursos com avaliações datadas: {len(datas_curso)}")
print("Nº de provas por curso:", dict(sorted({c: n for c, n in n_provas.items()}.items())))

# --- cliques (só dos alunos válidos)
vle = pd.read_csv(os.path.join(DATA_DIR, "studentVle.csv"),
                  usecols=["code_module","code_presentation","id_student","date","sum_click"])
vle["aluno"] = chave(vle)
vle = vle[vle["aluno"].isin(validos)].copy()
vle["curso"] = vle["code_module"] + "_" + vle["code_presentation"]

# --- atribuir cada clique à janela da PRÓXIMA prova (se dentro de W dias)
partes = []
for curso, sub in vle.groupby("curso"):
    if curso not in datas_curso: continue
    dts = datas_curso[curso]
    day = sub["date"].values
    idx = np.searchsorted(dts, day, side="left")     # 1ª prova com data >= dia
    ok = idx < len(dts)
    lead = np.full(len(day), -1)
    lead[ok] = dts[idx[ok]] - day[ok]                # dias até a prova
    inwin = ok & (lead >= 0) & (lead < W)
    if inwin.sum() == 0: continue
    partes.append(pd.DataFrame({
        "aluno": sub["aluno"].values[inwin],
        "prova": idx[inwin],                         # ordem da prova (0,1,2,...)
        "clicks": sub["sum_click"].values[inwin],
        "leadw": lead[inwin] * sub["sum_click"].values[inwin],
    }))
pre = pd.concat(partes, ignore_index=True)
pre = pre.groupby(["aluno","prova"]).agg(clicks=("clicks","sum"),
                                         leadw=("leadw","sum")).reset_index()

# --- montar features por aluno
piv = pre.pivot_table(index="aluno", columns="prova", values="clicks", fill_value=0)
leadtot = pre.groupby("aluno")["leadw"].sum()
clicktot = pre.groupby("aluno")["clicks"].sum()

feats = {}
curva_cluster = {}   # curva de preparação reamostrada (Q pontos) por aluno
for aluno in validos:
    curso = curso_de(aluno)
    n = n_provas.get(curso, 0)
    if n == 0:
        continue
    if aluno in piv.index:
        v = piv.loc[aluno, [i for i in range(n) if i in piv.columns]].reindex(range(n), fill_value=0).values.astype(float)
    else:
        v = np.zeros(n)
    soma = v.sum()
    if soma > 0:
        prop = v / soma
        x = np.arange(n)
        tend = np.polyfit(x, prop, 1)[0] if n >= 2 else 0.0     # inclinação da preparação
        conc = prop.std()                                       # empilhado vs espalhado
        cobf = prop[-1]                                         # fração na última prova
        durab = (v > 0).mean()                                  # fração de provas com pré-estudo
        ante = (leadtot.get(aluno, 0) / clicktot.get(aluno, 1)) / max(W-1, 1)  # 0=véspera,1=cedo
        # curva reamostrada p/ Q pontos
        pos = np.linspace(0, 1, n)
        curva = np.interp(np.linspace(0, 1, Q), pos, prop) if n >= 2 else np.full(Q, prop[0])
        inten = soma / n
    else:
        tend = conc = cobf = durab = ante = inten = 0.0
        curva = np.zeros(Q)
    feats[aluno] = [tend, ante, durab, conc, cobf, inten]
    curva_cluster[aluno] = curva

F = pd.DataFrame.from_dict(feats, orient="index",
        columns=["tendencia","antecedencia","durabilidade","concentracao","cobertura_final","intensidade_media"])
F = F.join(val[["final_result","faixa_volume"]])
curvas = pd.DataFrame.from_dict(curva_cluster, orient="index",
        columns=[f"prep_{i}" for i in range(Q)])
print(f"\nAlunos com features: {len(F):,}")
print("Estatística das features de clusterização:")
print(F[["tendencia","antecedencia","durabilidade","concentracao","cobertura_final"]].describe().round(3).to_string())

# --- clusterização (5 features de forma; intensidade fica de fora)
feat_cols = ["tendencia","antecedencia","durabilidade","concentracao","cobertura_final"]
Xz = StandardScaler().fit_transform(F[feat_cols].values)

inercia, silh = [], []
rng = np.random.RandomState(0)
amostra = rng.choice(len(Xz), size=min(5000, len(Xz)), replace=False)
print("\nEscolha de K:")
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
F["cluster"] = km.labels_

# --- 02: curva média de preparação por cluster (ao longo das provas)
eixo = np.linspace(0, 1, Q)
plt.figure(figsize=(9, 5))
for c in range(K):
    al = F.index[F["cluster"] == c]
    plt.plot(eixo, curvas.loc[al].mean().values, lw=2, marker="o",
             label=f"cluster {c} (n={len(al)})")
plt.xlabel("ordem das provas (0=1ª prova, 1=última)")
plt.ylabel("proporção média do pré-estudo")
plt.title(f"Perfis por avaliação (K={K}) — preparação ao longo das provas")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_perfis_preparacao.png"), dpi=130); plt.close()

# --- 02b: média das features por cluster (barras)
cen = F.groupby("cluster")[feat_cols].mean()
cen.T.plot(kind="bar", figsize=(10, 5))
plt.ylabel("valor médio"); plt.xlabel("feature"); plt.xticks(rotation=20)
plt.title("Média de cada feature por cluster"); plt.legend(title="cluster", bbox_to_anchor=(1.02, 1))
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "02b_features_por_cluster.png"), dpi=130); plt.close()

# --- perfil + validação
print("\n--- PERFIL DE CADA CLUSTER ---")
resumo = []
for c in range(K):
    m = F["cluster"] == c
    aprov = F.loc[m, "final_result"].isin(["Pass","Distinction"]).mean()*100
    linha = {"cluster": c, "n": int(m.sum()), "%base": round(100*m.mean(),1),
             "aprovacao_%": round(aprov,1)}
    for col in feat_cols + ["intensidade_media"]:
        linha[col] = round(F.loc[m, col].mean(), 3)
    resumo.append(linha)
resumo = pd.DataFrame(resumo)
print(resumo.to_string(index=False))

comp = pd.crosstab(F["cluster"], F["final_result"], normalize="index").mul(100).round(1)
print("\nResultado final por cluster (%):"); print(comp.to_string())
volx = pd.crosstab(F["cluster"], F["faixa_volume"], normalize="index").mul(100).round(1)
print("\nFaixa de volume por cluster (%):"); print(volx.to_string())

ordem = [c for c in ["Distinction","Pass","Fail"] if c in comp.columns]
comp[ordem].plot(kind="bar", stacked=True, figsize=(8.5,5),
                 color={"Distinction":"#2E7D32","Pass":"#8BC34A","Fail":"#FF7043"})
plt.ylabel("% dos alunos"); plt.xlabel("cluster"); plt.title("Resultado final por cluster (validação H2)")
plt.xticks(rotation=0); plt.legend(title="", bbox_to_anchor=(1.02,1))
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "03_resultado_por_cluster.png"), dpi=130); plt.close()

# --- 04: facas + PCA
sil_media = silhouette_score(Xz, F["cluster"].values, sample_size=6000, random_state=0)
amf = rng.choice(len(Xz), size=min(6000, len(Xz)), replace=False)
ls = F["cluster"].values[amf]; sp = silhouette_samples(Xz[amf], ls)
pca = PCA(n_components=2, random_state=0).fit(Xz); proj = pca.transform(Xz)
cen2 = pca.transform(km.cluster_centers_); var = pca.explained_variance_ratio_*100
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
yl = 10
for c in range(K):
    vals = np.sort(sp[ls == c]); yu = yl + len(vals); cor = cm.nipy_spectral(float(c)/K)
    ax1.fill_betweenx(np.arange(yl, yu), 0, vals, facecolor=cor, edgecolor=cor, alpha=0.7)
    ax1.text(-0.05, yl + 0.5*len(vals), str(c)); yl = yu + 10
ax1.axvline(x=sil_media, color="red", ls="--", label=f"média={sil_media:.3f}")
ax1.set_xlim([-0.2,1.0]); ax1.set_title(f"Silhueta por aluno (K={K})"); ax1.set_xlabel("coef. silhueta")
ax1.set_yticks([]); ax1.legend(loc="lower right")
cores = cm.nipy_spectral(F["cluster"].values.astype(float)/K)
ax2.scatter(proj[:,0], proj[:,1], c=cores, s=4, alpha=0.35)
ax2.scatter(cen2[:,0], cen2[:,1], marker="o", c="white", s=260, edgecolor="k", zorder=3)
for c in range(K): ax2.scatter(cen2[c,0], cen2[c,1], marker=f"${c}$", c="k", s=60, zorder=4)
ax2.set_title("Alunos em 2D (PCA)"); ax2.set_xlabel(f"PCA1 ({var[0]:.0f}%)"); ax2.set_ylabel(f"PCA2 ({var[1]:.0f}%)")
plt.suptitle(f"Diagnóstico de silhueta — K={K}", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "04_silhueta_facas_e_mapa.png"), dpi=130); plt.close()

# --- saídas
F.to_csv(os.path.join(OUT_DIR, "features_avaliacao.csv"))
resumo.to_csv(os.path.join(OUT_DIR, "perfil_dos_clusters.csv"), index=False)

print("\n" + "="*70)
print(f"Gravado em {os.path.abspath(OUT_DIR)}:")
print("  - 01_cotovelo_silhueta.png      02_perfis_preparacao.png")
print("  - 02b_features_por_cluster.png  03_resultado_por_cluster.png")
print("  - 04_silhueta_facas_e_mapa.png")
print("  - features_avaliacao.csv        perfil_dos_clusters.csv   saida_terminal.txt")
print(f"\nPasso 6 (opção A) OK. K={K}, janela={W} dias.")
sys.stdout = _orig
_log.close()