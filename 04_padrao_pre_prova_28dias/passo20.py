#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PASSO 20 — PADRÃO DIÁRIO DE CLIQUES ANTES DE CADA PROVA (Definição A)
Enquanto o passo 14 mede a curva do CURSO INTEIRO (Definição B — quando o
esforço se concentra ao longo do semestre), este passo mede outra coisa: o
padrão diário de cliques nos W dias que antecedem CADA prova (Definição A —
hábito de preparação repetido, prova a prova).

Motivação: um aluno pode ser "equilibrado" no curso inteiro (esforço
espalhado pelo semestre) e ainda assim ser um "crammer" que, toda vez, só
estuda pesado nos últimos dias antes de CADA prova -- os picos repetidos ficam
distribuídos pelo calendário e se diluem na curva do curso inteiro. Este passo
existe pra enxergar esse padrão que a Definição B esconde.

Como funciona:
  1. Pra cada prova de cada curso, olha os W=28 dias imediatamente anteriores
     (28 = menor gap real entre provas descontado o ruído de datas
     quase-duplicadas -- a maior janela que não invade a prova anterior na
     maioria dos cursos).
  2. Os 28 dias são agrupados em blocos de 4 dias (7 blocos), pra suavizar o
     ruído de clique dia-a-dia -- um único dia de pico isolado não domina
     mais a forma da curva.
  3. Pra cada aluno, em cada prova que ele tem cliques na janela: soma os
     cliques por bloco e normaliza em proporção (soma 1) -- é a "forma"
     suavizada da preparação daquela prova especificamente.
  4. Como cada aluno passa por várias provas, tira a MÉDIA das formas
     (proporção) de todas as provas em que ele teve alguma atividade -- vira
     o "padrão típico" de preparação daquele aluno, um ponto só de 7 números.
  5. K-means nesse ponto (7 dimensões) agrupa os alunos pelo arquétipo do
     padrão: crescente (cramming, cresce até a prova), constante (rotina
     estável) ou decrescente (estuda cedo, só revisa depois).

Importante: os nomes aqui (crescente/constante/decrescente) são
propositalmente diferentes de antecipado/equilibrado/fim-pesado (que são da
Definição B, curso inteiro) -- são dois eixos diferentes, não se espera que
batam um-com-um. O cruzamento entre os dois é o resultado mais interessante
deste passo.

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
DUR_CSV  = "./01_curva_semestre_preliminar/features_durabilidade.csv"
CURVA_B_CSV = "./saida_passo14/curvas_e_perfis.csv"   # legado removido; cruzamento é pulado (try/except)
OUT_DIR  = "./04_padrao_pre_prova_28dias"
W          = 28    # dias antes de cada prova observados (dia_offset 0..W-1)
                    # 28 = menor gap real entre provas, descontado o ruído de
                    # datas quase-duplicadas (ver saida_passo14/passo14.md) --
                    # maior janela sem invadir a prova anterior na maioria dos cursos.
BLOCO_DIAS = 4      # suaviza o ruído dia-a-dia agrupando em blocos de N dias
N_BLOCOS   = W // BLOCO_DIAS   # nº de pontos da curva suavizada
K_RANGE  = range(2, 9)
DEFAULT_K = 3      # crescente / constante / decrescente
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

print("="*70); print(f"PASSO 20 — PADRÃO ANTES DE CADA PROVA (W={W} dias, suavizado em blocos de {BLOCO_DIAS} dias)"); print("="*70)

def chave(df):
    return df["code_module"] + "_" + df["code_presentation"] + "_" + df["id_student"].astype(str)

# 1. Alunos válidos
val = pd.read_csv(DUR_CSV, index_col="aluno")
validos = set(val.index)
print(f"\nAlunos válidos: {len(validos):,}")

# 2. Datas de prova (únicas) por curso
asm = pd.read_csv(os.path.join(DATA_DIR, "assessments.csv"))
asm["curso"] = asm["code_module"] + "_" + asm["code_presentation"]
asm = asm.dropna(subset=["date"]); asm["date"] = asm["date"].astype(int)
datas_curso = {c: np.unique(g["date"].values) for c, g in asm.groupby("curso")}
print(f"Cursos com datas de prova: {len(datas_curso)}")

# 3. Cliques dos alunos válidos
vle = pd.read_csv(os.path.join(DATA_DIR, "studentVle.csv"),
                  usecols=["code_module","code_presentation","id_student","date","sum_click"])
vle["aluno"] = chave(vle)
vle["curso"] = vle["code_module"] + "_" + vle["code_presentation"]
vle = vle[vle["aluno"].isin(validos)].copy()

# 4. Pra cada clique: a que prova ele antecede (a mais próxima seguinte) e o
# dia_offset até ela (0 = dia da prova, W-1 = W-1 dias antes). Cliques fora da
# janela W são ignorados (não entram nesse cálculo -- só nos interessa a
# janela de preparação, diferente do passo14 que usa 100% dos cliques).
partes = []
for curso, sub in vle.groupby("curso"):
    if curso not in datas_curso: continue
    dts = datas_curso[curso]
    day = sub["date"].values
    idx = np.searchsorted(dts, day, side="left")
    ok = idx < len(dts)
    if ok.sum() == 0: continue
    offset = dts[idx[ok]] - day[ok]
    inwin = offset < W
    if inwin.sum() == 0: continue
    bloco = offset[inwin] // BLOCO_DIAS   # 0 = bloco mais perto da prova
    partes.append(pd.DataFrame({
        "aluno": sub["aluno"].values[ok][inwin], "curso": curso,
        "prova_idx": idx[ok][inwin], "bloco": bloco,
        "clicks": sub["sum_click"].values[ok][inwin],
    }))
pre = pd.concat(partes, ignore_index=True)
pre = pre.groupby(["aluno","curso","prova_idx","bloco"])["clicks"].sum().reset_index()

# 5. Forma (proporção) de cada (aluno, prova) -- N_BLOCOS valores que somam 1
# (blocos de BLOCO_DIAS dias em vez de dia a dia, pra suavizar o ruído diário)
piv = pre.pivot_table(index=["aluno","curso","prova_idx"], columns="bloco", values="clicks", fill_value=0)
piv = piv.reindex(columns=range(N_BLOCOS), fill_value=0)
soma_prova = piv.sum(axis=1)
piv = piv[soma_prova > 0]
forma_prova = piv.div(piv.sum(axis=1), axis=0)

# 6. Média das formas entre todas as provas de cada aluno -> 1 ponto por aluno
media_aluno = forma_prova.groupby(level="aluno").mean()
n_provas_usadas = forma_prova.groupby(level="aluno").size().rename("n_provas_com_clique")
feat_cols = [f"b{i}" for i in range(N_BLOCOS)]
media_aluno.columns = feat_cols

C = media_aluno.join(n_provas_usadas).join(val[["final_result","faixa_volume"]], how="inner")
C["aprovado"] = C["final_result"].isin(["Pass","Distinction"])
print(f"\nAlunos com padrão calculado (>=1 prova com clique na janela de {W} dias): {len(C):,}")
print(f"Distribuição de nº de provas usadas por aluno:\n{C['n_provas_com_clique'].describe().round(1).to_string()}")

# 7. K-means
Xz = StandardScaler().fit_transform(C[feat_cols].values)

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
C["cluster"] = km.labels_

# 8. Nomear os clusters pela INCLINAÇÃO da curva (bloco 0 = mais perto da
# prova -> bloco N-1 = mais longe): inclinação mais negativa (cai forte do
# bloco 0 pro N-1) = concentra perto da prova = "crescente" (cram); mais
# positiva = "decrescente" (estuda cedo, cai perto da prova); no meio = "constante".
x = np.arange(N_BLOCOS)  # 0=bloco mais perto da prova ... N_BLOCOS-1=mais longe
inclinacoes = {}
for c in range(K):
    media = C.loc[C["cluster"] == c, feat_cols].mean().values
    # regressão linear simples de "proporção" em função de dia_offset
    inclinacoes[c] = np.polyfit(x, media, 1)[0]
ordem = sorted(inclinacoes, key=inclinacoes.get)
# inclinação mais negativa (cai com o dia_offset, ou seja, sobe perto da prova) = crescente
nomes = {ordem[0]: "crescente", ordem[-1]: "decrescente"}
for c in ordem[1:-1]:
    nomes[c] = "constante"
C["padrao"] = C["cluster"].map(nomes)

print("\n--- Inclinação média de cada cluster (negativa = concentra perto da prova) ---")
for c in range(K):
    print(f"  cluster {c} ({nomes[c]}): inclinação={inclinacoes[c]:.4f}  n={int((C['cluster']==c).sum()):,}")

# 9. Gráfico do padrão de curvas — eixo em "dias antes da prova" (dia médio de
# cada bloco), suavizado (BLOCO_DIAS dias por ponto em vez de dia a dia)
dia_medio_bloco = np.arange(N_BLOCOS) * BLOCO_DIAS + (BLOCO_DIAS - 1) / 2.0  # bloco0 = mais perto
plt.figure(figsize=(10, 5))
cores = {"crescente": "#2E7D32", "constante": "#F9A825", "decrescente": "#1976D2"}
ordem_padroes = ["crescente", "constante", "decrescente"]
for p in ordem_padroes:
    al = C.index[C["padrao"] == p]
    medias = C.loc[al, feat_cols].mean().values  # b0 (perto da prova) .. b(N_BLOCOS-1) (longe)
    plt.plot(dia_medio_bloco, medias, lw=2, marker="o", color=cores[p], label=f"{p} (n={len(al):,})")
plt.xlabel(f"dias antes da prova (blocos de {BLOCO_DIAS} dias; 0 = dia da prova)")
plt.ylabel("proporção média dos cliques na janela")
plt.title(f"Padrão suavizado de preparação por prova (W={W} dias, {N_BLOCOS} blocos, K={K})")
plt.gca().invert_xaxis()
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_perfis_curvas.png"), dpi=130); plt.close()

# 10. Resultado final por padrão
resumo = []
for p in ordem_padroes:
    m = C["padrao"] == p
    resumo.append({"padrao": p, "n": int(m.sum()), "%base": round(100*m.mean(), 1),
                   "aprovacao_%": round(100*C.loc[m, "aprovado"].mean(), 1)})
resumo = pd.DataFrame(resumo)
print("\n--- PERFIL DE CADA PADRÃO (Definição A: hábito por prova) ---")
print(resumo.to_string(index=False))
resumo.to_csv(os.path.join(OUT_DIR, "perfil_dos_padroes.csv"), index=False)

comp = pd.crosstab(C["padrao"], C["final_result"], normalize="index").mul(100).round(1)
ordem_fr = [c for c in ["Distinction","Pass","Fail"] if c in comp.columns]
comp = comp.reindex(ordem_padroes)[ordem_fr]
print("\nResultado final por padrão (%):"); print(comp.to_string())
comp.plot(kind="bar", stacked=True, figsize=(8.5, 5),
         color={"Distinction":"#2E7D32","Pass":"#8BC34A","Fail":"#FF7043"})
plt.ylabel("% dos alunos"); plt.xlabel("padrão (por prova)"); plt.title("Resultado final por padrão de preparação (Definição A)")
plt.xticks(rotation=0); plt.legend(title="", bbox_to_anchor=(1.02,1))
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "03_resultado_por_cluster.png"), dpi=130); plt.close()

# 11. Silhueta (facas) + mapa PCA
sm = silhouette_score(Xz, C["cluster"].values, sample_size=6000, random_state=0)
amf = rng.choice(len(Xz), size=min(6000, len(Xz)), replace=False)
ls = C["cluster"].values[amf]; sp = silhouette_samples(Xz[amf], ls)
pca = PCA(n_components=2, random_state=0).fit(Xz); proj = pca.transform(Xz)
cen2 = pca.transform(km.cluster_centers_); var = pca.explained_variance_ratio_ * 100
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6)); yl = 10
for c in range(K):
    vals = np.sort(sp[ls == c]); yu = yl + len(vals); cor = cm.nipy_spectral(float(c)/K)
    ax1.fill_betweenx(np.arange(yl, yu), 0, vals, facecolor=cor, edgecolor=cor, alpha=0.7)
    ax1.text(-0.05, yl + 0.5*len(vals), nomes[c]); yl = yu + 10
ax1.axvline(x=sm, color="red", ls="--", label=f"média={sm:.3f}")
ax1.set_xlim([-0.2, 1.0]); ax1.set_title(f"Silhueta por aluno (K={K})"); ax1.set_xlabel("coef. silhueta")
ax1.set_yticks([]); ax1.legend(loc="lower right")
cores_pca = cm.nipy_spectral(C["cluster"].values.astype(float) / K)
ax2.scatter(proj[:,0], proj[:,1], c=cores_pca, s=4, alpha=0.35)
ax2.scatter(cen2[:,0], cen2[:,1], marker="o", c="white", s=260, edgecolor="k", zorder=3)
for c in range(K): ax2.scatter(cen2[c,0], cen2[c,1], marker=f"${nomes[c][0].upper()}$", c="k", s=60, zorder=4)
ax2.set_title("Alunos em 2D (PCA)"); ax2.set_xlabel(f"PCA1 ({var[0]:.0f}%)"); ax2.set_ylabel(f"PCA2 ({var[1]:.0f}%)")
plt.suptitle(f"Diagnóstico de silhueta — padrão diário por prova, K={K}", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "04_silhueta_facas_e_mapa.png"), dpi=130); plt.close()

# 12. Cruzamento com o perfil do CURSO INTEIRO (passo14, Definição B) -- não é
# validação (são eixos diferentes por design), é a comparação que interessa:
# mostra se um "equilibrado" (curso inteiro) esconde "crammers" (crescente).
try:
    Cb = pd.read_csv(CURVA_B_CSV, index_col=0)[["perfil"]]
    Cx = C.join(Cb, how="inner")
    cross = pd.crosstab(Cx["perfil"], Cx["padrao"])
    cross_pct = pd.crosstab(Cx["perfil"], Cx["padrao"], normalize="index").mul(100).round(1)
    print("\n--- CRUZAMENTO: perfil do curso inteiro (passo14) x padrão por prova (este passo) ---")
    print(cross.to_string())
    print("\n--- % por linha (dentro de cada perfil do curso inteiro, como se dividem os padrões por prova) ---")
    print(cross_pct.to_string())
    cross.to_csv(os.path.join(OUT_DIR, "cruzamento_com_passo14.csv"))
except FileNotFoundError:
    print("\n(passo14 não encontrado -- pulei o cruzamento)")

C.to_csv(os.path.join(OUT_DIR, "padroes_por_prova.csv"))

print("\n" + "="*70)
print(f"Gravado em {os.path.abspath(OUT_DIR)}:")
print("  - 01_cotovelo_silhueta.png       02_perfis_curvas.png")
print("  - 03_resultado_por_cluster.png   04_silhueta_facas_e_mapa.png")
print("  - padroes_por_prova.csv          perfil_dos_padroes.csv")
print("  - cruzamento_com_passo14.csv     passo20.md            saida_terminal.txt")
print(f"\nPasso 20 OK. W={W} dias, {N_BLOCOS} blocos de {BLOCO_DIAS} dias, K={K}.")
sys.stdout = _orig; _log.close()
