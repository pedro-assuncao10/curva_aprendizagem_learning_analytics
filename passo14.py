#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PASSO 14 — CURVA POR TEMPO (sem ancorar em prova)
Mesma pergunta do passo 12 (separar desistente/antecipado/equilibrado/
fim-pesado pela curva de esforço do CURSO INTEIRO — Definição B, ver
passo12.md), mas com o cálculo da curva simplificado:

  passo 12: intervalo = entre datas de prova (tamanho desigual) -> taxa
            (cliques/dia) -> reamostra em pontos fixos do calendário.
  passo 14: fatia = % fixo e IGUAL do calendário do curso (não depende de
            quantas provas o curso tem, nem de quando elas caem). Sem
            provas, sem duração desigual, sem taxa/dia -- é só a proporção
            de cliques que cai em cada 1/N do curso.

Por que trocar: descobrimos que o cálculo do passo 12 carregava um viés de
calendário (até 44% da variação de um ponto da curva era só "qual curso o
aluno fez", não comportamento dele -- porque as fatias nasciam ancoradas nas
datas de prova de CADA curso, que caem em lugares diferentes do calendário).
Fatias de tempo fixas removem essa dependência: 10%, 20%, 30% do curso
significam a mesma coisa pra qualquer curso, sem intermediário nenhum.
Testado com 5 sementes de K-means: ARI contra o passo 3 subiu de 0,175
(passo 12) para ~0,295 (aqui) -- ver passo14.md para os números completos.

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
OUT_DIR  = "./saida_passo14"
N        = 6      # nº de fatias iguais do calendário do curso (0-100%)
K_RANGE  = range(2, 9)
DEFAULT_K = 3      # perfis dentro de quem persiste: antecipado / equilibrado / fim-pesado
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

print("="*70); print("PASSO 14 — CURVA POR TEMPO (fatias iguais do calendário, sem prova)"); print("="*70)

def chave(df):
    return df["code_module"] + "_" + df["code_presentation"] + "_" + df["id_student"].astype(str)

# 1. Alunos válidos + sinal de persistência (passo 3)
val = pd.read_csv(DUR_CSV, index_col="aluno")
validos = set(val.index)
cp = pd.read_csv(CP_CSV, index_col=0)
print(f"\nAlunos válidos: {len(validos):,}")

# 2. Duração de cada curso (só isso é necessário -- nenhuma data de prova)
courses = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
courses["curso"] = courses["code_module"] + "_" + courses["code_presentation"]
comprimento = dict(zip(courses["curso"], courses["module_presentation_length"]))

# 3. Atribuir 100% dos cliques de cada aluno válido à fatia de tempo (0 a N-1)
vle = pd.read_csv(os.path.join(DATA_DIR, "studentVle.csv"),
                  usecols=["code_module","code_presentation","id_student","date","sum_click"])
vle["aluno"] = chave(vle)
vle["curso"] = vle["code_module"] + "_" + vle["code_presentation"]
vle = vle[vle["aluno"].isin(validos)].copy()
vle["comprimento"] = vle["curso"].map(comprimento)
vle["frac"] = (vle["date"] / vle["comprimento"]).clip(0, 0.999999)
vle["fatia"] = np.minimum((vle["frac"] * N).astype(int), N - 1)

capturado, disponivel = vle["sum_click"].sum(), vle["sum_click"].sum()  # 100% por construção (sem corte)
print(f"Cliques capturados: {capturado:,} (100% -- não há corte de janela nem de intervalo aqui)")

piv = vle.groupby(["aluno","fatia"])["sum_click"].sum().unstack(fill_value=0)
piv = piv.reindex(columns=range(N), fill_value=0)
soma = piv.sum(axis=1)
piv = piv[soma > 0]; soma = soma[soma > 0]
prop = piv.div(soma, axis=0)
feat_cols = [f"c{i}" for i in range(N)]
prop.columns = feat_cols

C = prop.join(val[["final_result","faixa_volume","ativo_no_fim"]], how="inner")
C["aprovado"] = C["final_result"].isin(["Pass","Distinction"])
print(f"Alunos com curva calculada: {len(C):,}")

# 4. ESTÁGIO 1 (regra): separar desistente de persistente
persist = C[C["ativo_no_fim"]].copy()
n_desist = (~C["ativo_no_fim"]).sum()
print(f"\nEstágio 1 (regra ativo_no_fim): desistentes={n_desist:,}  persistentes={len(persist):,}")
print(f"Aprovação dos desistentes: {100*C.loc[~C['ativo_no_fim'],'aprovado'].mean():.1f}%")
print(f"Aprovação dos persistentes: {100*persist['aprovado'].mean():.1f}%")

# 5. ESTÁGIO 2 (K-means): clusteriza a curva só entre quem persiste
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

# 6. Nomear cada cluster pela forma da curva (centro de massa temporal)
centro_massa = {}
for c in range(K):
    media = persist.loc[persist["cluster"] == c, feat_cols].mean().values
    centro_massa[c] = (media * np.arange(N)).sum() / media.sum()
ordem = sorted(centro_massa, key=centro_massa.get)
nomes = {ordem[0]: "antecipado", ordem[-1]: "fim-pesado"}
for c in ordem[1:-1]:
    nomes[c] = "equilibrado"
persist["perfil"] = persist["cluster"].map(nomes)

C["perfil"] = "desistente"
C.loc[persist.index, "perfil"] = persist["perfil"]
ordem_perfis = ["desistente", "antecipado", "equilibrado", "fim-pesado"]

# 7. Gráfico do PADRÃO DE CURVAS — mesmo estilo dos passos 6/7/12
eixo = (np.arange(N) + 0.5) / N   # centro de cada fatia, em fração do curso
plt.figure(figsize=(10, 5))
cores = {"desistente": "#C62828", "antecipado": "#1976D2", "equilibrado": "#F9A825", "fim-pesado": "#2E7D32"}
for p in ordem_perfis:
    al = C.index[C["perfil"] == p]
    plt.plot(eixo, C.loc[al, feat_cols].mean().values, lw=2, marker="o",
             color=cores[p], label=f"{p} (n={len(al):,})")
plt.xlabel("posição no curso (0 = início, 1 = fim oficial)")
plt.ylabel("proporção média dos cliques (fatias de tempo iguais)")
plt.title(f"Perfis de curva por TEMPO (estágio 1 regra + estágio 2 K={K}, N={N} fatias)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_perfis_curvas.png"), dpi=130); plt.close()

# 8. Validação: cruzamento com os perfis de referência do passo 3
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
print("\n--- VALIDAÇÃO: cruzamento perfil (passo14) x perfil (passo3, referência) ---")
print(cross.to_string())
print("\n--- mesmo cruzamento em % por linha (pureza) ---")
print(cross_pct.to_string())
print(f"\nARI do estágio 2 (K-means na curva por tempo) vs perfil de referência do passo3: {ari_estagio2:.3f}")
cross.to_csv(os.path.join(OUT_DIR, "validacao_crosstab.csv"))

# 9. Tabela-resumo por perfil
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
plt.ylabel("% dos alunos"); plt.xlabel("perfil"); plt.title("Resultado final por perfil de curva (por tempo)")
plt.xticks(rotation=0); plt.legend(title="", bbox_to_anchor=(1.02,1))
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "03_resultado_por_cluster.png"), dpi=130); plt.close()

# 10. Silhueta (facas) + mapa PCA — diagnóstico do estágio 2
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
plt.suptitle(f"Diagnóstico de silhueta — estágio 2, K={K}, curva por tempo", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "04_silhueta_facas_e_mapa.png"), dpi=130); plt.close()

# 11. Comparação com o método anterior (passo 12, ancorado em prova) para registro
comparacao = pd.DataFrame([
    {"metodo": "Ancorado em prova (passo12)", "ARI_vs_passo3": 0.175},
    {"metodo": f"Por tempo, N={N} (passo14, este)", "ARI_vs_passo3": round(ari_estagio2, 3)},
])
comparacao.to_csv(os.path.join(OUT_DIR, "comparacao_com_passo12.csv"), index=False)
print("\n--- Comparação com o método anterior ---")
print(comparacao.to_string(index=False))

C.to_csv(os.path.join(OUT_DIR, "curvas_e_perfis.csv"))

print("\n" + "="*70)
print(f"Gravado em {os.path.abspath(OUT_DIR)}:")
print("  - 01_cotovelo_silhueta.png       02_perfis_curvas.png")
print("  - 03_resultado_por_cluster.png   04_silhueta_facas_e_mapa.png")
print("  - curvas_e_perfis.csv            perfil_dos_grupos.csv")
print("  - validacao_crosstab.csv         comparacao_com_passo12.csv")
print("  - passo14.md                     saida_terminal.txt")
print(f"\nPasso 14 OK. Estágio1 (regra) + Estágio2 (K={K} na curva por tempo). ARI vs passo3={ari_estagio2:.3f}")
sys.stdout = _orig; _log.close()
