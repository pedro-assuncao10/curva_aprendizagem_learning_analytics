#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
PASSO 1 — ENGENHARIA DE FEATURES (curva temporal de engajamento)
Projeto: Curva Temporal de Engajamento — OULAD
=============================================================================

O que este script faz:

  1. Carrega os CSVs do OULAD.
  2. Transforma studentVle (log bruto de cliques) em uma SÉRIE SEMANAL por aluno.
  3. Normaliza o tempo pela DURAÇÃO de cada curso (cursos têm tamanhos diferentes),
     reamostrando toda curva para um número FIXO de baldes -> curvas comparáveis.
  4. Separa o grupo de DESISTENTES (Withdrawn) num arquivo à parte e os remove.
  5. Separa cliques PRÉ-CURSO (dia < 0) numa feature própria.
  6. Gera DUAS representações da curva, conforme decidido:
        (a) FORMA  -> proporção do esforço ao longo das semanas (neutraliza volume) [H1]
        (b) VOLUME -> intensidade total / por semana (quem clica pouco/mediano/muito)
  7. Extrai features interpretáveis (inclinação, centro de massa, semanas mortas...).
  8. PLOTA as curvas para inspeção visual.
  9. Salva a matriz aluno × features, pronta para o K-means.

COMO USAR:
  - Edite DATA_DIR abaixo para a pasta onde estão os CSVs do OULAD.
  - Rode:  python passo1.py
  - Depende só de: pandas, numpy, matplotlib
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # salva imagens sem precisar de tela
import matplotlib.pyplot as plt

# ===========================================================================
# CONFIGURAÇÃO  (edite aqui)
# ===========================================================================
DATA_DIR = "./dataset"
OUT_DIR    = "./saida_passo1"   # onde os resultados serão gravados
N_BUCKETS  = 30                 # nº de pontos por curva (resolução temporal fixa)
MIN_CLIQUES = 30                # piso de atividade: alunos abaixo disso são "esparsos"
MIN_SEMANAS_ATIVAS = 3          # piso adicional: precisa ter clicado em >= N semanas
PLOT_AMOSTRA = 200              # nº de alunos amostrados nos plots pesados (heatmap)

os.makedirs(OUT_DIR, exist_ok=True)


def carregar(nome, **kw):
    caminho = os.path.join(DATA_DIR, nome)
    print(f"  lendo {nome} ...")
    return pd.read_csv(caminho, **kw)


# ===========================================================================
# 1) CARGA
# ===========================================================================
print("[1/8] Carregando CSVs...")

courses = carregar("courses.csv")
# chave do curso = módulo + apresentação ; guardamos a duração
courses["curso"] = courses["code_module"] + "_" + courses["code_presentation"]
dur_curso = dict(zip(courses["curso"], courses["module_presentation_length"]))

info = carregar("studentInfo.csv",
                usecols=["code_module", "code_presentation", "id_student", "final_result"])
info["curso"] = info["code_module"] + "_" + info["code_presentation"]
info["aluno"] = info["curso"] + "_" + info["id_student"].astype(str)

reg = carregar("studentRegistration.csv",
               usecols=["code_module", "code_presentation", "id_student",
                        "date_registration", "date_unregistration"])
reg["aluno"] = (reg["code_module"] + "_" + reg["code_presentation"]
                + "_" + reg["id_student"].astype(str))

# studentVle é o arquivo pesado (~10,6M linhas). Lemos só o necessário e em dtypes enxutos.
vle = carregar(
    "studentVle.csv",
    usecols=["code_module", "code_presentation", "id_student", "date", "sum_click"],
    dtype={"id_student": np.int32, "date": np.int16, "sum_click": np.int32},
)
vle["curso"] = vle["code_module"] + "_" + vle["code_presentation"]
vle["aluno"] = vle["curso"] + "_" + vle["id_student"].astype(str)
print(f"     studentVle: {len(vle):,} linhas")


# ===========================================================================
# 2) SEPARAR DESISTENTES
#    (trajetória interrompida por evasão != perfil decrescente genuíno)
# ===========================================================================
print("[2/8] Separando desistentes (Withdrawn)...")

resultado = dict(zip(info["aluno"], info["final_result"]))
alunos_withdrawn = set(info.loc[info["final_result"] == "Withdrawn", "aluno"])
print(f"     desistentes: {len(alunos_withdrawn):,} alunos "
      f"({100*len(alunos_withdrawn)/info['aluno'].nunique():.1f}% da base)")


# ===========================================================================
# 3) SÉRIE SEMANAL NORMALIZADA PELA DURAÇÃO DO CURSO
#    - cliques pré-curso (date < 0) viram feature à parte
#    - tempo vira fração do curso [0,1] e cai em N_BUCKETS baldes fixos
# ===========================================================================
print("[3/8] Construindo a série temporal por aluno...")

vle["length"] = vle["curso"].map(dur_curso)

# --- cliques PRÉ-CURSO (material de boas-vindas etc.) -> feature separada
pre = vle.loc[vle["date"] < 0].groupby("aluno")["sum_click"].sum()
pre.name = "cliques_pre_curso"

# --- cliques DENTRO do curso [0, length]
dentro = vle.loc[(vle["date"] >= 0) & (vle["date"] <= vle["length"])].copy()
fracao = dentro["date"] / dentro["length"].clip(lower=1)         # 0..1
dentro["bucket"] = np.minimum((fracao * N_BUCKETS).astype(int), N_BUCKETS - 1)

# matriz aluno × bucket (soma de cliques por balde temporal)
serie = (dentro.groupby(["aluno", "bucket"])["sum_click"].sum()
                .unstack(fill_value=0)
                .reindex(columns=range(N_BUCKETS), fill_value=0))
serie.columns = [f"w{c:02d}" for c in serie.columns]
print(f"     {serie.shape[0]:,} alunos com pelo menos 1 clique no curso; "
      f"{serie.shape[1]} baldes temporais")


# ===========================================================================
# 4) FILTRO DE ESPARSIDADE E DESISTENTES
#    quem quase não clicou não tem "forma" — vira ruído na clusterização
# ===========================================================================
print("[4/8] Aplicando filtro de esparsidade e removendo desistentes...")

total_cliques  = serie.sum(axis=1)
semanas_ativas = (serie > 0).sum(axis=1)
mask_ok = (total_cliques >= MIN_CLIQUES) & (semanas_ativas >= MIN_SEMANAS_ATIVAS)

# ADIÇÃO: Filtrar os desistentes AGORA, antes de normalizar a forma
mask_nao_desistente = ~serie.index.isin(alunos_withdrawn)
mask_final = mask_ok & mask_nao_desistente

descartados_esparsos = (~mask_ok).sum()
descartados_withdrawn = mask_ok.sum() - mask_final.sum()

print(f"     descartados por esparsidade: {descartados_esparsos:,} "
      f"({100*descartados_esparsos/len(serie):.1f}%)  "
      f"[<{MIN_CLIQUES} cliques ou <{MIN_SEMANAS_ATIVAS} semanas ativas]")
print(f"     descartados por serem desistentes: {descartados_withdrawn:,} "
      f"({100*descartados_withdrawn/len(serie):.1f}%) "
      f"[removidos para não sujar a clusterização]")

serie = serie.loc[mask_final]
total_cliques  = total_cliques.loc[mask_final]
semanas_ativas = semanas_ativas.loc[mask_final]


# ===========================================================================
# 5) DUAS REPRESENTAÇÕES: FORMA e VOLUME
# ===========================================================================
print("[5/8] Gerando representações de FORMA e de VOLUME...")

# (a) FORMA: cada curva dividida pela própria soma -> proporção do esforço.
#     Dois alunos com mesmo ritmo, intensidades diferentes, ficam idênticos aqui.
forma = serie.div(serie.sum(axis=1), axis=0)
forma.columns = [f"forma_{c}" for c in serie.columns]

# (b) VOLUME: a série bruta tal como está (mantém magnitude).
volume_bruto = serie.copy()
volume_bruto.columns = [f"vol_{c}" for c in serie.columns]


# ===========================================================================
# 6) FEATURES INTERPRETÁVEIS (derivadas da curva)
# ===========================================================================
print("[6/8] Extraindo features interpretáveis...")

x = np.arange(N_BUCKETS)
P = forma.values                      # proporções (linhas somam ~1)

# inclinação: regressão linear simples da FORMA no tempo
#   >0 cresce ao longo do curso, <0 decresce
x_c = x - x.mean()
slope = (P * x_c).sum(axis=1) / (x_c**2).sum()

# centro de massa temporal: "quando" o esforço se concentra (0=início, 1=fim)
centro_massa = (P * x).sum(axis=1) / (N_BUCKETS - 1)

# semanas mortas: baldes sem nenhum clique
semanas_mortas = (serie.values == 0).sum(axis=1)

# regularidade: desvio-padrão da forma (alto = irregular/concentrado)
std_forma = P.std(axis=1)

# posição do pico (fração do curso onde mais clicou)
pico_pos = serie.values.argmax(axis=1) / (N_BUCKETS - 1)

# fração do esforço na metade final do curso (proxy de "deixa pra depois")
metade = N_BUCKETS // 2
frac_segunda_metade = P[:, metade:].sum(axis=1)

feat = pd.DataFrame({
    "total_cliques":        total_cliques.values,
    "media_semanal":        serie.mean(axis=1).values,
    "semanas_ativas":       semanas_ativas.values,
    "semanas_mortas":       semanas_mortas,
    "inclinacao":           slope,
    "centro_massa":         centro_massa,
    "std_forma":            std_forma,
    "pico_posicao":         pico_pos,
    "frac_segunda_metade":  frac_segunda_metade,
}, index=serie.index)

# pré-curso e resultado final (resultado NÃO entra na clusterização — só p/ validar depois)
feat["cliques_pre_curso"] = pre.reindex(feat.index).fillna(0).astype(int)
feat["final_result"]      = pd.Series(resultado).reindex(feat.index)

# rótulo de volume em tercis (o seu "clica pouco / mediano / muito") — descritivo
feat["faixa_volume"] = pd.qcut(feat["total_cliques"], 3,
                               labels=["baixo", "medio", "alto"])


# ===========================================================================
# 7) PLOTS PARA INSPEÇÃO VISUAL
# ===========================================================================
print("[7/8] Gerando plots...")

eixo_frac = x / (N_BUCKETS - 1)  # eixo x em fração do curso (0..1)

# --- 7.1 curva média geral (forma) + faixa de desvio (CORRIGIDO PARA PERCENTIS)
plt.figure(figsize=(9, 5))
mediana = np.median(P, axis=0)
p25 = np.percentile(P, 25, axis=0)
p75 = np.percentile(P, 75, axis=0)

plt.plot(eixo_frac, mediana, lw=2, label="forma mediana")
plt.fill_between(eixo_frac, p25, p75, alpha=0.2, label="Intervalo Interquartil (50% dos alunos)")
plt.xlabel("fração do curso (0=início, 1=fim)")
plt.ylabel("proporção do esforço (forma)")
plt.title("Curva representativa de engajamento (Mediana e IQR)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_curva_media_forma.png"), dpi=130)
plt.close()

# --- 7.2 amostra de curvas individuais (forma) — pra você "ver" a variedade
plt.figure(figsize=(9, 5))
amostra_idx = np.random.RandomState(0).choice(len(P), size=min(40, len(P)), replace=False)
for i in amostra_idx:
    plt.plot(eixo_frac, P[i], lw=0.8, alpha=0.5)
plt.xlabel("fração do curso"); plt.ylabel("proporção do esforço")
plt.title(f"40 curvas individuais (forma) — variedade de trajetórias")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_curvas_individuais_forma.png"), dpi=130)
plt.close()

# --- 7.3 distribuição de volume com os cortes pouco/medio/muito
plt.figure(figsize=(9, 5))
plt.hist(np.log10(feat["total_cliques"].clip(lower=1)), bins=50, color="#4C72B0")
for q, lab in zip([1/3, 2/3], ["corte baixo|medio", "corte medio|alto"]):
    v = np.log10(feat["total_cliques"].quantile(q))
    plt.axvline(v, color="crimson", ls="--")
    plt.text(v, plt.ylim()[1]*0.9, lab, rotation=90, va="top", fontsize=8)
plt.xlabel("log10(total de cliques)"); plt.ylabel("nº de alunos")
plt.title("Distribuição de VOLUME total (com tercis)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_distribuicao_volume.png"), dpi=130)
plt.close()

# --- 7.4 heatmap aluno × semana (amostra, ordenado por centro de massa)
amostra = feat.sample(min(PLOT_AMOSTRA, len(feat)), random_state=0).index
ordem = feat.loc[amostra, "centro_massa"].sort_values().index
plt.figure(figsize=(9, 6))
plt.imshow(forma.loc[ordem].values, aspect="auto", cmap="magma",
           extent=[0, 1, 0, len(ordem)])
plt.colorbar(label="proporção do esforço")
plt.xlabel("fração do curso"); plt.ylabel("alunos (ordenados por centro de massa)")
plt.title(f"Heatmap forma — {len(ordem)} alunos amostrados")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_heatmap_forma.png"), dpi=130)
plt.close()

# --- 7.5 curva média por FAIXA DE VOLUME (mostra se forma muda com volume)
plt.figure(figsize=(9, 5))
for faixa in ["baixo", "medio", "alto"]:
    idx = feat.index[feat["faixa_volume"] == faixa]
    plt.plot(eixo_frac, forma.loc[idx].values.mean(axis=0), lw=2, label=f"volume {faixa}")
plt.xlabel("fração do curso"); plt.ylabel("proporção do esforço (forma)")
plt.title("Forma média por faixa de volume")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_forma_por_volume.png"), dpi=130)
plt.close()


# ===========================================================================
# 8) SALVAR AS MATRIZES (entrada do Passo 2 — clusterização)
# ===========================================================================
print("[8/8] Salvando matrizes de saída...")

# matriz de FORMA (curva normalizada) — entrada principal da clusterização (H1)
forma.to_csv(os.path.join(OUT_DIR, "matriz_forma.csv"))
# matriz de VOLUME (série bruta)
volume_bruto.to_csv(os.path.join(OUT_DIR, "matriz_volume.csv"))
# features interpretáveis + rótulos descritivos
feat.to_csv(os.path.join(OUT_DIR, "features_interpretaveis.csv"))

# matriz combinada pronta pra experimentar:
#   forma + features. final_result fica de FORA da clusterização (base "não-anotada").
combinada = pd.concat([forma, feat.drop(columns=["final_result", "faixa_volume"])], axis=1)
combinada.to_csv(os.path.join(OUT_DIR, "matriz_clusterizacao.csv"))

# lista de desistentes (analisados em pipeline separado)
pd.Series(sorted(alunos_withdrawn), name="aluno").to_csv(
    os.path.join(OUT_DIR, "alunos_desistentes.csv"), index=False)

print("\n=== RESUMO ===")
print(f"Alunos válidos (pós-filtro):     {len(feat):,}")
print(f"Baldes temporais por curva:      {N_BUCKETS}")
print(f"Distribuição final_result (SEM desistentes):")
print(feat['final_result'].value_counts().to_string())
print(f"\nArquivos gravados em: {os.path.abspath(OUT_DIR)}")
print("  - matriz_forma.csv            (curva-formato, entrada principal H1)")
print("  - matriz_volume.csv           (série bruta de volume)")
print("  - features_interpretaveis.csv (inclinação, centro de massa, etc.)")
print("  - matriz_clusterizacao.csv    (forma + features, pronta p/ K-means)")
print("  - alunos_desistentes.csv      (pipeline separado)")
print("  - 01..05_*.png                (gráficos para inspeção)")
print("\nPASSO 1 concluído. Próximo passo: clusterização sobre matriz_clusterizacao.csv.")