# -*- coding: utf-8 -*-
"""
Dia 1 — Exploração inicial do dataset OULAD
Objetivo: conhecer cada tabela, ver tamanho, tipos, nulos, duplicados,
checar como as tabelas se conectam (chaves), e gerar gráficos básicos.

IMPORTANTE sobre os nomes em inglês:
As colunas dos CSVs originais (ex: final_result, sum_click, code_module)
permanecem em inglês no código — são os nomes reais das colunas no arquivo,
e mudar isso quebraria a leitura/junção dos dados. O que foi traduzido para
português é apenas a CAMADA DE EXIBIÇÃO: os textos impressos no .txt e os
títulos/eixos/legendas dos gráficos, para facilitar sua leitura e interpretação.

Estrutura esperada de pastas:
ANALITICA_APREN_TRAB_FINAL/
    dataset/
        assessments.csv
        courses.csv
        studentAssessment.csv
        studentInfo.csv
        studentRegistration.csv
        studentVle.csv
        vle.csv
    explorar_dataset.py   <- este arquivo

Saídas geradas (tudo dentro de uma pasta nova "exploracao/"):
    exploracao/resultado_exploracao.txt
    exploracao/01_distribuicao_resultado_final.png
    exploracao/02_alunos_por_curso.png
    exploracao/03_distribuicao_cliques_amostra.png
    exploracao/04_distribuicao_notas.png
    exploracao/05_cliques_por_dia_amostra.png
"""

import sys
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem interface gráfica, só para salvar arquivos
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)

DATASET_DIR = "dataset"        # pasta onde estão os CSVs originais
OUT_DIR = "exploracao"         # pasta de saída (texto + gráficos)
TXT_PATH = os.path.join(OUT_DIR, "resultado_exploracao.txt")

os.makedirs(OUT_DIR, exist_ok=True)

sns.set_style("whitegrid")


# ---------------------------------------------------------------------
# Dicionários de tradução (somente para exibição em texto e gráficos)
# ---------------------------------------------------------------------

# Tradução dos valores de final_result, a variável-alvo do projeto
TRADUCAO_RESULTADO = {
    "Pass": "Aprovado",
    "Fail": "Reprovado",
    "Withdrawn": "Desistente",
    "Distinction": "Aprovado com distinção",
}

# Tradução dos nomes das colunas, usada apenas para imprimir tabelas
# mais legíveis no .txt (não altera os nomes reais no dataframe)
TRADUCAO_COLUNAS = {
    "code_module": "curso",
    "code_presentation": "edicao_do_curso",
    "id_student": "id_aluno",
    "gender": "genero",
    "region": "regiao",
    "highest_education": "escolaridade_anterior",
    "imd_band": "faixa_de_renda_da_regiao",
    "age_band": "faixa_etaria",
    "num_of_prev_attempts": "tentativas_anteriores",
    "studied_credits": "creditos_cursados",
    "disability": "possui_deficiencia",
    "final_result": "resultado_final",
    "date_registration": "data_matricula",
    "date_unregistration": "data_desistencia",
    "id_assessment": "id_avaliacao",
    "date_submitted": "data_entrega",
    "is_banked": "nota_aproveitada_de_tentativa_anterior",
    "score": "nota",
    "id_site": "id_material",
    "activity_type": "tipo_de_atividade",
    "week_from": "semana_inicio",
    "week_to": "semana_fim",
    "date": "dia_relativo_ao_inicio_do_curso",
    "sum_click": "total_de_cliques",
    "assessment_type": "tipo_de_avaliacao",
    "weight": "peso_na_nota_final",
    "module_presentation_length": "duracao_em_dias",
}


def traduzir_colunas(df):
    """Retorna uma cópia do dataframe só com os nomes de coluna traduzidos,
    usada apenas para exibição no texto (não afeta o df original)."""
    return df.rename(columns=TRADUCAO_COLUNAS)


def traduzir_resultado(serie):
    """Traduz os valores de final_result para português, mantendo
    categorias não mapeadas como estão (segurança contra valor inesperado)."""
    return serie.map(TRADUCAO_RESULTADO).fillna(serie)


# ---------------------------------------------------------------------
# Redirecionamento de toda a saída de print() para o arquivo .txt
# (mantém também uma cópia no terminal, para acompanhar em tempo real)
# ---------------------------------------------------------------------
class TeeOutput:
    """Escreve simultaneamente no terminal e em um arquivo."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


txt_file = open(TXT_PATH, "w", encoding="utf-8")
sys.stdout = TeeOutput(sys.__stdout__, txt_file)


def linha(titulo):
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)


def resumo_tabela(nome, df):
    """Aplica o checklist padrão da Entrega 2 (Seção 2) a uma tabela,
    exibindo nomes de coluna traduzidos para facilitar a leitura."""
    df_exibicao = traduzir_colunas(df)
    linha(f"TABELA: {nome}")
    print(f"Formato (linhas, colunas): {df.shape}")
    print(f"\nTipos de dado:\n{df_exibicao.dtypes}")
    print(f"\nValores nulos por coluna:\n{df_exibicao.isnull().sum()}")
    print(f"\nLinhas duplicadas: {df.duplicated().sum()}")
    print(f"\nPrimeiras 5 linhas:\n{df_exibicao.head()}")


def salvar_grafico(fig, nome_arquivo):
    caminho = os.path.join(OUT_DIR, nome_arquivo)
    fig.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[gráfico salvo] {caminho}")


# ---------------------------------------------------------------------
# 1. Carregar as tabelas pequenas por completo
# ---------------------------------------------------------------------
courses = pd.read_csv(f"{DATASET_DIR}/courses.csv")
assessments = pd.read_csv(f"{DATASET_DIR}/assessments.csv")
vle = pd.read_csv(f"{DATASET_DIR}/vle.csv")
student_info = pd.read_csv(f"{DATASET_DIR}/studentInfo.csv")
student_registration = pd.read_csv(f"{DATASET_DIR}/studentRegistration.csv")
student_assessment = pd.read_csv(f"{DATASET_DIR}/studentAssessment.csv")

for nome, df in [
    ("courses (cursos)", courses),
    ("assessments (avaliacoes)", assessments),
    ("vle (materiais do ambiente virtual)", vle),
    ("studentInfo (dados do aluno + resultado)", student_info),
    ("studentRegistration (matricula/desistencia)", student_registration),
    ("studentAssessment (notas dos alunos)", student_assessment),
]:
    resumo_tabela(nome, df)


# ---------------------------------------------------------------------
# 2. studentVle.csv — a tabela grande (10M+ linhas)
#    Em vez de pegar as primeiras linhas, lemos em chunks e pegamos
#    1% de cada chunk aleatoriamente para evitar viés de ordenação.
# ---------------------------------------------------------------------
linha("TABELA: studentVle - cliques dos alunos (amostra aleatória de ~1% para inspeção)")

chunks_amostra = []
total_linhas = 0

# Lendo de 1 em 1 milhão e pegando 1% aleatório
for chunk in pd.read_csv(f"{DATASET_DIR}/studentVle.csv", chunksize=1_000_000):
    total_linhas += len(chunk)
    chunks_amostra.append(chunk.sample(frac=0.01, random_state=42))

student_vle_amostra = pd.concat(chunks_amostra, ignore_index=True)

print(f"Total de linhas reais no arquivo: {total_linhas}")
print(f"Formato da amostra extraída: {student_vle_amostra.shape}")
print(f"\nTipos de dado:\n{traduzir_colunas(student_vle_amostra).dtypes}")
print(f"\nPrimeiras 5 linhas:\n{traduzir_colunas(student_vle_amostra).head()}")
print(f"\nEstatísticas de 'dia_relativo_ao_inicio_do_curso' na amostra (cuidado: pode haver valores negativos):")
print(student_vle_amostra["date"].describe())
print(f"\nEstatísticas de 'total_de_cliques' na amostra:")
print(student_vle_amostra["sum_click"].describe())


# ---------------------------------------------------------------------
# 3. Perguntas centrais do Dia 1: quantos alunos, distribuição de resultado
# ---------------------------------------------------------------------
linha("Quantos alunos únicos existem (id_aluno)")
print(f"Linhas em studentInfo (aluno x curso): {len(student_info)}")
print(f"Alunos únicos (id_student.nunique()): {student_info['id_student'].nunique()}")
print("\n-> A diferença entre os dois números mostra quantos alunos fizeram mais de um curso/tentativa.")

linha("Distribuição do resultado final (variável-alvo)")
resultado_traduzido = traduzir_resultado(student_info["final_result"])
print(resultado_traduzido.value_counts())
print("\nProporção (%):")
print((resultado_traduzido.value_counts(normalize=True) * 100).round(2))

# Gráfico 1: distribuição do resultado final
fig, ax = plt.subplots(figsize=(7, 4.5))
ordem = resultado_traduzido.value_counts().index
sns.countplot(x=resultado_traduzido, order=ordem, ax=ax)
ax.set_title("Distribuição do resultado final (todos os alunos e cursos)")
ax.set_xlabel("Resultado final")
ax.set_ylabel("Número de registros")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=9)
salvar_grafico(fig, "01_distribuicao_resultado_final.png")

linha("Quantos registros (aluno x curso) por curso (code_module)")
print(student_info["code_module"].value_counts())

linha("Quantos registros por curso x edição (code_module + code_presentation)")
print(student_info.groupby(["code_module", "code_presentation"]).size())

# Gráfico 2: registros por curso (code_module)
fig, ax = plt.subplots(figsize=(7, 4.5))
contagem_curso = student_info["code_module"].value_counts().sort_index()
sns.barplot(x=contagem_curso.index, y=contagem_curso.values, ax=ax)
ax.set_title("Número de registros (aluno x curso) por curso")
ax.set_xlabel("Curso")
ax.set_ylabel("Número de registros")
salvar_grafico(fig, "02_alunos_por_curso.png")


# ---------------------------------------------------------------------
# 4. Checagem de chaves entre tabelas
# ---------------------------------------------------------------------
linha("Checagem de chaves: studentInfo vs studentVle")
alunos_info = set(student_info["id_student"].unique())

alunos_vle = set()
for chunk in pd.read_csv(f"{DATASET_DIR}/studentVle.csv", usecols=["id_student"], chunksize=1_000_000):
    alunos_vle.update(chunk["id_student"].unique())

print(f"Alunos únicos em studentInfo: {len(alunos_info)}")
print(f"Alunos únicos em studentVle: {len(alunos_vle)}")
print(f"Alunos em studentInfo SEM nenhum clique em studentVle: {len(alunos_info - alunos_vle)}")
print(f"Alunos em studentVle SEM registro em studentInfo: {len(alunos_vle - alunos_info)}")

linha("Checagem de chaves: studentInfo vs studentAssessment")
alunos_assessment = set(student_assessment["id_student"].unique())
print(f"Alunos únicos em studentAssessment: {len(alunos_assessment)}")
print(f"Alunos em studentInfo SEM nenhuma nota em studentAssessment: {len(alunos_info - alunos_assessment)}")


# ---------------------------------------------------------------------
# 5. Alunos que desistiram (Withdrawal) — checagem cruzada com registration
# ---------------------------------------------------------------------
linha("Alunos desistentes x data de desistência preenchida")
withdrawn = student_info[student_info["final_result"] == "Withdrawn"]
reg_withdrawn = student_registration.merge(
    withdrawn[["code_module", "code_presentation", "id_student"]],
    on=["code_module", "code_presentation", "id_student"],
)
print(f"Total de registros de alunos desistentes em studentInfo: {len(withdrawn)}")
print(f"Desses, com data de desistência preenchida: {reg_withdrawn['date_unregistration'].notna().sum()}")
print(f"Desses, SEM data de desistência preenchida: {reg_withdrawn['date_unregistration'].isna().sum()}")


# ---------------------------------------------------------------------
# 6. Gráficos adicionais a partir da amostra de studentVle e das notas
# ---------------------------------------------------------------------
linha("Gráficos adicionais (baseados na amostra de studentVle e nas notas)")

# Gráfico 3: distribuição de cliques (amostra aleatória)
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.histplot(student_vle_amostra["sum_click"], bins=30, kde=True, ax=ax)
ax.set_title("Distribuição de cliques por interação (amostra aleatória de ~1%)")
ax.set_xlabel("Cliques por interação registrada")
ax.set_ylabel("Frequência")
salvar_grafico(fig, "03_distribuicao_cliques_amostra.png")

# Gráfico 4: distribuição de notas (studentAssessment)
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.histplot(student_assessment["score"].dropna(), bins=30, kde=True, ax=ax)
ax.set_title("Distribuição de notas das avaliações")
ax.set_xlabel("Nota (0 a 100)")
ax.set_ylabel("Frequência")
salvar_grafico(fig, "04_distribuicao_notas.png")

# Gráfico 5: cliques agregados por dia (amostra aleatória)
fig, ax = plt.subplots(figsize=(8, 4.5))
cliques_por_dia = student_vle_amostra.groupby("date")["sum_click"].sum().sort_index()
ax.plot(cliques_por_dia.index, cliques_por_dia.values, linewidth=1)
ax.axvline(0, color="red", linestyle="--", linewidth=1, label="Início do curso (dia 0)")
ax.set_title("Soma de cliques por dia (amostra aleatória de ~1%)")
ax.set_xlabel("Dia (relativo ao início do curso)")
ax.set_ylabel("Soma de cliques")
ax.legend()
salvar_grafico(fig, "05_cliques_por_dia_amostra.png")

print("\n\nExploração do Dia 1 concluída.")
print(f"Texto completo salvo em: {TXT_PATH}")
print(f"Gráficos salvos na pasta: {OUT_DIR}/")

sys.stdout = sys.__stdout__  # restaura a saída padrão antes de fechar o arquivo
txt_file.close()