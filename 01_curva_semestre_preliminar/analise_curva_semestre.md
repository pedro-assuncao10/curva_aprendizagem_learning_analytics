# Análise Preliminar — A Curva do Semestre Inteiro

**O que é:** o estudo exploratório que precedeu o pipeline final do projeto, rodado sobre **todo o OULAD** (22 apresentações, ~26 mil alunos com atividade). Funde os antigos passos 1 (forma), 2 (volume) e 3 (durabilidade) num script só: `analise_curva_semestre.py` (rodar da raiz do projeto: `python 01_curva_semestre_preliminar/analise_curva_semestre.py`).

Aqui a curva analisada é a do **semestre inteiro** — o tempo de cada curso é normalizado pela duração (30 fatias de 0 a 100%) e a curva de cada aluno é dividida pela própria soma, isolando a *forma* do *volume*. Foi este estudo que revelou as limitações da visão agregada e motivou a migração para as **curvas por ciclo de avaliação** do pipeline final (`kmeans_ciclos.py` e derivados).

## Parte A — Volume (o baseline do projeto)

Soma todos os cliques de cada aluno, corta em tercis (baixo/médio/alto) e cruza com o resultado final.

- `01_distribuicao_volume.png` — histograma do volume (escala log) com os cortes dos tercis.
- `02_resultado_por_volume.png` — **o insight-baseline**: a aprovação cresce fortemente do tercil baixo para o alto. "Quem clica mais, passa mais" é o óbvio que o resto do projeto tenta superar — toda contribuição da *forma* é medida contra isto.

## Parte B — Forma (a curva do semestre, sem desistentes)

Remove desistentes (Withdrawn) e alunos esparsos (<30 cliques ou <3 fatias ativas), que não têm "forma" mensurável, e olha a distribuição do esforço ao longo do curso.

- `03_curva_mediana_forma.png` — a curva típica: esforço decrescente ao longo do semestre.
- `04_heatmap_forma.png` — 200 alunos ordenados pelo centro de massa temporal: a variedade de trajetórias (cedo → tarde) é contínua, não separada em blocos óbvios.
- `05_forma_por_faixa_volume.png` — a forma média é quase idêntica nos três tercis de volume: **forma e volume são dimensões distintas**, a premissa central do projeto (H1).

## Parte C — Durabilidade (a descoberta que corrigiu o rumo)

O grupo "começo-pesado" (esforço concentrado no início) misturava dois alunos **opostos** com o mesmo centro de massa:

- **abandonou_cedo** — clicou no começo e zerou o resto (o "esforço inicial" é só o rastro de quem parou);
- **antecipou_sustentou** — pico no começo, mas ativo até o fim.

O que os separa é a **durabilidade**: semanas mortas (fatias com zero clique) e atividade no último terço. `06_abandonou_vs_sustentou.png` mostra que a aprovação dos dois é radicalmente diferente em todas as faixas de volume — provar que "quando começa" não basta sem "até quando dura" foi o que motivou o filtro por submissão de avaliação usado no pipeline final.

## Arquivo gerado além dos gráficos

- `features_durabilidade.csv` — uma linha por aluno (sem desistentes): total, proporção por terço, semanas mortas, atividade no fim, timing e resultado. **É insumo dos estudos seguintes**: `03_preparacao_pre_prova_7dias/passo6.py` e `04_padrao_pre_prova_28dias/passo20.py` o leem para definir os alunos válidos.

## Lições que este estudo deixou para o pipeline final

1. A curva do semestre **esconde** o ritmo de preparação por prova (um aluno que concentra na véspera de *todas* as provas parece "equilibrado" no agregado) → análise final é por **ciclo de avaliação**.
2. Curvas truncadas de desistentes imitam perfis "decrescentes" → o pipeline final só usa ciclos cujo TMA foi **submetido**.
3. Forma ≠ volume (Parte B) → no pipeline final, forma normalizada clusteriza e volume vira covariável dos modelos preditivos.
