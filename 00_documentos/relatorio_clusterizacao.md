# Perfis Temporais de Preparação para Avaliações: Clusterização de Curvas de Esforço no OULAD com K-means, K-medoids e Agrupamento Hierárquico

**Disciplina:** Aprendizagem de Máquina — PPGCC/UFMA
**Professor:** Luciano Reis Coutinho
**Aluno:** Pedro Assunção
**Lista de Atividades 2 — Questões 2 (K-means) e 3 (Agrupamento Hierárquico), apresentadas em relatório único por se tratarem de clusterização da mesma base com métodos diferentes.**

---

## 1. Introdução

### 1.1 Contexto

A maior parte das análises de engajamento em ambientes virtuais de aprendizagem resume o comportamento do aluno a um número agregado — total de cliques, total de acessos no semestre. Essa redução esconde uma diferença importante: dois alunos com o mesmo volume de interações podem ter distribuído seu esforço de formas completamente distintas ao longo do tempo. Este trabalho investiga a **forma da curva de esforço** do aluno, e não seu volume: interessa *quando* o aluno estuda, não *quanto*.

O trabalho se insere no projeto de pesquisa "Curva Temporal de Engajamento: Identificação de Perfis de Estudo e seu Efeito sobre o Desempenho" (ficha inicial da disciplina IA Aplicada à Educação), cuja primeira hipótese (H1) postula a existência de perfis distintos e identificáveis de trajetória de engajamento — por exemplo, alunos que **antecipam** a preparação, alunos de ritmo **equilibrado** e alunos que **deixam para o final** — mesmo entre alunos com volume total de acessos semelhante.

### 1.2 Problema

Dado o registro diário de cliques de cada aluno em um ambiente virtual, deseja-se descobrir, **sem rótulos prévios** (aprendizado não supervisionado), se existem grupos naturais de "formato de preparação" para as avaliações — e quantos são. Duas decisões de modelagem são centrais e foram tratadas como parte do problema:

1. **Curva por avaliação, não por curso inteiro.** Um aluno que concentra o estudo na véspera de *todas* as provas apresenta, na visão agregada do semestre, uma curva de volume aproximadamente constante — e seria classificado, erroneamente, como "equilibrado". A curva correta para capturar os perfis de interesse é a distribuição do esforço **dentro do ciclo de cada avaliação**.
2. **Separar forma de volume.** Clusterizar contagens brutas agrupa por quantidade (quem clica muito vs. pouco). Para agrupar por forma, cada curva é normalizada pela soma do próprio aluno no ciclo, tornando-se uma distribuição de proporções.

### 1.3 Objetivos

- Construir uma base não anotada de **curvas de esforço por ciclo de avaliação** a partir do OULAD;
- Realizar experimentos de **K-means** variando o número de grupos (k = 2..8), o número de atributos (4 e 8 bins temporais) e a medida de distância/similaridade (euclidiana, Manhattan, cosseno e Chebyshev, via **K-medoids/PAM implementado do zero**, já que o K-means clássico é euclidiano por construção);
- Realizar experimentos de **agrupamento hierárquico** variando a medida de distância entre grupos (linkage Ward, completo, médio e simples) e, como enriquecimento, a métrica ponto-a-ponto;
- Avaliar os particionamentos por **cotovelo, silhueta (média e por amostra), correlação cofenética, ARI e balanceamento dos grupos**, e decidir de forma justificada o melhor particionamento;
- Verificar a associação preliminar entre perfil encontrado e resultado final do aluno (aprovação/reprovação/evasão).

### 1.4 Ferramentas escolhidas

| Ferramenta | Versão | Uso |
|---|---|---|
| Python | 3.13 | linguagem de todo o pipeline |
| pandas | 3.0 | leitura em chunks, agregações, tabelas |
| scikit-learn | 1.9 | `KMeans`, silhueta, ARI, PCA |
| SciPy | — | `pdist`, `linkage`, dendrogramas, cofenética |
| matplotlib | 3.11 | todas as figuras |
| **K-medoids (PAM)** | implementação própria | variação da medida de distância (atende ao "implemente ou escolha implementação") |

Scripts do trabalho: `kmeans_ciclos.py` (pipeline principal), `kmeans_divisivo.py` (divisão recursiva), `hierarquico.py` (linkages), `distancias.py` (K-medoids × métricas) e `hierarquico_distancias.py` (hierárquico × métricas).

---

## 2. Bases de Dados

### 2.1 Descrição

Foi utilizado o **OULAD — Open University Learning Analytics Dataset** (Kuzilek, Hlosta & Zdrahal, 2017), dataset público (licença CC-BY 4.0) da Open University (Reino Unido), com dados de 32.593 matrículas em 7 módulos (AAA–GGG) e 22 apresentações (2013–2014). Tabelas usadas:

| Arquivo | Conteúdo | Papel neste trabalho |
|---|---|---|
| `studentVle.csv` (~433 MB, ≈10,65 milhões de linhas) | cliques por aluno × material × dia | matéria-prima das curvas |
| `assessments.csv` | avaliações, tipo, data (dia relativo), peso | delimita os ciclos |
| `studentAssessment.csv` | submissões e notas | filtra ciclos válidos |
| `studentInfo.csv` | demografia e `final_result` | análise perfil × resultado |
| `courses.csv`, `studentRegistration.csv`, `vle.csv` | apoio | exploração inicial |

### 2.2 Recorte: por que o módulo FFF, apresentação 2014J

Para não misturar calendários de avaliação distintos (os picos de esforço só são interpretáveis contra as datas de prova do próprio curso), a análise foi restrita a **uma única apresentação de um único módulo**. A escolha foi guiada pelos dados:

- FFF é o segundo maior módulo em matrículas (7.762; BBB tem 7.909, empate técnico) e **FFF 2014J é sua maior turma: 2.365 alunos matriculados**, dos quais 2.121 com atividade registrada no VLE;
- FFF 2014J tem **5 TMAs com peso** (Tutor-Marked Assessments) em calendário bem espaçado — dias 24, 52, 94, 136 e 199 — com pesos 12,5/12,5/25/25/25, além do exame final (~dia 241). Os CMAs têm peso zero e foram ignorados;
- o menor intervalo entre TMAs é de **28 dias**, o que baliza a granularidade temporal viável.

### 2.3 Estatísticas do recorte

- 167.164 pares aluno × dia com atividade;
- 7.118 ciclos válidos (aluno × TMA) após os filtros da Seção 3, oriundos de 1.806 alunos;
- distribuição de `final_result` em FFF 2014J: Pass, Distinction, Fail e Withdrawn (evasão), com evasão concentrada nos alunos de poucos ciclos válidos (ver Seção 6.4).

### 2.4 Questões de qualidade

1. **Evasão trunca as curvas.** Alunos `Withdrawn` param de clicar antes do fim; na visão por curso inteiro isso se confunde com um perfil "decrescente" espúrio. O desenho por ciclo *com exigência de submissão do TMA* elimina o problema na origem: quem desistiu contribui apenas com os ciclos que de fato viveu.
2. **Cliques antes do início do curso** (`date < 0`) existem e foram excluídos — não fazem parte do ritmo durante o curso.
3. **Cauda pesada**: dias com centenas de cliques distorceriam as proporções; tratado com `log1p` (Seção 3.3).
4. **Ciclos quase vazios**: proporções calculadas sobre meia dúzia de cliques são ruído; tratado com filtro de atividade mínima.

---

## 3. Preparação

### 3.1 Seleção

- Somente FFF 2014J; somente os 5 TMAs com peso;
- **Ciclo de avaliação** = intervalo entre o deadline anterior e o atual: (0→24], (24→52], (52→94], (94→136], (136→199] — durações de 24 a 63 dias;
- Um ciclo de um aluno só entra na base se o aluno **submeteu o TMA daquele ciclo** (cruzamento com `studentAssessment.csv`).

### 3.2 Limpeza

- Exclusão de cliques com `date < 0`;
- Filtro de atividade mínima por ciclo: **≥ 5 cliques e ≥ 2 dias ativos** (80 ciclos descartados);
- Agregação prévia: soma de cliques por aluno × dia (reduz 10,65 M linhas do VLE para 167 mil pares no recorte; cache local para reprodutibilidade).

### 3.3 Atributos derivados (features)

Cada ciclo vira um vetor de dimensão fixa em três passos:

1. **Tempo relativo**: como os ciclos têm durações diferentes, o tempo de cada um é normalizado para [0, 1] e dividido em **B bins** (configuração principal B = 4: *início, meio 1, meio 2, véspera*; sensibilidade B = 8);
2. **Amortecimento da cauda**: os cliques diários passam por `log1p` antes de somar por bin;
3. **Normalização de forma**: o vetor é dividido pela própria soma → proporções que somam 1. O volume sai do vetor (fica guardado como covariável); **só a forma entra na clusterização**.

Resultado: matriz **7.118 × 4** (ou 7.118 × 8), não anotada, contínua, uma linha por ciclo.

---

## 4. Design de Experimentos

### 4.1 Unidade de análise e divisão dos dados

A unidade clusterizada é o **ciclo** (aluno × TMA), não o aluno: cada aluno contribui com até 5 observações. O perfil do aluno é derivado *depois*, pela **consistência dos rótulos** de seus ciclos (regra: mesmo rótulo em ≥ 60% de ≥ 3 ciclos; caso contrário "Misto"; menos de 3 ciclos válidos → "Insuficiente"). Isso responde empiricamente à pergunta "o aluno repete o comportamento?" em vez de assumi-la — e evita o artefato de médias (a média de um aluno errático seria uma curva chapada, falsamente "equilibrada").

Por ser aprendizado não supervisionado, não há divisão treino/teste; a validação é interna (métricas abaixo) e por **estabilidade entre métodos** (triangulação). `final_result` nunca entra na clusterização — é usado apenas a posteriori, como validação externa de utilidade.

### 4.2 Métricas

- **Cotovelo**: inércia/WCSS (K-means) ou custo total ao medoide (K-medoids);
- **Silhueta média e por amostra** (Rousseeuw, 1987) — sempre calculada **na própria métrica** do experimento;
- **Correlação cofenética** (hierárquico);
- **ARI** (Adjusted Rand Index) entre soluções, para medir concordância entre métodos/métricas;
- **Balanceamento**: partições com grupo menor que 5% da base (< 356 ciclos) são marcadas **degeneradas** — critério essencial, pois silhueta alta com grupos de 1–13 pontos é ilusória (Seção 6.5);
- Reprodutibilidade: `random_state = 42`, `n_init = 50` (K-means) / 5 inicializações k-means++ (PAM).

### 4.3 Grade de experimentos

| Fator | Valores |
|---|---|
| Método | K-means, K-means divisivo (2→2), K-medoids (PAM), hierárquico |
| k | 2 a 8 (varredura); 3 (confirmatório) |
| Atributos | 4 bins e 8 bins por ciclo |
| Distância ponto-a-ponto | euclidiana, Manhattan, cosseno, Chebyshev |
| Distância entre grupos (linkage) | Ward, completo, médio, simples (+ Ward-corda para cosseno) |

### 4.4 O protocolo de escolha de k (como se chegou ao k = 3)

O número de grupos foi tratado com um protocolo em três camadas, para conciliar honestidade matemática com a hipótese de pesquisa:

1. **Melhor k matemático**: em toda varredura, o k de maior silhueta média é reportado e recebe pasta própria de resultados (`melhor_k_*`);
2. **Divisão recursiva (divisiva)**: se o melhor k é 2, aplica-se o mesmo protocolo *dentro* do maior agrupamento — se o melhor sub-k também for 2, a aplicação recursiva do critério "melhor k" termina em 3 grupos sem nunca forçá-los;
3. **k = 3 confirmatório**: solução com 3 centróides fixados, justificada pela H1 (três perfis hipotetizados) e validada pelo dendrograma do Ward e pela convergência entre todas as rotas.

---

## 5. Modelagem / Experimentação

### 5.1 K-means — varredura de k (Questão 2)

Configuração principal (4 bins, euclidiana). Resultados da varredura:

| k | Inércia | Silhueta média |
|---|---|---|
| **2** | 385,0 | **0,401** |
| 3 | 312,3 | 0,246 |
| 4 | 266,3 | 0,239 |
| 5 | 234,5 | 0,240 |
| 6 | 210,0 | 0,248 |
| 7 | 191,5 | 0,250 |
| 8 | 176,7 | 0,240 |

*Figuras: `../05_kmeans_curvas_ciclo/4bins/selecao_k/cotovelo.png`, `silhueta_media.png` e `silhueta_facas_todos_k.png` (diagrama de facas para **cada** k).*

**Leitura.** O melhor k matemático é **2** (silhueta 0,401): a estrutura mais forte dos dados separa os ciclos de preparação tardia extrema de todo o resto. O cotovelo tem dobra suave em 3–4. A grade de facas mostra que k = 2 e k = 3 são as últimas partições "limpas" — de k = 4 em diante surgem grupos pequenos e caudas negativas crescentes.

**k = 2 (melhor matemático)**: Cluster "tardio extremo" com 63% do esforço na véspera (n = 1.545 ciclos) vs. resto (n = 5.573). *(`../05_kmeans_curvas_ciclo/4bins/melhor_k_2/`)*

**k = 3 (confirmatório)** — os três perfis hipotetizados emergem com clareza *(`../05_kmeans_curvas_ciclo/4bins/k3/centroides.png`)*:

| Perfil | Curva média (início / meio 1 / meio 2 / véspera) | n ciclos |
|---|---|---|
| Adiantado | **0,377** / 0,233 / 0,139 / 0,251 | 2.360 |
| Equilibrado | 0,188 / 0,246 / 0,264 / 0,302 | 3.475 |
| Tardio | 0,131 / 0,104 / 0,101 / **0,664** | 1.283 |

### 5.2 K-means divisivo: dividindo o k duas vezes

Aplicando a varredura *dentro* do cluster "resto" do k = 2: **o melhor sub-k é novamente 2** (silhueta 0,250), e os dois subgrupos são exatamente Adiantado (n = 1.959) e Equilibrado (n = 3.614). A solução combinada (2→2) tem silhueta global **0,2486, ligeiramente superior ao k = 3 direto (0,2458)**, concordância de 91,1% com ele (ARI 0,74) e preserva 100% do cluster Tardio. *(`../05_kmeans_curvas_ciclo/4bins/divisivo/`)*

**Este é o resultado central do protocolo de k:** respeitando sempre o melhor k matemático e reaplicando o critério recursivamente, chega-se a três grupos sem fixá-los a priori. O k = 3 não é imposto — emerge.

### 5.3 Variação do número de atributos: 4 vs. 8 bins

Repetindo todo o pipeline com 8 bins (`../05_kmeans_curvas_ciclo/8bins/`):

| Configuração | Silhueta k=2 | Silhueta k=3 |
|---|---|---|
| 4 bins | 0,401 | 0,246 |
| 8 bins | 0,355 | 0,139 |

As curvas ficam visualmente mais informativas (o Tardio "hiberna" até 75% do ciclo e explode na última fatia), mas **a qualidade da partição cai** — mais dimensões, mais ruído por bin, grupos mais difusos, e a fronteira Adiantado↔Equilibrado fica instável. Conclusão de projeto: **4 bins para clusterizar, 8 bins para descrever**.

### 5.4 Variação da medida de distância: K-medoids (PAM) — Questão 2

O K-means clássico é euclidiano por construção (o centróide-média só minimiza soma de quadrados euclidiana). Para variar a métrica corretamente foi implementado do zero um **K-medoids (PAM)** — inicialização k-means++, alternância atribuição/atualização de medoides sobre matriz de distâncias pré-computada — e aplicado com 4 métricas, replicando o desenho completo (varredura com facas por k, k = 3, divisivo 2→2) para cada uma. *(`../07_kmedoids_distancias/4bins/`, um subdiretório por métrica)*

| Métrica | Melhor k | Sil. melhor k | Sil. k=3 | Melhor sub-k (divisivo) | Grupos k=3 | ARI k=3 vs baseline |
|---|---|---|---|---|---|---|
| Euclidiana | 2 | 0,397 | 0,230 | 2 | 3.040/2.633/1.445 | 0,59 |
| Manhattan | 2 | 0,362 | 0,227 | 2 | 3.229/2.394/1.495 | 0,51 |
| **Cosseno** | 2 | **0,477** | **0,388** | 2 | 2.880/2.606/1.632 | 0,58 |
| Chebyshev | 2 | 0,366 | 0,242 | 2 | 3.482/2.287/1.349 | 0,63 |

*(baseline = K-means euclidiano k=3; silhuetas calculadas na própria métrica; figuras-síntese: `comparativo_curvas_k3.png` e `comparativo_cotovelo_silhueta.png`)*

**Leituras.** (i) O padrão estrutural é invariante à métrica: melhor k = 2, sub-k = 2, e as curvas médias dos 3 grupos são quase indistinguíveis entre as quatro métricas — os perfis **não são artefato da distância escolhida**. (ii) O **cosseno domina** (0,388 no k = 3, melhor valor de toda a grade): como as curvas são vetores de proporção (forma pura), a similaridade angular é a métrica "nativa" do problema. (iii) Em 8 bins (`../07_kmedoids_distancias/8bins/`) tudo se repete em versão mais ruidosa: cosseno 0,243 > Chebyshev 0,161 > euclidiana 0,152 > Manhattan 0,131, com o Tardio sobrevivendo nítido em todas.

### 5.5 Agrupamento hierárquico — variação de linkage (Questão 3)

Sobre a mesma matriz 7.118 × 4, quatro linkages, cortes k = 2..8. *(`../06_hierarquico_curvas/4bins/`: dendrogramas com corte em 3 tracejado, silhueta por linkage, curvas do Ward k=3)*

| Linkage | Cofenética | Silhueta k=3 | Menor grupo (k=3) | Diagnóstico |
|---|---|---|---|---|
| **Ward** | 0,598 | **0,262** | **1.226** | grupos balanceados e interpretáveis |
| Médio | 0,767 | 0,439 | 13 | isola micro-grupos de outliers |
| Simples | 0,606 | 0,503 | 3 | encadeamento: 7.110 num grupo só |
| Completo | 0,328 | — | — | não produz corte com 2–3 grupos (empates) |

**Ward k = 3** recupera os mesmos três perfis — Tardio [0,114/0,104/0,120/**0,661**] n = 1.254; Adiantado [**0,441**/0,198/0,147/0,214] n = 1.226; Equilibrado n = 4.638 — com silhueta **0,2615, superior ao K-means (0,2458)**, concordância de 77,7% (ARI 0,43; divergência concentrada na fronteira Adiantado↔Equilibrado; o Tardio é preservado quase integralmente). Em 8 bins: Ward 0,169 > K-means 0,139, mesmo padrão.

### 5.6 Hierárquico × métrica de distância (enriquecimento)

Como Ward exige geometria euclidiana, para o cosseno usou-se o recurso clássico do **Ward-corda** (Ward sobre os vetores normalizados em L2; a distância de corda é monotônica ao cosseno). *(`../08_hierarquico_distancias/4bins/`)*

Resultado (4 bins): entre 13 combinações métrica × linkage, **apenas duas produzem 3 grupos utilizáveis** (menor grupo ≥ 5%): **Ward euclidiano (silhueta 0,262)** e **Ward-corda no cosseno (0,291)** — o cosseno vence de novo, repetindo o ranking do K-medoids. Manhattan e Chebyshev não têm nenhum linkage válido: médio/simples isolam 1–137 outliers (com silhuetas ilusórias de 0,4–0,5) e o completo frequentemente nem divide. Em 8 bins, idem: só Ward (0,169) e Ward-corda (0,175).

---

## 6. Avaliação

### 6.1 Análise comparativa consolidada (k = 3, 4 bins)

| Rota | Silhueta k=3 | Encontra os 3 perfis? |
|---|---|---|
| K-medoids cosseno | **0,388** | sim |
| Ward-corda (cosseno) | 0,291 | sim |
| Ward euclidiano | 0,262 | sim |
| K-means divisivo 2→2 | 0,249 | sim |
| K-means direto | 0,246 | sim |
| K-medoids Chebyshev / euclidiano / Manhattan | 0,242 / 0,230 / 0,227 | sim |

**Melhor particionamento (decisão final):** os **3 perfis (Adiantado / Equilibrado / Tardio) em curvas de 4 bins**, tendo como referência operacional o K-means euclidiano k = 3 (compatível com Ward e com o baseline preditivo futuro) e como melhor separação absoluta o **K-medoids com cosseno**. A decisão se justifica por quatro evidências independentes: (i) o divisivo 2→2 chega a 3 grupos aplicando sempre o melhor k; (ii) o dendrograma do Ward sustenta o corte em 3; (iii) as curvas são estáveis sob 4 métricas × 2 famílias de métodos × 2 resoluções (ARIs 0,43–0,74 entre rotas, sempre com o Tardio preservado); (iv) k = 3 é a última partição não fragmentada na grade de facas.

### 6.2 Por que não simplesmente k = 2?

O k = 2 tem a maior silhueta em todas as rotas (0,36–0,48) e é reportado como melhor k matemático. Ele captura a oposição mais forte — "tardio extremo vs. resto" — mas colapsa dois comportamentos pedagogicamente distintos (antecipar vs. distribuir) num único grupo. A segunda divisão é estatisticamente legítima (é o melhor k *dentro* do resto, em todas as métricas) e é ela que produz a tipologia útil.

### 6.3 Estabilidade dos perfis entre métodos

Concordâncias no k = 3 (4 bins): divisivo vs. direto **91%** (ARI 0,74); Ward vs. K-means 78% (ARI 0,43); K-medoids (4 métricas) vs. baseline ARI 0,51–0,63; Ward-corda vs. baseline 0,47. Em todas, as divergências são deslizamentos da fronteira Adiantado↔Equilibrado; o Tardio é o perfil mais robusto da tipologia.

### 6.4 Validação externa: perfil × resultado final

Consolidando ciclos em perfis de aluno (regra de consistência da Seção 4.1) e cruzando com `final_result` — que nunca participou da clusterização *(`../05_kmeans_curvas_ciclo/4bins/k3/resultado_por_perfil.png` e `perfil_x_resultado.csv`)*:

| Perfil do aluno (4 bins) | n | Aprovação (Pass+Distinction) | Fail | Withdrawn |
|---|---|---|---|---|
| Adiantado | 285 | **86%** | 11% | 4% |
| Equilibrado | 657 | 85% | 10% | 4% |
| Misto | 277 | 72% | 21% | 7% |
| Tardio | 156 | **70%** | 24% | 6% |
| Insuficiente (< 3 ciclos) | 431 | ~0,5% | 33% | **67%** |

Três leituras: (i) o gradiente é monotônico — quanto mais cedo o esforço no ciclo, melhor o desfecho; o Tardio consistente reprova mais que o dobro do Adiantado; (ii) na resolução de 8 bins o contraste é ainda maior (Adiantado 87% com 25% de distinção vs. Tardio 60% com 34% de Fail); (iii) a categoria "Insuficiente" concentra quase toda a evasão — a estratégia de exigir submissão do TMA isolou os desistentes numa categoria própria em vez de deixá-los contaminar os perfis, exatamente como planejado. Esses resultados dão suporte preliminar às hipóteses H1 e H2 do projeto.

### 6.5 Lições metodológicas (anti-armadilhas)

1. **Silhueta sem inspeção de tamanhos engana**: linkages médio/simples atingem 0,4–0,5 isolando 1–137 pontos — partições formalmente "melhores" e praticamente inúteis. Todo ranking deste trabalho exigiu grupos ≥ 5% da base.
2. **K-means com distância trocada não é K-means**: a variação de métrica exige K-medoids (ou normalizações com garantia, como o Ward-corda).
3. **Mais atributos ≠ melhor**: dobrar os bins suavizou as curvas e piorou todas as silhuetas.
4. **Agregação esconde forma**: a decisão de modelar por ciclo (e não por curso) foi a que tornou os perfis detectáveis.

---

## 7. Conclusão

Este trabalho construiu, a partir do OULAD (FFF 2014J), uma base não anotada de 7.118 curvas de esforço por ciclo de avaliação e a submeteu a uma grade de 16 configurações de clusterização (métodos × métricas × resoluções × linkages). As conclusões:

1. **Existem três perfis temporais de preparação** — Adiantado, Equilibrado e Tardio — detectáveis de forma estável por K-means, K-means divisivo, K-medoids em quatro métricas e agrupamento hierárquico Ward, em duas resoluções de atributos (H1 do projeto suportada por triangulação);
2. **O melhor k matemático é sempre 2** (tardio extremo vs. resto), e a aplicação recursiva do mesmo critério (2→2) faz os 3 perfis emergirem sem imposição — protocolo que concilia a exigência de "respeitar o melhor k" com a hipótese de pesquisa;
3. **A similaridade de cosseno é a melhor métrica para curvas de forma** (silhueta 0,388, melhor valor da grade), coerente com a natureza dos dados (vetores de proporção);
4. **Ward é o único linkage utilizável** nesses dados; os demais produzem partições degeneradas com silhuetas enganosamente altas;
5. Os perfis têm **validade externa**: a taxa de aprovação cai monotonicamente do Adiantado (86%) ao Tardio (70%), com a evasão isolada na categoria de dados insuficientes — indício favorável à H2, a ser formalizado.

**Limitações**: um único módulo/apresentação (generalização não testada); associação perfil × resultado é correlacional, não causal (confundidores possíveis: habilidade prévia, carga de trabalho); a fronteira Adiantado↔Equilibrado é sensível ao método (±300–1.100 ciclos); o limiar de consistência (60% em ≥ 3 ciclos) é uma escolha de projeto.

**Trabalhos futuros**: testar o ganho preditivo do perfil sobre volume + nota intermediária (H3, regressão logística/AUC); atribuição suave via GMM; replicação nos demais módulos do OULAD; uso do calendário real de TMAs como bins pedagógicos; recomendação de ritmo baseada no perfil historicamente vencedor (H4).

---

## 8. Referências

- Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171. (OULAD, CC-BY 4.0 — analyse.kmi.open.ac.uk/open_dataset)
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proc. 5th Berkeley Symposium*, 281–297.
- Kaufman, L., & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis*. Wiley. (PAM/K-medoids)
- Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.
- Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *JASA*, 58(301), 236–244.
- Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2, 193–218. (ARI)
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
- Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272.

---

## Apêndice A — Mapa de resultados (pastas e figuras)

| Pasta | Conteúdo |
|---|---|
| `../05_kmeans_curvas_ciclo/4bins/` | K-means 4 bins: `selecao_k/` (cotovelo, silhueta média, facas por k), `melhor_k_2/`, `k3/` (centróides, facas+PCA, resultado por perfil), `divisivo/` |
| `../05_kmeans_curvas_ciclo/8bins/` | réplica em 8 bins |
| `../06_hierarquico_curvas/4bins/` e `8bins/` | 4 linkages: dendrogramas, silhueta por linkage, Ward k=3, comparação com K-means |
| `../07_kmedoids_distancias/4bins/` e `8bins/` | K-medoids × 4 métricas, cada uma com `selecao_k/`, `k3/`, `divisivo/`; comparativos na raiz |
| `../08_hierarquico_distancias/4bins/` e `8bins/` | hierárquico × métrica; comparativo de partições utilizáveis |

Todos os experimentos são reproduzíveis com os scripts Python dentro de cada pasta numerada, executados da raiz do projeto (semente fixa 42; cache `dataset/cache_vle_fff2014j.csv` acelera reexecuções).
