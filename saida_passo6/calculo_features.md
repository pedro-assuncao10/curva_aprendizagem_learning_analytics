Memorial Descritivo: Engenharia de Features Ancoradas em Avaliações

Projeto: Curva Temporal de Engajamento — Identificação de Perfis de Estudo (OULAD)
Disciplina: IA Aplicada à Educação | UFMA

1. O Problema da Dimensionalidade Arbitrária

Em passos anteriores do projeto, a trajetória de engajamento do aluno foi representada fatiando-se a duração total do curso em 30 janelas fixas de tempo (ex: w00 a w29).

Embora essa abordagem tenha padronizado o tempo, ela introduziu ruído ao algoritmo de clusterização (Curse of Dimensionality). O K-Means precisava lidar com 30 dimensões onde picos de interação ocorriam em momentos diferentes para cada curso, simplesmente porque as datas de provas não coincidiam nas semanas arbitrárias.

Para melhorar a clareza e o significado pedagógico (semântica) do agrupamento, a análise temporal foi refatorada. Em vez de perguntar "O que o aluno faz na semana 15?", o pipeline passou a extrair métricas sobre "Como o aluno se comporta nas vésperas de qualquer prova oficial?".

2. A Janela de Preparação (W)

Todas as features descritas abaixo partem de uma premissa comum: o conceito de "Pré-Estudo".

O script localiza no dataset oficial (assessments.csv) a data de entrega de todas as avaliações do curso em que o aluno está matriculado.

Define-se uma janela de antecedência constante, W = 7 dias.

Varre-se o histórico de logs do aluno (studentVle.csv). Se um clique ocorre dentro dos 7 dias que antecedem uma prova, ele é computado como "preparação para a Prova X". Qualquer clique fora dessas janelas (ex: logo após a prova) é descartado para o cálculo do perfil de estudo.

Com isso, o aluno passa a ser representado por um Vetor de Preparação (ex: num curso com 4 provas, o vetor [50, 20, 0, 5] indica a quantidade de cliques que o aluno deu nos 7 dias que antecederam cada uma das provas, respectivamente).

A partir deste vetor, derivam-se as 5 features imunes ao tamanho do curso, utilizadas no K-Means.

3. As 5 Features Derivadas (Matemática e Lógica)

Para evitar que o "Volume Bruto" influencie a clusterização, os cliques de cada prova no vetor são divididos pela soma total de cliques dados em todas as janelas de preparação, transformando o vetor bruto em um Vetor de Proporção (prop).

A) Durabilidade (durabilidade)

Lógica Pedagógica: O aluno se prepara para todas as avaliações do curso ou estuda apenas para a primeira prova e depois desaparece?

Fórmula Matemática: Conta quantas provas no vetor do aluno possuem valor maior que zero, e divide pelo total de provas oferecidas no curso.

Exemplo: Num curso de 4 provas, se o vetor é [50, 20, 0, 5], o aluno se preparou para 3 delas. Durabilidade = 3/4 = 0.75 (ou 75%).

Escala: Varia de 0.0 (abandonou totalmente) a 1.0 (focou em todas as provas).

B) Tendência (tendencia)

Lógica Pedagógica: O ritmo de preparação do aluno está acelerando, se mantendo constante ou perdendo fôlego à medida que o semestre avança?

Fórmula Matemática: O algoritmo aplica uma Regressão Linear Simples (np.polyfit(ordem_da_prova, prop, 1)[0]) sobre o vetor de proporção. O valor extraído é o coeficiente angular (a inclinação da reta).

Exemplo: Se as proporções são [0.4, 0.3, 0.2, 0.1], a linha está caindo fortemente, gerando um valor negativo.

Escala: Valores negativos indicam perda de ritmo (desengajamento progressivo); valores positivos indicam aumento de esforço na reta final do curso. Valores próximos a zero indicam constância.

C) Concentração (concentracao)

Lógica Pedagógica: O aluno distribui seu esforço de forma equilibrada entre as avaliações, ou dá um "tiro curto" investindo toda sua energia em uma única prova?

Fórmula Matemática: É o Desvio Padrão populacional (prop.std()) dos valores do vetor de proporção.

Exemplo: Se o vetor de proporções for perfeitamente distribuído [0.25, 0.25, 0.25, 0.25], a variação é nula, logo a concentração é 0.0. Se ele joga tudo na primeira prova [0.90, 0.10, 0.0, 0.0], o desvio padrão explode, resultando num valor alto.

Escala: Quanto maior o valor, mais "empilhado" e errático é o padrão de estudo do aluno.

D) Cobertura Final (cobertura_final)

Lógica Pedagógica: O aluno sobreviveu academicamente até o fim do curso? Ele ainda possuía "combustível" de preparação para a última avaliação do semestre?

Fórmula Matemática: Extrai o valor exato da última posição do vetor de proporção (prop[-1]).

Exemplo: Se o vetor do aluno for [0.70, 0.20, 0.10, 0.00], a cobertura final dele é 0.0 (zero).

Escala: Varia de 0.0 a 1.0. Valores baixos são um forte preditor de reprovação ou "abandono funcional".

E) Antecedência (antecedencia)

Lógica Pedagógica: Dentro daquela janela de 7 dias antes da prova, o aluno distribui os cliques organizadamente (ex: estuda um pouco todo dia) ou ele acessa todo o material de forma desesperada na véspera (dia -1)?

Fórmula Matemática: É calculada como uma média ponderada dos dias de antecedência. Para cada clique, o script multiplica 1 pelo número de dias restantes para a prova (leadw). A soma total desses "cliques ponderados" é dividida pelo total de cliques brutos e ajustada para uma escala normalizada.

Escala: Varia de 0.0 (perfil "deixa tudo para a véspera / desespero") a 1.0 (perfil "preparação precoce e constante").

4. Conclusão Metodológica

A adoção destas cinco features semânticas elevou significativamente a precisão do agrupamento não-supervisionado. O coeficiente de Silhueta Média saltou de patamares inadequados (< 0.15) no cenário de 30 dimensões espaciais brutas para 0.304 no K-Means ancorado em avaliações (K=3), garantindo grupos comportamentais coesos e perfeitamente interpretáveis no contexto pedagógico.