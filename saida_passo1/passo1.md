Memorial Descritivo: Engenharia de Features (Passo 1)

Projeto: Curva Temporal de Engajamento — Identificação de Perfis de Estudo (OULAD)
Disciplina: IA Aplicada à Educação | UFMA

1. O Problema e o Objetivo da Engenharia de Features

Os dados brutos de interação em ambientes virtuais de aprendizagem (como a tabela studentVle.csv do OULAD) são puramente transacionais. Eles funcionam como um grande "extrato bancário", registrando eventos isolados ("O aluno X, no dia Y, deu Z cliques").

Nesse formato bruto, um algoritmo de Machine Learning não consegue enxergar um "padrão de comportamento" ou uma "rotina de estudos". O objetivo deste Passo 1 foi traduzir eventos isolados em representações matemáticas de comportamento, permitindo testar a Hipótese 1 (H1) — a premissa de que a "forma" temporal do engajamento importa mais que o mero volume total.

Para isso, o pipeline derivou diferentes matrizes e tabelas.

2. A Régua de Tempo Padronizada (Os Buckets)

Cursos diferentes têm durações em dias distintas, e alunos entram e saem em momentos diferentes. Para comparar o comportamento de todos, criamos uma régua de tempo padronizada fatiando a duração total de cada curso em 30 janelas de tempo iguais (variável N_BUCKETS = 30).

Cada janela foi rotulada como w (de window ou week), indo de w00 (início do curso) a w29 (final do curso). Com isso, podemos comparar objetivamente o "meio do curso" (w15) de todos os estudantes, independentemente da disciplina que cursam.

3. Matrizes Derivadas: Volume vs. Forma

A partir da régua padronizada, dividimos o conceito de engajamento em duas matrizes distintas:

A) Matriz de Volume (matriz_volume.csv)

O que contém: As colunas vão de vol_w00 a vol_w29.

O que significa: Representa a quantidade absoluta de cliques do aluno naquela janela de tempo.

Exemplo: Se o Aluno João tem vol_w01 = 150, ele realizou 150 interações naquela fatia de tempo.

Objetivo no projeto: Servir como baseline. Sabendo que "quem clica mais tende a passar mais", usaremos o volume para provar (na Hipótese H3) que a "Forma" adiciona um ganho preditivo além do óbvio.

B) Matriz de Forma (matriz_forma.csv) — O Coração do Modelo

O que contém: As colunas vão de forma_w00 a forma_w29.

O que significa: Representa a proporção (percentual) do esforço total daquele aluno gasto em cada janela de tempo.

Como é calculada: O total de cliques da janela é dividido pelo total de cliques do aluno no curso todo.

O Insight: Se o Aluno João deu 1.500 cliques no total, e 150 no w01, ele tem forma_w01 = 0.10 (gastou 10% da sua energia ali). Se o Aluno Pedro deu apenas 100 cliques no total, mas 10 no w01, ele também tem forma_w01 = 0.10.

Objetivo no projeto: Isolar a rotina do estudante. Ao neutralizar a intensidade absoluta, o algoritmo de clusterização (K-Means) poderá agrupar o Aluno João e o Aluno Pedro no mesmo cluster de comportamento, validando a Hipótese 1 (existência de perfis baseados no tempo).

4. Variáveis Interpretáveis (features_interpretaveis.csv)

Algoritmos de clusterização retornam resultados matematicamente precisos, mas abstratos (ex: "Cluster A vs. Cluster B"). Para dar semântica humana a esses grupos (necessário para H2 e H4), derivamos variáveis que "resumem" a curva de forma de cada aluno:

pico_posicao: Identifica em qual janela (w00 a w29) o aluno concentrou sua maior proporção de esforço. (Ajuda a identificar perfis procrastinadores).

centro_massa: Calcula se o peso médio do engajamento está concentrado no início (valores próximos a 0) ou no final do curso (valores próximos a 1).

semanas_mortas: Conta quantas janelas de tempo o aluno passou com exatamente zero interações (indicador de inconstância).

inclinacao: Uma regressão linear simples sobre a curva. Uma inclinação positiva significa que o aluno aumentou o ritmo ao longo do curso; negativa, que ele "esfriou" com o tempo.

5. Matriz Final de Clusterização (matriz_clusterizacao.csv)

Esta é a união da matriz de Forma com as Features Interpretáveis, consolidada em um único arquivo limpo, sem valores ausentes.

Decisões Críticas de Engenharia:

Remoção de Evasão: Os alunos com resultado "Withdrawn" (evadidos) foram filtrados desta matriz. Se mantidos, suas interrupções abruptas confundiriam o algoritmo, criando falsos clusters de "perfil decrescente".

Exclusão da Variável-Alvo: A coluna final_result (se passou ou reprovou) foi omitida intencionalmente. Isso evita vazamento de dados, forçando o algoritmo a agrupar os alunos exclusivamente por seu comportamento de estudo, permitindo que a associação com o desempenho (H2) seja testada posteriormente.