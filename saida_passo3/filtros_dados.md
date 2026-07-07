Memorial Descritivo: Critérios de Inclusão, Exclusão e Filtragem de Dados

Projeto: Curva Temporal de Engajamento — OULAD
Disciplina: IA Aplicada à Educação | UFMA

1. O Problema e a Base Original

O Open University Learning Analytics Dataset (OULAD) contém registros detalhados de interações de estudantes em ambientes virtuais de aprendizagem. A tabela principal de informações dos alunos (studentInfo.csv), antes de qualquer tratamento, contava com 32.593 registros (aluno × curso).

No entanto, o objetivo desta pesquisa não é analisar "tudo o que acontece" na plataforma, mas sim aplicar Machine Learning (Clusterização) para descobrir perfis e rotinas de estudo.

Para que os algoritmos de agrupamento consigam identificar trajetórias geométricas (a "forma" do engajamento ao longo do curso) sem serem enganados por ruídos estatísticos, foi necessário aplicar três filtros metodológicos rigorosos. Esses filtros reduziram a base para um conjunto final robusto de aproximadamente 21.318 alunos.

Este documento explica os critérios técnicos e metodológicos para a remoção desses ~11.000 registros.

2. Os Três Filtros Aplicados

Filtro 1: Exclusão de Evadidos (Withdrawals)

Critério: Foram removidos todos os alunos cujo status na variável-alvo (final_result) era "Withdrawn" (Desistente).

Justificativa Metodológica: A premissa central do projeto é investigar o comportamento ao longo de todo o curso. Um aluno que tranca ou evade na metade da disciplina não apresenta um "perfil de engajamento decrescente" autêntico; sua trajetória foi interrompida de forma administrativa. Se mantidos na base, algoritmos como o K-Means criariam, erroneamente, um cluster de alunos que "desaparecem no meio do semestre", confundindo um abandono formal com uma rotina de estudos irregular.

Impacto: Este grupo representava cerca de 30% da base original (aproximadamente 10.156 registros), sendo a maior redução do dataset.

Filtro 2: Exclusão de Alunos "Fantasmas" (Sem Registro VLE)

Critério: Foram descartados os alunos presentes na tabela de matrículas (studentInfo), mas que não possuíam nenhuma linha de interação na tabela de logs brutos do sistema (studentVle).

Justificativa Metodológica: Para o cálculo das features de "Volume" e "Forma", é matematicamente impossível derivar curvas ou proporções temporais de um estudante sem nenhum clique. Alunos fantasmas não geram dados de comportamento.

Filtro 3: Filtro de Esparsidade (Remoção de Ruído Extremo)

Critério: Foram retidos apenas os alunos que atenderam a dois pisos mínimos de atividade:

MIN_CLIQUES = 30 (Pelo menos 30 cliques totais no curso inteiro).

MIN_SEMANAS_ATIVAS = 3 (Interações registradas em pelo menos 3 fatias temporais distintas).

Justificativa Estatística: O modelo de "Forma" fatia a duração do curso em 30 janelas de tempo iguais. Se um aluno interagiu apenas duas vezes (por exemplo, baixando a ementa no primeiro dia e acessando uma nota no último dia), a representação gráfica de seu esforço será um pico isolado sem conexão com uma rotina. Esse tipo de comportamento esparso introduz variância artificial (outliers de proporção) que desestabiliza o centroide do K-Means. A clusterização precisa de uma "curva" mínima para ser desenhada.

Impacto: Removeu alunos que formalmente reprovaram por falta de comparecimento digital, mas que não configuram um padrão comportamental passível de análise pela máquina.

3. Resumo da Base Final Validada

Após as etapas descritas, a base de dados resultante (representada nos arquivos matriz_clusterizacao.csv e matriz_forma.csv) possui as seguintes características:

Total Validado: ~21.318 alunos.

Qualidade dos Dados: Todos os estudantes analisados chegaram até o final da disciplina (foram Aprovados, Reprovados ou Aprovados com Distinção).

Aptidão para Machine Learning: Todos possuem densidade de dados suficiente (volume e frequência) para que suas trajetórias sejam comparadas matematicamente pelo algoritmo K-Means, garantindo que o agrupamento seja feito puramente pela semântica de suas rotinas de estudo.