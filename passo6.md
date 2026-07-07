# Passo 6 — Features ancoradas nas avaliações (opção A)

## A mudança de ideia
Até aqui a curva era cortada em 30 fatias **arbitrárias** do curso. Agora ela é ancorada nos **marcos reais**: as avaliações. O comportamento do aluno se organiza em torno das provas (estuda antes, relaxa depois), então medir em torno delas é mais fiel — e era a hipótese original da ficha ("concentra nas vésperas de avaliação").

## O problema dos cursos diferentes (e a solução)
Cada curso tem um número diferente de provas (5, 7, ...). Se cada aluno virasse uma lista do tamanho do nº de provas do seu curso, alunos de cursos diferentes teriam vetores de tamanhos diferentes e o K-means não poderia compará-los.

**Solução (opção A):** resumir a "lista de provas" em **indicadores de tamanho fixo**, iguais para todo aluno, independentes de quantas provas o curso teve.

## Como cada feature é construída
Para cada avaliação com data conhecida (`assessments.csv`), definimos a janela dos **7 dias antes** dela e somamos os cliques do aluno nessa janela = "pré-estudo" daquela prova. Disso saem:

- **tendencia** — a inclinação do pré-estudo ao longo das provas. Positiva = se prepara cada vez mais; negativa = vai largando.
- **antecedencia** — quando, dentro da janela, ele clica. Perto de 1 = estuda dias antes; perto de 0 = só na véspera.
- **durabilidade** — em que fração das provas ele teve algum pré-estudo. Baixa = parou de estudar para as provas em algum momento.
- **concentracao** — se o pré-estudo é espalhado pelas provas ou empilhado numa só.
- **cobertura_final** — quanto do pré-estudo caiu na última prova. Captura quem engajou até o fim vs. quem abandonou.

O **volume** (intensidade média de cliques por prova) é calculado mas **não entra na clusterização** — fica só para validar, como nos passos anteriores. Os **desistentes (Withdrawn)** continuam de fora (a base vem filtrada do Passo 4.2).

## Por que só 5 features (e não 34)
Cada uma mede uma propriedade **diferente** do comportamento — sem a redundância das 34 anteriores (onde p_terço e semanas_mortas eram deriváveis da forma). Menos dimensões e sem redundância tende a dar distâncias mais informativas e silhueta menos achatada. E o centro de cada cluster vira legível: "tendência negativa + durabilidade baixa = aluno que vai largando".

## Arquivos gerados
- **02_perfis_preparacao.png** — a curva média de pré-estudo ao longo das provas, por cluster (o análogo da curva de forma, agora ancorado nas provas).
- **02b_features_por_cluster.png** — média de cada uma das 5 features por cluster (ajuda a nomear).
- **01_cotovelo_silhueta.png**, **04_silhueta_facas_e_mapa.png** — escolha e diagnóstico de K, igual ao Passo 5.
- **03_resultado_por_cluster.png** — validação H2 (cada perfil passa mais ou menos).
- **features_avaliacao.csv** — uma linha por aluno com as 5 features + intensidade + resultado.
- **perfil_dos_clusters.csv** — resumo numérico de cada cluster.

## O que comparar com o Passo 5
Olhe a silhueta: se ela subir em relação ao Passo 5 (que era ~0,04 em K=4), as features por avaliação separam melhor. Olhe `02_perfis_preparacao`: se os clusters tiverem curvas de preparação visivelmente diferentes (um decrescente, um crescente, um que zera no fim), os perfis ficaram mais nítidos e mais fiéis à hipótese.

## Parâmetros
- `W = 7` (janela de pré-estudo em dias) — troque para 14 para testar.
- `DEFAULT_K = 4` para comparar com o Passo 5; ajuste após ver cotovelo/silhueta.