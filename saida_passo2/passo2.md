# Passo 2 — Volume

## O que este passo faz
Responde **uma** pergunta: *quanto cada aluno clicou no total?* — e a partir disso, *quem clica pouco, médio e muito*, e *isso tem a ver com passar?*

Aqui ainda **não** olhamos *quando* o aluno clicou (isso é a forma, Passo 3). Só o total.

## Como o volume é calculado
1. Abrimos a `studentVle.csv` (o log de cliques).
2. Cada aluno é identificado pela chave `code_module + code_presentation + id_student`.
3. **Somamos** todos os `sum_click` de cada aluno. Esse total é o "volume".

É só isso: volume = soma de todos os cliques do aluno no curso inteiro.

## As faixas (pouco / médio / muito)
Cortamos os alunos em **tercis**: ordenamos pelo total de cliques e dividimos em três grupos de tamanho igual — o terço que menos clicou (`baixo`), o do meio (`medio`) e o que mais clicou (`alto`). Os valores exatos dos cortes aparecem no terminal.

## Arquivos que este passo gera

### `volume_por_aluno.csv`
**Origem:** somamos `sum_click` por aluno na `studentVle.csv`, classificamos em faixa por tercil, e **cruzamos com `final_result` da `studentInfo.csv`** (ligados pela chave do aluno).
**Colunas:** `aluno`, `total_cliques`, `faixa_volume` (baixo/medio/alto), `final_result`.
**Para que serve:** é a tabela-base do volume — uma linha por aluno dizendo quanto clicou e como terminou.

### `01_distribuicao_volume.png`
**Origem:** histograma do `total_cliques` (em escala log, porque alguns alunos clicam muitíssimo mais que a maioria). As linhas vermelhas são os cortes dos tercis.
**Para que serve:** mostra que a maioria dos alunos se concentra numa faixa central, com uma cauda de "super-usuários".

### `02_resultado_por_volume.png`  ← o insight
**Origem:** para cada faixa de volume, calculamos a % de alunos em cada `final_result`.
**Cruzamento:** `total_cliques` (vem da `studentVle`) × `final_result` (vem da `studentInfo`).
**Para que serve:** é a primeira resposta de verdade do projeto — *clicar mais está associado a passar mais?* Se as barras verdes (Pass/Distinction) crescem da faixa baixa para a alta, sim.

## O que observar
- Provavelmente a faixa `alto` tem mais aprovação e menos desistência que a `baixo`. Isso confirma o óbvio (quem usa mais a plataforma se sai melhor) — e é exatamente o "óbvio" que a forma (Passo 3) vai tentar superar, mostrando algo que o volume sozinho não revela.
- Guarde este resultado: ele é o **baseline**. Tudo que a forma trouxer de novo será comparado contra isto.