# Passo 4.2 — Abandonador vs Antecipador

## O problema que este passo resolve
No Passo 4, o grupo "começo-pesado" (faixa baixo) tinha só 34% de aprovação — mas era um grupo **suspeito**, porque juntava dois tipos de aluno opostos sob o mesmo rótulo:

- **Abandonou cedo:** clicou bastante no começo e depois **zerou** o curso. O "esforço no começo" é só o rastro de quem parou.
- **Antecipou e sustentou:** teve um pico no começo, mas **continuou aparecendo** até o fim. Esforço cedo de verdade, por antecipação.

Os dois têm o mesmo "centro de massa no início", mas são fenômenos diferentes — e a régua grossa do Passo 4 não os separava.

## O que distingue os dois: durabilidade
A diferença não está em *quando começou*, e sim em *até quando durou*. Medimos isso com duas features:

- **semanas mortas:** quantas das 30 fatias do curso o aluno passou com **zero** cliques. Abandonador tem muitas; quem sustenta tem poucas.
- **atividade no último terço:** se o aluno teve **algum** clique no terço final do curso. Abandonador zerou; quem sustenta manteve.

Dentro de "começo-pesado", quem **não** teve atividade no último terço é rotulado `abandonou_cedo`; quem teve é `antecipou_sustentou`.

## Arquivos gerados

### `features_durabilidade.csv`
**Origem:** série de cliques da `studentVle` em 30 fatias (normalizada pela duração de `courses.csv`), de onde tiramos semanas mortas, proporção por terço e atividade no fim; cruzado com `final_result` (`studentInfo`, sem desistentes).
**Colunas:** `aluno`, `total`, `p_comeco/p_meio/p_fim`, `semanas_mortas`, `cliques_ultimo_terco`, `ativo_no_fim`, `faixa_volume`, `timing`, `final_result`.
**Para que serve:** é a base de features de durabilidade que vai **alimentar o K-means** no Passo 5, dando a ele a informação necessária para separar abandonador de antecipador sozinho.

### `comeco_pesado_detalhado.csv`
**Origem:** apenas os alunos "começo-pesado", já com a coluna `subtipo` (abandonou vs sustentou).

### `01_abandonou_vs_sustentou.png`
**Origem:** taxa de aprovação dos dois subtipos, dentro de cada faixa de volume.
**Para que serve:** mostra que separar os dois faz diferença — se as barras vermelha (abandonou) e verde (sustentou) tiverem alturas bem distintas, a mitigação funcionou.

## Conexão com o K-means (Passo 5)
Este passo não é o agrupamento final — é a preparação que faltava. Ele prova que "durabilidade" é um eixo real e cria as features que o K-means vai usar para encontrar, **sozinho e na curva fina**, a distinção que aqui fizemos à mão.