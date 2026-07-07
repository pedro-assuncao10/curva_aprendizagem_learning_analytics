# Passo 5 — K-means (achar os perfis)

## O que muda aqui
Todos os passos anteriores foram **preparação**. Este é o primeiro que faz a **clusterização** de verdade: o algoritmo agrupa os alunos por conta própria e revela os perfis de engajamento. É a resposta da H1 ("existem perfis distintos?") e a base da Lista 2 (Questão 2).

## O que entra no algoritmo
A matriz de cada aluno = **forma (30 fatias)** + **durabilidade** (`semanas_mortas`, `p_comeco`, `p_meio`, `p_fim`). O volume total fica **de fora** de propósito — queremos agrupar por *como* o aluno estuda, não por *quanto*. O resultado final (`final_result`) também fica de fora: a clusterização tem que ser cega ao desfecho, senão seria trapaça. Ele só volta no fim, para validar.

Antes de rodar, padronizamos tudo (`StandardScaler`: cada feature com média 0 e desvio 1), senão features em escalas diferentes (proporções de 0 a 1 vs. semanas mortas de 0 a 30) seriam comparadas de forma injusta.

## Como escolhemos o número de perfis (K)
Rodamos o K-means para K de 2 a 8 e olhamos dois critérios:
- **Cotovelo (inércia):** o quão "apertados" ficam os grupos. Cai sempre que aumenta K; procuramos o ponto onde para de cair rápido (a "dobra").
- **Silhueta:** quão bem separado cada aluno está do cluster vizinho (de -1 a 1; maior é melhor). Procuramos o pico.

O script sugere o K de melhor silhueta, mas você decide olhando os dois gráficos. Para fixar um K à mão, edite `DEFAULT_K` no topo.

## Arquivos gerados

### `02_perfis_curva_media.png`  ← o principal
A curva média de cada cluster. **É aqui que você vê os perfis**: um cluster pode ser uma curva que despenca (abandono), outra constante, outra com pico no fim. Cada linha é um "tipo de aluno".

### `01_cotovelo_silhueta.png`
Os dois critérios de escolha de K, lado a lado. É a justificativa metodológica do número de perfis (a Lista pede isso explicitamente).

### `03_resultado_por_cluster.png`  ← validação (H2)
Para cada cluster, a % de Pass/Distinction/Fail. Liga o perfil ao desempenho: *o tipo de trajetória tem a ver com passar?* Como o `final_result` não entrou na clusterização, se os clusters tiverem aprovação diferente, isso é um achado real.

### `04_pca_clusters.png`
Os alunos comprimidos em 2 dimensões (PCA) e coloridos por cluster — só para visualizar se os grupos se separam no espaço.

### `perfil_dos_clusters.csv` e `alunos_clusters.csv`
Resumo numérico de cada cluster (tamanho, semanas mortas média, proporção por terço, taxa de aprovação) e o cluster de cada aluno. A tabela de perfil é o que você usa para **nomear** os clusters ("constante", "abandono precoce", "vésperas"...).

## Como ler o resultado
1. Olhe `02` e dê um nome a cada curva.
2. Confira em `perfil_dos_clusters.csv` se os números batem com o nome (ex.: o cluster "abandono" deve ter muitas semanas mortas e `p_fim` perto de 0).
3. Olhe `03`: os perfis têm aprovação diferente? Esse é o elo perfil → desempenho.
4. Olhe a tabela cluster × faixa de volume (no terminal): se um perfil for quase todo de um volume só, lembre que forma e volume ainda andam um pouco juntos.

## Limite honesto
A silhueta em dados assim (perfis que são um gradiente, não ilhas) costuma ser modesta (~0,2-0,4). Isso **não** invalida os clusters — significa que as fronteiras são suaves. A decisão de K leva em conta também a **interpretabilidade**: poucos perfis nomeáveis valem mais que muitos clusters difíceis de explicar.