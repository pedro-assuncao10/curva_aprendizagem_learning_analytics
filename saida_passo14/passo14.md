# Passo 14 — Curva por tempo (sem ancorar em prova)

## Definição do projeto (herdada do passo 12)

Curva de aprendizagem = trajetória de esforço ao longo do **curso inteiro**
(Definição B — ver `saida_passo12/passo12.md`), não hábito de preparação por
prova individual. Os 4 perfis (desistente, antecipado, equilibrado,
fim-pesado) descrevem essa trajetória.

## O que mudou em relação ao passo 12

| | Passo 12 | Passo 14 (este) |
|---|---|---|
| Pontos da curva definidos por | datas de prova de cada curso (nº e posição variam por curso) | fatias de tempo **iguais** (10%, 20%, ... do curso) |
| Precisa de `assessments.csv`? | sim | **não** |
| Duração dos intervalos | desigual (19 a 76 dias) → precisa converter em taxa (cliques/dia) | igual por construção → não precisa de taxa, só proporção |
| Intervalo final pós-última-prova | tratado à parte (intervalo extra) | não existe esse problema, o curso é dividido direto em N fatias |

## Por que a mudança

Depois de fechar a Definição B, ficou claro que as datas de prova não
precisavam mais definir os pontos da curva — elas só faziam isso por herança
da ideia original (medir em torno da prova). Testamos e encontramos um
problema real com a versão ancorada em prova: até **44% da variação** de um
ponto da curva era explicada só por "qual curso o aluno fez" (calendários de
prova diferentes colocam o "meio da curva" em lugares diferentes do
calendário real), não pelo comportamento do aluno.

Fatias de tempo fixas removem essa dependência por construção: 30% do curso
significa o mesmo pedaço de calendário pra qualquer curso, não importa quantas
provas ele tem ou quando elas caem. É também exatamente como o passo 3 (nossa
referência independente) já mede — por terços do curso, não por prova.

## Teste de validação (antes de trocar)

Comparamos as duas formas de montar a curva, clusterizando (K=3, K-means) e
medindo a concordância (ARI) com o rótulo independente do passo 3, em 5
sementes aleatórias diferentes:

```
Ancorado em prova (passo12):        ARI = 0,175
Por tempo, N=6 fatias (passo14):    ARI = 0,287 a 0,297 (média 0,295)
Por tempo, N=20 fatias:             ARI = 0,294 a 0,318 (média 0,306)
```

N=20 é marginalmente melhor que N=6, mas a diferença é pequena perto do ganho
de trocar o método (0,175 → ~0,30). Ficamos com **N=6** por simplicidade e
para manter continuidade com a quantidade de pontos usada nos passos
anteriores.

## Resultado

| perfil | n | % da base | aprovação |
|---|---|---|---|
| desistente | 3.135 | 14,7% | 1,2% |
| antecipado | 7.983 | 37,4% | 78,1% |
| equilibrado | 7.955 | 37,3% | 90,0% |
| fim-pesado | 2.245 | 10,5% | 85,9% |

Pureza contra o rótulo de referência do passo 3 (dos alunos que o passo3
rotula como X, quanto % cai no grupo certo aqui):

- **desistente**: 100% (mesma regra, não é validação independente)
- **antecipado**: **95,3%** (era 72,9% no passo 12 — ganho grande)
- **fim-pesado**: 69,9% (era 84,7% no passo 12 — caiu um pouco)
- **equilibrado**: 34,4% (era 27,0% no passo 12 — melhorou um pouco, mas
  continua sendo o perfil mais fraco/residual)

ARI final = **0,296** (era 0,175 no passo 12). Ganho líquido claro, mesmo com
`fim-pesado` tendo piorado um pouco — o ganho em `antecipado` compensa
bastante.

## Arquivos gerados

- `01_cotovelo_silhueta.png` — escolha de K para o estágio 2 (nota: a
  silhueta pura prefere K=6 aqui, mas mantemos K=3 fixo pela mesma razão dos
  passos anteriores — comparar com os 3 perfis conhecidos)
- `02_perfis_curvas.png` — as 4 curvas médias, agora por fatia de tempo igual
- `03_resultado_por_cluster.png` — aprovação/reprovação por perfil
- `04_silhueta_facas_e_mapa.png` — diagnóstico de silhueta do estágio 2 + PCA
- `curvas_e_perfis.csv` — 6 pontos de curva (por tempo) + perfil final por aluno
- `perfil_dos_grupos.csv` — tabela-resumo (n, % base, aprovação)
- `validacao_crosstab.csv` — cruzamento com os perfis de referência do passo 3
- `comparacao_com_passo12.csv` — ARI do método antigo vs este
