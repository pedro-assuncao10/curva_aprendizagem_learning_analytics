# -*- coding: utf-8 -*-
"""
Validacao estatistica dos perfis de preparacao (FFF 2014J, curvas de 4 bins).

Cada afirmacao do trabalho ganha um teste-guardiao:
  T1 Permutacao da silhueta ... "os 3 grupos existiriam em dados sem estrutura temporal?"
  T2 ANOVA/Kruskal do volume .. "a forma e so volume disfarcado?"
  T3 Qui-quadrado + Cramer .... "perfil e aprovacao andam juntos?"
  T4 ANOVA notas + Tukey ...... "perfil muda a nota? entre quais perfis?"
  T5 Kaplan-Meier + log-rank .. "o comportamento inicial preve QUANDO o aluno desiste?"

Regras de honestidade:
  - Testes T2..T5 usam apenas variaveis EXTERNAS a clusterizacao (volume, nota,
    resultado final, data de desistencia) - nunca as curvas, para evitar
    circularidade. T1 e a unica excecao e por isso usa permutacao, nao ANOVA.
  - Uma observacao por aluno (ciclos do mesmo aluno sao dependentes).
  - Withdrawn fica fora dos testes de aprovacao/nota; ganha o teste proprio (T5).

Saida: resultados_validacao/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import kmeans_ciclos as kc
from kmeans_ciclos import (BASE_DIR, DATA_DIR, MODULE, PRES, SEED,
                           SURF, INK, SEC, MUT, GRID, BASE, SERIES,
                           CORES_PERFIL, COR_RESULTADO)

OUT = os.path.join(BASE_DIR, "resultados_validacao")
N_PERM = 200
COR_MISTO = "#898781"
PERFIS_ORDEM = ["Adiantado", "Equilibrado", "Tardio", "Misto"]


def cor_perfil(p):
    return CORES_PERFIL.get(p, COR_MISTO)


# ---------------------------------------------------------------- base
def montar_base():
    """Matriz de curvas + tabela por aluno com variaveis externas."""
    diario = kc.carregar_vle_diario()
    tmas, submissores = kc.montar_ciclos()
    X, meta, _ = kc.construir_matriz(diario, tmas, submissores)

    km3 = KMeans(n_clusters=3, n_init=50, random_state=SEED).fit(X)
    nomes, _ = kc.nomear_perfis(km3.cluster_centers_)
    atrib = meta.copy()
    atrib["perfil_ciclo"] = pd.Series(km3.labels_).map(nomes).values

    # perfil consolidado (mesma regra do pipeline: >=60% em >=3 ciclos)
    def consolidar(g):
        cont = g["perfil_ciclo"].value_counts()
        moda, freq, n = cont.index[0], cont.iloc[0], len(g)
        perfil = (moda if n >= 3 and freq / n >= 0.6
                  else ("Misto" if n >= 3 else "Insuficiente"))
        return pd.Series({"perfil": perfil, "n_ciclos": n,
                          "volume_total": g["total_clicks"].sum()})

    alunos = (atrib.groupby("id_student")
              .apply(consolidar, include_groups=False).reset_index())

    # nota media nos TMAs do FFF 2014J
    sub = pd.read_csv(os.path.join(DATA_DIR, "studentAssessment.csv"),
                      usecols=["id_assessment", "id_student", "score"])
    sub = sub[sub["id_assessment"].isin(tmas["id_assessment"])]
    notas = (sub.dropna(subset=["score"]).groupby("id_student")["score"]
             .mean().rename("nota_media").reset_index())
    alunos = alunos.merge(notas, on="id_student", how="left")

    info = pd.read_csv(os.path.join(DATA_DIR, "studentInfo.csv"),
                       usecols=["code_module", "code_presentation",
                                "id_student", "final_result"])
    info = info[(info["code_module"] == MODULE) &
                (info["code_presentation"] == PRES)][["id_student",
                                                      "final_result"]]
    alunos = alunos.merge(info, on="id_student", how="left")

    reg = pd.read_csv(os.path.join(DATA_DIR, "studentRegistration.csv"))
    reg = reg[(reg["code_module"] == MODULE) &
              (reg["code_presentation"] == PRES)][["id_student",
                                                   "date_unregistration"]]
    alunos = alunos.merge(reg, on="id_student", how="left")

    cursos = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
    fim_curso = int(cursos[(cursos["code_module"] == MODULE) &
                           (cursos["code_presentation"] == PRES)]
                    ["module_presentation_length"].iloc[0])
    return X, atrib, alunos, tmas, fim_curso


# ---------------------------------------------------------------- T1 permutacao
def t1_permutacao(X):
    print("T1: permutação da silhueta...")
    rng = np.random.default_rng(SEED)
    km = KMeans(n_clusters=3, n_init=50, random_state=SEED).fit(X)
    sil_real = silhouette_score(X, km.labels_)

    nulos = []
    for i in range(N_PERM):
        Xp = X.copy()
        # embaralha a ORDEM dos bins dentro de cada curva:
        # destroi a estrutura temporal, preserva a composicao de valores
        for linha in Xp:
            rng.shuffle(linha)
        kmp = KMeans(n_clusters=3, n_init=10, random_state=int(rng.integers(1e9)))
        rotp = kmp.fit_predict(Xp)
        nulos.append(silhouette_score(Xp, rotp))
    nulos = np.array(nulos)
    p = (1 + (nulos >= sil_real).sum()) / (N_PERM + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(nulos, bins=30, color=SERIES[0], alpha=0.75,
            edgecolor=SURF, label="silhueta com bins embaralhados")
    ax.axvline(sil_real, color=SERIES[5], linewidth=2.5)
    ax.annotate(f"silhueta real = {sil_real:.3f}", (sil_real, ax.get_ylim()[1]),
                textcoords="offset points", xytext=(-8, -14), color=SERIES[5],
                fontsize=10, fontweight="bold", ha="right")
    ax.set_xlabel("silhueta média (k=3)")
    ax.set_ylabel("frequência")
    ax.set_title(f"T1 — Teste de permutação ({N_PERM} embaralhamentos)")
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(SEC)
    fig.savefig(os.path.join(OUT, "t1_permutacao_silhueta.png"))
    plt.close(fig)
    return sil_real, nulos, p


# ---------------------------------------------------------------- helpers
def eta_quadrado(grupos):
    tudo = np.concatenate(grupos)
    media = tudo.mean()
    ss_total = ((tudo - media) ** 2).sum()
    ss_entre = sum(len(g) * (g.mean() - media) ** 2 for g in grupos)
    return ss_entre / ss_total


# ---------------------------------------------------------------- T2 volume
def t2_volume(alunos):
    print("T2: ANOVA/Kruskal do volume por perfil...")
    base = alunos[alunos["perfil"].isin(PERFIS_ORDEM)].copy()
    base["log_vol"] = np.log10(base["volume_total"])
    grupos = [base.loc[base["perfil"] == p, "log_vol"].to_numpy()
              for p in PERFIS_ORDEM]

    F, p_anova = stats.f_oneway(*grupos)
    H, p_kw = stats.kruskal(*grupos)
    _, p_lev = stats.levene(*grupos)
    eta2 = eta_quadrado(grupos)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bp = ax.boxplot([10 ** g for g in grupos], tick_labels=[
        f"{p}\n(n={len(g)})" for p, g in zip(PERFIS_ORDEM, grupos)],
        patch_artist=True, showfliers=False, widths=0.55)
    for patch, p_nome in zip(bp["boxes"], PERFIS_ORDEM):
        patch.set_facecolor(cor_perfil(p_nome))
        patch.set_alpha(0.75)
        patch.set_edgecolor(SURF)
    for el in ("medians",):
        for linha in bp[el]:
            linha.set_color(INK)
    ax.set_yscale("log")
    ax.set_ylabel("volume total de cliques (escala log)")
    ax.set_title(f"T2 — Volume por perfil  (η² = {eta2:.3f}: perfil explica "
                 f"{eta2 * 100:.1f}% da variância do volume)")
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(OUT, "t2_volume_por_perfil.png"))
    plt.close(fig)
    return F, p_anova, H, p_kw, p_lev, eta2


# ---------------------------------------------------------------- T3 qui-quadrado
def t3_quiquadrado(alunos):
    print("T3: qui-quadrado perfil x aprovação...")
    base = alunos[(alunos["perfil"].isin(PERFIS_ORDEM)) &
                  (alunos["final_result"].isin(["Pass", "Distinction",
                                                "Fail"]))].copy()
    base["aprovado"] = base["final_result"].isin(["Pass", "Distinction"])
    tabela = pd.crosstab(base["perfil"], base["aprovado"]).reindex(PERFIS_ORDEM)
    chi2, p, gl, esperado = stats.chi2_contingency(tabela)
    n = tabela.to_numpy().sum()
    cramer = np.sqrt(chi2 / (n * (min(tabela.shape) - 1)))

    # residuos ajustados
    obs = tabela.to_numpy().astype(float)
    linha_tot = obs.sum(axis=1, keepdims=True)
    col_tot = obs.sum(axis=0, keepdims=True)
    resid = (obs - esperado) / np.sqrt(
        esperado * (1 - linha_tot / n) * (1 - col_tot / n))
    resid = pd.DataFrame(resid, index=tabela.index,
                         columns=["Reprovado", "Aprovado"]
                         if tabela.columns[0] == False else
                         ["Aprovado", "Reprovado"])

    taxa = base.groupby("perfil")["aprovado"].mean().reindex(PERFIS_ORDEM) * 100
    geral = base["aprovado"].mean() * 100
    ns = base.groupby("perfil").size().reindex(PERFIS_ORDEM)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = np.arange(len(PERFIS_ORDEM))
    for i, p_nome in enumerate(PERFIS_ORDEM):
        ax.bar(i, taxa[p_nome], width=0.6, color=cor_perfil(p_nome),
               edgecolor=SURF, linewidth=2)
        ax.text(i, taxa[p_nome] + 1.2, f"{taxa[p_nome]:.0f}%", ha="center",
                fontsize=10, fontweight="bold", color=SEC)
    ax.axhline(geral, color=INK, linewidth=1.2, linestyle="--")
    ax.annotate(f"taxa geral = {geral:.0f}%", (len(PERFIS_ORDEM) - 0.5, geral),
                textcoords="offset points", xytext=(0, 6), color=SEC,
                fontsize=9, ha="right")
    ax.set_xticks(xs, [f"{p}\n(n={ns[p]})" for p in PERFIS_ORDEM])
    ax.set_ylabel("% de aprovação (Pass + Distinction)")
    ax.set_ylim(0, 100)
    ax.set_title(f"T3 — Aprovação por perfil  (χ²={chi2:.1f}, p={p:.2e}, "
                 f"V de Cramér={cramer:.3f})")
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(OUT, "t3_quiquadrado_aprovacao.png"))
    plt.close(fig)
    return tabela, chi2, p, cramer, resid, taxa, geral


# ---------------------------------------------------------------- T4 notas
def t4_notas(alunos):
    print("T4: ANOVA + Tukey das notas por perfil...")
    base = alunos[(alunos["perfil"].isin(PERFIS_ORDEM)) &
                  alunos["nota_media"].notna() &
                  (alunos["final_result"] != "Withdrawn")].copy()
    grupos = [base.loc[base["perfil"] == p, "nota_media"].to_numpy()
              for p in PERFIS_ORDEM]

    F, p_anova = stats.f_oneway(*grupos)
    H, p_kw = stats.kruskal(*grupos)
    _, p_lev = stats.levene(*grupos)
    eta2 = eta_quadrado(grupos)
    tukey = stats.tukey_hsd(*grupos)
    ic = tukey.confidence_interval()

    pares, difs, los, his, ps = [], [], [], [], []
    for i in range(len(PERFIS_ORDEM)):
        for j in range(i + 1, len(PERFIS_ORDEM)):
            pares.append(f"{PERFIS_ORDEM[i]} − {PERFIS_ORDEM[j]}")
            difs.append(grupos[i].mean() - grupos[j].mean())
            los.append(ic.low[i, j])
            his.append(ic.high[i, j])
            ps.append(tukey.pvalue[i, j])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bp = axes[0].boxplot(grupos, tick_labels=[
        f"{p}\n(n={len(g)}, μ={g.mean():.1f})"
        for p, g in zip(PERFIS_ORDEM, grupos)],
        patch_artist=True, showfliers=False, widths=0.55)
    for patch, p_nome in zip(bp["boxes"], PERFIS_ORDEM):
        patch.set_facecolor(cor_perfil(p_nome))
        patch.set_alpha(0.75)
        patch.set_edgecolor(SURF)
    for linha in bp["medians"]:
        linha.set_color(INK)
    axes[0].set_ylabel("nota média nos TMAs")
    axes[0].set_title(f"Notas por perfil (ANOVA F={F:.1f}, p={p_anova:.2e})")
    axes[0].grid(axis="x", visible=False)

    ys = np.arange(len(pares))[::-1]
    for y, d, lo, hi, pv in zip(ys, difs, los, his, ps):
        cor = SERIES[5] if pv < 0.05 else MUT
        axes[1].plot([lo, hi], [y, y], color=cor, linewidth=2)
        axes[1].scatter([d], [y], color=cor, s=45, zorder=5)
        axes[1].annotate(f"p={pv:.3f}" if pv >= 0.001 else "p<0.001",
                         (hi, y), textcoords="offset points", xytext=(8, -3),
                         fontsize=8, color=SEC)
    axes[1].axvline(0, color=BASE, linewidth=1)
    axes[1].set_yticks(ys, pares, fontsize=9)
    axes[1].set_xlabel("diferença de médias (pontos de nota) com IC 95%")
    axes[1].set_title("Tukey HSD — pares (vermelho = significativo)")
    axes[1].grid(axis="y", visible=False)
    fig.suptitle("T4 — Perfil muda a nota? Entre quais perfis?",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "t4_notas_anova_tukey.png"))
    plt.close(fig)

    tukey_df = pd.DataFrame({"par": pares, "diferenca": difs,
                             "ic_low": los, "ic_high": his, "p_ajustado": ps})
    return F, p_anova, H, p_kw, p_lev, eta2, tukey_df, grupos


# ---------------------------------------------------------------- T5 sobrevivencia
def km_curva(tempos, eventos):
    """Kaplan-Meier simples: devolve (t, S(t))."""
    ordem = np.argsort(tempos)
    t, e = tempos[ordem], eventos[ordem]
    uniq = np.unique(t[e == 1])
    S, ts, s_atual = [1.0], [0.0], 1.0
    for ti in uniq:
        em_risco = (t >= ti).sum()
        mortes = ((t == ti) & (e == 1)).sum()
        s_atual *= (1 - mortes / em_risco)
        ts.append(ti)
        S.append(s_atual)
    return np.array(ts), np.array(S)


def logrank_multigrupo(tempos, eventos, grupos_arr, rotulos):
    k = len(rotulos)
    uniq = np.sort(np.unique(tempos[eventos == 1]))
    O = np.zeros(k)
    E = np.zeros(k)
    V = np.zeros((k, k))
    for t in uniq:
        risco = tempos >= t
        n = risco.sum()
        d = ((tempos == t) & (eventos == 1)).sum()
        if n <= 1 or d == 0:
            continue
        n_g = np.array([(risco & (grupos_arr == g)).sum() for g in rotulos])
        d_g = np.array([((tempos == t) & (eventos == 1) &
                         (grupos_arr == g)).sum() for g in rotulos])
        O += d_g
        E += d * n_g / n
        for i in range(k):
            for j in range(k):
                delta = 1.0 if i == j else 0.0
                V[i, j] += (d * (n_g[i] / n) * (delta - n_g[j] / n)
                            * (n - d) / (n - 1))
    z = (O - E)[:-1]
    chi2 = float(z @ np.linalg.solve(V[:-1, :-1], z))
    p = float(stats.chi2.sf(chi2, k - 1))
    return chi2, p, O, E


def t5_sobrevivencia(atrib, alunos, tmas, fim_curso):
    print("T5: Kaplan-Meier + log-rank (perfil precoce)...")
    marco = int(tmas["date"].iloc[1])            # deadline do TMA2 (dia 52)

    # perfil precoce: rotulos dos ciclos 1 e 2 (quem tem os dois)
    c12 = atrib[atrib["ciclo"].isin([1, 2])].pivot(
        index="id_student", columns="ciclo", values="perfil_ciclo").dropna()
    precoce = pd.Series(np.where(c12[1] == c12[2], c12[1], "Misto"),
                        index=c12.index, name="perfil_precoce").reset_index()

    base = precoce.merge(alunos[["id_student", "date_unregistration"]],
                         on="id_student", how="left")
    unreg = pd.to_numeric(base["date_unregistration"], errors="coerce")
    base = base[(unreg.isna()) | (unreg > marco)].copy()
    unreg = pd.to_numeric(base["date_unregistration"], errors="coerce")
    base["evento"] = unreg.notna().astype(int)
    base["tempo"] = np.where(base["evento"] == 1, unreg - marco,
                             fim_curso - marco)

    rotulos = [p for p in PERFIS_ORDEM
               if (base["perfil_precoce"] == p).sum() >= 30]
    base = base[base["perfil_precoce"].isin(rotulos)]
    tempos = base["tempo"].to_numpy(dtype=float)
    eventos = base["evento"].to_numpy()
    grupos_arr = base["perfil_precoce"].to_numpy()

    chi2, p, O, E = logrank_multigrupo(tempos, eventos, grupos_arr, rotulos)

    fig, ax = plt.subplots(figsize=(8, 5))
    for pnome in rotulos:
        m = grupos_arr == pnome
        ts, S = km_curva(tempos[m], eventos[m])
        ax.step(ts + marco, S * 100, where="post", color=cor_perfil(pnome),
                linewidth=2)
        ax.annotate(f"{pnome} (n={m.sum()}, evasões={int(eventos[m].sum())})",
                    (ts[-1] + marco, S[-1] * 100),
                    textcoords="offset points", xytext=(8, 0),
                    color=cor_perfil(pnome), fontsize=9, fontweight="bold",
                    va="center")
    for d in tmas["date"].astype(int).tolist()[2:]:
        ax.axvline(d, color=GRID, linewidth=1, zorder=0)
        ax.annotate("TMA", (d, ax.get_ylim()[0]), textcoords="offset points",
                    xytext=(2, 4), fontsize=7, color=MUT)
    ax.set_xlabel("dia do curso (análise a partir do dia 52 = TMA2)")
    ax.set_ylabel("% ainda matriculado (Kaplan-Meier)")
    ax.set_title(f"T5 — Sobrevivência por perfil precoce  "
                 f"(log-rank χ²={chi2:.1f}, p={p:.4f})")
    ax.set_ylim(bottom=None, top=101)
    fig.subplots_adjust(right=0.74)
    fig.savefig(os.path.join(OUT, "t5_sobrevivencia_km.png"))
    plt.close(fig)
    return chi2, p, base, rotulos, O, E


# ---------------------------------------------------------------- tabela final
def tabela_final(resultados):
    """Tabela-sintese desenhada manualmente (altura de linha por conteudo)."""
    import textwrap

    def quebrar(txt, largura):
        linhas = []
        for l in str(txt).split("\n"):
            linhas.extend(textwrap.wrap(l, largura) or [""])
        return linhas

    df = pd.DataFrame(resultados)
    df.to_csv(os.path.join(OUT, "tabela_resultados.csv"), index=False)

    linhas_tab = []
    for r in df.to_dict("records"):
        cels = [quebrar(r["teste"], 18), quebrar(r["pergunta"], 30),
                quebrar(r["numeros"], 48), quebrar(r["veredito"], 44)]
        linhas_tab.append((cels, max(len(c) for c in cels),
                           bool(r["sucesso"])))

    alturas = [n + 0.9 for _, n, _ in linhas_tab]
    total = sum(alturas) + 1.5
    fig, ax = plt.subplots(figsize=(15, 0.26 * total + 0.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(total, 0)
    ax.axis("off")
    col_x = [0.01, 0.145, 0.385, 0.72]
    cabecalhos = ["Teste", "Pergunta", "Resultado numérico", "Veredito"]

    ax.add_patch(plt.Rectangle((0, 0), 1, 1.3, facecolor="#f0efec",
                               edgecolor="none"))
    for x, h in zip(col_x, cabecalhos):
        ax.text(x, 0.65, h, fontweight="bold", color=INK, va="center",
                fontsize=10.5)
    ax.plot([0, 1], [1.3, 1.3], color=BASE, linewidth=1)

    y = 1.5
    for (cels, _, ok), h in zip(linhas_tab, alturas):
        for j, (x, cel) in enumerate(zip(col_x, cels)):
            if j == 3:
                cor, peso = ("#006300" if ok else "#b3261e"), "bold"
            elif j == 0:
                cor, peso = INK, "bold"
            else:
                cor, peso = SEC, "normal"
            ax.text(x, y + 0.15, "\n".join(cel), va="top", fontsize=9.5,
                    color=cor, fontweight=peso, linespacing=1.45)
        y += h
        ax.plot([0, 1], [y, y], color=GRID, linewidth=0.8)
    ax.set_title("Validação estatística dos perfis — síntese",
                 fontweight="bold", color=INK, pad=14)
    fig.savefig(os.path.join(OUT, "tabela_resultados.png"))
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    X, atrib, alunos, tmas, fim_curso = montar_base()
    print(f"Base: {len(X):,} ciclos, {len(alunos):,} alunos, "
          f"fim do curso = dia {fim_curso}")

    resultados = []

    sil_real, nulos, p1 = t1_permutacao(X)
    ok1 = p1 < 0.01
    resultados.append({
        "teste": "T1 Permutação\n(silhueta)",
        "pergunta": "Os 3 grupos existiriam em dados\nsem estrutura temporal?",
        "numeros": (f"silhueta real = {sil_real:.3f}\n"
                    f"nulos: μ={nulos.mean():.3f}, máx={nulos.max():.3f}\n"
                    f"p = {p1:.4f} ({N_PERM} permutações)"),
        "veredito": ("SIM, estrutura genuína: nenhuma permutação\n"
                     "alcançou a silhueta real"
                     if ok1 else "Inconclusivo: silhueta real dentro do nulo"),
        "sucesso": ok1})

    F2, pa2, H2v, pk2, plev2, eta2 = t2_volume(alunos)
    ok2 = eta2 < 0.06   # efeito pequeno (convencao de Cohen)
    resultados.append({
        "teste": "T2 ANOVA/Kruskal\n(volume)",
        "pergunta": "A forma é só volume\ndisfarçado?",
        "numeros": (f"ANOVA F={F2:.1f} (p={pa2:.2e})\n"
                    f"Kruskal H={H2v:.1f} (p={pk2:.2e})\n"
                    f"η² = {eta2:.3f} ({eta2 * 100:.1f}% da variância)"),
        "veredito": (f"NÃO: perfis diferem pouco em volume —\n"
                     f"perfil explica só {eta2 * 100:.1f}% do volume "
                     f"(efeito pequeno)" if ok2 else
                     f"PARCIAL: volume difere entre perfis (η²="
                     f"{eta2:.2f});\ncontrolar volume nos modelos preditivos"),
        "sucesso": ok2})

    tabela3, chi3, p3, cramer3, resid3, taxa3, geral3 = t3_quiquadrado(alunos)
    ok3 = p3 < 0.05
    resultados.append({
        "teste": "T3 Qui-quadrado\n(perfil × aprovação)",
        "pergunta": "Perfil e aprovação\nandam juntos?",
        "numeros": (f"χ² = {chi3:.1f}, gl=3, p = {p3:.2e}\n"
                    f"V de Cramér = {cramer3:.3f}\n"
                    f"taxas: " + ", ".join(f"{p} {taxa3[p]:.0f}%"
                                           for p in PERFIS_ORDEM)),
        "veredito": ("SIM: associação significativa —\n"
                     "perfil e resultado final não são independentes"
                     if ok3 else "NÃO detectada associação"),
        "sucesso": ok3})

    F4, pa4, H4v, pk4, plev4, eta4, tukey_df, grupos4 = t4_notas(alunos)
    ok4 = pa4 < 0.05 and pk4 < 0.05
    sig_pares = tukey_df[tukey_df["p_ajustado"] < 0.05]["par"].tolist()
    resultados.append({
        "teste": "T4 ANOVA + Tukey\n(notas)",
        "pergunta": "Perfil muda a nota?\nEntre quais perfis?",
        "numeros": (f"ANOVA F={F4:.1f} (p={pa4:.2e}), η²={eta4:.3f}\n"
                    f"Kruskal H={H4v:.1f} (p={pk4:.2e})\n"
                    f"Tukey: {len(sig_pares)}/6 pares significativos"),
        "veredito": ("SIM: notas diferem por perfil.\nPares: " +
                     "; ".join(sig_pares) if ok4
                     else "NÃO detectada diferença de notas"),
        "sucesso": ok4})

    chi5, p5, base5, rot5, O5, E5 = t5_sobrevivencia(atrib, alunos, tmas,
                                                     fim_curso)
    ok5 = p5 < 0.05
    obs_esp = ", ".join(f"{r}: {int(o)}obs/{e:.0f}esp"
                        for r, o, e in zip(rot5, O5, E5))
    resultados.append({
        "teste": "T5 Kaplan-Meier\n+ log-rank (evasão)",
        "pergunta": "O comportamento inicial prevê\nQUANDO o aluno desiste?",
        "numeros": (f"log-rank χ² = {chi5:.1f}, p = {p5:.4f}\n"
                    f"n = {len(base5):,} vivos no dia 52\n"
                    f"evasões obs/esperadas: {obs_esp}"),
        "veredito": ("SIM: curvas de sobrevivência diferem —\n"
                     "o perfil dos 2 primeiros ciclos antecipa a evasão"
                     if ok5 else
                     "NÃO: evasão posterior não difere por perfil precoce"),
        "sucesso": ok5})

    tabela_final(resultados)

    with open(os.path.join(OUT, "resumo.txt"), "w", encoding="utf-8") as f:
        f.write("VALIDAÇÃO ESTATÍSTICA DOS PERFIS (FFF 2014J, 4 bins)\n\n")
        for r in resultados:
            f.write(f"== {r['teste'].replace(chr(10), ' ')} ==\n")
            f.write(f"Pergunta: {r['pergunta'].replace(chr(10), ' ')}\n")
            f.write(f"Números:  {r['numeros'].replace(chr(10), ' | ')}\n")
            f.write(f"Veredito: {r['veredito'].replace(chr(10), ' ')}\n\n")
        f.write("Detalhe T3 — resíduos ajustados (|z|>2 = desvio real):\n")
        f.write(resid3.round(2).to_string())
        f.write("\n\nDetalhe T4 — Tukey HSD completo:\n")
        f.write(tukey_df.round(4).to_string(index=False))
        f.write(f"\n\nLevene: volume p={plev2:.3f} | notas p={plev4:.3f}\n")

    print("\n" + "=" * 60)
    for r in resultados:
        print(f"{r['teste'].replace(chr(10), ' ')}: "
              f"{'OK' if r['sucesso'] else 'ATENÇÃO'}")
    print(f"Concluído. Resultados em: {OUT}")


if __name__ == "__main__":
    main()
