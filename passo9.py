#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PASSO 9 — H3: PODER PREDITIVO DA FORMA (sem nota, sem vazamento)
A nota foi REMOVIDA: no OULAD o resultado é derivado das notas, então usá-las
seria prever X com algo que é X (target leakage). Sem nota, a H3 fica honesta.

Compara 3 conjuntos de features:
  V   = só volume
  F   = só forma (5 descritores da curva)
  V+F = volume + forma
Em 2 janelas: EARLY (1º terço = alerta precoce) e FULL (curso todo).
Testa 4 arquiteturas de MLP + Gradient Boosting, e plota matriz de confusão.
Alvo: aprovado (Pass/Distinction) vs reprovado (Fail). Desistentes fora.

Requer: pandas, numpy, matplotlib, scikit-learn
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import joblib
import time

DATA_DIR = "./dataset"
DUR_CSV  = "./saida_passo4_2/features_durabilidade.csv"
OUT_DIR  = "./saida_passo9"
MODELS_DIR = "./saida_passo9/modelos"
N_BINS   = 10
N_SPLITS = 5
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def nome_arquivo(*partes):
    s = "_".join(str(p) for p in partes)
    for ch in " ()+,": s = s.replace(ch, "")
    return s

class Tee:
    def __init__(self, *s): self.s = s
    def write(self, m):
        for x in self.s: x.write(m)
    def flush(self):
        for x in self.s: x.flush()
_orig = sys.stdout
_log = open(os.path.join(OUT_DIR, "saida_terminal.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, _log)

print("="*70); print("PASSO 9 — H3 sem nota (4 arquiteturas + GBoost)"); print("="*70)

def chave(df):
    return df["code_module"] + "_" + df["code_presentation"] + "_" + df["id_student"].astype(str)

val = pd.read_csv(DUR_CSV, index_col=0)
val = val[val["final_result"].isin(["Pass","Distinction","Fail"])]
y = val["final_result"].isin(["Pass","Distinction"]).astype(int).values
validos = set(val.index)
print(f"\nAlunos: {len(val):,}  | aprovados: {y.mean()*100:.1f}%")

courses = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
courses["curso"] = courses["code_module"] + "_" + courses["code_presentation"]
dur = dict(zip(courses["curso"], courses["module_presentation_length"]))

vle = pd.read_csv(os.path.join(DATA_DIR, "studentVle.csv"),
                  usecols=["code_module","code_presentation","id_student","date","sum_click"])
vle["aluno"] = chave(vle); vle = vle[vle["aluno"].isin(validos)].copy()
vle["curso"] = vle["code_module"] + "_" + vle["code_presentation"]
vle["length"] = vle["curso"].map(dur)
vle = vle[(vle["date"] >= 0) & (vle["date"] <= vle["length"])]
vle["frac"] = vle["date"] / vle["length"].clip(lower=1)

def features_janela(a, b):
    sub = vle[(vle["frac"] >= a) & (vle["frac"] <= b)].copy()
    sub["slice"] = np.minimum(((sub["frac"] - a) / (b - a) * N_BINS).astype(int), N_BINS - 1)
    M = (sub.groupby(["aluno","slice"])["sum_click"].sum().unstack(fill_value=0)
            .reindex(index=list(validos), columns=range(N_BINS), fill_value=0))
    V = M.values.astype(float); total = V.sum(1); safe = np.where(total > 0, total, 1)
    prop = V / safe[:, None]; x = np.arange(N_BINS); xc = x - x.mean()
    F = pd.DataFrame({
        "volume": total,
        "f_slope": (prop*xc).sum(1)/(xc**2).sum(),
        "f_centro": (prop*x).sum(1)/(N_BINS-1),
        "f_dead": (V == 0).sum(1)/N_BINS,
        "f_std": prop.std(1),
        "f_peak": V.argmax(1)/(N_BINS-1),
    }, index=M.index)
    return F.loc[val.index]

JANELAS = {"EARLY": (0.0, 1/3), "FULL": (0.0, 1.0)}
CONJ = {"V (volume)": ["volume"],
        "F (forma)":  ["f_slope","f_centro","f_dead","f_std","f_peak"],
        "V+F":        ["volume","f_slope","f_centro","f_dead","f_std","f_peak"]}
ARQS = {"rasa(16)": (16,), "media(32,16)": (32,16),
        "profunda(64,32,16)": (64,32,16), "larga(128,64)": (128,64)}

def build(kind):
    if kind == "GBoost":
        return HistGradientBoostingClassifier(random_state=0)
    return make_pipeline(StandardScaler(),
            MLPClassifier(hidden_layer_sizes=ARQS[kind], max_iter=400,
                          early_stopping=True, alpha=1e-3, random_state=0))

def avalia(X, y, kind, contexto="", salvar=True):
    t0 = time.time()
    print(f"  >> treinando {kind:20s} [{contexto}] ...", flush=True)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    aucs, f1s = [], []
    for i, (tr, te) in enumerate(skf.split(X, y), 1):
        clf = build(kind); clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        a = roc_auc_score(y[te], p); f = f1_score(y[te], (p >= 0.5).astype(int), average="macro")
        aucs.append(a); f1s.append(f)
        print(f"       fold {i}/{N_SPLITS}: AUC={a:.3f} F1={f:.3f}", flush=True)
    auc_m, f1_m = float(np.mean(aucs)), float(np.mean(f1s))
    if salvar:
        # re-treina na base TODA e salva o modelo em disco
        modelo_final = build(kind); modelo_final.fit(X, y)
        caminho = os.path.join(MODELS_DIR, nome_arquivo("modelo", contexto, kind) + ".joblib")
        joblib.dump(modelo_final, caminho)
        print(f"     [salvo] {os.path.basename(caminho)}", flush=True)
    print(f"     => média AUC={auc_m:.3f} F1={f1_m:.3f}  ({time.time()-t0:.0f}s)\n", flush=True)
    return auc_m, f1_m

# pré-computa features das duas janelas
FEAT = {nome: features_janela(a, b) for nome, (a, b) in JANELAS.items()}

# ---------- PARTE 1: comparar arquiteturas (no conjunto V+F) ----------
print("\n### PARTE 1 — arquiteturas de rede (conjunto V+F) ###")
linhas_arq = []
for jan in JANELAS:
    X = FEAT[jan][CONJ["V+F"]].values
    for kind in list(ARQS) + ["GBoost"]:
        auc, f1 = avalia(X, y, kind, contexto=f"{jan}_VF")
        linhas_arq.append({"janela": jan, "modelo": kind, "AUC": round(auc,3), "F1": round(f1,3)})
arq = pd.DataFrame(linhas_arq)
melhor_mlp = (arq[arq["modelo"] != "GBoost"].groupby("modelo")["AUC"].mean().idxmax())
print(f"\nMelhor arquitetura MLP (AUC média): {melhor_mlp}")

# ---------- PARTE 2: V vs F vs V+F (a H3) com a melhor MLP e GBoost ----------
print("\n### PARTE 2 — V vs F vs V+F (a H3) ###")
linhas_h3 = []
for jan in JANELAS:
    for conj_nome, cols in CONJ.items():
        X = FEAT[jan][cols].values
        for kind in [melhor_mlp, "GBoost"]:
            auc, f1 = avalia(X, y, kind, contexto=f"{jan}_{conj_nome}")
            linhas_h3.append({"janela": jan, "conjunto": conj_nome, "modelo": kind,
                              "AUC": round(auc,3), "F1": round(f1,3)})
h3 = pd.DataFrame(linhas_h3)
h3.to_csv(os.path.join(OUT_DIR, "h3_resultados.csv"), index=False)
arq.to_csv(os.path.join(OUT_DIR, "arquiteturas.csv"), index=False)

# ---------- gráfico 1: arquiteturas ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, jan in zip(axes, JANELAS):
    sub = arq[arq["janela"] == jan]
    ax.bar(sub["modelo"], sub["AUC"], color="#5C6BC0")
    ax.set_title(f"Arquiteturas — {jan} (V+F)"); ax.set_ylabel("AUC"); ax.set_ylim(0.5, 1.0)
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(sub["AUC"]): ax.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "01_arquiteturas.png"), dpi=130); plt.close()

# ---------- gráfico 2: V vs F vs V+F (a H3) ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cores = {"V (volume)": "#90A4AE", "F (forma)": "#42A5F5", "V+F": "#2E7D32"}
for ax, jan in zip(axes, JANELAS):
    sub = h3[(h3["janela"] == jan) & (h3["modelo"] == melhor_mlp)]
    ax.bar(sub["conjunto"], sub["AUC"], color=[cores[c] for c in sub["conjunto"]])
    ax.set_title(f"{jan} — MLP {melhor_mlp}"); ax.set_ylabel("AUC"); ax.set_ylim(0.5, 1.0)
    for i, v in enumerate(sub["AUC"]): ax.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=9)
plt.suptitle("H3 — só volume vs só forma vs volume+forma (sem nota)", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "02_h3_volume_vs_forma.png"), dpi=130); plt.close()

# ---------- gráfico 3: matrizes de confusão (melhor MLP, V+F) ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
for ax, jan in zip(axes, JANELAS):
    X = FEAT[jan][CONJ["V+F"]].values
    pred = cross_val_predict(build(melhor_mlp), X, y, cv=skf, method="predict")
    cm = confusion_matrix(y, pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["reprovou","aprovou"]); ax.set_yticklabels(["reprovou","aprovou"])
    ax.set_xlabel("previsto"); ax.set_ylabel("real"); ax.set_title(f"Matriz de confusão — {jan} (V+F)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "03_matriz_confusao.png"), dpi=130); plt.close()

print("\n>>> RESUMO H3:"); print(h3.to_string(index=False))
print("\n" + "="*70)
print(f"Gravado em {os.path.abspath(OUT_DIR)}:")
print("  - 01_arquiteturas.png        (4 redes + GBoost)")
print("  - 02_h3_volume_vs_forma.png  (a H3: V vs F vs V+F)")
print("  - 03_matriz_confusao.png     (onde o modelo erra)")
print("  - h3_resultados.csv  arquiteturas.csv  saida_terminal.txt")
print(f"  - modelos/  ({len(os.listdir(MODELS_DIR))} modelos .joblib salvos)")
print("\nLeitura: se V+F > V, a forma adiciona. Se F sozinho já vai bem, a forma tem sinal próprio.")
sys.stdout = _orig; _log.close()