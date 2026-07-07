#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PASSO 11 — H3 com AS FEATURES DOS PERFIS (coerência total com a clusterização)
Usa EXATAMENTE as 5 features que geraram os perfis (passos 6-8):
  tendencia, antecedencia, durabilidade, concentracao, cobertura_final
Objetivo: provar que as curvas/perfis que achamos têm sentido PREDITIVO.

Dois enquadramentos:
  FULL     : as 5 features sobre o curso inteiro -> valida o poder preditivo.
  PARCIAL  : as MESMAS 5 features, mas recalculadas só com as avaliações do
             1º 70% do curso -> alerta precoce (sem olhar o futuro).
Compara V (volume) / F (as 5 features) / V+F, com MLP media(32,16) + GBoost.
Trata o DESBALANCEAMENTO com peso de classe (balanced, derivado da frequência
real) e mostra recall/precision da classe REPROVADO + curva precision-recall.
Alvo: aprovado (Pass/Distinction) vs reprovado (Fail). Desistentes fora.

Requer: pandas, numpy, matplotlib, scikit-learn, joblib
"""

import os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (roc_auc_score, f1_score, recall_score, precision_score,
                             confusion_matrix, precision_recall_curve)
import joblib

DATA_DIR = "./dataset"
DUR_CSV  = "./saida_passo4_2/features_durabilidade.csv"
OUT_DIR  = "./saida_passo11"
W        = 7
N_SPLITS = 5
os.makedirs(OUT_DIR, exist_ok=True)

class Tee:
    def __init__(self, *s): self.s = s
    def write(self, m):
        for x in self.s: x.write(m)
    def flush(self):
        for x in self.s: x.flush()
_orig = sys.stdout
_log = open(os.path.join(OUT_DIR, "saida_terminal.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, _log)

print("="*70); print("PASSO 11 — H3 com as 5 features dos perfis"); print("="*70)

def chave(df):
    return df["code_module"] + "_" + df["code_presentation"] + "_" + df["id_student"].astype(str)
def curso_de(a):
    p = a.split("_"); return p[0] + "_" + p[1]

val = pd.read_csv(DUR_CSV, index_col=0)
val = val[val["final_result"].isin(["Pass","Distinction","Fail"])]
y = val["final_result"].isin(["Pass","Distinction"]).astype(int).values
validos = set(val.index)
print(f"\nAlunos: {len(val):,}  | aprovados: {y.mean()*100:.1f}%  | reprovados: {(1-y.mean())*100:.1f}%")

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

asm = pd.read_csv(os.path.join(DATA_DIR, "assessments.csv"),
                  usecols=["id_assessment","code_module","code_presentation","date"]).dropna(subset=["date"])
asm["curso"] = asm["code_module"] + "_" + asm["code_presentation"]
asm["length"] = asm["curso"].map(dur)
asm["pos"] = asm["date"].astype(int) / asm["length"].clip(lower=1)

def features_5(thr):
    """As 5 features dos perfis, usando só avaliações com pos<=thr e cliques com frac<=thr."""
    asm_w = asm[asm["pos"] <= thr]
    datas_curso = {c: np.sort(g["date"].astype(int).values) for c, g in asm_w.groupby("curso")}
    n_provas = {c: len(d) for c, d in datas_curso.items()}
    vsub = vle[vle["frac"] <= thr]
    partes = []
    for curso, sub in vsub.groupby("curso"):
        if curso not in datas_curso: continue
        dts = datas_curso[curso]; day = sub["date"].values
        idx = np.searchsorted(dts, day, side="left"); ok = idx < len(dts)
        lead = np.full(len(day), -1); lead[ok] = dts[idx[ok]] - day[ok]
        inwin = ok & (lead >= 0) & (lead < W)
        if inwin.sum() == 0: continue
        partes.append(pd.DataFrame({"aluno": sub["aluno"].values[inwin], "prova": idx[inwin],
                                    "clicks": sub["sum_click"].values[inwin],
                                    "leadw": lead[inwin]*sub["sum_click"].values[inwin]}))
    pre = pd.concat(partes, ignore_index=True)
    pre = pre.groupby(["aluno","prova"]).agg(clicks=("clicks","sum"), leadw=("leadw","sum")).reset_index()
    piv = pre.pivot_table(index="aluno", columns="prova", values="clicks", fill_value=0)
    leadtot = pre.groupby("aluno")["leadw"].sum(); clicktot = pre.groupby("aluno")["clicks"].sum()
    volume = vsub.groupby("aluno")["sum_click"].sum()

    feats = {}
    for aluno in validos:
        n = n_provas.get(curso_de(aluno), 0)
        if n == 0:
            feats[aluno] = [0,0,0,0,0]; continue
        if aluno in piv.index:
            v = piv.loc[aluno, [i for i in range(n) if i in piv.columns]].reindex(range(n), fill_value=0).values.astype(float)
        else:
            v = np.zeros(n)
        s = v.sum()
        if s > 0:
            prop = v/s; x = np.arange(n)
            tend = np.polyfit(x, prop, 1)[0] if n >= 2 else 0.0
            conc = prop.std(); cobf = prop[-1]; durab = (v > 0).mean()
            ante = (leadtot.get(aluno,0)/clicktot.get(aluno,1))/max(W-1,1)
        else:
            tend=conc=cobf=durab=ante=0.0
        feats[aluno] = [tend, ante, durab, conc, cobf]
    F = pd.DataFrame.from_dict(feats, orient="index",
        columns=["tendencia","antecedencia","durabilidade","concentracao","cobertura_final"])
    F["volume"] = volume.reindex(F.index).fillna(0)
    return F.loc[val.index]

FEATS = {"FULL": features_5(1.0), "PARCIAL(70%)": features_5(0.70)}
CONJ = {"V (volume)": ["volume"],
        "F (5 features)": ["tendencia","antecedencia","durabilidade","concentracao","cobertura_final"],
        "V+F": ["volume","tendencia","antecedencia","durabilidade","concentracao","cobertura_final"]}

def build(kind):
    if kind == "GBoost":
        return HistGradientBoostingClassifier(random_state=0)
    return Pipeline([("sc", StandardScaler()),
                     ("mlp", MLPClassifier(hidden_layer_sizes=(32,16), max_iter=400,
                                           early_stopping=True, alpha=1e-3, random_state=0))])

def cv(X, y, kind, balancear=False):
    """CV manual; retorna métricas médias e predições out-of-fold."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    oof_prob = np.zeros(len(y)); oof_pred = np.zeros(len(y), dtype=int)
    aucs=[]; f1s=[]; rec=[]; prec=[]
    for tr, te in skf.split(X, y):
        clf = build(kind)
        sw = compute_sample_weight("balanced", y[tr]) if balancear else None
        if kind == "GBoost":
            clf.fit(X[tr], y[tr], sample_weight=sw)
        else:
            clf.fit(X[tr], y[tr], mlp__sample_weight=sw)
        p = clf.predict_proba(X[te])[:,1]; pred = (p>=0.5).astype(int)
        oof_prob[te]=p; oof_pred[te]=pred
        aucs.append(roc_auc_score(y[te],p)); f1s.append(f1_score(y[te],pred,average="macro"))
        rec.append(recall_score(y[te],pred,pos_label=0)); prec.append(precision_score(y[te],pred,pos_label=0,zero_division=0))
    return {"AUC":np.mean(aucs),"F1":np.mean(f1s),"recall_rep":np.mean(rec),"prec_rep":np.mean(prec),
            "oof_prob":oof_prob,"oof_pred":oof_pred}

# ===== PARTE A: as 5 features preveem? (V vs F vs V+F, sem balancear) =====
print("\n### PARTE A — as 5 features dos perfis preveem? ###")
linhasA=[]
for fr in FEATS:
    for cn, cols in CONJ.items():
        X = FEATS[fr][cols].values
        for kind in ["media(32,16)","GBoost"]:
            r = cv(X, y, kind)
            print(f"  {fr:12s} | {cn:14s} | {kind:12s} AUC={r['AUC']:.3f} F1={r['F1']:.3f}")
            linhasA.append({"janela":fr,"conjunto":cn,"modelo":kind,"AUC":round(r['AUC'],3),"F1":round(r['F1'],3)})
A = pd.DataFrame(linhasA); A.to_csv(os.path.join(OUT_DIR,"parteA_features.csv"), index=False)

# ===== PARTE B: desbalanceamento (V+F, sem peso vs balanced) =====
print("\n### PARTE B — desbalanceamento: sem peso vs balanced (V+F) ###")
linhasB=[]; oof_store={}
for fr in FEATS:
    X = FEATS[fr][CONJ["V+F"]].values
    for kind in ["media(32,16)","GBoost"]:
        for bal in [False, True]:
            r = cv(X, y, kind, balancear=bal)
            tag = "balanced" if bal else "sem_peso"
            print(f"  {fr:12s} | {kind:12s} | {tag:9s} recall_reprov={r['recall_rep']:.3f} "
                  f"prec_reprov={r['prec_rep']:.3f} F1={r['F1']:.3f} AUC={r['AUC']:.3f}")
            linhasB.append({"janela":fr,"modelo":kind,"peso":tag,
                            "recall_reprov":round(r['recall_rep'],3),"prec_reprov":round(r['prec_rep'],3),
                            "F1":round(r['F1'],3),"AUC":round(r['AUC'],3)})
            if kind=="media(32,16)": oof_store[(fr,tag)] = r
B = pd.DataFrame(linhasB); B.to_csv(os.path.join(OUT_DIR,"parteB_desbalanceamento.csv"), index=False)

# salva os modelos finais V+F (balanced) treinados na base toda
for fr in FEATS:
    X = FEATS[fr][CONJ["V+F"]].values
    sw = compute_sample_weight("balanced", y)
    m = build("media(32,16)"); m.fit(X, y, mlp__sample_weight=sw)
    joblib.dump(m, os.path.join(OUT_DIR, f"modelo_{fr.replace('%','').replace('(','').replace(')','')}_VF_balanced.joblib"))

# ---------- gráfico 1: as 5 features preveem (A vs F vs V+F) ----------
fig, axes = plt.subplots(1, 2, figsize=(13,5))
cores = {"V (volume)":"#90A4AE","F (5 features)":"#42A5F5","V+F":"#2E7D32"}
for ax, fr in zip(axes, FEATS):
    sub = A[(A["janela"]==fr)&(A["modelo"]=="media(32,16)")]
    ax.bar(sub["conjunto"], sub["AUC"], color=[cores[c] for c in sub["conjunto"]])
    ax.set_title(f"{fr} — MLP (32,16)"); ax.set_ylabel("AUC"); ax.set_ylim(0.5,1.0)
    for i,v in enumerate(sub["AUC"]): ax.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=9)
plt.suptitle("H3 — as 5 features dos perfis preveem o resultado?", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"01_features_preveem.png"), dpi=130); plt.close()

# ---------- gráfico 2: efeito do balanceamento no recall de reprovado ----------
fig, ax = plt.subplots(figsize=(10,5))
sub = B[B["modelo"]=="media(32,16)"].copy()
sub["rot"] = sub["janela"]+" / "+sub["peso"]
x = np.arange(len(sub)); w=0.35
ax.bar(x-w/2, sub["recall_reprov"], w, label="recall reprovado", color="#C62828")
ax.bar(x+w/2, sub["prec_reprov"], w, label="precision reprovado", color="#1565C0")
ax.set_xticks(x); ax.set_xticklabels(sub["rot"], rotation=15); ax.set_ylim(0,1)
ax.set_title("Classe REPROVADO: efeito do peso de classe (MLP)"); ax.legend()
for i,(rr,pp) in enumerate(zip(sub["recall_reprov"],sub["prec_reprov"])):
    ax.text(i-w/2, rr+0.01, f"{rr:.2f}", ha="center", fontsize=8)
    ax.text(i+w/2, pp+0.01, f"{pp:.2f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"02_balanceamento_recall.png"), dpi=130); plt.close()

# ---------- gráfico 3: matriz de confusão balanced (V+F) ----------
fig, axes = plt.subplots(1, 2, figsize=(11,5))
for ax, fr in zip(axes, FEATS):
    r = oof_store[(fr,"balanced")]; cm = confusion_matrix(y, r["oof_pred"])
    err_rep = cm[0,1]/cm[0].sum()*100; err_apr = cm[1,0]/cm[1].sum()*100
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["reprovou","aprovou"]); ax.set_yticklabels(["reprovou","aprovou"])
    ax.set_xlabel("previsto"); ax.set_ylabel("real")
    ax.set_title(f"{fr} balanced\nerro reprovou={err_rep:.0f}% erro aprovou={err_apr:.0f}%", fontsize=10)
    for i in range(2):
        for j in range(2):
            pct = cm[i,j]/cm[i].sum()*100
            ax.text(j,i,f"{cm[i,j]}\n({pct:.0f}%)",ha="center",va="center",
                    color="white" if cm[i,j]>cm.max()/2 else "black", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"03_confusao_balanced.png"), dpi=130); plt.close()

# ---------- gráfico 4: curva precision-recall da classe REPROVADO ----------
fig, ax = plt.subplots(figsize=(8,6))
for fr in FEATS:
    r = oof_store[(fr,"balanced")]
    y_rep = 1-y; prob_rep = 1-r["oof_prob"]
    pr, rc, _ = precision_recall_curve(y_rep, prob_rep)
    ax.plot(rc, pr, lw=2, label=f"{fr}")
ax.set_xlabel("recall (reprovados capturados)"); ax.set_ylabel("precision (alarmes corretos)")
ax.set_title("Trade-off real da classe REPROVADO (os dados decidem o limiar)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"04_precision_recall.png"), dpi=130); plt.close()

print("\n>>> PARTE A:"); print(A.to_string(index=False))
print("\n>>> PARTE B:"); print(B.to_string(index=False))
print("\n" + "="*70)
print(f"Gravado em {os.path.abspath(OUT_DIR)}:")
print("  - 01_features_preveem.png       (as 5 features preveem? V/F/V+F)")
print("  - 02_balanceamento_recall.png   (peso de classe melhora recall de reprovado)")
print("  - 03_confusao_balanced.png      (matriz com peso, % por classe)")
print("  - 04_precision_recall.png       (trade-off real; o dado decide o limiar)")
print("  - parteA_features.csv  parteB_desbalanceamento.csv  saida_terminal.txt")
sys.stdout = _orig; _log.close()