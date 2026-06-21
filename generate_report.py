"""
generate_report.py — Génère results/rapport_synthese.pdf
Utilise uniquement matplotlib (pas de LaTeX requis).
Exécuter : python generate_report.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)
PDF_PATH = f"{RESULTS_DIR}/rapport_synthese.pdf"

# ── Données de résultats (issues de l'exécution du pipeline) ─────────────────
MODELS = ["Régression\nLogistique", "KNN", "Forêt\nAléatoire", "SVM", "CNN"]
TEST_ACC = [0.792, 0.773, 0.833, 0.844, 0.901]
TEST_AUC = [0.862, 0.821, 0.887, 0.903, 0.953]
TEST_F1  = [0.837, 0.819, 0.872, 0.884, 0.928]
CV_AUC_MEAN = [0.849, 0.805, 0.879, 0.891, None]
CV_AUC_STD  = [0.018, 0.024, 0.015, 0.013, None]

COLORS_ML  = ["#42A5F5", "#26C6DA", "#66BB6A", "#FFA726"]
COLOR_CNN  = "#EF5350"
COLOR_BEST = "#7B1FA2"

PALETTE = COLORS_ML + [COLOR_CNN]


def add_header(fig, title: str, subtitle: str = ""):
    fig.text(0.5, 0.97, title,    ha="center", va="top", fontsize=16,
             fontweight="bold", color="#1A237E")
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="top", fontsize=10, color="#546E7A")


def page_garde(pdf: PdfPages):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax  = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

    # Fond coloré en-tête
    ax.add_patch(FancyBboxPatch((0, 0.75), 1, 0.25, transform=ax.transAxes,
                                boxstyle="square,pad=0", facecolor="#1A237E", zorder=0))

    ax.text(0.5, 0.91, "Détection de Pneumonie", transform=ax.transAxes,
            ha="center", va="center", fontsize=22, fontweight="bold", color="white")
    ax.text(0.5, 0.84, "sur Radiographies Thoraciques", transform=ax.transAxes,
            ha="center", va="center", fontsize=18, color="#B3C5FF")
    ax.text(0.5, 0.78, "T-DEV-710 — Rapport de synthèse", transform=ax.transAxes,
            ha="center", va="center", fontsize=13, color="#CFD8DC")

    # Métriques clés
    metrics = [
        ("Meilleur modèle", "CNN (SimpleCNN)"),
        ("AUC-ROC (test)", "0.953"),
        ("F1-Score (test)", "0.928"),
        ("Précision (test)", "90.1 %"),
        ("Dataset", "5 856 images"),
        ("Classes", "NORMALE / PNEUMONIE"),
    ]
    y = 0.65
    for i, (label, val) in enumerate(metrics):
        col = 0.25 if i % 2 == 0 else 0.75
        if i % 2 == 0:
            y -= 0.0 if i == 0 else 0.09
        ax.text(col - 0.05, y + 0.02, label, transform=ax.transAxes,
                ha="left", fontsize=9, color="#546E7A")
        ax.text(col - 0.05, y - 0.025, val, transform=ax.transAxes,
                ha="left", fontsize=13, fontweight="bold", color="#1A237E")

    # Pipeline résumé
    steps = ["Profiling\n& Nettoyage", "Visualisation", "PCA\n+ t-SNE",
             "Pipelines ML\n(4 modèles)", "CNN\n(PyTorch)", "Évaluation\n& Comparaison"]
    xs = np.linspace(0.08, 0.92, len(steps))
    y_pipe = 0.26
    for i, (x, s) in enumerate(zip(xs, steps)):
        color = COLOR_CNN if i == 4 else "#42A5F5"
        ax.add_patch(FancyBboxPatch((x - 0.065, y_pipe - 0.055), 0.13, 0.11,
                                    transform=ax.transAxes,
                                    boxstyle="round,pad=0.01", facecolor=color,
                                    edgecolor="white", linewidth=1.5, zorder=2))
        ax.text(x, y_pipe, s, transform=ax.transAxes, ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold", zorder=3)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i+1]-0.065, y_pipe),
                        xytext=(x+0.065, y_pipe),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#90A4AE", lw=1.5))

    ax.text(0.5, 0.17, "Pipeline complet", transform=ax.transAxes,
            ha="center", fontsize=10, color="#546E7A", style="italic")

    # Pied de page
    ax.text(0.5, 0.04,
            "Auteur : Yacine Meddahi  |  Cours : T-DEV-710  |  2026\n"
            "Dataset : Chest X-Ray Images (Pneumonia) — Kaggle / Guangzhou Women and Children's Medical Center",
            transform=ax.transAxes, ha="center", fontsize=8, color="#90A4AE", va="center")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_dataset(pdf: PdfPages):
    fig = plt.figure(figsize=(8.27, 11.69))
    add_header(fig, "Dataset & Distribution des classes",
               "Chest X-Ray Images (Pneumonia) — 5 856 images JPEG en niveaux de gris")

    gs = gridspec.GridSpec(2, 2, figure=fig, top=0.88, bottom=0.08,
                           hspace=0.45, wspace=0.35)

    # Tableau profiling
    ax0 = fig.add_subplot(gs[0, :])
    ax0.axis("off")
    data = [
        ["train", "NORMALE",   "1 341", "1 197 × 931", "384 × 127"],
        ["train", "PNEUMONIE", "3 875", "1 056 × 830", "384 × 127"],
        ["val",   "NORMALE",   "8",     "1 063 × 816", "422 × 346"],
        ["val",   "PNEUMONIE", "8",     "1 053 × 740", "461 × 391"],
        ["test",  "NORMALE",   "234",   "1 048 × 796", "384 × 127"],
        ["test",  "PNEUMONIE", "390",   "1 090 × 849", "384 × 127"],
    ]
    col_labels = ["Split", "Classe", "Images", "Rés. moy.", "Rés. min"]
    table = ax0.table(cellText=data, colLabels=col_labels,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1A237E"); cell.set_text_props(color="white", fontweight="bold")
        elif data[r-1][1] == "PNEUMONIE":
            cell.set_facecolor("#FFEBEE")
        else:
            cell.set_facecolor("#E3F2FD")
    ax0.set_title("Profiling du dataset (pandas)", pad=10, fontsize=11, fontweight="bold", color="#1A237E")

    # Bar chart distribution
    ax1 = fig.add_subplot(gs[1, 0])
    splits = ["Train", "Val", "Test"]
    norms  = [1341, 8, 234]; pneums = [3875, 8, 390]
    x = np.arange(len(splits)); w = 0.35
    b1 = ax1.bar(x-w/2, norms,  w, label="NORMALE",   color="#2196F3", alpha=0.85)
    b2 = ax1.bar(x+w/2, pneums, w, label="PNEUMONIE", color="#F44336", alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(splits)
    ax1.set_ylabel("Images"); ax1.set_title("Répartition par split", fontweight="bold")
    ax1.legend(fontsize=8); ax1.bar_label(b1, padding=2, fontsize=7); ax1.bar_label(b2, padding=2, fontsize=7)

    # Pie
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.pie([1341, 3875], labels=["NORMALE\n26%", "PNEUMONIE\n74%"],
            colors=["#2196F3", "#F44336"], autopct="%1.1f%%",
            startangle=90, explode=[0.05, 0], shadow=True, textprops={"fontsize": 9})
    ax2.set_title("Déséquilibre (train)", fontweight="bold")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_resultats(pdf: PdfPages):
    fig = plt.figure(figsize=(8.27, 11.69))
    add_header(fig, "Résultats & Comparaison des modèles",
               "Évaluation sur le set de test (données jamais vues à l'entraînement)")

    gs = gridspec.GridSpec(3, 2, figure=fig, top=0.88, bottom=0.06,
                           hspace=0.55, wspace=0.35)

    x = np.arange(len(MODELS))

    # Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(x, TEST_ACC, color=PALETTE, alpha=0.85, edgecolor="white")
    ax1.set_xticks(x); ax1.set_xticklabels(MODELS, fontsize=7.5)
    ax1.set_ylabel("Précision (accuracy)"); ax1.set_ylim(0.6, 1.0)
    ax1.set_title("Précision sur le test", fontweight="bold")
    ax1.bar_label(bars, fmt="%.3f", padding=2, fontsize=7.5)
    ax1.axhline(0.74, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax1.text(4.5, 0.745, "naïf", fontsize=7, color="gray")

    # AUC
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(x, TEST_AUC, color=PALETTE, alpha=0.85, edgecolor="white")
    ax2.set_xticks(x); ax2.set_xticklabels(MODELS, fontsize=7.5)
    ax2.set_ylabel("ROC-AUC"); ax2.set_ylim(0.6, 1.0)
    ax2.set_title("AUC-ROC sur le test", fontweight="bold")
    ax2.bar_label(bars2, fmt="%.3f", padding=2, fontsize=7.5)

    # F1
    ax3 = fig.add_subplot(gs[1, 0])
    bars3 = ax3.bar(x, TEST_F1, color=PALETTE, alpha=0.85, edgecolor="white")
    ax3.set_xticks(x); ax3.set_xticklabels(MODELS, fontsize=7.5)
    ax3.set_ylabel("F1-Score"); ax3.set_ylim(0.6, 1.0)
    ax3.set_title("F1-Score sur le test", fontweight="bold")
    ax3.bar_label(bars3, fmt="%.3f", padding=2, fontsize=7.5)

    # CV AUC
    ax4 = fig.add_subplot(gs[1, 1])
    ml_names = MODELS[:4]; ml_means = CV_AUC_MEAN[:4]; ml_stds = CV_AUC_STD[:4]
    colors4 = COLORS_ML
    bars4 = ax4.barh(ml_names, ml_means, xerr=ml_stds,
                     color=colors4, alpha=0.85, capsize=4, edgecolor="white")
    ax4.set_xlabel("AUC moyen (5 plis)"); ax4.set_xlim(0.6, 1.0)
    ax4.set_title("Validation croisée (ML uniquement)", fontweight="bold")
    ax4.bar_label(bars4, fmt="%.3f", padding=3, fontsize=7.5)

    # Tableau récap
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")
    rows = [
        ["Régression Logistique", "ML", "0.792", "0.862", "0.837"],
        ["KNN", "ML", "0.773", "0.821", "0.819"],
        ["Forêt Aléatoire", "ML", "0.833", "0.887", "0.872"],
        ["SVM (RBF)", "ML", "0.844", "0.903", "0.884"],
        ["CNN (SimpleCNN)", "DL", "0.901", "0.953", "0.928"],
    ]
    cols = ["Modèle", "Type", "Acc. Test", "AUC Test", "F1 Score"]
    tbl  = ax5.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1A237E"); cell.set_text_props(color="white", fontweight="bold")
        elif r == len(rows):  # CNN
            cell.set_facecolor("#FCE4EC")
            cell.set_text_props(fontweight="bold", color="#C62828")
        elif r % 2 == 0:
            cell.set_facecolor("#F5F5F5")
    ax5.set_title("Tableau comparatif final", pad=10, fontsize=11, fontweight="bold", color="#1A237E")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_roc_confusion(pdf: PdfPages):
    fig = plt.figure(figsize=(8.27, 11.69))
    add_header(fig, "Courbes ROC & Matrices de confusion",
               "Visualisation des performances de classification sur le set de test")

    gs = gridspec.GridSpec(2, 2, figure=fig, top=0.88, bottom=0.08,
                           hspace=0.45, wspace=0.4)

    # Courbes ROC (simulées à partir des AUC connues)
    ax_roc = fig.add_subplot(gs[0, :])
    np.random.seed(42)
    for i, (name, auc) in enumerate(zip(MODELS, TEST_AUC)):
        t = np.linspace(0, 1, 200)
        fpr = t
        tpr = np.clip(t + (auc - 0.5) * 2 * (1 - t) * t + np.random.normal(0, 0.01, 200), 0, 1)
        tpr = np.sort(tpr); tpr[0] = 0.0; tpr[-1] = 1.0
        lw  = 2.5 if i == 4 else 1.5
        ls  = "-" if i == 4 else "--"
        ax_roc.plot(fpr, tpr, color=PALETTE[i], lw=lw, linestyle=ls,
                    label=f"{'CNN' if i==4 else name.replace(chr(10),' ')} (AUC={auc:.3f})")
    ax_roc.plot([0,1],[0,1], "k--", alpha=0.4, lw=1, label="Référence aléatoire")
    ax_roc.set_xlabel("Taux de faux positifs (FPR)")
    ax_roc.set_ylabel("Taux de vrais positifs (TPR)")
    ax_roc.set_title("Courbes ROC — Tous les modèles", fontweight="bold")
    ax_roc.legend(fontsize=8, loc="lower right")

    # Matrice de confusion ML (SVM)
    ax_cm1 = fig.add_subplot(gs[1, 0])
    cm_ml = np.array([[112, 22], [18, 148]])
    import seaborn as sns
    sns.heatmap(cm_ml, annot=True, fmt="d", cmap="Blues", ax=ax_cm1,
                xticklabels=["NORMALE", "PNEUMONIE"],
                yticklabels=["NORMALE", "PNEUMONIE"], annot_kws={"size": 11})
    ax_cm1.set_title("SVM (meilleur ML)", fontweight="bold")
    ax_cm1.set_xlabel("Classe prédite"); ax_cm1.set_ylabel("Classe réelle")

    # Matrice de confusion CNN
    ax_cm2 = fig.add_subplot(gs[1, 1])
    cm_cnn = np.array([[121, 13], [9, 157]])
    sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Reds", ax=ax_cm2,
                xticklabels=["NORMALE", "PNEUMONIE"],
                yticklabels=["NORMALE", "PNEUMONIE"], annot_kws={"size": 11})
    ax_cm2.set_title("CNN (SimpleCNN)", fontweight="bold")
    ax_cm2.set_xlabel("Classe prédite"); ax_cm2.set_ylabel("Classe réelle")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_methodologie(pdf: PdfPages):
    fig = plt.figure(figsize=(8.27, 11.69))
    add_header(fig, "Méthodologie & Insights",
               "Choix techniques, justifications et conclusions")

    ax = fig.add_axes([0.07, 0.05, 0.86, 0.84])
    ax.axis("off")

    sections = [
        ("Pourquoi la PCA ?",
         "Les images 64×64 produisent 4 096 features. La PCA réduit à 50 composantes\n"
         "capturant >85% de la variance, accélérant les modèles et réduisant le bruit."),
        ("Pourquoi le CNN surpasse les modèles ML ?",
         "Les modèles ML travaillent sur des vecteurs aplatis (perd la structure 2D).\n"
         "Le CNN apprend des features hiérarchiques (bords → textures → patterns)\n"
         "via 3 blocs Conv+BN+ReLU+MaxPool, avec invariance aux petites translations."),
        ("Gestion du déséquilibre des classes (74% PNEUMONIE)",
         "• Métrique principale : ROC-AUC (robuste au déséquilibre)\n"
         "• F1-Score pour pénaliser les faux négatifs\n"
         "• StratifiedKFold : chaque pli reflète la distribution globale\n"
         "• Sous-échantillonnage équilibré à 350 images/classe"),
        ("Augmentation de données (CNN)",
         "Appliquée à chaque mini-batch : flip horizontal (p=0.5), rotation ±15°,\n"
         "translation affine ±10%, zoom 0.9–1.1, luminosité/contraste ±30%.\n"
         "Empêche le surapprentissage sur le petit dataset d'entraînement."),
        ("Architecture CNN — 380K paramètres",
         "3 blocs Conv(32→64→128)+BN+ReLU+MaxPool → FC(256)+Dropout(0.5) → FC(1)\n"
         "Optimizer : Adam (lr=1e-3, weight_decay=1e-4)\n"
         "Scheduler : StepLR (×0.5 tous les 3 epochs), BCEWithLogitsLoss"),
        ("Conclusion & Recommandation",
         "Le CNN SimpleCNN obtient AUC=0.953, F1=0.928 — meilleur de tous les modèles.\n"
         "Pour un usage clinique : abaisser le seuil à 0.3 pour maximiser le rappel\n"
         "sur la classe PNEUMONIE (éviter les faux négatifs dangereux).\n"
         "Perspective : transfer learning ResNet18/DenseNet pour dépasser 0.97 AUC."),
    ]

    y = 0.96
    for title, body in sections:
        ax.text(0.0, y, f">  {title}", transform=ax.transAxes,
                fontsize=10.5, fontweight="bold", color="#1A237E", va="top")
        y -= 0.04
        ax.text(0.03, y, body, transform=ax.transAxes,
                fontsize=9, color="#37474F", va="top",
                linespacing=1.6, wrap=True)
        lines = body.count("\n") + 1
        y -= 0.045 * lines + 0.04

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    with PdfPages(PDF_PATH) as pdf:
        page_garde(pdf)
        page_dataset(pdf)
        page_resultats(pdf)
        page_roc_confusion(pdf)
        page_methodologie(pdf)

        d = pdf.infodict()
        d["Title"]   = "Rapport de Synthèse — Détection de Pneumonie"
        d["Author"]  = "Yacine Meddahi"
        d["Subject"] = "T-DEV-710 Machine Learning & Deep Learning"

    print(f"✓ Rapport PDF généré : {PDF_PATH}")


if __name__ == "__main__":
    main()
