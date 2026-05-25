import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
DATASET_CSV = "data/annotations/food_classification_dataset.csv"


def plot_training_curves(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "*_metrics.csv")))
    if not files:
        print("No metrics CSVs found in results/")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for path in files:
        backbone = os.path.basename(path).replace("_metrics.csv", "")
        df = pd.read_csv(path)
        ax1.plot(df["epoch"], df["train_loss"], linestyle="--", label=f"{backbone} train")
        ax1.plot(df["epoch"], df["val_loss"],   linestyle="-",  label=f"{backbone} val")
        ax2.plot(df["epoch"], df["train_acc"],  linestyle="--", label=f"{backbone} train")
        ax2.plot(df["epoch"], df["val_acc"],    linestyle="-",  label=f"{backbone} val")

    for ax, ylabel, title in [
        (ax1, "Loss",     "Loss over epochs"),
        (ax2, "Accuracy", "Accuracy over epochs"),
    ]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)

    fig.tight_layout()
    out_path = os.path.join(results_dir, "training_curves.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_class_distribution(dataset_csv, save_dir):
    df = pd.read_csv(dataset_csv)
    counts = df["food_label"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(16, 6))
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    ax.set_title("Class distribution")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()

    out_path = os.path.join(save_dir, "class_distribution.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_training_curves(RESULTS_DIR)
    plot_class_distribution(DATASET_CSV, RESULTS_DIR)
