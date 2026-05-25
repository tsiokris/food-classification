import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
DATASET_CSV = "data/annotations/food_classification_dataset.csv"


def plot_training_curves(metrics_path, save_dir):
    df = pd.read_csv(metrics_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(df["epoch"], df["train_loss"], label="Train")
    ax1.plot(df["epoch"], df["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss over epochs")
    ax1.legend()

    ax2.plot(df["epoch"], df["train_acc"], label="Train")
    ax2.plot(df["epoch"], df["val_acc"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy over epochs")
    ax2.legend()

    fig.tight_layout()
    out_path = os.path.join(save_dir, "training_curves.png")
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
    plot_training_curves(os.path.join(RESULTS_DIR, "metrics.csv"), RESULTS_DIR)
    plot_class_distribution(DATASET_CSV, RESULTS_DIR)
