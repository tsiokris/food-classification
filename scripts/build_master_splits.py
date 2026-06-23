import pandas as pd


def build_master_splits(output_path="data/annotations/master_splits.csv"):
    frames = []

    for fold in range(3):
        df = pd.read_csv(f"cnn_splits/fold_{fold}.csv")
        chunk = df[["image_path", "source", "label", "split"]].copy()
        chunk["fold_split"] = f"fold_{fold}_" + chunk["split"]
        frames.append(chunk)

    holdout = pd.read_csv("cnn_splits/holdout_final.csv")
    chunk = holdout[["image_path", "source", "label"]].copy()
    chunk["split"] = "holdout_final"
    chunk["fold_split"] = "holdout_final"
    frames.append(chunk)

    master = pd.concat(frames, ignore_index=True)
    master.to_csv(output_path, index=False)

    print(f"Saved {len(master)} rows to {output_path}")
    print(master["fold_split"].value_counts().sort_index().to_string())
    return master


if __name__ == "__main__":
    build_master_splits()
