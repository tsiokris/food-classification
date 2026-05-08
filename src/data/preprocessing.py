import pandas as pd

def build_classification_dataset(labels_path: str, splits_path: str, dataset_path: str) -> pd.DataFrame:
    """Merge labels into a single food label and splits CSVs into a classification dataset and save to dataset_path.

    Args:
        labels_path: path to the metadata csv. Must contain columns: food_label_1, food_label_2, food_label_3, images
        splits_path: path to the csv containing the splits. Must contain columns: images, split
        dataset_path: path for the classification dataset csv to be saved.
    Returns:
        pd.DataFrame of the classification dataset
    """

    def combine_labels(row):
        """combines multiple food labels into a single string to be used as a single label."""
        
        labels = [row["food_label_1"], row["food_label_2"], row["food_label_3"]]
        labels = [str(label).strip() for label in labels if pd.notna(label)]
        return "_".join(labels)


    labels_df = pd.read_csv(labels_path)
    splits_df = pd.read_csv(splits_path)
    df = pd.merge(labels_df, splits_df, on='images')
    df["food_label"] = df.apply(combine_labels, axis=1)
    final_df = df[["images", "split", "food_label"]]
    final_df.to_csv(dataset_path, index=False)

    return final_df