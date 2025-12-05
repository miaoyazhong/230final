import pandas as pd
from sklearn.model_selection import train_test_split
from data_load import bucketize_view_count
import csv

def write_train_test_csv(original_csv_path, train_csv_path, test_csv_path, train_frac=0.96, random_state=42):
    df = pd.read_csv(original_csv_path,
                     header=0,
                     names=["views", "image_path", "title"],
                     quotechar='"',
                     escapechar='\\',
                     engine="python")
    df["bucket"] = df["views"].apply(bucketize_view_count).astype("int32")
    indices = list(range(len(df)))
    train_idx, test_idx = train_test_split(indices, train_size=train_frac, stratify=df["bucket"].tolist(), random_state=random_state)
    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]

    # Save to disk with all fields quoted, same quote/escape chars as your reader
    # Use header=False if the original CSV had no header row (you read with names=...).
    df_train.to_csv(train_csv_path,
                    index=False,
                    header=True,
                    quoting=csv.QUOTE_ALL,
                    quotechar='"',
                    escapechar='\\')
    df_test.to_csv(test_csv_path,
                   index=False,
                   header=True,
                   quoting=csv.QUOTE_ALL,
                   quotechar='"',
                   escapechar='\\')

    print(f"Wrote {len(df_train)} train rows to {train_csv_path}")
    print(f"Wrote {len(df_test)} test rows to {test_csv_path}")

if __name__ == "__main__":
    write_train_test_csv("data/clean_dataset.csv", "data/train_split.csv", "data/test_split.csv")