import os
import shutil
import pandas as pd

def copy_viral_covers(csv_path, output_folder):

    df = pd.read_csv(csv_path, header=None)

    # Delete output folder if it exists, then recreate it
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)        # delete folder and its contents
    os.makedirs(output_folder, exist_ok=True)

    for idx, row in df.iterrows():
        cover_path = "data/" + row[0]  # only column: path to cover image

        if os.path.exists(cover_path):
            # extract filename from the path
            filename = os.path.basename(cover_path)

            dest_path = os.path.join(output_folder, filename)
            shutil.copy(cover_path, dest_path)

            print(f"Copied: {cover_path} → {dest_path}")
        else:
            print(f"❌ Path not found: {cover_path}")

# --- CONFIG ---
csv_path = "data/viral_cover.csv"      # path to your CSV file
output_folder = "dataset/viral"         # folder where covers will be copied
copy_viral_covers(csv_path, output_folder)

csv_path = "data/no_viral_cover.csv"      # path to your CSV file
output_folder = "dataset/no_viral"         # folder where covers will be copied
copy_viral_covers(csv_path, output_folder)