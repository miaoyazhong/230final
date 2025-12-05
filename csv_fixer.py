import csv

def fix_csv(input_path, output_path):
    fixed_rows = []

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in list(reader):
            if not row:
                continue

            # If row has more than 3 columns, join extras into title
            if len(row) > 3:
                views = row[0].strip()
                image_path = row[1].strip()

                # Everything else belongs to the title
                title_parts = row[2:]
                title = ",".join([x.strip() for x in title_parts])

                fixed_rows.append([views, image_path, title])

            elif len(row) == 3:
                # Already correct
                fixed_rows.append([row[0].strip(), row[1].strip(), row[2].strip()])

            else:
                # Broken row fallback
                print("Warning: Skipping malformed row:", row)


    # Write fixed CSV with proper quoting
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["views", "image_path", "title"])   # header
        writer.writerows(fixed_rows)

    print(f"Fixed CSV saved to: {output_path}")
    print(f"Rows fixed: {len(fixed_rows)}")


if __name__ == "__main__":
    fix_csv("data/bilibili_raw.csv", "data/clean_dataset.csv")
