
import os
import csv
from tqdm import tqdm
import sys
from collections import defaultdict

csv.field_size_limit(2**20)  

MIN_REVIEWS = 500

data_dir = "data"                
preprocessed_dir = "preprocessed" 
os.makedirs(preprocessed_dir, exist_ok=True)

COLUMNS = [
    "review_id",
    "product_id",
    "product_title",
    "star_rating",
    "helpful_votes",
    "total_votes",
    "verified_purchase",
    "review_body"
]

for fname in os.listdir(data_dir):
    if not fname.lower().endswith(".tsv"):
        continue

    infile  = os.path.join(data_dir, fname)
    outfile = os.path.join(preprocessed_dir, f"preprocessed_{fname}")

    print(f"\nProcessing {fname!r} ...")

    counts = defaultdict(int)
    with open(infile, newline="", encoding="utf-8") as inf:
        reader = csv.DictReader(inf, delimiter="\t")
        for row in tqdm(reader, desc="  Counting reviews", leave=False):
            pid = row.get("product_id")
            if pid:
                counts[pid] += 1

    num_products = sum(1 for cnt in counts.values() if cnt >= MIN_REVIEWS)
    print(f"  → {num_products} products have ≥ {MIN_REVIEWS} reviews")

    with open(infile,  newline="", encoding="utf-8") as inf, \
         open(outfile, "w", newline="", encoding="utf-8") as outf:

        reader = csv.DictReader(inf, delimiter="\t")
        writer = csv.DictWriter(outf, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()

        for row in tqdm(reader, desc="  Writing filtered", leave=False):
            pid = row.get("product_id")
            if pid and counts[pid] >= MIN_REVIEWS:
                writer.writerow({col: row.get(col, "") for col in COLUMNS})

    print(f"  → Saved filtered file to {outfile!r}")

print("\nAll done!")  
