import csv
from collections import defaultdict
from tqdm import tqdm

# input and output files
input = "data/filtered/filtered_toys.tsv"
output = "data/counts/toys_sorted.tsv"

# create counts for all product ids
counts = defaultdict(int)
with open(input, newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in tqdm(reader, desc="counting"):
        counts[row["product_id"]] += 1

# write the counts out to a file in sorted order (most first)
with open(output, mode="w", newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(["product_id", "count"]) 
    for id, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        writer.writerow([id, count])

