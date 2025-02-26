import csv
from collections import defaultdict
from tqdm import tqdm

# input and output files
input = "data/filtered/filtered_toy.tsv"
output = "data/counts/toys_ratings.tsv"

# create counts for all product ids
counts = defaultdict(int)
ratings = defaultdict(float)
with open(input, newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in tqdm(reader, desc="counting"):
        counts[row["product_id"]] += 1
        ratings[row["product_id"]] += float(row["star_rating"])

# write the counts out to a file in sorted order (most first)
with open(output, mode="w", newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(["product_id", "count", "avg_rating"]) 
    # for id, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    #     writer.writerow([id, count])
    for id, count in sorted(counts.items(), key=lambda x: (ratings[x[0]] / x[1], x[1]), reverse=False):
        if count >= 500:
            average_rating = ratings[id] / count
            writer.writerow([id, count, average_rating])

