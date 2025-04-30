import csv
from collections import defaultdict
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--infile', type=str, required=True, help='Input File (file.tsv)')
parser.add_argument('--outfile', type=str, required=True, help='Output File (file.tsv)')
args = parser.parse_args()
input = args.infile
output = args.outfile

# input and output files
# input = "data/filtered/filtered_multi.tsv"
# output = "data/counts/multi_ratings.tsv"

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
    for id, count in sorted(counts.items(), key=lambda x: x[1], reverse=False):
        if count >= 500:
            average_rating = ratings[id] / count
            writer.writerow([id, count, average_rating])

