import csv
from tqdm import tqdm
import sys
from collections import defaultdict

# allow large files
csv.field_size_limit(sys.maxsize)

# number of reviews required
MIN_REVIEWS = 500

# input and output files
infile = "data/amazon_reviews_us_Electronics_v1_00.tsv"
outfile = "data/filtered/filtered_ electronics.tsv"

# columns to keep
columns = ["review_id", "product_id", "product_title", "star_rating", "helpful_votes", "total_votes", "verified_purchase", "review_body"]


# create counts for all product ids
counts = defaultdict(int)
with open(infile, newline='', encoding='utf-8') as input_file:
    reader = csv.DictReader(input_file, delimiter='\t')
    for row in tqdm(reader, desc="counting"):
        counts[row["product_id"]] += 1


# write the filtered data to the output file
with open(infile, newline='', encoding='utf-8') as input, open(outfile, mode='w', newline='', encoding='utf-8') as output:
    reader = csv.DictReader(input, delimiter='\t')
    writer = csv.DictWriter(output, fieldnames=columns, delimiter='\t')
    
    writer.writeheader()
    # for i, row in enumerate(tqdm(reader, desc="writing", total=len(list(reader)))):
    for row in tqdm(reader, desc="writing"):
        # only keep reviews where the product has at least 500 reviews
        if counts[row["product_id"]] >= MIN_REVIEWS:
            # only keep the columns previously stated 
            writer.writerow({col: row[col] for col in columns})
