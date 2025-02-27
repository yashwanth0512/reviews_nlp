import csv
from collections import defaultdict
from tqdm import tqdm
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")
sid = SentimentIntensityAnalyzer()

personal_con = {
    "I", "me", "my", "mine", "we", "us", "our", "ours",
    "child", "kid", "kids", "son", "daughter", "grandson", "granddaughter",
    "birthday", "gift", "love", "loved", "elated", "mother", "father",
    "husband", "wife", "he", "she","they", 
}

def is_product_focused(sentence):
    tokens = sentence.lower().split()
    for word in personal_con:
        if word in tokens:
            return False
    return True

input_file = "D:/nlpproject/filtered_toys.tsv"
output_file = "D:/nlpproject/toys.txt"

products = defaultdict(lambda: {"title": "", "reviews": []})
with open(input_file, newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile, delimiter='\t')
    for row in tqdm(reader, desc="Grouping reviews by product"):
        prod_id = row["product_id"]
        products[prod_id]["title"] = row["product_title"]
        products[prod_id]["reviews"].append(row["review_body"])

product_summaries = {}
per_product_metrics = {}
for idx, (prod_id, data) in enumerate(tqdm(products.items(), desc="Processing products")):
    if idx >= 20:
        break

    positive_features = set()
    negative_features = set()
    product_title = data["title"]
    review_count = len(data["reviews"])

    for review in data["reviews"]:
        doc = nlp(review)
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            if len(sentence_text.split()) < 5:
                continue
            if not is_product_focused(sentence_text):
                continue
            sentiment = sid.polarity_scores(sentence_text)
            if sentiment['compound'] >= 0.05:
                positive_features.add(sentence_text)
            elif sentiment['compound'] <= -0.05:
                negative_features.add(sentence_text)

    product_summaries[prod_id] = {
        "title": product_title,
        "positive": list(positive_features),
        "negative": list(negative_features),
        "review_count": review_count
    }
    
    pos_count = len(positive_features)
    neg_count = len(negative_features)
    total_features = pos_count + neg_count
    unique_pos = len(set(positive_features))
    unique_neg = len(set(negative_features))

    if total_features > 0:
        sentiment_ratio = (pos_count - neg_count) / total_features
        diversity = (unique_pos + unique_neg) / total_features
        pos_percentage = (pos_count / total_features) * 100
        neg_percentage = (neg_count / total_features) * 100
    else:
        sentiment_ratio = 0
        diversity = 0
        pos_percentage = 0
        neg_percentage = 0

    per_product_metrics[prod_id] = {
        "review_count": review_count,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "percentage_positive": pos_percentage,
        "percentage_negative": neg_percentage,
    }

with open(output_file, "w", encoding="utf-8") as out:
    for prod_id, summary in product_summaries.items():
        metrics = per_product_metrics[prod_id]
        out.write(f"Product ID: {prod_id}\n")
        out.write(f"Product Title: {summary['title']}\n")
        out.write(f"Product Name: {summary['title']}\n\n")
        
        
        out.write("Positive Features:\n")
        if summary["positive"]:
            for feat in summary["positive"]:
                out.write(f"• \"{feat}\"\n")
        else:
            out.write("• No positive features detected.\n")
        
        out.write("\nNegative Features:\n")
        if summary["negative"]:
            for feat in summary["negative"]:
                out.write(f"• \"{feat}\"\n")
        else:
            out.write("• No negative features detected.\n")
        
        out.write("\nMetrics (Per Product):\n")
        out.write(f"Total Reviews: {metrics['review_count']}\n")
        out.write(f"Total Features: {metrics['positive_count'] + metrics['negative_count']}\n")
        out.write(f"Positive Features Count: {metrics['positive_count']}\n")
        out.write(f"Negative Features Count: {metrics['negative_count']}\n")
        out.write(f"Percentage Positive: {metrics['percentage_positive']:.2f}%\n")
        out.write(f"Percentage Negative: {metrics['percentage_negative']:.2f}%\n")
        out.write("\n" + "-" * 50 + "\n\n")

print("reviews of the first 20 products!")
