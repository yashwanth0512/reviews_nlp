'''
Sentiment Analysis baseline on text (focusing on reviews).
Code sourced from: https://www.kaggle.com/code/justingunderson/customer-reviews-sentiment-analysis-nlp/notebook#SECTION-2:


'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


plt.style.use('ggplot')

import nltk

# DOWNLOAD VADER
from nltk.sentiment import SentimentIntensityAnalyzer

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

# scipy for outputs // smooth out between 0 - 1
from scipy.special import softmax

from transformers import pipeline

# # only need to download these ONCE
# # needed for tokenization
# nltk.download('punkt_tab')
# # needed for tagging
# nltk.download('averaged_perceptron_tagger_eng')
# # needed for chunking
# nltk.download('maxent_ne_chunker_tab')
# nltk.download('words')

# # needed for SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')

# Analyze the reviews for only one product
PRODUCT_ID = 'B004S8F7QM'

# load the file in
# df = pd.read_csv('data/filtered/filtered_toy.tsv',sep='\t')
df = pd.read_csv(f'data/extract/toy_{PRODUCT_ID}.tsv',sep='\t')
# convert to text to process 
df['review_body'] = df['review_body'].astype(str)

sia = SentimentIntensityAnalyzer()

sent_pipeline = pipeline("sentiment-analysis")
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

def polarity_scores_sent_pipeline(example):
    result = sent_pipeline(example)[0]
    return {
        'sent_pipeline_label': result['label'],
        'sent_pipeline_score': result['score']
    }

def process_long_text(text, analyzer):
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + 512] for i in range(0, len(tokens), 512)]
    parts = [" ".join(chunk) for chunk in chunks]

    aggregated_scores = {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0, 'count': 0}

    for p in parts:
        if analyzer == 'vader':
            scores = sia.polarity_scores(p)
        elif analyzer == 'roberta':
            scores = polarity_scores_roberta(p)
        else:
            raise ValueError("Unknown analyzer")

        aggregated_scores['compound'] += scores['compound']
        aggregated_scores['pos'] += scores['pos']
        aggregated_scores['neu'] += scores['neu']
        aggregated_scores['neg'] += scores['neg']
        aggregated_scores['count'] += 1

    final_scores = {k: v / aggregated_scores['count'] for k, v in aggregated_scores.items() if k != 'count'}
    return final_scores

res = {}
for i, row in df.iterrows():
    if (row['product_id'] == PRODUCT_ID):
        try:
            text = row['review_body']
            myid = row['review_id']
            vader_result = sia.polarity_scores(text)
            vader_result_rename = {}
            for key, value in vader_result.items():
                vader_result_rename[f"vader_{key}"] = value
            roberta_result = polarity_scores_roberta(text)
            sent_pipeline_result = polarity_scores_sent_pipeline(text)
            
            # combine all result dictionaries
            # results = {**vader_result_rename, **roberta_result}
            results = {**vader_result_rename, **roberta_result, **sent_pipeline_result}
            res[myid] = results
        except RuntimeError as e:
            print(f'Broke for id {myid}')
            print(e, '\n')


results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'review_id'})
results_df = results_df.merge(df, how='left')

# print(results_df.columns)
# print(results_df.mean())

numeric_cols = ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound',
                'roberta_neg', 'roberta_neu', 'roberta_pos', 'sent_pipeline_score']
# results_df[numeric_cols] = results_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# print(results_df.columns)
# Calculate average scores
avg_scores = results_df[numeric_cols].mean()
print(f"\nFor {PRODUCT_ID} ({df.shape[0]} reviews):\n")
print("Average Scores:")
print(avg_scores)
print(results_df["sent_pipeline_label"].mode()[0])
print()

print("Average Star Rating:")
real_comp = results_df["star_rating"].mean() / 5
print("  mean: ", real_comp)
real_neg = results_df['star_rating'].isin([1, 2]).sum() / results_df['star_rating'].count()
real_neu = results_df['star_rating'].isin([3]).sum() / results_df['star_rating'].count()
real_pos = results_df['star_rating'].isin([4, 5]).sum() / results_df['star_rating'].count()
print("  1/2 stars (neg): ", real_neg)
print("  3 stars (neu): ", real_neu)
print("  4/5 stars (pos: ", real_pos)
print()

# calc error percentages
def calc_error(real, observed):
    return abs(observed - real) / real

# calc error percentages for VADER
vader_neg_error = calc_error(real_neg, avg_scores['vader_neg'])
vader_neu_error = calc_error(real_neu, avg_scores['vader_neu'])
vader_pos_error = calc_error(real_pos, avg_scores['vader_pos'])
vader_comp_error = calc_error(real_comp, avg_scores['vader_compound'])

# calc error percentages for Roberta
roberta_neg_error = calc_error(real_neg, avg_scores['roberta_neg']) 
roberta_neu_error = calc_error(real_neu, avg_scores['roberta_neu']) 
roberta_pos_error = calc_error(real_pos, avg_scores['roberta_pos'])

if results_df["sent_pipeline_label"].mode()[0] == 'NEGATIVE':
    sent_error = calc_error(real_comp, 1-avg_scores['sent_pipeline_score'])
else:
    sent_error = calc_error(real_comp, avg_scores['sent_pipeline_score'])

# print error %
print("Error Percentages:")
print(f"VADER:\n  Positive: {vader_pos_error * 100:.2f}%, Neutral: {vader_neu_error * 100:.2f}%, Negative: {vader_neg_error * 100:.2f}%, Compound: {vader_comp_error * 100:.2f}%")
print(f"Roberta:\n  Positive: {roberta_pos_error * 100:.2f}%, Neutral: {roberta_neu_error * 100:.2f}%, Negative: {roberta_neg_error * 100:.2f}%")
print(f"Sent Pipeline Score:\n  {sent_error * 100:.2f}%")
print(f"Sent Pipeline Label:\n  {(real_comp >= .5 and results_df["sent_pipeline_label"].mode()[0] == 'POSITIVE')or (real_comp <= .5 and results_df["sent_pipeline_label"].mode()[0] == 'NEGATIVE')}")


# sns.pairplot(data=results_df,
#              vars=['vader_neg', 'vader_neu', 'vader_pos',
#                    'roberta_neg', 'roberta_neu', 'roberta_pos',
#                    'sent_pipeline_score'],
#              hue='star_rating',
#              palette='tab10')
# plt.show()

# # Query: Roberta highest positive text with lowest possible star rating
# high1 = results_df.query('star_rating == 1') \
#     .sort_values('roberta_pos', ascending=False)[['review_body', 'roberta_pos', 'roberta_neg']].head(5)

# print('Roberta High 1 stars:')
# for idx, row in high1.iterrows():
#     print(f"Review: {row['review_body']}\nRoberta Positive: {row['roberta_pos']}\nRoberta Negative: {row['roberta_neg']}\n")

# # Query: Roberta lowest negative text with highest possible star rating
# low5 = results_df.query('star_rating == 5') \
#     .sort_values('roberta_neg', ascending=False)[['review_body', 'roberta_pos', 'roberta_neg']].head(5)

# print('Roberta Low 5 stars:')
# for idx, row in low5.iterrows():
#     print(f"Review: {row['review_body']}\nRoberta Positive: {row['roberta_pos']}\nRoberta Negative: {row['roberta_neg']}\n")
# print()


# # Query: VADER highest positive text with lowest possible star rating
# high1 = results_df.query('star_rating == 1') \
#     .sort_values('vader_pos', ascending=False)[['review_body', 'vader_pos', 'vader_neg']].head(5)

# print('VADER High 1 stars:')
# for idx, row in high1.iterrows():
#     print(f"Review: {row['review_body']}\nVADER Positive: {row['vader_pos']}\nVADER Negative: {row['vader_neg']}\n")

# # Query: VADER lowest negative text with highest possible star rating
# low5 = results_df.query('star_rating == 5') \
#     .sort_values('vader_neg', ascending=False)[['review_body', 'vader_pos', 'vader_neg']].head(5)

# print('VADER Low 5 stars:')
# for idx, row in low5.iterrows():
#     print(f"Review: {row['review_body']}\nVADER Positive: {row['vader_pos']}\nVADER Negative: {row['vader_neg']}\n")