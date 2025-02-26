import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


plt.style.use('ggplot')

import nltk

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
PRODUCT_ID = 'B000BN8Y8G'

# load the file in
df = pd.read_csv('data/filtered/filtered_toy.tsv',sep='\t')

# # insert ID column for later use
# df.insert(0, 'id', range(1, 1 + len(df)))

# print(df.shape)

# # prints the entries and all columns
# print(df.head(5))

# # prints column info
# df.info()

# # create graph of ALL star ratings
# # TODO: should filter by product id later
# ax = df['star_rating'].value_counts().sort_index() \
#     .plot(kind='bar',
#           title='Count of Reviews by Stars',
#           figsize=(10, 5))
# ax.set_xlabel('Review Stars')
# plt.show()

# # get all entries of this column
# example = df['review_body']

# get a specific entry
example = df['review_body'][2]

# tokenize it
tokens = nltk.word_tokenize(example)
# tag based on part of speech
tagged = nltk.pos_tag(tokens)

# Put tagged parts of speech into ENTITIES
# NLTK Chunk  //  ne_chunk()

entities = nltk.chunk.ne_chunk(tagged)
# entities.pprint()


# DOWNLOAD VADER
from nltk.sentiment import SentimentIntensityAnalyzer

# # PROGRESS BAR TRACKER
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()


# print('I am so happy!')
# print(sia.polarity_scores('I am so happy!'), '\n')


# for i in range(0, 10):
#     example = df['review_body'][i]
#     print(example)
#     print(sia.polarity_scores(example), '\n')

# convert to text to process 
df['review_body'] = df['review_body'].astype(str)


# Run the polarity score on the entire dataset
# create Result dictionary res{} to
# store individual polarity scores
res = {}
for i, row in df.iterrows():
# for i, row in df.iloc[:50].iterrows():
    if (row['product_id'] == PRODUCT_ID):
        text = row['review_body']
        id = row['review_id']
        res[id] = sia.polarity_scores(text)
# print(res)


# # store avg polarity scores
# avg = {}
# sum = {}
# # for i, row in df.iterrows():
# for i, row in df.iloc[:50].iterrows():
#     text = row['review_body']
#     id = row['product_id']

#     if id not in sum:
#         sum[id] = {
#             'compound': 0,
#             'pos': 0,
#             'neu': 0,
#             'neg': 0,
#             'count': 0
#         }

#     scores = sia.polarity_scores(text)

#     sum[id]['compound'] += scores['compound']
#     sum[id]['pos'] += scores['pos']
#     sum[id]['neu'] += scores['neu']
#     sum[id]['neg'] += scores['neg']
#     sum[id]['count'] += 1


# for id, scores in sum.items():
#     avg[id] = {
#         'compound': scores['compound'] / scores['count'],
#         'pos': scores['pos'] / scores['count'],
#         'neu': scores['neu'] / scores['count'],
#         'neg': scores['neg'] / scores['count']
#     }

# # print(avg)





# Convert dictionary to dataframe

# dataframe is oriented wrong way, run .T to pivot table
vaders = pd.DataFrame(res).T
# vaders = pd.DataFrame(avg).T

# Merge onto original dataframe
vaders = vaders.reset_index().rename(columns={'index': 'review_id'})

# left merge
vaders = vaders.merge(df, how='left')


# print()
# print(vaders.head())

# fig, axs = plt.subplots(1, 4, figsize=(12, 3))
# sns.barplot(data=vaders, x='star_rating', y='pos', ax=axs[0])
# sns.barplot(data=vaders, x='star_rating', y='neu', ax=axs[1])
# sns.barplot(data=vaders, x='star_rating', y='neg', ax=axs[2])
# sns.barplot(data=vaders, x='star_rating', y='compound', ax=axs[3])
# axs[0].set_title('Positive')
# axs[1].set_title('Neutral')
# axs[2].set_title('Negative')
# axs[3].set_title('Compound')

# # RUN   plt.tight_layout()  to fix overlapping yaxis labels
# plt.tight_layout()


# plt.show()



from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

# scipy for outputs // smooth out between 0 - 1
from scipy.special import softmax

from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

tokenizer(example, return_tensors='pt')
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)

# Convert output from tensor to numpy to store locally
scores = output[0][0].detach().numpy()

# Smooth out score values between 0 - 1
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}

# print('\n\nexample =',example)

# print('VADER Score:')
# print(sia.polarity_scores(example))

# print('Roberta Scores Dictionary:')
# print(scores_dict)

# print('Sent Pipeline:')
# print(sent_pipeline(example))
# print()


# example = df['review_body'][6]
# print('example =',example)

# print('VADER Score:')
# print(sia.polarity_scores(example))

# print('Roberta Scores Dictionary:')
# print(scores_dict)

# print('Sent Pipeline:')
# print(sent_pipeline(example))
# print()


# example = df['review_body'][15]
# print('example =',example)

# print('VADER Score:')
# print(sia.polarity_scores(example))

# print('Roberta Scores Dictionary:')
# print(scores_dict)

# print('Sent Pipeline:')
# print(sent_pipeline(example))
# print()


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
            # sent_pipeline_result = polarity_scores_sent_pipeline(text)
            
            # combine all result dictionaries
            results = {**vader_result_rename, **roberta_result}
            # both = {**vader_result_rename, **roberta_result, **sent_pipeline_result}
            res[myid] = results
        except RuntimeError:
            print(f'Broke for id {myid}')


results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'review_id'})
results_df = results_df.merge(df, how='left')

print(results_df.groupby('product_id').mean())


# sns.pairplot(data=results_df,
#              vars=['vader_neg', 'vader_neu', 'vader_pos',
#                   'roberta_neg', 'roberta_neu', 'roberta_pos'],
#                 #   'sent_pipeline_score'],
#             hue='star_rating',
#             palette='tab10')
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