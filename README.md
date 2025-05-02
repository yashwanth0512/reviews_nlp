# Amazon Review NLP Analysis 
For the full project description, view [**proposal.pdf**](https://github.com/afinn12/reviews_nlp/blob/main/proposal.pdf).


## Data
The datasets used are sourced from [Kaggle](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset?resource=download&select=amazon_reviews_us_Toys_v1_00.tsv). 
They are stored locally because the file sizes exceed Github's limit (even with LFS).

### Counts
In the **/counts** directory, the product ID is recorded with the number of reviews associated with it. They are sorted in descending order. 

### .py Files
#### filter.py
This file filters the data columns, removing 7 columns to keep only review_id, product_id, product_title, star_rating, helpful_votes, total_votes, verified_purchase, and review_body. It also filters based on the number of reviews for the product, only keeping reviews for products with at least 500 reviews. The output file is stored in **/data/filtered.**

#### count.py
This file outputs the counts and average ratings for any products with at least 500 reviews. It is sorted by the lowest average star rating and can be modified to sort by the number of reviews. The output file is stored in **/data/counts.**

#### extract.py
This file extracts all of the rows for a given product. It keeps the previous 8 columns that were filtered by *filter.py* since it takes the output from that file as input and allows for easier product analysis. The output file is stored in **/data/extract.**

### How to preprocess the data
To process the data, run *filter.py*, followed by *extract.py*. Then, sentiment_baseline can be run. 

To help identify products of interest, run *count.py* in no particular order. This is needed for sentiment_analysis as well as it contains each product's average star rating. 

**PRODUCT_ID and output/input file names must be updated in the files.**

## Baseline 

### Sentiment Analysis
The file *sentiment_baseline.py* acts as the baseline for sentiment analysis. It employs 3 different techniques: VADER, Roberta, and the Sent Pipeline to analyze the reviews for a given product. 

It computes and outputs the average sentiment scores, as well as the error percentage. As a basis, the star ratings were used where 1-2 stars were considered as negative, 3 as neutral, and 4-5 as positive. 

The (commented out) queries at the bottom can be used to identify cases where the models incorrectly identified the sentiment. 

### Classification

### Summarization 

## Sentiment Analysis
Run : 
```
python .\sentiment_analysis.py --input input.csv --output output.csv --model_dir model_dir
```
The BERT-mini model was trained on review data from Kaggle and then used to predict the sentiment of reviews for a given product. The models are given preprocessed data files, which filter out unused columns and extract reviews for a specific product ID. Then, it outputs the individual reviews and overall product summaries. The individual reviews summary includes the original fields as well as fields such as the positive and negative sentiment scores, the predicted label, and whether it was correctly predicted for debugging and to allow additional postprocessing. The product summary contains the average positive/negative sentiments, the predicted star ratings, metrics, and emotional statistics.



