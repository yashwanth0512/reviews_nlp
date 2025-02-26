# Amazon Review NLP Analysis 
For the full project descripting, view **proposal.pdf**.


## Data
The datasets used are sourced from [Kaggle](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset?resource=download&select=amazon_reviews_us_Toys_v1_00.tsv). The raw datasets are stored in **/data**.


### Counts
In the **/counts** directory, the product ID is recorded with the number of reviews associated with it. They are sorted in descending order. 

*After baseline testing, we discovered that **NUMBER** reviews were needed for sufficient NLP analysis.* 


### Filtered Data
In the **/filtered** directory, datasets are filtered to remove several columns, only keeping **product_id, product_title, star_rating, helpful_votes, total_votes, verified_purchase, and review_body.**

*The data will also be trimmed based on the number of reviews associated with a product once that threshold is discovered. For now, we've trimmed to at least 500 reviews*


## Baseline 

### Classification

### Sentiment Analysis

### Summarization 