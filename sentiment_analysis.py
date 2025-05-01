import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from scipy.special import softmax
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import argparse
import os
from collections import defaultdict

# python .\sentiment_analysis.py --input datafile --output output.tsv --model_dir sentiment_output6
# if model_dir isn't found, it's trained

# ensure no empty fields and ignore neutral reviews
def preprocess_data(input_file):
    df = pd.read_csv(input_file, sep='\t')
    df['review_body'] = df['review_body'].fillna('').astype(str)
    df = df[df['star_rating'].isin([1, 2, 4, 5])]
    df['label'] = df['star_rating'].apply(lambda x: 1 if x >= 4 else 0)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    return Dataset.from_pandas(train_df[['review_body', 'label']]), Dataset.from_pandas(val_df[['review_body', 'label']])

# tokenize the text
def tokenize(examples, tokenizer, max_length=512):
    return tokenizer(examples['review_body'], padding='max_length', truncation=True, max_length=max_length)

# train the model
def train_model(input_file, model_name, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, val_dataset = preprocess_data(input_file)
    # tokenize the data
    train_dataset = train_dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize(x, tokenizer), batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        accuracy = (predictions == torch.tensor(labels)).float().mean()
        return {"accuracy": accuracy.item()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

# train or load a model
def init_model(input_file, model_name, model_dir):
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        print(f"Loading model from {model_dir}")
    else:
        print(f"Model {model_dir} not found, training a new one...")
        train_model(input_file, model_name, model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

# predict the post/neg perctanages and the label
def predict_sentiment(text, tokenizer, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
    with torch.no_grad():
        output = model(**encoded_input)
    logits = output.logits[0].detach().cpu().numpy()
    probs = softmax(logits)
    pred_label = 'positive' if probs[1] > probs[0] else 'negative'
    return {
        'positive_score': float(probs[1]),
        'negative_score': float(probs[0]),
        'predicted_label': pred_label
    }


def analyze_reviews(input_file, model_dir, model_name, output_file):

    # used for pos/neg analysis
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found.")
    df = pd.read_csv(input_file, sep='\t')
    df['review_body'] = df['review_body'].fillna('').astype(str)

    tokenizer, model = init_model(input_file, model_name, model_dir)
    model = model.to(device)

    # used for emotion analysis
    emotion_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    emotion_model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    emotion_model = emotion_model.to(device)
    # model_config = emotion_model.config
    # full_emotion_labels = list(model_config.id2label.values())
    # print(full_emotion_labels)
    emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    emotion_counter = {label: 0.0 for label in emotion_labels}
    top_emotion_scores = {label: 0.0 for label in emotion_labels}
    top_emotion_reviews = {label: {'text': '', 'probs': []} for label in emotion_labels}

    # iterate through reviews
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting sentiment"):
        # predict
        sentiment = predict_sentiment(row['review_body'], tokenizer, model, device=device)
        emotion_probs = predict_emotion(row['review_body'], emotion_tokenizer, emotion_model, device=device)
        
        # keep track of review with highest score for each emotion 
        for i, label in enumerate(emotion_labels):
            emotion_counter[label] += emotion_probs[i]
            if emotion_probs[i] > top_emotion_scores[label]:
                top_emotion_scores[label] = emotion_probs[i]
                top_emotion_reviews[label]['text'] = row['review_body']
                top_emotion_reviews[label]['probs'] = emotion_probs

        # convert star to post/neg
        star = row.get('star_rating', None)
        if star in [1, 2]:
            ground_truth = 'negative'
        elif star == 3:
            ground_truth = None
        elif star in [4, 5]:
            ground_truth = 'positive'
        else:
            ground_truth = None
        # set whether the prediction was right
        correct = int(sentiment['predicted_label'] == ground_truth) if ground_truth in ['positive', 'negative'] else None

        results.append({
            'review_id': row.get('review_id', idx),
            'product_id': row['product_id'],
            'review_body': row['review_body'],
            'star_rating': star,
            'ground_truth': ground_truth,
            'positive_score': sentiment['positive_score'],
            'negative_score': sentiment['negative_score'],
            'predicted_label': sentiment['predicted_label'],
            'correct': correct
        })

    results_df = pd.DataFrame(results)

    # save individual review results to file
    results_df.to_csv(output_file,sep='\t', index=False)
    print(f"Saved review results to {output_file}")

    # calc stats    
    total_reviews = len(results_df)
    avg_pos = (results_df['predicted_label'] == 'positive').mean()
    avg_neg = (results_df['predicted_label'] == 'negative').mean()

    valid_rows = results_df['correct'].notna()
    accuracy = results_df.loc[valid_rows, 'correct'].mean()

    filtered = results_df[results_df['ground_truth'].isin(['positive', 'negative'])]
    expected = filtered['ground_truth'].map({'negative': 0, 'positive': 1}).values
    pred = filtered['predicted_label'].map({'negative': 0, 'positive': 1}).values

    precision = precision_score(expected, pred, zero_division=0)
    recall = recall_score(expected, pred, zero_division=0)
    f1 = f1_score(expected, pred, zero_division=0)

    # look up star rating in ratings folder
    # run counts.py to get, otherwise will be N/A
    ratings_path = None
    if 'toy' in input_file.lower():
        ratings_path = './data/counts/toys_ratings.tsv'
    elif 'multi' in input_file.lower():
        ratings_path = './data/counts/multi_ratings.tsv'
    else:
        print("Ratings file not found.")
    
    star_rating = "N/A"
    error = 0.0  
    product_id = results_df['product_id'].iloc[0]
    if ratings_path:
        ratings_df = pd.read_csv(ratings_path, sep='\t', header=None, names=['product_id', 'count', 'avg_rating'])
        # look up the star value of the product
        star = ratings_df[ratings_df['product_id'] == product_id]
        if not star.empty:
            star_rating = star['avg_rating'].values[0]
            pos_to_star = avg_pos*10/2
            error = (abs(pos_to_star - float(star_rating)) / float(star_rating)) * 100

    # get the values for the emotiosn
    dist = {label: val / total_reviews for label, val in emotion_counter.items()}
    emotion_lines = "\n".join([f"{label.capitalize()}: {score:.5f}" for label, score in dist.items() if label != 'Fear'])

    # create summary
    summary_text = (
        f"Summary Statistics:\n\n"
        f"Product ID: {product_id}\n\n"
        f"Total Reviews: {total_reviews}\n\n"

        f"Average Positive Rate: {avg_pos:.5f}\n"
        f"   Star Rating: {star_rating}\n"
        f"   Star Prediction: {avg_pos*10/2:.5f}\n"
        f"      Error percentage: {error:.3f}%\n"
        f"Average Negative Rate: {avg_neg:.5f}\n\n"

        f"Accuracy: {accuracy:.5f}\n"
        f"Precision: {precision:.5f}\n"
        f"Recall: {recall:.5f}\n"
        f"F1 Score: {f1:.5f}\n\n\n"

        f"Emotions:\n{emotion_lines}\n\n"
        f"Top Reviews per Emotion:\n"
    )

    # add the review with the top score for each emotion
    for label in emotion_labels:
        if label != "fear":
            summary_text += f"Highest for {label.capitalize()}\n"
            top_probs = top_emotion_reviews[label]['probs']
            for i, l in enumerate(emotion_labels):
                summary_text += f"  {l.capitalize()}: {top_probs[i]:.5f}\n"
            summary_text += f"Review: {top_emotion_reviews[label]['text']}\n\n"

    summary_file = output_file.replace('.tsv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    
    print(f"Saved product results to {summary_file}")
    print(summary_text)

# predict the emotion
def predict_emotion(text, tokenizer, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
    with torch.no_grad():
        output = model(**encoded_input)
    logits = output.logits[0].detach().cpu().numpy()
    probs = softmax(logits)
    return probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input TSV review file')
    parser.add_argument('--model', type=str, default='prajjwal1/bert-mini', help='Base model to fine-tune')
    # parser.add_argument('--model', type=str, default='distilbert-base-uncased', help='Base model to fine-tune')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory to save/load fine-tuned model')
    parser.add_argument('--output', type=str, required=True, help='Path to output CSV')

    args = parser.parse_args()
    # init_model(args.input, args.model, args.model_dir)
    analyze_reviews(args.input, args.model_dir, args.model, args.output)

if __name__ == "__main__":
    main()