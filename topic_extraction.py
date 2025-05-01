import pandas as pd
import argparse
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from difflib import SequenceMatcher

nlp = spacy.load("en_core_web_sm")

# clean text for special characters
def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"'", '', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

# if a phrase has a noun in it
def contains_noun(phrase):
    doc = nlp(phrase)
    return any(token.pos_ in {"NOUN", "PROPN"} for token in doc)

# if a sentence isn't completely incoherent
def has_meaning(phrase):
    doc = nlp(phrase)
    tokens = list(doc)

    if not contains_noun(phrase) or not any(t.pos_ in {"VERB", "ADJ"} for t in tokens):
        return False

    # reject if incomplete ending
    if tokens[-1].pos_ in {"ADP", "DET", "CCONJ", "SCONJ", "PART", "PRON"}:
        return False

    # check for verb with no objects
    if sum(t.pos_ == "VERB" for t in tokens) >= 2 and not any(t.dep_ in {"dobj", "attr", "nsubj"} for t in tokens):
        return False

    # check for verb with no subject
    if len([t for t in tokens if t.pos_ == "VERB"]) > 1 and not any(t.dep_ in {"dobj", "attr", "pobj"} for t in tokens):
        return False
    
    # allow short phrases that describe a function
    if len(tokens) <= 4 and any(t.pos_ == "VERB" for t in tokens) and any(t.pos_ == "ADV" for t in tokens):
        return True
    
    # if too many nouns, ignore it 
    # some output was noun heavy
    num_nouns = sum(1 for t in tokens if t.pos_ in {"NOUN", "PROPN"})
    if num_nouns >= len(tokens) - 1:
        return False

    # check relationships between words
    good_deps = {"amod", "attr", "dobj", "nsubj", "prep", "pobj", "advmod"}
    has_dependency = any(t.dep_ in good_deps for t in tokens)
    if not has_dependency:
        return False

    good_deps = {"amod", "attr", "dobj", "nsubj", "prep"}
    return any(t.dep_ in good_deps for t in tokens)

# see how similar phrases are for uniqueness
def is_similar(a, b, threshold=0.6):
    return SequenceMatcher(None, a, b).ratio() > threshold

# get ngrams for topic extraction
def get_ngrams(texts, n=8):
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english')
    X = vec.fit_transform(texts)
    ngrams = vec.get_feature_names_out()
    freqs = X.toarray().sum(axis=0)
    return dict(zip(ngrams, freqs))

# clean the phrases
def clean_phrases(phrases, threshold=0.6):
    def similarity(a, b):
        set_a, set_b = set(a.split()), set(b.split())
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0

    def is_valid(phrase):
        words = phrase.split()
        alpha_ratio = sum(w.isalpha() for w in words) / len(words)
        return alpha_ratio >= 0.6 and not re.search(r'\b(\d+)\b.*\b\1\b', phrase) and has_meaning(phrase)

    selected = []
    for phrase, count in sorted(phrases, key=lambda x: -x[1]):
        if not is_valid(phrase):
            continue
        if all(
            similarity(phrase, p) < threshold and not is_similar(phrase, p)
            for p, _ in selected
        ):
            selected.append((phrase, count))
    return selected


def normalize_topic(topic):
    doc = nlp(topic)
    lemmas = [token.lemma_.lower() for token in doc if token.pos_ in {"NOUN", "ADJ"} and not token.is_stop]
    return " ".join(sorted(set(lemmas)))


def clean_topics(topic_dict, threshold=0.60):
    
    topic_norm_map = {topic: (topic) for topic in topic_dict}
    grouped = defaultdict(list)
    used = set()

    for topic, norm in topic_norm_map.items():
        if topic in used:
            continue
        for other_topic, other_norm in topic_norm_map.items():
            if other_topic in used:
                continue
            similarity = SequenceMatcher(None, norm, other_norm).ratio()
            if similarity >= threshold:
                grouped[topic].extend(topic_dict[other_topic])
                used.add(other_topic)

    return grouped



def get_topics(df, top_k_keywords=50, min_phrase_len=3, max_phrase_len=8, top_phrases_per_topic=10):

    print("Finding topics...")
    df['review_body'] = df['review_body'].fillna('').astype(str).map(clean_text)
    all_text = df['review_body'].tolist()


    ignore_keywords = {'year old', 'yr old', 'gift', 'like', 'year', 'old', 'bought','son', 'daughter'}

    # title_words = set()
    # if 'product_title' in df.columns:
    #     df['product_title'] = df['product_title'].fillna('').astype(str).map(clean_text)
    #     for title in df['product_title']:
    #         title_words.update(title.split())


    short_vectorizer = CountVectorizer(ngram_range=(2, 5), stop_words='english', max_features=200)
    short_X = short_vectorizer.fit_transform(all_text)
    short_keywords = short_vectorizer.get_feature_names_out()
    short_freqs = short_X.toarray().sum(axis=0)

    top_keywords = [
        kw for _, kw in sorted(zip(short_freqs, short_keywords), reverse=True)[:top_k_keywords]
        if not any(suffix in kw for suffix in ignore_keywords)
        # and contains_noun(kw)
        # and all(token.pos_ in {"NOUN", "PROPN"} for token in nlp(kw))
        # and not all(word in title_words for word in kw.split())
    ]

    all_ngrams = {}
    for n in range(min_phrase_len, max_phrase_len + 1):
        ngram_counts = get_ngrams(all_text, n=n)
        for ng, count in ngram_counts.items():
            if count >= 2:
                all_ngrams[ng] = count

    topic_to_phrases = defaultdict(list)
    for topic in top_keywords:
        for phrase, count in all_ngrams.items():
            if topic in phrase:
                topic_to_phrases[topic].append((phrase, count))
    topic_to_phrases = clean_topics(topic_to_phrases)

    output_lines = []
    cleaned_topic_dict = {}

    for topic, phrases in topic_to_phrases.items():
        cleaned = clean_phrases(phrases)
        # cleaned = phrases
        if cleaned:
            cleaned = [(p, int(c)) for p, c in cleaned]
            cleaned_topic_dict[topic] = cleaned
            output_lines.append(f"Topic: {topic.title()}")
            for phrase, count in cleaned[:top_phrases_per_topic]:
                output_lines.append(f'  - "{phrase}" ({count} reviews)')
            output_lines.append("")

    return '\n'.join(output_lines), cleaned_topic_dict

def generate_feature(topic, phrases, max_sentences=1):
    topic_doc = nlp(topic)
    noun_chunk = next((chunk.text for chunk in topic_doc.noun_chunks), topic)
    noun_lemma = [token.lemma_ for token in topic_doc if token.pos_ == "NOUN"]
    noun = noun_lemma[-1] if noun_lemma else topic.split()[-1]

    mods = set()
    verbs = set()

    for phrase, _ in phrases[:5]:
        doc = nlp(phrase)
        for token in doc:
            if token.pos_ == "ADJ":
                mods.add(token.text)
            if token.pos_ == "VERB":
                verbs.add(token.lemma_)

    mods = list(mods)[:2]
    verbs = list(verbs)[:1]

    mod_part = " and ".join(mods) if mods else ""
    verb_part = verbs[0] if verbs else ""

    if mod_part and verb_part:
        return [f"The {mod_part} {noun_chunk} {verb_part}s."]
    elif mod_part:
        return [f"It's a {mod_part} {noun_chunk}."]
    elif verb_part:
        return [f"The {noun_chunk} {verb_part}s."]
    else:
        return [f"This review discusses {noun_chunk}."]
   
def clean_sentence(summary_text):
    from difflib import SequenceMatcher
    import re

    def is_meaningful(sentence):
        return len(sentence.split()) > 2 and any(word.isalpha() for word in sentence.split())

    raw_sentences = re.findall(r'[^.?!]*[.?!]', summary_text)
    cleaned_sentences = [s.strip().capitalize() for s in raw_sentences if is_meaningful(s)]

    unique_sentences = []
    for sent in cleaned_sentences:
        if all(not is_similar(sent, existing) for existing in unique_sentences):
            unique_sentences.append(sent)

    return ' '.join(unique_sentences).strip()


def extractive_summary(topic, phrases, summarizer):
    phrase_texts = [p for p, _ in phrases[:5]]
    input_text = ". ".join(phrase_texts)
    if not input_text.endswith('.'):
        input_text += '.'
    try:
        result = summarizer(input_text, max_length=len(input_text.split()), min_length=2, do_sample=False)
        if isinstance(result, list) and "summary_text" in result[0]:
            summary = result[0]["summary_text"]
            return clean_sentence(summary)
        else:
            print(f"ERROR: '{topic}': {result}")
            return f"People commented on {topic}."
    except Exception as e:
        print(f"ERROR Summarization failed for topic '{topic}': {e}")
        return f"People commented on {topic}."


def generate_lm_summary(topic, phrases, summarizer):
    return extractive_summary(topic, phrases, summarizer)



def main(input_file, output_file, summarize=False):
    # model_name = "google/flan-t5-base" 

    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    df = pd.read_csv(input_file, sep='\t')
    # print(df['product_title'][0])

    topic_output, topic_dict = get_topics(df)
    topic_path = output_file.replace('.tsv', '_phrases.txt')

    with open(topic_path, 'w') as f:
        f.write(topic_output)

    print(f"topic summary saved to: {topic_path}")

    if summarize:
        print("Loading Sentence-BERT model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        topics = {k: [(p, int(c)) for p, c in v] for k, v in topic_dict.items()}

        summary_path = output_file.replace('.tsv', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Here's what customers said:\n\n")
            for topic, phrases in topics.items():
                summaries = generate_lm_summary(topic, phrases, summarizer)
                # summaries = generate_feature(topic, phrases, model)
                # summaries = generate_nlp_feature_sentences(topic, phrases, model)
                # cleaned_summaries = ''.join(''.join(char for char in summary if char.isalpha() or char.isspace()) for summary in summaries)
                if summaries:
                    f.write(f"Topic: {topic.title()}\n")
                    # for s in summaries:
                    # if len(cleaned_summaries) > 1:
                    f.write(f"  - {summaries}\n")
                    f.write("\n")

        print(f"Topic summary saved to: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input TSV review file')
    parser.add_argument('--output', type=str, required=True, help='Path to output TSV with keywords')
    args = parser.parse_args()
    main(args.input, args.output, summarize=True)
