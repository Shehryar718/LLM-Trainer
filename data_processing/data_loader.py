import re
import pandas as pd
from datasets import Dataset
from .prompts import format_sql_text_to_prompt, format_open_ended_text_to_prompt
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_helper(data_path, func):
    df = pd.read_csv(data_path)
    df.fillna("", inplace=True)
    df = df.sample(frac=1)

    df['Prompt'] = df.apply(func, axis=1)
    return df

def load_and_preprocess_dataset(sql_dataset_path, open_ended_dataset_path):

    df_sql = preprocess_helper(sql_dataset_path, format_sql_text_to_prompt)
    df_open_ended = preprocess_helper(open_ended_dataset_path, format_open_ended_text_to_prompt)

    combined_df = pd.concat([df_sql.Prompt, df_open_ended.Prompt], axis=0).to_frame()
    combined_df.reset_index(drop=True, inplace=True)

    dataset = Dataset.from_pandas(combined_df)

    return dataset

def load_and_preprocess_intent_dataset(dataset_path, max_words=10000):
    df = pd.read_csv(dataset_path)
    rows, cols = df.shape
    df = df.sample(rows)
    texts = df['Text'].astype(str)
    labels = df['Class']

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    texts = texts.apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=max_words, stop_words='english')
    X_tfidf = vectorizer.fit_transform(texts)

    return X_tfidf, labels, label_encoder, vectorizer

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
