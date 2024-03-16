import os
import json
import pickle
import argparse
import psycopg2
import pandas as pd
from data_processing import clean_text

def load_intent(model_path="models/intent_classifier.pkl"):
    with open(model_path, 'rb') as model_file:
      loaded_classifier, loaded_vectorizer, loaded_label_encoder = pickle.load(model_file)

    return loaded_classifier, loaded_vectorizer, loaded_label_encoder

def infer_intent(text, classifier, vectorizer, label_encoder):
    cleaned_text = clean_text(text)
    tfidf = vectorizer.transform([cleaned_text])
    prediction = classifier.predict(tfidf)
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    return predicted_class


def run_query(query, db_params):
    assert isinstance(query, str)

    try:
        with psycopg2.connect(**db_params) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

        return pd.DataFrame(result, columns=columns).to_html()
    except psycopg2.Error as error:
        raise Exception(error)

def llm_response(model, tokenizer, input_text):

    input_texts = config["input_text"]
    
    eos_token = [";", "';", " ;"]

    model.eval()

    with torch.no_grad():

        prompt = format_inference_prompt(input_text)

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_token_count = input_ids.size(1)

        generated_tokens = []
        next_token_id = None

        start = time.time()
        for _ in range(input_token_count + 32):
            if next_token_id is not None:
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(1)], dim=-1)

            next_token_logits = model(input_ids).logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            next_token = tokenizer.decode(next_token_id)
            if next_token in eos_token: break
            generated_tokens.append(next_token)

        stop = time.time()

        generated_text = "".join(generated_tokens) + next_token
        output_token_count = len(tokenizer.tokenize(generated_text))

        print(f"Input Text: {input_text}")
        print(f"Generated Text:\n{generated_text}")
        print(f'\nInput Token Length: {input_token_count}')
        print(f'Output Token Length: {output_token_count}')
        print(f'Query executed in {stop - start:.2f} seconds')
        print('-' * 50)



