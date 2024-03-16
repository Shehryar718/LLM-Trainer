import os
import pickle
import torch
from data_processing import clean_text
from utils import infer_intent, run_query
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from data_processing import format_open_ended_inference_prompt, format_sql_inference_prompt
from train_llm import load_model

query = "SELECT title, description FROM public.training_templates WHERE thumbnail_id IS NOT NULL;"
db_params = {
    'dbname': 'phishcode',
    'user': 'postgres',
    'password': 'shehryar123',
    'host': 'localhost',  # Use 'localhost' if the database is on your local machine
    'port': '5432'   # Typically, PostgreSQL uses port 5432
}
# df = run_query(query, db_params)
# df

def out_of_context():
    return "As an AI model, Syntax Pilot, trained exclusively on PHISHCODE data, I'm unable to provide an answer to this question."

def open_ended(text, model, tokenizer, func):
    eos_token = [";", "';", " ;", "\n"]
    prompt = func(text)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    generated_tokens = []
    next_token_id = None

    for _ in range(64):
        if next_token_id is not None:
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(1)], dim=-1)

        next_token_logits = model(input_ids).logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        next_token = tokenizer.decode(next_token_id)
        if not generated_tokens:
            if next_token == "\n":
                continue

        if next_token in eos_token: break

        generated_tokens.append(next_token)

#    generated_text = "".join(generated_tokens) + next_token
    return "".join(generated_tokens) + next_token

if __name__ == '__main__':
    
    # Loading the intent Classifier
    model_path = "models/intent_classifier.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: Model file not found at {model_path}")
        
    with open(model_path, 'rb') as model_file:
        loaded_classifier, loaded_vectorizer, loaded_label_encoder = pickle.load(model_file)
    
    # Loading Syntex Pilot
    model_path = "models/codegen-2B-mono"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: Model file not found at {model_path}")
        
    model, tokenizer = load_model(model_path)
    
    model.eval()
    with torch.no_grad():
        while True:
            text = input('Enter you text: ')
            if text == "EXIT": break
            
            intent = infer_intent(text, loaded_classifier, loaded_vectorizer, loaded_label_encoder)
            
            if intent == "OC": result = out_of_context()
            elif intent == "OE": result = open_ended(text, model, tokenizer, format_open_ended_inference_prompt)
            else: result = open_ended(text, model, tokenizer, format_sql_inference_prompt)
            
            print(result)
            print('--------------------------------------------------')
            
        
        
