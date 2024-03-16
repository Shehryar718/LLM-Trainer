import os
import time
import torch
import json
import argparse
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from data_processing import format_sql_inference_prompt, format_open_ended_inference_prompt
from train_llm import load_model
from utils import load_intent, infer_intent

def parse_arguments():
    parser = argparse.ArgumentParser(description="Perform inference using a model based on an inference configuration file.")
    parser.add_argument("--infer_config", type=str, required=True, help="Path to the JSON inference configuration file.")
    return parser.parse_args()

def main(args):
    with open(args.infer_config, 'r') as json_file:
        config = json.load(json_file)

    model_path = config["model_checkpoint_path"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: Model file not found at {model_path}")
    
    model, tokenizer = load_model(model_path)

    input_texts = config["input_text"]
    
    eos_token = [";", "';", " ;", "\n"]

    model.eval()

    with torch.no_grad():
        for input_text in input_texts:
            classifier, vectorizer, label_encoder = load_intent()
            intent = infer_intent(input_text, classifier, vectorizer, label_encoder)

            if intent == "OC":
                print("As an AI model, Syntax Pilot, trained exclusively on PHISHCODE data, I'm unable to provide an answer to this question.")
            else:
                if intent == "OE":
                    prompt = format_open_ended_inference_prompt(input_text)
                else:
                    prompt = format_sql_inference_prompt(input_text)

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
                    if not generated_tokens:
                        if next_token == "\n":
                            continue

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



if __name__ == '__main__':
    args = parse_arguments()
    main(args)
