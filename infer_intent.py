import os
import json
import pickle
import argparse
from data_processing import clean_text
from utils import infer_intent

def parse_arguments():
    parser = argparse.ArgumentParser(description="Perform inference using a model based on an inference configuration file.")
    parser.add_argument("--infer_intent_config", type=str, required=True, help="Path to the JSON inference configuration file.")
    return parser.parse_args()
    
def main(args):

    with open(args.infer_intent_config, 'r') as json_file:
        config = json.load(json_file)
        
    model_path = config["model_checkpoint_path"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: Model file not found at {model_path}")
        
    with open(model_path, 'rb') as model_file:
        loaded_classifier, loaded_vectorizer, loaded_label_encoder = pickle.load(model_file)

    input_texts = config["input_text"]
    for input_text in input_texts:
        predicted_class = infer_intent(input_text, loaded_classifier, loaded_vectorizer, loaded_label_encoder)
        print(f"Input text: {input_text}\nPredicted class: {predicted_class}")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
