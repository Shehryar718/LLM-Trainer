import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_processing import load_and_preprocess_dataset
from configs import bnb_config, peft_config, get_training_args
from trl import SFTTrainer

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.output_hidden_states = False
    model.config.output_attentions = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference using a causal LM model.")
    parser.add_argument("--sql_data_path", type=str, required=True, help="Path to the sql dataset file.")
    parser.add_argument("--oe_data_path", type=str, required=True, help="Path to the open ended dataset file.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the pretrained model checkpoint.")
    parser.add_argument("--model_name", type=str, default="Salesforce/codegen-2B-mono", help="Model name if model_path is not provided.")
    return parser.parse_args()

def main(args):
    sql_data_path = args.sql_data_path
    open_ended_data_path = args.oe_data_path

    if not os.path.exists(sql_data_path):
        raise FileNotFoundError(f"Error: Data file not found at {sql_data_path}")

    if not os.path.exists(open_ended_data_path):
        raise FileNotFoundError(f"Error: Data file not found at {open_ended_data_path}")

    dataset = load_and_preprocess_dataset(sql_data_path, open_ended_data_path)

    model_name = args.model_name
    model_path = args.model_path

    if model_path is not None:
        if not os.path.exists(model_path):
            print(f"Model checkpoint for {model_name} not found at {model_path}. Downloading {model_name}...")
            model_path = model_name
        else:
            print(f"Using the provided model checkpoint at {model_path}.")
    else:
        model_path = model_name

    model, tokenizer = load_model(model_path)

    torch.cuda.empty_cache()

    max_seq_length = 512

    output_dir = f"models/{model_name.split('/')[-1]}/checkpoints"
    training_arguments = get_training_args(output_dir)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="Prompt",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module.to(torch.float32)

    trainer.train()
    trainer.save_model(f"./models/{model_name.split('/')[-1]}")

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
