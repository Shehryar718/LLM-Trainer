# LLMTrainer

## Overview
LLMTrainer is a versatile pipeline for training and fine-tuning Large Language Models (LLMs) in natural language processing (NLP). It offers a streamlined process for customizing LLMs, optimizing hyperparameters, and evaluating model performance. With support for data preprocessing, scalability, and model deployment, LLMTrainer simplifies LLM development for various NLP applications, from chatbots to text analysis.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
    - [Training](#training)
    - [Inference](#inference)
- [Project Structure](#project-structure)
- [Configuration Files](#configuration-files)
- [Data](#data)
  
## Installation
Before you can run the code, you need to install the required dependencies. You can do this using the Makefile:
```bash
make install
```

## Usage

### Training
To train the model, use the train.py script. Here's how you can run it:
```bash
python train.py --data_path /path/to/data/file --model_name model_name --model_path /path/to/model/checkpoint
```
- `--data_path`: Path to the data file.
- `--model_path` (optional): Path to the pretrained model checkpoint. If not provided, the default model will be used.
- `--model_name` (optional): Name of the model if `--model_path` is not provided. The code will attempt to use a model with this name.

### Inference
To perform inference using the trained model, use the `infer.py` script. You'll need an inference configuration file (e.g., `infer_config.json`) with the model checkpoint path and input text. Here's how you can run it:
```bash
python infer.py --infer_config /path/to/infer_config.json
```
An example `infer_config.json` file:
```json
{
    "model_checkpoint_path": "./models/codegen-2B-mono",
    "input_text": [
        "List all the campaigns.",
        "List number of admins in the database."
    ]
}
```

## Project Structure
The project has the following file structure:
```markdown
my_project/
├── data/
│ └── SQLExamples.csv
├── configs/
│ ├── __init__.py
│ ├── bnb_config.py
│ ├── infer_config.json
│ ├── train_config.py
│ └── peft_config.py
├── data_processing/
│ ├── __init__.py
│ ├── data_loader.py
│ └── prompts.py
├── models/
├── infer.py
├── train.py
└── README.md
```

## Configuration Files
- `bnb_config.py`: This file contains the configuration settings for BitsAndBytes.
- `peft_config.py`: This file contains the configuration settings for PEFT (Partial Evidence Fine-tuning).
- `train_config.py`: This file holds the configuration for training the model.
- `infer_config.json`: This JSON file contains the configuration for performing inference with the model.

## Data
The project relies on a meticulously curated dataset stored in the data directory, specifically in the SQLExamples.csv file. This dataset contains a rich set of information, including context, natural language questions, and their corresponding SQL queries. The data has been carefully hand-prepared, ensuring accuracy and reliability. All SQL queries within the dataset have been thoroughly validated and tested, making it a valuable resource for training and evaluating language models.
