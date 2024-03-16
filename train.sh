echo "Training intent classifier..."
python train_intent.py -v \
--data_path data/intent_dataset.csv

echo "Training LLM..."
python train_llm.py \
--sql_data_path data/sql_dataset.csv \
--oe_data_path data/open_ended_dataset.csv  \
--model_path models/codegen-2B-mono
