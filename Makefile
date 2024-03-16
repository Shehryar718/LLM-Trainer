install:
	@echo "Installing project dependencies..."
	pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
	pip install -q datasets bitsandbytes einops wandb langchain
	pip install -q scikit-learn
	pip install -q psycopg2
	pip install -q tensorrt
	@echo "Installation complete."
