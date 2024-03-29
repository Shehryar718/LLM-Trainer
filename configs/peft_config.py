from peft import LoraConfig

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
#    target_modules=[
#        "query_key_value",
#        "dense",
#        "dense_h_to_4h",
#        "dense_4h_to_h"
#  ]
)
