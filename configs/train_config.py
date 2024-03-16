from transformers import TrainingArguments

def get_training_args(output_dir):
    
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    optim = "paged_adamw_32bit"
    save_steps = 50
    logging_steps = 1
    learning_rate = 2e-4
    max_grad_norm = 0.3
    max_steps = 5
    warmup_ratio = 0.03
    lr_scheduler_type = "constant"
    num_train_epochs = 10

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=True,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        num_train_epochs=num_train_epochs,
        report_to="none",
    )
    
    return training_arguments
