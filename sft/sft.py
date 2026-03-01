from dataset import prepare_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from preprocess import setup_chat_format
from trl import SFTConfig, SFTTrainer
import torch
import debugpy
import os

# debugpy.listen(("0.0.0.0", 5678))
# print(f"Debugger listening on {os.uname()[1]}:5678. Waiting for attach…")
# debugpy.wait_for_client()

DATA_PATH = "./sft/data/training_syllogism.json"

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configure model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
# Setup chat template
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

dataset = prepare_dataset(DATA_PATH, tokenizer)

# Configure trainer
training_args = SFTConfig(
    output_dir="./sft_output",
    max_steps=1000,
    per_device_train_batch_size=32,
    learning_rate=5e-5,
    logging_steps=50,
    save_steps=200,
    bf16=True,
    # eval_strategy="steps",
    # eval_steps=50,
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)

# Start training
trainer.train()