import json
from datasets import Dataset
from transformers import PreTrainedTokenizer

def format_syllogism_record(record: dict) -> list[dict]:
    """
    Transforms raw syllogism variables into a standard conversation format.
    """
    user_content = record["question"]
    assistant_content = record["response"]
    
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    return messages

def prepare_dataset(data_path: str, tokenizer: PreTrainedTokenizer) -> Dataset:
    """
    Loads JSON data, applies role mappings, and formats via the tokenizer template.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        # Assumes data is a JSON array. If using JSONL, iterate and load line by line.
        raw_data = json.load(f)

    dataset = Dataset.from_list(raw_data)

    def apply_chat_template(example: dict) -> dict:
        messages = format_syllogism_record(example)
        
        # Tokenize=False returns a string instead of token IDs
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": formatted_text}

    processed_dataset = dataset.map(
        apply_chat_template,
        remove_columns=dataset.column_names,
        desc="Applying chat template to syllogism dataset"
    )

    return processed_dataset