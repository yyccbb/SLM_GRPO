from transformers import PreTrainedModel, PreTrainedTokenizer

def setup_chat_format(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Assign the explicit Jinja2 template string
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )

    # Synchronize configuration variables between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # Synchronize the generation config variables
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer