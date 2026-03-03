import re
from transformers import AutoTokenizer

def generate_model_response(model, tokenizer, messages):
    chat_template = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # print(f"{chat_template}")

    # tokenize
    inputs = tokenizer(chat_template, return_tensors="pt")
    # print(f"Input ids shape: {inputs['input_ids'].shape}, first 10 tokens: {inputs['input_ids'][:, :10]}")

    # generate response
    outputs = model.generate(**inputs, max_new_tokens=512)
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[:, input_length:]

    decoded_output = tokenizer.batch_decode(generated_tokens)[0]
    return decoded_output

def extract_answer(response: str):
    answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer:
        return answer.group(1)
    return answer
