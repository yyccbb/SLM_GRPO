import os
import re
import reasoning_gym
import openai
import dotenv
import torch
import debugpy
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich import print

from generation.utils import generate_model_response

debugpy.listen(("0.0.0.0", 5678))
print(f"Debugger listening on {os.uname()[1]}:5678. Waiting for attach…")
debugpy.wait_for_client()

SEED = 42
# model_name_or_path = "HuggingFaceTB/SmolLM-135M-Instruct"
model_name_or_path = "./sft_output/checkpoint-1000"
env_name = "propositional_logic"

dataset = reasoning_gym.create_dataset(
    env_name, seed=SEED, size=5
)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# dotenv.load_dotenv()
# client = openai.Client(
#     api_key=os.getenv('OPENROUTER_API_KEY'),
#     base_url="https://openrouter.ai/api/v1"
# )

encouraging_words = "You are an expert at answering logic questions."
sys_prompt = """Generate an answer after thinking. You must use the following template:
<think>your thinking steps</think>
<answer>the answer</answer>"""

final_sys_prompt = sys_prompt # TODO

def extract_answer(response: str):
    answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer:
        return answer.group(1)
    return answer

for example in dataset:
    question = example["question"]
    answer = example["answer"]

    print(f"[bold white]System: {final_sys_prompt}[/bold white]")
    print(f"[bold blue]Question: [/bold blue]\n" + question)
    if answer:
        print(f"\n[bold green]Answer: [/bold green]\n" + answer)
    print('-' * 5)

    # generation
    messages = [
        {"role": "system", "content": final_sys_prompt},
        {"role": "user", "content": question}
    ]

    llm_response = generate_model_response(model, tokenizer, messages)
    # llm_response = client.chat.completions.create(
    #     model="openai/gpt-5-mini",
    #     messages=messages
    # ).choices[0].message.content
    answer = extract_answer(llm_response)
    score_func = reasoning_gym.get_score_answer_fn(
        example["metadata"]["source_dataset"]
    )

    print(f"Extracted answer: ", answer)
    reward = score_func(answer, example)

    if reward > 0:
        print(f"[bold yellow]Correct![/bold yellow]")
    else:
        print(f"[bold red]Incorrect![/bold red]")