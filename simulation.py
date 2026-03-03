import os
import reasoning_gym
import openai
import dotenv
import torch
import debugpy
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from inference.utils import generate_model_response, extract_answer

console = Console()

# debugpy.listen(("0.0.0.0", 5678))
# print(f"Debugger listening on {os.uname()[1]}:5678. Waiting for attach…")
# debugpy.wait_for_client()

SEED = None
# model_name_or_path = "HuggingFaceTB/SmolLM-135M-Instruct"
model_name_or_path = "./outputs/sft/checkpoint-600"
env_name = "syllogism" #"propositional_logic"

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

for idx, example in enumerate(dataset):
    question = example["question"]
    gt_answer = example["answer"]

    console.print(Rule(f"[bold cyan]Example {idx}[/bold cyan]"))

    # System Prompt Panel
    console.print(
        Panel(
            final_sys_prompt,
            title="[bold white]System Prompt[/bold white]",
            border_style="white",
            expand=True
        )
    )

    # Question Panel
    console.print(
        Panel(
            question,
            title="[bold blue]Question[/bold blue]",
            border_style="blue",
            expand=True
        )
    )

    # Ground Truth Panel (optional)
    if gt_answer:
        console.print(
            Panel(
                str(gt_answer),
                title="[bold green]Ground Truth Answer[/bold green]",
                border_style="green",
                expand=True
            )
        )

    # ===== Generation =====
    messages = [
        {"role": "system", "content": final_sys_prompt},
        {"role": "user", "content": question}
    ]

    llm_response = generate_model_response(model, tokenizer, messages)
    # llm_response = client.chat.completions.create(
    #     model="openai/gpt-5-mini",
    #     messages=messages
    # ).choices[0].message.content

    console.print(
        Panel(
            llm_response,
            title="[bold magenta]Model Output[/bold magenta]",
            border_style="magenta",
            expand=True
        )
    )

    extracted = extract_answer(llm_response)

    console.print(
        Panel(
            str(extracted),
            title="[bold yellow]Extracted Answer[/bold yellow]",
            border_style="yellow",
            expand=True
        )
    )

    score_func = reasoning_gym.get_score_answer_fn(
        example["metadata"]["source_dataset"]
    )

    reward = score_func(extracted, example)

    if reward > 0:
        result_text = "[bold green]Correct ✓[/bold green]"
        border_color = "green"
    else:
        result_text = "[bold red]Incorrect ✗[/bold red]"
        border_color = "red"

    console.print(
        Panel(
            result_text,
            title="[bold white]Result[/bold white]",
            border_style=border_color,
            expand=True
        )
    )

    console.print("\n")