import openai
from reasoning_gym import create_dataset
import dotenv
import os
import asyncio
import json
import backoff
import sys

'''
create a .env file and paste your OPENAI_API_KEY there.
Run this script using: `python data_generation.py syllogism 200` to generate 200 examples of the syllogism task
'''


dotenv.load_dotenv()
client = openai.AsyncClient(
    api_key = os.getenv('OPENROUTER_API_KEY'),
    base_url="https://openrouter.ai/api/v1"
)
ENVIRONMENT = "syllogism"
model = "openai/gpt-5-mini"
model_name = model.rsplit('/', 1)[1]
semaphore = asyncio.Semaphore(50)
num_datapoints = 500 # By default generates 1000 examples
system_prompt = """
You are an Assistant good at solving logical reasoning problems.

For each problem, follow this exact response structure:

<think>
Write your step-by-step reasoning here. Clearly analyze the premises and explain how they relate to the conclusion.
</think>
<answer>
Write only the final answer here.
</answer>

Guidelines:

- Always include both <think> and <answer> blocks.
- The <think> block must contain a complete reasoning process.
- The <answer> block must contain only the final answer (e.g., Yes or No).
- Do not include anything outside these two blocks.
- Do not generate code.
- Solve the given problem specifically, even if examples are shown.
- You will also be provided with the correct answer. Ensure your reasoning logically leads to that answer.
- Keep the reasoning clear, structured, and logically consistent.
"""

dataloader = create_dataset(name=ENVIRONMENT, size=num_datapoints)


@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def generate_response(item):
    async with semaphore:  # Use the global semaphore
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
    Question: {item['question']}
    Metadata: {item['metadata']}
    Answer: {item['answer']}
                    """,
            },
        ]
        response = await client.chat.completions.create(
            messages=messages, 
            model=model
        )
        print("-" * 5)
        print(response.choices[0].message.content)
        print("GT answer: ", item["answer"])
        print("-" * 5)
        return {
            "question": item["question"],
            "metadata": item["metadata"],
            "answer": item["answer"],
            "response": response.choices[0].message.content,
        }


async def main():
    responses = await asyncio.gather(*[generate_response(item) for item in dataloader])
    fname = f"./sft/data/responses_{ENVIRONMENT}_{model_name}.json"    
    json.dump(responses, open(fname, "w"), indent=4)
    print(f"Saved responses to {fname}")


if __name__ == "__main__":
    asyncio.run(main())