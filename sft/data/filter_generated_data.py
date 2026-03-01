import json
import re

SOURCE_PATH = "responses_syllogism_gpt-5-mini.json"
FILTERED_TRAINING_DATA_PATH = "training_syllogism_2.json"

def is_valid_format(text):
    if not isinstance(text, str):
        return False, "Not a string"

    # Must contain exactly one think block and one answer block
    think_blocks = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
    answer_blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)

    if len(think_blocks) != 1:
        return False, "Invalid number of <think> blocks"

    if len(answer_blocks) != 1:
        return False, "Invalid number of <answer> blocks"

    think_content = think_blocks[0].strip()
    answer_content = answer_blocks[0].strip()

    if len(think_content) == 0:
        return False, "<think> block empty"

    if len(answer_content) == 0:
        return False, "<answer> block empty"

    # Ensure ordering is correct
    think_pos = text.find("<think>")
    answer_pos = text.find("<answer>")

    if think_pos > answer_pos:
        return False, "<answer> appears before <think>"

    # Ensure nothing exists outside the two blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<answer>.*?</answer>", "", cleaned, flags=re.DOTALL)

    if cleaned.strip() != "":
        return False, "Extra text outside tags"

    return True, "Valid"

def main():
    with open(SOURCE_PATH, "r") as file:
        json_data = json.load(file)
    
    valid_samples = []
    for sample in json_data:
        response = sample.get("response")
        is_valid, reason = is_valid_format(response)
        if is_valid:
            valid_samples.append(sample)

    print(f"Valid samples: {len(valid_samples)}")
    with open(FILTERED_TRAINING_DATA_PATH, "w") as file:
        json.dump(valid_samples, file, indent=4)

if __name__ == "__main__":
    main()