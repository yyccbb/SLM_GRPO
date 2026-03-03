# buffer.py

import torch
from utils import left_pad

def build_experience(
    full_response,
    log_probs,
    advantages,
    tokenizer,
    input_size
):
    inverted_padding_mask = (full_response != tokenizer.pad_token_id).int()

    response_end_idx = (
        full_response.shape[1]
        - torch.flip(inverted_padding_mask, dims=[1]).argmax(dim=1)
    )

    response_mask = torch.zeros_like(inverted_padding_mask)

    for i in range(response_mask.shape[0]):
        response_mask[i, input_size:response_end_idx[i]] = 1

    experience = [
        {
            "input_sequence": full_response[i],
            "log_probs": log_probs[i],
            "response_mask": response_mask[i],
            "advantages": advantages[i],
        }
        for i in range(advantages.shape[0])
    ]

    return experience


def collate_experience(experience, accelerator):
    all_sequences = left_pad(
        [e["input_sequence"] for e in experience]
    ).to(accelerator.device)

    attention_mask = left_pad(
        [torch.ones_like(e["input_sequence"]) for e in experience], 0
    ).to(accelerator.device)

    old_log_probs = left_pad(
        [e["log_probs"] for e in experience]
    ).to(accelerator.device)

    response_mask = left_pad(
        [e["response_mask"] for e in experience]
    ).to(accelerator.device)

    advantage = torch.cat(
        [e["advantages"] for e in experience], dim=0
    ).unsqueeze(-1).to(accelerator.device)

    return all_sequences, attention_mask, old_log_probs, response_mask, advantage