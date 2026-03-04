import torch
import numpy as np
import grpo_utils
from config import EPS

def collect_rollouts(
    llm,
    tokenizer,
    batch,
    n_rollouts,
    max_new_tokens
):
    input_ids = batch['inputs']['input_ids'] # (B, L)
    attention_mask = batch['inputs']['attention_mask'] # (B, L)
    validator = batch['validator'] # (B, )
    input_size = input_ids.shape[1] # L

    with torch.no_grad():
        full_response = llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            num_return_sequences=n_rollouts,
            temperature=1,
            eos_token_id=tokenizer.eos_token_id
        ) # (B * n_rollouts, L')

        response_attention_mask = (full_response != tokenizer.pad_token_id).long()

        assistant_responses = full_response[:, input_size:] # (B * n_rollouts, L:L')

        log_probs = grpo_utils.calculate_logits(llm, full_response, response_attention_mask) # (B * n_rollouts, L')

        decoded_responses = tokenizer.batch_decode(
            assistant_responses, skip_special_tokens=True
        ) # List(B * n_rollouts, ), str

        rewards = grpo_utils.calculate_rewards(
            decoded_responses,
            np.repeat(validator, n_rollouts)
        ) # List(B * n_rollouts, ), float

        batch_size = input_ids.shape[0] # B
        rewards = np.reshape(rewards, (batch_size, n_rollouts)) # (B, n_rollouts) float

        advantages = (
            rewards - np.mean(rewards, axis=1, keepdims=True)
        ) / (np.std(rewards, axis=1, keepdims=True) + EPS) # (B, n_rollouts)

        advantages = torch.tensor(
            advantages.reshape(-1, 1),
            dtype=torch.float32,
            device=llm.device
        ) # (B * n_rollouts, )

    return full_response, log_probs, advantages, rewards