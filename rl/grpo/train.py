import os
import debugpy
import torch
import grpo_utils
from accelerate import Accelerator
from config import *
from utils import load_model, load_tokenizer, get_data_loader
from rollout import collect_rollouts
from buffer import build_experience, collate_experience

ENABLE_DEBUGPY = True

def grpo_step(
    llm,
    optimizer,
    accelerator,
    all_sequences,
    attention_mask,
    old_log_probs,
    response_mask,
    advantage
):
    cur_log_probs = grpo_utils.calculate_logits(
        llm,
        all_sequences,
        attention_mask
    )

    loss = grpo_utils.calculate_grpo_loss(
        cur_log_probs,
        old_log_probs,
        response_mask,
        advantage
    )

    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()

    return loss.item()

def main():
    if ENABLE_DEBUGPY:
        debugpy.listen(("0.0.0.0", 5678))
        print(f"Debugger listening on {os.uname()[1]}:5678. Waiting for attach…")
        debugpy.wait_for_client()

    accelerator = Accelerator()

    llm = load_model(MODEL_NAME)
    tokenizer = load_tokenizer(MODEL_NAME)
    dataloader = get_data_loader("syllogism", tokenizer)
    optimizer = torch.optim.Adam(llm.parameters(), lr=LR)

    llm, dataloader, optimizer = accelerator.prepare(
        llm, dataloader, optimizer
    )

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):

            if step >= STEPS_PER_EPOCH:
                break

            # ===== Rollout =====
            full_response, log_probs, advantages = collect_rollouts(
                llm,
                tokenizer,
                batch,
                N_ROLLOUTS,
                MAX_NEW_TOKENS
            )

            input_size = batch['inputs']['input_ids'].shape[1]

            # ===== Build Experience =====
            experience = build_experience(
                full_response,
                log_probs,
                advantages,
                tokenizer,
                input_size
            )

            # ===== Collate =====
            all_sequences, attention_mask, old_log_probs, response_mask, advantage = \
                collate_experience(experience, accelerator)

            # ===== Policy Update =====
            loss = grpo_step(
                llm,
                optimizer,
                accelerator,
                all_sequences,
                attention_mask,
                old_log_probs,
                response_mask,
                advantage
            )

            if accelerator.is_main_process:
                print(f"Step {step} | Loss: {loss:.4f}")

    accelerator.wait_for_everyone()
    accelerator.save_model(llm, "./outputs/grpo")

if __name__ == "__main__":
    main()