import os
import debugpy
import torch
import wandb
import time
import grpo_utils
import argparse
from accelerate import Accelerator
from transformers.utils import logging
from config import *
from utils import load_model, load_tokenizer, get_dataloader
from rollout import collect_rollouts
from buffer import build_experience, collate_experience

ENABLE_DEBUGPY = False

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training")

    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help="Path or name of model checkpoint"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Prompt batch size"
    )

    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=N_ROLLOUTS,
        help="Number of sampled responses per prompt"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=STEPS_PER_EPOCH,
        help="Training steps per epoch"
    )

    return parser.parse_args()

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
    args = parse_args()
    if ENABLE_DEBUGPY:
        debugpy.listen(("0.0.0.0", 5678))
        print(f"Debugger listening on {os.uname()[1]}:5678. Waiting for attach…")
        debugpy.wait_for_client()

    accelerator = Accelerator()

    run_name = f"grpo_bs{args.batch_size}_roll{args.n_rollouts}_ckpt{args.model_name.split('-')[-1]}"
    
    if accelerator.is_main_process:
        wandb.init(project="slm-grpo", name=run_name, config=vars(args))

    llm = load_model(args.model_name)
    tokenizer = load_tokenizer(args.model_name)
    dataloader = get_dataloader("syllogism", tokenizer, args.batch_size)
    accelerator.print(f"Dataset length: {len(dataloader.dataset)}")
    optimizer = torch.optim.Adam(llm.parameters(), lr=LR)

    llm, dataloader, optimizer = accelerator.prepare(
        llm, dataloader, optimizer
    )

    global_step = 0

    for epoch in range(args.num_epochs):
        accelerator.print(f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):

            if step >= args.steps_per_epoch:
                break

            # ===== Rollout =====
            full_response, log_probs, advantages, rewards = collect_rollouts(
                llm,
                tokenizer,
                batch,
                args.n_rollouts,
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
                mean_reward = rewards.mean()
                print(f"Step {global_step} | Mean reward: {mean_reward} | Loss: {loss:.4f}")
                wandb.log({
                    "loss": loss, 
                    "mean_advantage": advantage.mean().item(),
                    "mean_reward": mean_reward
                }, step=global_step)

            global_step += 1

    if accelerator.is_main_process:
        wandb.finish()

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        save_dir = f"./outputs/grpo/{run_name}"
        model_to_save = accelerator.unwrap_model(llm)
        model_to_save.save_pretrained(
            save_dir,
            safe_serialization=True
        )
        tokenizer.save_pretrained(save_dir)
if __name__ == "__main__":
    main()