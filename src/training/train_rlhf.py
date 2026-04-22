import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rlhf_utils import (
    BLOCK_SIZE,
    BOS_TOKEN,
    is_valid,
    load_reward_model,
    load_transformer_model,
    predict_reward_from_sequences,
    set_seed,
    tokens_to_sequence,
    validity_penalty,
)


def top_k_filter(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None:
        return logits
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    filtered = logits.clone()
    filtered[filtered < v[:, [-1]]] = float("-inf")
    return filtered



def sample_batch_with_policy_stats(
    model,
    reference_model,
    batch_size: int,
    total_steps: int,
    temperature: float,
    top_k: int | None,
):
    device = next(model.parameters()).device
    idx = torch.full((batch_size, 1), BOS_TOKEN, dtype=torch.long, device=device)
    seq_logprob = torch.zeros(batch_size, device=device)
    seq_kl = torch.zeros(batch_size, device=device)

    for _ in range(total_steps):
        idx_cond = idx[:, -BLOCK_SIZE:]

        logits_cur, _ = model(idx_cond)
        logits_cur_last = logits_cur[:, -1, :]

        with torch.no_grad():
            logits_ref, _ = reference_model(idx_cond)
            logits_ref_last = logits_ref[:, -1, :]

        full_logp_cur = F.log_softmax(logits_cur_last, dim=-1)
        full_probs_cur = full_logp_cur.exp()
        full_logp_ref = F.log_softmax(logits_ref_last, dim=-1)
        kl_step = torch.sum(full_probs_cur * (full_logp_cur - full_logp_ref), dim=-1)
        seq_kl = seq_kl + kl_step

        sample_logits = logits_cur_last / temperature
        sample_logits = top_k_filter(sample_logits, top_k)
        sample_logp = F.log_softmax(sample_logits, dim=-1)
        next_token = torch.multinomial(sample_logp.exp(), num_samples=1)
        token_logprob = sample_logp.gather(dim=-1, index=next_token).squeeze(-1)
        seq_logprob = seq_logprob + token_logprob

        idx = torch.cat([idx, next_token], dim=1)

    sampled_tokens = idx[:, 1:]
    sequences = [tokens_to_sequence(tokens.detach().cpu().numpy()) for tokens in sampled_tokens]
    return sampled_tokens, sequences, seq_logprob / float(total_steps), seq_kl / float(total_steps)



def main():
    parser = argparse.ArgumentParser(description="Policy-gradient style RLHF fine-tuning for the drum Transformer.")
    parser.add_argument("--rl-steps", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--total-steps", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--beta-kl", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--invalid-penalty-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path("data")
    reward_model_path = data_dir / "reward_model.npz"
    if not reward_model_path.exists():
        raise FileNotFoundError(f"Missing {reward_model_path}. Run train_reward_model.py first.")

    reward_model = load_reward_model(reward_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, base_checkpoint = load_transformer_model(data_dir / "transformer_model.pth", device=device, train=True)
    reference_model, _ = load_transformer_model(data_dir / "transformer_model.pth", device=device, train=False)
    for param in reference_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = []
    best_state_dict = None
    best_reward = float("-inf")

    for step in range(1, args.rl_steps + 1):
        tokens, sequences, seq_logprob, seq_kl = sample_batch_with_policy_stats(
            model=model,
            reference_model=reference_model,
            batch_size=args.batch_size,
            total_steps=args.total_steps,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        reward_pred = predict_reward_from_sequences(sequences, reward_model, output_space="z").astype(np.float32)
        invalid_penalties = np.array([validity_penalty(seq) for seq in sequences], dtype=np.float32)
        validity_flags = np.array([1.0 if is_valid(seq) else 0.0 for seq in sequences], dtype=np.float32)
        final_rewards = reward_pred - args.invalid_penalty_weight * invalid_penalties

        rewards_t = torch.tensor(final_rewards, dtype=torch.float32, device=device)
        advantages = rewards_t - rewards_t.mean()

        policy_loss = -(advantages.detach() * seq_logprob).mean()
        kl_loss = args.beta_kl * seq_kl.mean()
        loss = policy_loss + kl_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        mean_reward = float(np.mean(final_rewards))
        mean_pred = float(np.mean(reward_pred))
        mean_penalty = float(np.mean(invalid_penalties))
        valid_frac = float(np.mean(validity_flags))
        mean_kl = float(seq_kl.detach().mean().item())

        info = {
        "step": step,
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "kl_loss": float(kl_loss.detach().item()),
        "mean_reward": mean_reward,
        "mean_predicted_reward_z": mean_pred,
        "mean_invalid_penalty": mean_penalty,
        "valid_fraction": valid_frac,
        "mean_kl": mean_kl,
        }
        history.append(info)
        print(info)

        checkpoint_dir = data_dir / "rlhf_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if step % 5 == 0:
            torch.save(
                {
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "step": step,
                    "config": vars(args),
                    "info": info,
                },
                checkpoint_dir / f"rlhf_step_{step}.pth",
            )

        if mean_reward > best_reward:
            best_reward = mean_reward
            best_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    save_obj = {
        "model_state_dict": best_state_dict if best_state_dict is not None else model.state_dict(),
        "base_model": "transformer_model.pth",
        "objective": "maximize E[r(X)] with policy gradient approximation using reward-model scores and KL regularization",
        "equation": "loss = -E[(r-b) * log p_theta(X)] + beta * KL(pi_theta || pi_ref)",
        "config": vars(args),
        "best_mean_reward": best_reward,
        "history": history,
    }
    torch.save(save_obj, data_dir / "rlhf_model.pth")

    with open(data_dir / "rlhf_training_info.json", "w") as f:
        json.dump({
            "config": vars(args),
            "best_mean_reward": best_reward,
            "num_updates": len(history),
            "history": history,
            "base_checkpoint_keys": list(base_checkpoint.keys()) if isinstance(base_checkpoint, dict) else [],
        }, f, indent=2)

    print("Saved data/rlhf_model.pth and data/rlhf_training_info.json")


if __name__ == "__main__":
    main()
