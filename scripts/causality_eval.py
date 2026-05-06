"""Evaluate causality (predictive sufficiency) for each TR type and think_steps.

Metric: KL(P_θ(Z|Y) || P_θ(Z|T)) where θ is the frozen LLaMA-1B inside the trained
same-task discriminator. Both arms share identical frozen parameters — the KL is a direct
measure of how well T functionally substitutes Y within M_θ's computational graph.

Z is defined as the last `z_split_tokens` tokens of each generated sequence.
Y is the preceding prefix (reasoning portion).

Per-example outputs include `kl_windows`: a dict {W: scalar_kl} computed under
the proper Y/Z split for each W in `WINDOWS` (one thought-arm forward per window;
the explicit arm is a single pass over the full context, sliced per window). This
lets appendix analyses report causality at any listed W without rerunning on GPU.
The legacy scalar `kl` field equals `kl_windows[z_split_tokens]` for backward
compatibility with existing downstream analyses.
"""

import json
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from src.dataset.tr_loader import ThoughtRepresentationLoader
from src.model.discriminator import LlamaDiscriminator
from src.utils import set_seed_all
from src.utils.config import CausalityEvalConfig, ThoughtRepresentation, register_configs, resolve_other_vector_dim
from src.utils.logging import Logger

register_configs()

WINDOWS: tuple[int, ...] = (10, 25, 50, 100, 200)

def resolve_num_rep_spaces(tr_type: ThoughtRepresentation, source_num_layers: int) -> int:
    """Resolve discriminator num_rep_spaces for causality evaluation.

    LAST_INPUT_TOKEN depends on the source model layer count used during
    discriminator training (num_hidden_layers + 1). Other TR types keep their
    enum-defined feature count.
    """
    if tr_type == ThoughtRepresentation.LAST_INPUT_TOKEN:
        return source_num_layers
    return tr_type.num_features

def patch_disc_with_min_proj(
    discriminator: LlamaDiscriminator,
    cfg: CausalityEvalConfig,
    tr_type: ThoughtRepresentation,
    logger: Logger,
) -> None:
    """Overwrite the discriminator's projection components with a minimality
    CE(Y|T) probe's trainable params, in-place.

    Loads ``<min_proj_root>/<min_proj_run_label>/probe_<tr>_<steps>/trainable_params.pt``
    and copies:
      shared_projection.{weight,bias} → other_projection.{weight,bias}
      post_projection_norm.{weight,bias} → post_projection_norm.{weight,bias}
      vec_emb (broadcast over num_rep_spaces) → other_type_emb

    other_norm is left untouched. In the standard discriminator config
    (use_deep_projection=False) other_norm is a non-affine LayerNorm with
    no learnable parameters and is numerically identical to the minimality
    probe's input_norm.
    """
    probe_dir = (
        Path(cfg.min_proj_root)
        / cfg.min_proj_run_label
        / f"probe_{tr_type.value}_{cfg.think_steps}"
    )
    pt_path = probe_dir / "trainable_params.pt"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Minimality projection not found: {pt_path}. "
            f"Set proj_source=disc to use the discriminator projection instead."
        )
    logger.info(f"Patching discriminator with minimality projection from {pt_path}")
    sd = torch.load(pt_path, map_location=discriminator.other_projection.weight.device)

    needed = {
        "shared_projection.weight",
        "shared_projection.bias",
        "post_projection_norm.weight",
        "post_projection_norm.bias",
        "vec_emb",
    }
    missing = needed - set(sd.keys())
    if missing:
        raise KeyError(f"Minimality probe is missing keys: {sorted(missing)}")

    target_dtype = discriminator.other_projection.weight.dtype
    discriminator.other_projection.weight.data.copy_(
        sd["shared_projection.weight"].to(target_dtype)
    )
    discriminator.other_projection.bias.data.copy_(
        sd["shared_projection.bias"].to(target_dtype)
    )
    discriminator.post_projection_norm.weight.data.copy_(
        sd["post_projection_norm.weight"].to(target_dtype)
    )
    discriminator.post_projection_norm.bias.data.copy_(
        sd["post_projection_norm.bias"].to(target_dtype)
    )
    vec_emb = sd["vec_emb"].to(target_dtype)
    broadcast = vec_emb.unsqueeze(0).expand_as(discriminator.other_type_emb).contiguous()
    discriminator.other_type_emb.data.copy_(broadcast)
    logger.info(
        f"Projection swap complete (num_rep_spaces={discriminator.num_rep_spaces}, "
        f"vec_emb broadcast across {discriminator.other_type_emb.shape[0]} positions)"
    )

def project_thought(discriminator: LlamaDiscriminator, vecs: torch.Tensor) -> torch.Tensor:
    """Map T through the discriminator's projection into LLaMA-1B embedding space.

    Mirrors the "thought_vecs" path in LlamaDiscriminator.forward():
        other_norm → other_projection → post_projection_norm → + other_type_emb

    Args:
        vecs: [1, seq_len, other_vector_dim]
    Returns:
        projected: [1, seq_len, model_embedding_dim]
    """
    device = vecs.device
    T = discriminator.other_norm(vecs)
    T = discriminator.other_projection(T)
    T = discriminator.post_projection_norm(T)

    seq_len = T.size(1)
    if discriminator.num_rep_spaces > 1:
        num_repeats = (seq_len + discriminator.num_rep_spaces - 1) // discriminator.num_rep_spaces
        pos_emb = discriminator.other_type_emb.repeat(num_repeats, 1)[:seq_len].to(device)
    else:
        pos_emb = discriminator.other_type_emb[:seq_len].to(device)

    return T + pos_emb                                     

def compute_kl_windows(
    discriminator: LlamaDiscriminator,
    projected_T: torch.Tensor,
    all_ids: torch.Tensor,
    windows: tuple[int, ...],
    device: str,
) -> dict[int, float]:
    """Compute KL(P(Z|Y) || P(Z|T)) at each window in `windows`.

    The explicit arm is a single forward pass over all_ids[:-1] (shared by every
    window via logit slicing). The thought arm is one forward pass per window
    because the `T` prefix must sit immediately before Z: widening Z for a larger
    window would inject extra real-reasoning tokens between T and the evaluation
    span, which is a different metric. Windows larger than len(all_ids)-1 are
    skipped (None).

    Args:
        projected_T: [1, seq_len, model_dim]
        all_ids:     [N] full output token IDs (N >= 2 required)
        windows:     sorted tuple of positive ints
    Returns:
        {W: mean_kl_over_W-1_positions_or_None}.
    """
    llm = discriminator.model
    results: dict[int, float] = {}
    N = len(all_ids)

    with torch.no_grad():
                                                                                
        context_ids = all_ids[:-1].unsqueeze(0).to(device)
        logits_explicit_full = llm(input_ids=context_ids).logits[0]                

        for W in windows:
            if N <= W + 1:
                results[W] = None  # type: ignore[assignment]
                continue
            z_ids = all_ids[-W:]
            n_z = len(z_ids)
                                                                         
            logits_explicit = logits_explicit_full[-(n_z - 1) :, :]

            z_embeds = llm.get_input_embeddings()(z_ids[:-1].unsqueeze(0).to(device))
            full_embeds = torch.cat([projected_T, z_embeds], dim=1)
            logits_thought = llm(inputs_embeds=full_embeds).logits[0, -(n_z - 1) :, :]

            log_p = F.log_softmax(logits_explicit.float(), dim=-1)
            log_q = F.log_softmax(logits_thought.float(), dim=-1)
            kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean()
            results[W] = float(kl.item())

    return results

@hydra.main(version_base=None, config_path="../configs", config_name="causality_eval")
def main(cfg: CausalityEvalConfig) -> None:
    logger = Logger.from_config(cfg.logging)
    logger.info("=" * 70)
    logger.info("Causality Eval")
    logger.info(OmegaConf.to_yaml(cfg))
    if cfg.wandb.use_wandb:
        import wandb as _wandb
        _wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    logger.info("=" * 70)

    if cfg.seed is not None:
        set_seed_all(cfg.seed)

    device = cfg.discriminator.device
    tr_type = ThoughtRepresentation(cfg.tr_type)
    other_vector_dim = resolve_other_vector_dim(tr_type, cfg.source_hidden_size)
    num_rep_spaces = resolve_num_rep_spaces(tr_type, cfg.source_num_layers)

    logger.info(f"Loading discriminator backbone from {cfg.discriminator.model_name}")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg.discriminator.torch_dtype, torch.bfloat16)

    discriminator = LlamaDiscriminator(
        logger=logger,
        model_name=cfg.discriminator.model_name,
        other_vector_dim=other_vector_dim,
        device=device,
        load_in_8bit=cfg.discriminator.load_in_8bit,
        load_in_4bit=cfg.discriminator.load_in_4bit,
        torch_dtype=torch_dtype,
        trust_remote_code=cfg.discriminator.trust_remote_code,
        max_memory=cfg.discriminator.max_memory,
        freeze_base_model=cfg.discriminator.freeze_base_model,
        num_rep_spaces=num_rep_spaces,
    )
    discriminator.load()

    ckpt_path = Path(cfg.disc_dir) / "checkpoints" / "checkpoint_final.pt"
    logger.info(f"Loading checkpoint from {ckpt_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    discriminator.load_trainable_state_dict(state_dict)
    discriminator.eval()
    discriminator.model.gradient_checkpointing_disable()

    if cfg.proj_source == "minimality_output":
        patch_disc_with_min_proj(discriminator, cfg, tr_type, logger)
    elif cfg.proj_source != "disc":
        raise ValueError(
            f"Unknown proj_source: {cfg.proj_source!r}. Use 'disc' or 'minimality_output'."
        )

    logger.info(f"Loading TR data for {tr_type.value} (think_steps={cfg.think_steps})")
    tr_loader = ThoughtRepresentationLoader(
        logger=logger,
        tr_type=tr_type,
        cached_data_dir=cfg.tr_data_dir,
        split_name="test",
        vec_dim=other_vector_dim,
        think_steps=cfg.think_steps,
    )

    g_indices = list(tr_loader.generated_texts_cache.keys())
    if cfg.num_examples > 0:
        g_indices = g_indices[: cfg.num_examples]
    logger.info(f"Evaluating on {len(g_indices)} examples × {cfg.num_return_sequences} beams")

    backbone_tokenizer = discriminator.tokenizer

    kl_values = []
    per_example_records: list[dict] = []
    skipped = 0
    total_beams = sum(
        len(tr_loader.get_group_data(g_idx)["generated_texts"]) for g_idx in g_indices
    )
    pbar = tqdm(total=total_beams, desc=f"causality [{cfg.tr_type}]", unit="beam")

    for g_idx in g_indices:
        group_data = tr_loader.get_group_data(g_idx)
        thought_vec = group_data["thought_vec"]                                               
        generated_texts = group_data["generated_texts"]                      

        for beam_idx in range(len(generated_texts)):
            all_ids = torch.tensor(
                backbone_tokenizer.encode(generated_texts[beam_idx], add_special_tokens=False),
                dtype=torch.long,
            )

            if len(all_ids) <= cfg.z_split_tokens + 1:
                skipped += 1
                per_example_records.append({
                    "g_idx": int(g_idx),
                    "beam_idx": int(beam_idx),
                    "kl": None,
                    "y_len": 0,
                    "z_len": int(len(all_ids)),
                    "skipped": True,
                    "reason": "too_short",
                    "kl_windows": {str(W): None for W in WINDOWS},
                })
                pbar.update(1)
                continue

            if tr_type in (ThoughtRepresentation.RANDOM_VECTOR, ThoughtRepresentation.EMBEDDING_NO_POOLING):
                t_vec = thought_vec[beam_idx]                         
            else:
                t_vec = thought_vec                                               

            vecs = t_vec.unsqueeze(0).to(device, dtype=torch_dtype)                   
            if cfg.tile_to_length is not None:
                L_target = int(cfg.tile_to_length)
                if L_target <= 0:
                    raise ValueError(f"tile_to_length must be positive, got {L_target}")
                L_cur = vecs.size(1)
                n_rep = (L_target + L_cur - 1) // L_cur
                vecs = vecs.repeat(1, n_rep, 1)[:, :L_target, :]
            projected_T = project_thought(discriminator, vecs)

            kl_windows = compute_kl_windows(
                discriminator, projected_T, all_ids, WINDOWS, device
            )
                                                                               
            if cfg.z_split_tokens in kl_windows and kl_windows[cfg.z_split_tokens] is not None:
                kl = float(kl_windows[cfg.z_split_tokens])
            else:
                                                             
                candidates = [(abs(W - cfg.z_split_tokens), W, v) for W, v in kl_windows.items() if v is not None]
                if not candidates:
                    skipped += 1
                    per_example_records.append({
                        "g_idx": int(g_idx),
                        "beam_idx": int(beam_idx),
                        "kl": None,
                        "y_len": int(len(all_ids) - cfg.z_split_tokens),
                        "z_len": int(cfg.z_split_tokens),
                        "skipped": True,
                        "reason": "all_windows_skipped",
                        "kl_windows": {str(W): None for W in WINDOWS},
                    })
                    pbar.update(1)
                    continue
                _, _, kl = min(candidates)
                kl = float(kl)

            kl_values.append(kl)
            per_example_records.append({
                "g_idx": int(g_idx),
                "beam_idx": int(beam_idx),
                "kl": kl,
                "y_len": int(len(all_ids) - cfg.z_split_tokens),
                "z_len": int(cfg.z_split_tokens),
                "skipped": False,
                "kl_windows": {str(W): (None if v is None else float(v)) for W, v in kl_windows.items()},
            })

            running_mean = sum(kl_values) / len(kl_values)
            pbar.set_postfix(mean_kl=f"{running_mean:.4f}", skipped=skipped)
            pbar.update(1)

            if cfg.wandb.use_wandb:
                _wandb.log({"kl": kl, "running_mean_kl": running_mean, "step": len(kl_values)})

    pbar.close()

    if not kl_values:
        logger.error(
            f"No valid examples found — all sequences shorter than "
            f"z_split_tokens ({cfg.z_split_tokens}) + 1."
        )
        return

    mean_kl = sum(kl_values) / len(kl_values)
    logger.info(
        f"Causality Error (mean KL): {mean_kl:.4f}  "
        f"[n={len(kl_values)}, skipped={skipped}]"
    )

    result = {
        "tr_type": cfg.tr_type,
        "think_steps": cfg.think_steps,
        "mean_kl": mean_kl,
        "n": len(kl_values),
        "skipped": skipped,
    }

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
                                                                        
    out_file = out_dir / f"{cfg.tr_type}_{cfg.think_steps}.json"
    out_file.write_text(json.dumps(result, indent=2))
    logger.info(f"Results written to {out_file}")

    per_example_file = out_dir / f"{cfg.tr_type}_{cfg.think_steps}_per_example.jsonl"
    with per_example_file.open("w") as fh:
        for rec in per_example_records:
            fh.write(json.dumps(rec) + "\n")
    logger.info(f"Per-example KL written to {per_example_file}")

    if cfg.wandb.use_wandb:
        _wandb.log(result)
        _wandb.finish()

if __name__ == "__main__":
    main()
