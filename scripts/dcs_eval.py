"""Compute Distributional Consistency Score (DCS) for each TR type.

DCS measures whether a thought representation T reflects the model's true semantic distribution
across K beam-search outputs — invariant to lexical variation, dispersed when outputs diverge.

Two variants of the binary Semantic Equivalence Matrix E:
  E_emb:   cos(nemotron_emb_i, nemotron_emb_j) > tau  (embedding-based)
  E_parse: BBEH answer extraction (data/bbeh/evaluate.py logic), fallback to E_emb[i,j]
           for beams where extraction fails (no "The answer is:" prefix found).

Functional Similarity Matrix M (same trained discriminator as Separability):
  M[i,j] = 0.5 * (f_disc(T_i, y_j) + f_disc(T_j, y_i))

DCS = 1 - MAE(M, E)  averaged over examples.  ∈ [0, 1], higher is better.

Outputs one JSON file per job to output_dir/{tr_type}_{think_steps}.json.
"""

import json
from math import ceil
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from src.utils.bbeh_eval import fuzzy_match, preprocess_sample
from src.dataset.tr_loader import ThoughtRepresentationLoader
from src.model.discriminator import LlamaDiscriminator
from src.utils import set_seed_all
from src.utils.config import DCSEvalConfig, ThoughtRepresentation, register_configs, resolve_other_vector_dim
from src.utils.logging import Logger

register_configs()

_ANSWER_PREFIXES = [
    "The answer is:",
    "The final answer is ",
    "The final answer is: ",
    "The answer is ",
]

def _has_answer_prefix(text: str) -> bool:
    return any(p in text for p in _ANSWER_PREFIXES)

def compute_cos_sim(emb_vecs: torch.Tensor) -> torch.Tensor:
    """Raw cosine similarity matrix from L2-normalised embeddings.

    Args:
        emb_vecs: [K, d] already L2-normalised
    Returns:
        sim: [K, K] float, diagonal zero
    """
    sim = emb_vecs @ emb_vecs.T          
    sim = sim.float().clone()
    sim.fill_diagonal_(0.0)
    return sim

def compute_E_emb(emb_vecs: torch.Tensor, tau: float) -> torch.Tensor:
    """Binary equivalence matrix from L2-normalised nemotron embeddings.

    Args:
        emb_vecs: [K, d] already L2-normalised
        tau: cosine similarity threshold
    Returns:
        E: [K, K] float, diagonal zero
    """
    sim = emb_vecs @ emb_vecs.T          
    E = (sim > tau).float()
    E.fill_diagonal_(0.0)
    return E

def compute_E_parse(
    decoded_texts: list[str],
    E_emb: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Binary equivalence matrix from BBEH answer parsing, fallback to E_emb.

    Uses data/bbeh/evaluate.py (preprocess_sample + fuzzy_match) for extraction and
    comparison. Falls back to E_emb[i,j] when extraction fails on either beam i or j.

    Args:
        decoded_texts: List[K] decoded beam strings
        E_emb: [K, K] embedding-based equivalence matrix (fallback)
    Returns:
        (E_parse [K, K], n_failed) — n_failed = number of beams with no answer prefix
    """
    K = len(decoded_texts)
    answers: list[str | None] = []
    failed: list[bool] = []

    for text in decoded_texts:
        if _has_answer_prefix(text):
            answers.append(preprocess_sample(text))
            failed.append(False)
        else:
            answers.append(None)
            failed.append(True)

    n_failed = sum(failed)
    E = torch.zeros(K, K)

    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            if failed[i] or failed[j]:
                E[i, j] = E_emb[i, j]            
            else:
                E[i, j] = float(fuzzy_match(answers[i], answers[j]))

    return E, n_failed

def _expand_thought(t_vec: torch.Tensor, expand_dim: int) -> torch.Tensor:
    """Expand/repeat thought vector to expand_dim rows, mirroring DiscriminatorDataset.

    Args:
        t_vec: [seq_len, d]
    Returns:
        [expanded_len, d]  where expanded_len >= expand_dim (not hard-truncated)
    """
    if t_vec.shape[0] == 1:
        return t_vec.expand(expand_dim, -1)
    elif t_vec.shape[0] < expand_dim:
        rep_needed = ceil(expand_dim / t_vec.shape[0])
        return t_vec.repeat(rep_needed, 1)
    return t_vec

def compute_M(
    thought_vecs: torch.Tensor,
    generated_texts: list[str],
    discriminator: LlamaDiscriminator,
    tr_type: ThoughtRepresentation,
    expand_dim: int,
    max_token_length: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute K×K functional similarity matrix via trained discriminator.

    M[i,j] = 0.5 * (f_disc(T_i, y_j) + f_disc(T_j, y_i))
    where f_disc(T, y) = sigmoid probability from LlamaDiscriminator.predict().

    Args:
        thought_vecs: [seq_len, d] (shared) or [K, 1, d] (per-beam for RV/ENOPOOL)
        generated_texts: List[K] of decoded beam strings
        discriminator: trained LlamaDiscriminator (eval mode)
        tr_type: selects per-beam vs shared thought vector
        expand_dim: must match discriminator training
        max_token_length: truncation limit for y_j token sequences (backbone vocab)
        device, dtype: target device and dtype
    Returns:
        (M, p): M is the symmetrized K×K matrix with diagonal zero;
        p is the raw asymmetric K×K matrix where p[i,j] = f_disc(T_i, y_j).
        Returning both lets downstream analysis separate directional from
        symmetric effects without recomputing.
    """
    K = len(generated_texts)
    per_beam = tr_type in (
        ThoughtRepresentation.RANDOM_VECTOR,
        ThoughtRepresentation.EMBEDDING_NO_POOLING,
    )

    tokenizer = discriminator.tokenizer
    y_ids: list[torch.Tensor] = []
    for text in generated_texts:
        ids = tokenizer.encode(
            text, add_special_tokens=False, truncation=True, max_length=max_token_length
        )
        y_ids.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device))

    p = torch.zeros(K, K)
    for i in range(K):
        if per_beam:
            t_raw = thought_vecs[i]          
        else:
            t_raw = thought_vecs                   

        t_expanded = _expand_thought(t_raw.to(device=device, dtype=dtype), expand_dim)
        t_batched = t_expanded.unsqueeze(0)                        

        for j in range(K):
            if i == j:
                continue
            with torch.no_grad():
                _, probs = discriminator.predict(
                    token_ids=y_ids[j],
                    thought_vecs=t_batched,
                )
            p[i, j] = probs[0].item()

    M = 0.5 * (p + p.T)
    M.fill_diagonal_(0.0)
    return M, p

def dcs_from_matrices(M: torch.Tensor, E: torch.Tensor) -> float:
    """DCS = 1 - MAE over off-diagonal elements of M and E."""
    K = M.shape[0]
    mask = ~torch.eye(K, dtype=torch.bool)
    mae = (M[mask] - E[mask]).abs().mean().item()
    return 1.0 - mae

@hydra.main(version_base=None, config_path="../configs", config_name="dcs_eval")
def main(cfg: DCSEvalConfig) -> None:
    logger = Logger.from_config(cfg.logging)
    logger.info("=" * 70)
    logger.info("DCS Eval")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 70)

    if cfg.wandb.use_wandb:
        import wandb as _wandb
        _wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    if cfg.seed is not None:
        set_seed_all(cfg.seed)

    device = cfg.discriminator.device
    tr_type = ThoughtRepresentation(cfg.tr_type)

    source_hidden_size = cfg.get("source_hidden_size", 4096)
    source_num_layers = cfg.get("source_num_layers", 33)
    other_vector_dim = resolve_other_vector_dim(tr_type, source_hidden_size)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(cfg.discriminator.torch_dtype, torch.bfloat16)

    logger.info(f"Loading discriminator from {cfg.disc_dir}")
    discriminator = LlamaDiscriminator(
        logger=logger,
        model_name=cfg.discriminator.model_name,
        other_vector_dim=other_vector_dim,
        device=device,
        load_in_8bit=cfg.discriminator.load_in_8bit,
        load_in_4bit=cfg.discriminator.load_in_4bit,
        torch_dtype=torch_dtype,
        trust_remote_code=cfg.discriminator.trust_remote_code,
        freeze_base_model=cfg.discriminator.freeze_base_model,
        num_rep_spaces=(source_num_layers if tr_type == ThoughtRepresentation.LAST_INPUT_TOKEN else tr_type.num_features),
    )
    discriminator.load()

    ckpt_path = Path(cfg.disc_dir) / "checkpoints" / "checkpoint_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    discriminator.load_trainable_state_dict(state_dict)
    discriminator.eval()
    discriminator.model.gradient_checkpointing_disable()

    logger.info(f"Loading TR data: {tr_type.value}, think_steps={cfg.think_steps}")
    tr_loader = ThoughtRepresentationLoader(
        logger=logger,
        tr_type=tr_type,
        cached_data_dir=cfg.tr_data_dir,
        split_name="test",
        vec_dim=other_vector_dim,
        think_steps=cfg.think_steps,
    )

    logger.info("Loading embedding_no_pooling for E_emb...")
    emb_loader = ThoughtRepresentationLoader(
        logger=logger,
        tr_type=ThoughtRepresentation.EMBEDDING_NO_POOLING,
        cached_data_dir=cfg.tr_data_dir,
        split_name="test",
        vec_dim=4096,                                                      
    )

    g_indices = list(tr_loader.generated_texts_cache.keys())
    if cfg.num_examples > 0:
        g_indices = g_indices[: cfg.num_examples]
    logger.info(f"Evaluating {len(g_indices)} examples (tau={cfg.tau}, expand_dim={cfg.expand_dim})")

    dcs_emb_values: list[float] = []
    dcs_parse_values: list[float] = []
    per_example_records: list[dict] = []
    total_failed = 0
    total_beams = 0
    skipped = 0

    pbar = tqdm(g_indices, desc=f"DCS [{cfg.tr_type}]", unit="example")
    for g_idx in pbar:
        group_data = tr_loader.get_group_data(g_idx)
        thought_vec = group_data["thought_vec"]
        generated_texts = group_data["generated_texts"]             
        K = len(generated_texts)

        try:
            emb_data = emb_loader.get_group_data(g_idx)
            emb_vecs = emb_data["thought_vec"].squeeze(1).float()          
            emb_vecs = F.normalize(emb_vecs, dim=-1)
        except KeyError:
            logger.warning(f"Embedding data missing for group {g_idx}, skipping")
            skipped += 1
            continue

        cos_sim = compute_cos_sim(emb_vecs)

        E_emb = compute_E_emb(emb_vecs, tau=cfg.tau)
        E_parse, n_failed = compute_E_parse(generated_texts, E_emb)
        total_failed += n_failed
        total_beams += K

        failed_mask = [1 if not _has_answer_prefix(t) else 0 for t in generated_texts]

        M, p = compute_M(
            thought_vecs=thought_vec,
            generated_texts=generated_texts,
            discriminator=discriminator,
            tr_type=tr_type,
            expand_dim=cfg.expand_dim,
            max_token_length=cfg.max_token_length,
            device=device,
            dtype=torch_dtype,
        )

        dcs_emb = dcs_from_matrices(M, E_emb)
        dcs_parse = dcs_from_matrices(M, E_parse)
        dcs_emb_values.append(dcs_emb)
        dcs_parse_values.append(dcs_parse)

        per_example_records.append({
            "g_idx": int(g_idx),
            "K": int(K),
            "cos_sim": cos_sim.to(torch.float32).cpu(),
            "p": p.to(torch.float32).cpu(),
            "M": M.to(torch.float32).cpu(),
            "E_emb": E_emb.to(torch.float32).cpu(),
            "E_parse": E_parse.to(torch.float32).cpu(),
            "failed_mask": failed_mask,
            "n_failed": int(n_failed),
            "dcs_emb": float(dcs_emb),
            "dcs_parse": float(dcs_parse),
        })

        n = len(dcs_emb_values)
        failure_rate = total_failed / max(total_beams, 1)
        pbar.set_postfix(
            dcs_emb=f"{sum(dcs_emb_values)/n:.4f}",
            dcs_parse=f"{sum(dcs_parse_values)/n:.4f}",
            fail=f"{failure_rate:.1%}",
        )

        if cfg.wandb.use_wandb:
            _wandb.log({
                "dcs_emb": dcs_emb,
                "dcs_parse": dcs_parse,
                "running_mean_dcs_emb": sum(dcs_emb_values) / n,
                "running_mean_dcs_parse": sum(dcs_parse_values) / n,
                "extraction_failure_rate": failure_rate,
                "step": n,
            })

    pbar.close()

    if not dcs_emb_values:
        logger.error("No valid examples evaluated.")
        return

    n = len(dcs_emb_values)
    mean_dcs_emb = sum(dcs_emb_values) / n
    mean_dcs_parse = sum(dcs_parse_values) / n
    failure_rate = total_failed / max(total_beams, 1)

    logger.info(f"DCS (E_emb):   {mean_dcs_emb:.4f}")
    logger.info(f"DCS (E_parse): {mean_dcs_parse:.4f}")
    logger.info(
        f"Answer extraction failure rate: {failure_rate:.2%} "
        f"({total_failed}/{total_beams} beams)"
    )
    logger.info(f"Examples evaluated: {n}, skipped: {skipped}")

    result = {
        "tr_type": cfg.tr_type,
        "think_steps": cfg.think_steps,
        "mean_dcs_emb": mean_dcs_emb,
        "mean_dcs_parse": mean_dcs_parse,
        "tau": cfg.tau,
        "n": n,
        "skipped": skipped,
        "extraction_failure_rate": failure_rate,
        "total_failed_extractions": total_failed,
        "total_beams": total_beams,
    }

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{cfg.tr_type}_{cfg.think_steps}.json"
    out_file.write_text(json.dumps(result, indent=2))
    logger.info(f"Results written to {out_file}")

    per_example_file = out_dir / f"{cfg.tr_type}_{cfg.think_steps}_per_example.pt"
    torch.save(
        {
            "tr_type": cfg.tr_type,
            "think_steps": cfg.think_steps,
            "tau": cfg.tau,
            "records": per_example_records,
        },
        per_example_file,
    )
    logger.info(f"Per-example matrices written to {per_example_file}")

    if cfg.wandb.use_wandb:
        _wandb.log(result)
        _wandb.finish()

if __name__ == "__main__":
    main()
