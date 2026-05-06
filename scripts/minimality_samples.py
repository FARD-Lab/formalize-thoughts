"""Generate sample outputs from a trained minimality probe and compare to references."""

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.dataset.minimality_dataset import MinimalityDataset
from src.model.minimality_probe import ThoughtDescriptor
from src.utils.config import ThoughtRepresentation
from src.utils.logging import Logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to final_model dir")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--tr_type", default="last_input_token")
    parser.add_argument("--think_steps", type=int, default=1)
    parser.add_argument("--tr_data_dir", required=True)
    parser.add_argument("--llm_data_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--output_file", default="./outputs/minimality_samples.txt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = Logger("minimality_samples")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    tr_type = ThoughtRepresentation(args.tr_type)

    logger.info(f"Loading tokenizer from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Building test dataset...")
    dataset = MinimalityDataset(
        logger=logger,
        tokenizer=tokenizer,
        llm_data_dir=args.llm_data_dir,
        tr_data_dir=args.tr_data_dir,
        tr_type=tr_type,
        split_name="test",
        num_return_sequences=8,
        shard_size=1024,
        think_steps=args.think_steps,
        max_input_length=512,
    )
    logger.info(f"Test set size: {len(dataset)}")

    logger.info(f"Loading ThoughtDescriptor from {args.base_model}")
    probe = ThoughtDescriptor(
        logger=logger,
        model_name=args.base_model,
        vector_dim=dataset[0]["input_vecs"].shape[-1],
        device=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        freeze_base_model=True,
    )
    probe.load()

    trained_weights = Path(args.model_dir) / "pytorch_model.bin"
    logger.info(f"Loading trained weights from {trained_weights}")
    state_dict = torch.load(trained_weights, map_location=device)
    missing, unexpected = probe.load_state_dict(state_dict, strict=False)
    logger.info(f"Missing keys: {missing}")
    logger.info(f"Unexpected keys: {unexpected}")
    probe.eval()
    probe.to(device)

    output_lines = []
    num_samples = min(args.num_samples, len(dataset))

    for idx in range(num_samples):
        sample = dataset[idx]
        input_vecs = sample["input_vecs"].unsqueeze(0).to(device, dtype=torch.bfloat16)                 
        target_ids = sample["target_token_ids"]
        reference = tokenizer.decode(target_ids, skip_special_tokens=True)

        with torch.no_grad():
            generated_ids = probe.generate(
                vecs=input_vecs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        block = (
            f"=== Sample {idx + 1} ===\n"
            f"[REFERENCE]\n{reference}\n\n"
            f"[PREDICTED]\n{prediction}\n"
        )
        print(block)
        output_lines.append(block)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"TR type: {args.tr_type}, think_steps: {args.think_steps}\n")
        f.write(f"Model: {args.model_dir}\n\n")
        f.write("\n".join(output_lines))
    logger.info(f"Saved samples to {out_path}")

if __name__ == "__main__":
    main()
