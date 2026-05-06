from fmri_mapping.io.nsd import get_resource

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm, trange
import os
import argparse
from pathlib import Path

BATCH_SIZE = 64

MODELS = {
    "gtr": "sentence-transformers/gtr-t5-base",
    "e5": "intfloat/e5-base-v2",
    "stella": "infgrad/stella-base-en-v2",
    "granite": "ibm-granite/granite-embedding-278m-multilingual",
    "granite-small": "ibm-granite/granite-embedding-30m-english",
    "e5-small": "intfloat/e5-small",
    "all-mini": "all-MiniLM-L6-v2",
}


def get_embeddings(model, caption, batch_size=32, device="cpu"):
    all_embeddings = []
    for i in trange(0, len(caption), batch_size):
        batch_texts = caption[i : i + batch_size]
        embeddings = model.encode(batch_texts, convert_to_tensor=True, device=device)
        all_embeddings.append(embeddings)

    embeddings_tensor = torch.cat(all_embeddings)
    return embeddings_tensor


def process_embedding(model_key, captions, labels, batch_size, device):
    model_name = MODELS[model_key]
    encoder = SentenceTransformer(model_name).to(device)
    sent_embeds = get_embeddings(
        encoder, captions, batch_size=batch_size, device=device
    ).cpu()

    pck = {
        "Z": sent_embeds,
        "labels": labels.cpu(),
        "model_name": model_name,
        # "captions": captions,
    }

    return pck


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate language embeddings for NSD stimuli."
    )
    parser.add_argument(
        "--model_keys",
        nargs="+",
        default=None,
        help="List of model keys to generate embeddings for.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to the folder where the embeddings will be saved.",
        default="language_embeddings",
    )

    parser.add_argument(
        "--template",
        type=str,
        default="nsd_language_embeddings_{model_key}.pt",
    )

    args = parser.parse_args()

    args.output_folder = Path(args.output_folder)

    if args.model_keys is None:
        args.model_keys = list(MODELS.keys())
    else:
        for key in args.model_keys:
            if key not in MODELS:
                raise ValueError(
                    f"Model key {key} is not valid. Valid keys are: {list(MODELS.keys())}"
                )

    return args


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the captions
    captions = get_resource("coco-captions")
    caption = captions.captions.str.split(";").str[0].tolist()
    labels = torch.tensor(captions.nsd_id.values, device="cpu")

    args = parse_args()
    model_keys = args.model_keys
    output_folder = args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    for model_key in tqdm(model_keys):
        output_file = output_folder / args.template.format(model_key=model_key)
        pck = process_embedding(
            model_key=model_key,
            captions=caption,
            labels=labels,
            batch_size=BATCH_SIZE,
            device=device,
        )
        torch.save(pck, output_file)


if __name__ == "__main__":
    main()
