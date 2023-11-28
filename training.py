import random
from functools import partial

import datasets.io.parquet
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import gc
import numpy as np

from model import load_model, save_model, push_to_hub
import accelerate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Run on device:", device)


def load_tokenizer():
    return AutoTokenizer.from_pretrained("roberta-base")


class IterDataset(IterableDataset):
    def __init__(self, generator, size):
        self.generator = generator
        self.size = size

    def __iter__(self):
        return self.generator

    def __len__(self):
        return self.size


def load_dataset_sroie(tokenizer=None):
    dataset = load_dataset("arvindrajan92/sroie_document_understanding", split="train")
    dataset = IterDataset(preprocessing(dataset, tokenizer=tokenizer), len(dataset))
    train_loader = DataLoader(dataset=dataset, batch_size=1)
    return train_loader


def preprocessing(dataset, tokenizer=None, max_length=512):
    if tokenizer is None:
        tokenizer = load_tokenizer()

    for line in dataset:
        ocr_tokens = tokenizer(
            " ".join([f"<{elem['label']}>{elem['text']}<{elem['label']}/>" for elem in line["ocr"]])
        )["input_ids"]

        if len(ocr_tokens) > max_length:
            ocr_tokens = ocr_tokens[:max_length]
        if len(ocr_tokens) < max_length:
            ocr_tokens = ocr_tokens + [tokenizer.pad_token_id] * (max_length - len(ocr_tokens))
        ocr_tokens = torch.tensor(ocr_tokens)

        image = line["image"]
        img = np.array(image)
        img = img.transpose((2, 0, 1))
        img = img.astype(np.float32)
        img = img / 255.
        img = torch.tensor(img)

        yield (img, ocr_tokens)


def train(epochs, model, tokenizer, training_dataloader, optimizer, scheduler, accelerator):
    for epoch in range(epochs):
        losses = []
        for batch in tqdm(training_dataloader):
            accelerator.free_memory()
            optimizer.zero_grad()

            pixel_values, labels = batch

            output = model(pixel_values=pixel_values, labels=labels)

            loss = output.loss
            losses.append(loss.item())

            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

        if epoch % 10 == 0:
            print(''.join(tokenizer.batch_decode(labels)))
            print(''.join(tokenizer.batch_decode(output.logits.argmax(dim=-1))))
            print(f"epoch {epoch} : {sum(losses) / len(losses)}")
            push_to_hub(model)


        if device == "cuda":
            output.detach()
            pixel_values.detach()
            labels.detach()
            loss.detach()
            torch.cuda.empty_cache()
        # del output
        del loss
        del pixel_values
        del output
        del labels

        gc.collect()




def evaluate(model, img_test):
    model.eval()
    print(model(img_test))


def main():
    # load tokenizer
    tokenizer = load_tokenizer()

    # load the dataset
    training_dataloader = load_dataset_sroie(tokenizer=tokenizer)

    # load the model
    model = load_model().to(device)
    model.train()

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # optimizer for nlp
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    accelerator = accelerate.Accelerator()
    model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        model, optimizer, training_dataloader, scheduler
    )
    n = 1001
    train(n, model, tokenizer, training_dataloader, optimizer, scheduler, accelerator)


if __name__ == '__main__':
    main()
