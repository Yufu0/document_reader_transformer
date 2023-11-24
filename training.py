from functools import partial

from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer
import gc
import numpy as np

from pretrained_model.pretrained_model import load_model, save_model
import accelerate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Run on", device)


def load_tokenizer():
    return AutoTokenizer.from_pretrained("roberta-base")


def load_dataset_sroie(tokenizer=None):
    dataset = load_dataset("arvindrajan92/sroie_document_understanding", split="train")
    dataset = dataset.shard(num_shards=100, index=0)
    dataset = dataset.map(partial(preprocessing, tokenizer=tokenizer))
    dataset.set_format("torch", columns=["image", "ocr"])
    # dataset = dataset.map(lambda elem: {"image": elem["image"].permute(2, 0, 1).float() / 255.})
    #
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    return train_loader


def preprocessing(image_ocr, tokenizer=None, max_length=512):
    if tokenizer is None:
        tokenizer = load_tokenizer()

    ocr_tokens = tokenizer(
        " ".join([f"<{elem['label']} > {elem['text']} <{elem['label']} />" for elem in image_ocr["ocr"]])
    )["input_ids"]

    if len(ocr_tokens) > max_length:
        ocr_tokens = ocr_tokens[:max_length]
    if len(ocr_tokens) < max_length:
        ocr_tokens = ocr_tokens + [tokenizer.pad_token_id] * (max_length - len(ocr_tokens))

    img = np.array(image_ocr["image"])
    img = img / 255.
    img = img.astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    return {
        "image": img,
        "ocr": ocr_tokens
    }


def train(epochs, model, tokenizer, training_dataloader, optimizer, scheduler, accelerator):
    for epoch in range(epochs):
        losses = []
        for batch in training_dataloader:
            accelerator.free_memory()
            optimizer.zero_grad()

            pixel_values, labels = batch["image"], batch["ocr"]

            output = model(pixel_values=pixel_values, labels=labels)

            loss = output.loss
            losses.append(loss.item())

            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            # del output
            del loss
            del pixel_values
            del labels

            gc.collect()

        if epoch % 100 == 0:
            print(''.join(tokenizer.batch_decode(labels)))
            print(''.join(tokenizer.batch_decode(output.logits.argmax(dim=-1))))
            print(f"epoch {epoch} : {sum(losses) / len(losses)}")
            save_model(model)


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
    n = 10
    train(n, model, tokenizer, training_dataloader, optimizer, scheduler, accelerator)


if __name__ == '__main__':
    main()
