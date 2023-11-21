from functools import partial

from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer
import gc

from pretrained_model.pretrained_model import load_model, save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Run on", device)


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    return tokenizer


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

    return {
        "ocr": ocr_tokens
    }


def train(epochs, model, tokenizer, train_loader, optimizer):
    input_ids = torch.ones(1, 512).long().to(device)
    for epoch in range(epochs):
        losses = []
        for batch in train_loader:
            pixel_values, labels = batch["image"], batch["ocr"]
            pixel_values = pixel_values.permute(0, 3, 1, 2).float()

            output = model(pixel_values=pixel_values, decoder_input_ids=input_ids, labels=labels)

            loss = output.loss
            losses.append(loss)

            print(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            del output
            del loss
            
            gc.collect()
        
        # if epoch % 5 == 0:
        #     save_model(model)
        print(f"epoch {epoch} : {sum(losses) / len(losses)}")

def evaluate (model, img_test):
    model.eval()
    print(model(img_test))


if __name__ == '__main__':
    # load the model
    model = load_model().to(device)

    # load tokenizer
    tokenizer = load_tokenizer()

    # load the dataset
    dataset = load_dataset("arvindrajan92/sroie_document_understanding", split="train")
    dataset = dataset.map(partial(preprocessing, tokenizer=tokenizer))
    dataset.set_format("torch", columns=["image", "ocr"])

    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # optimizer for nlp
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)


    train(1, model, tokenizer, train_loader, optimizer)

