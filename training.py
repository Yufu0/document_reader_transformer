from functools import partial

from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer
import gc

from pretrained_model.pretrained_model import load_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Run on", device)


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    return tokenizer


def preprocessing(image_ocr, tokenizer):
    return {
        "image": image_ocr["image"],
        "ocr": tokenizer(
            " ".join([f"<{elem['label']} > {elem['text']} <{elem['label']} />" for elem in image_ocr["ocr"]])
        )["input_ids"]
    }


def train(epochs, model, tokenizer, train_loader, optimizer):
    for epoch in range(epochs):
        losses = []
        for batch in train_loader:
            pixel_values = batch["image"]
            pixel_values = pixel_values.permute(0, 3, 1, 2)
            pixel_values = pixel_values.float()
            labels = preprocessing(batch["ocr"], tokenizer)

            if len(labels) > 512:
                labels = labels[:300]
            length = len(labels)
            print(device)
            labels = torch.Tensor(labels, device=device).long()

            input_ids = torch.ones(1, length).long().to(device)
            output = model(pixel_values=pixel_values, decoder_input_ids=input_ids, labels=labels)

            loss = output.loss
            losses.append(loss)

            print(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            gc.collect()
        
        if epoch % 5 == 0:
        	save_model(model)
        print(f"epoch {epoch} : {sum(losses) / len(losses)}")

def evaluate (model, img_test):
	model.eval()
	print(model(img_test))


if __name__ == '__main__':
    # load the model
    # model = load_model().to(device)

    # load tokenizer
    tokenizer = load_tokenizer()

    # load the dataset
    dataset = load_dataset("arvindrajan92/sroie_document_understanding", split="train")
    preprocessing_ocr = partial(preprocessing, tokenizer=tokenizer)
    dataset.map(preprocessing_ocr)
    print(dataset[0])
    #
    # data.set_format("torch", columns=["image", "ocr"])
    # train_loader = DataLoader(dataset=data, batch_size=1, shuffle=True)
    #
    #
    # print(device)
    #
    #
    # model.config.decoder_start_token_id = tokenizer.cls_token_id
    # model.config.pad_token_id = tokenizer.pad_token_id
    #
    # # optimizer for nlp
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    #
    #
    # train(1, model, tokenizer, train_loader, optimizer)

