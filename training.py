from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, VisionEncoderDecoderConfig, VisionEncoderDecoderModel, AutoTokenizer
import torch
from MyDocParser.DocParser import DocParserConfig, DocParserModel


def load_model():
    AutoConfig.register("docparser", DocParserConfig)
    AutoModel.register(DocParserConfig, DocParserModel)

    config = VisionEncoderDecoderConfig.from_pretrained("./")  # fichier config.json

    model = VisionEncoderDecoderModel(config=config)

    return model


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    return tokenizer


def preprocessing(ocr, tokenizer):
    return tokenizer(
        " ".join([f"<{elem['label']} > {elem['text']} <{elem['label']} />" for elem in ocr])
    )["input_ids"]


def train(epochs, model, tokenizer, train_loader, criterion, optimizer):
    for epoch in range(epochs):
        for batch in train_loader:
            pixel_values = batch["image"]
            pixel_values = pixel_values.permute(0, 3, 1, 2)
            pixel_values = pixel_values.float().to("cuda")
            labels = preprocessing(batch["ocr"], tokenizer)

            if len(labels) > 512:
                labels = labels[:300]
            length = len(labels)
            labels = torch.Tensor(labels).long().to("cuda")

            input_ids = torch.ones(1, length).long().to("cuda")
            output = model(pixel_values=pixel_values, decoder_input_ids=input_ids, labels=labels)

            loss = output.loss

            print(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            break


if __name__ == '__main__':
    data = load_dataset("arvindrajan92/sroie_document_understanding", split="train")
    data.set_format("torch", columns=["image", "ocr"])
    # data_ocr = preprocessing(data["ocr"])
    # data_image = torch.Tensor([img for img in data["image"]])
    train_loader = DataLoader(dataset=data, batch_size=1, shuffle=True)

    model = load_model().to("cuda")
    tokenizer = load_tokenizer()

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # optimizer for nlp
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # criterion for nlp
    criterion = torch.nn.CrossEntropyLoss()

    train(1, model, tokenizer, train_loader, criterion, optimizer)

