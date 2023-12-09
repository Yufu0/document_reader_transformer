from typing import re

from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, VisionEncoderDecoderModel, DonutProcessor
import gc
import numpy as np

from model import load_model, save_model, push_to_hub
import accelerate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Run on device:", device)


def load_tokenizer():
    return AutoTokenizer.from_pretrained("roberta-base")


class IterDataset(IterableDataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.size = len(dataset)
        self.generator = preprocessing(dataset, tokenizer=tokenizer)
        self.tokenizer = tokenizer

    def __iter__(self):
        if self.generator.gi_frame is None:
            self.generator = preprocessing(self.dataset, tokenizer=self.tokenizer)
        return self.generator

    def __len__(self):
        return self.size


def load_dataset_sroie(tokenizer=None):
    dataset = load_dataset("arvindrajan92/sroie_document_understanding", split="train").shuffle()
    # dataset = dataset.shard(num_shards=100, index=0)
    dataset = IterDataset(dataset, tokenizer)
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
        # if image.width * image.height > 1_000_000:
        #     ratio = np.sqrt(1_000_000 / (image.width * image.height))
        #     image = image.resize((int(image.width * ratio), int(image.height * ratio)))
        img = np.array(image)
        img = img.transpose((2, 0, 1))
        img = img.astype(np.float32)
        # img = img / 255.
        # img = torch.tensor(img)

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

            # del loss
            # del pixel_values
            # del output
            # del labels

            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        print(f"epoch {epoch} : {sum(losses) / len(losses)}")
        if epoch % 10 == 0:
            save_model(model)
            push_to_hub(model)

            evaluate(model, pixel_values, labels, tokenizer)


def evaluate(model, pixel_values, labels, tokenizer):

    with torch.no_grad():
        decoder_input_ids = torch.ones(1, 512).int()
        output = model(pixel_values=pixel_values, labels=labels)
        output = output.logits.detach().cpu()
        output = output.argmax(dim=-1)
        print("True :", tokenizer.batch_decode(labels))
        print("Predicted :", tokenizer.batch_decode(output))


def main():
    # load tokenizer
    # tokenizer = load_tokenizer()

    # load the dataset
    # training_dataloader = load_dataset_sroie(tokenizer=tokenizer)

    # load the model
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip", use_fast=False)
    # model.train()
#AutoTokenizer
    # model.config.decoder_start_token_id = tokenizer.cls_token_id
    # model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset("hf-internal-testing/example-documents", split="test")
    image = dataset[1]["image"]

    # prepare decoder inputs
    task_prompt = "<s_rvlcdip>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    pixel_values = processor(image, return_tensors="pt").pixel_values
    print(pixel_values.shape)

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    print(processor.token2json(sequence))


    # accelerator = accelerate.Accelerator()
    # model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    #     model, optimizer, training_dataloader, scheduler
    # )
    n = 1001
    # train(n, model, tokenizer, training_dataloader, optimizer, scheduler, accelerator)


if __name__ == '__main__':
    main()
