import random
from typing import Any, List, Tuple

from nltk import edit_distance
from torch.utils.data import DataLoader, Dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
import torch

import re
import json

from tqdm import tqdm
import numpy as np


# from donut import JSONParseEvaluator
added_tokens = []
class DonutDataset(Dataset):
    """
    PyTorch Dataset for Donut. This class takes a HuggingFace Dataset as input.

    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into pixel_values (vectorized image) and labels (input_ids of the tokenized string).

    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        max_length: the max number of tokens for the target sequences
        split: whether to load "train", "validation" or "test" split
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
        prompt_end_token: the special token at the end of the sequences
        sort_json_key: whether or not to sort the JSON keys
    """

    def __init__(
            self,
            dataset_name_or_path: str,
            max_length: int,
            split: str = "train",
            ignore_id: int = -100,
            task_start_token: str = "<s>",
            prompt_end_token: str = None,
            sort_json_key: bool = True,
            model: VisionEncoderDecoderModel = None,
            processor: DonutProcessor = None,
    ):
        super().__init__()

        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self.sort_json_key = sort_json_key

        self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        self.dataset_length = len(self.dataset)
        self.model = model
        self.processor = processor

        self.gt_token_sequences = []
        for sample in self.dataset:
            ground_truth = json.loads(sample["ground_truth"])
            if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]

            self.gt_token_sequences.append(
                [
                    self.json2token(
                        gt_json,
                        update_special_tokens_for_json_key=self.split == "train",
                        sort_json_key=self.sort_json_key,
                    )
                    + self.processor.tokenizer.eos_token
                    for gt_json in gt_jsons  # load json from list of json
                ]
            )

        self.add_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.add_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                    output += (
                            fr"<s_{k}>"
                            + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                            + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in added_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings of the decoder
        """
        newly_added_num = self.processor.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))
            added_tokens.extend(list_of_tokens)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels
        Convert gt data into input_ids (tokenized string)
        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[idx]

        # inputs
        pixel_values = self.processor(sample["image"], random_padding=self.split == "train",
                                      return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()

        # targets
        target_sequence = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        input_ids = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[
            labels == self.processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token
        # labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
        return pixel_values, labels, target_sequence


def main():
    LOCATION = 'naver-clova-ix/donut-base' #'./donut/pretrained'
    processor = DonutProcessor.from_pretrained(LOCATION)
    model = VisionEncoderDecoderModel.from_pretrained(LOCATION)
    dataset = load_dataset("naver-clova-ix/cord-v2", split="validation")
    max_length = model.config.decoder.max_length

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    train_dataset = DonutDataset(
        "naver-clova-ix/cord-v2",
        max_length=max_length,
        split="train",
        task_start_token="<s_cord-v2>",
        prompt_end_token="<s_cord-v2>",
        sort_json_key=False,
        processor=processor,
        model=model
    )

    val_dataset = DonutDataset(
        "naver-clova-ix/cord-v2",
        max_length=max_length,
        split="validation",
        task_start_token="<s_cord-v2>",
        prompt_end_token="<s_cord-v2>",
        sort_json_key=False,
        processor=processor,
        model=model
    )

    with torch.no_grad():
        for data in val_dataset:
            image = data["image"]
            print(image)
            # prepare decoder inputs
            task_prompt = "<s_cord-v2>"
            decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False,
                                                    return_tensors="pt").input_ids

            pixel_values = processor(image, return_tensors="pt").pixel_values

            outputs = model.generate(
                pixel_values.to(device),
                decoder_input_ids=decoder_input_ids.to(device),
                max_length=model.decoder.config.max_position_embeddings,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=False,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=False,
            )

            sequence = processor.batch_decode(outputs.sequences)[0]
            print(sequence)
            sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
            sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
            print(processor.token2json(sequence))
            print('\n')

    # model.config.pad_token_id = processor.tokenizer.pad_token_id
    # model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s_cord-v2>'])[0]
    #
    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    #
    # for batch in tqdm(train_dataloader):
    #     pixel_values, labels, answers = batch
    #     outputs = model(pixel_values, labels=labels)
    #     loss = outputs.loss
    #     loss.backward()
    #     print(loss.item())
    #     sequences = outputs.logits.argmax(dim=-1)
    #     predictions = []
    #     for seq in processor.tokenizer.batch_decode(sequences):
    #         seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    #         seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
    #         print(seq)
    #     predictions.append(seq)
    #
    #     scores = []
    #     for pred, answer in zip(predictions, answers):
    #         pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
    #         # NOT NEEDED ANYMORE
    #         # answer = re.sub(r"<.*?>", "", answer, count=1)
    #         answer = answer.replace(processor.tokenizer.eos_token, "")
    #         scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))
    #
    #         if len(scores) == 1:
    #             print(f"Prediction: {pred}")
    #             print(f"    Answer: {answer}")
    #             print(f" Normed ED: {scores[0]}")
    #
    #
    # output_list = []
    # accs = []
    # for batch in tqdm(val_dataloader):
    #     pixel_values, labels, answers = batch
    #     batch_size = pixel_values.shape[0]
    #     # we feed the prompt to the model
    #     decoder_input_ids = torch.full((batch_size, 1), model.config.decoder_start_token_id, device=device)
    #
    #     outputs = model.generate(pixel_values,
    #                              decoder_input_ids=decoder_input_ids,
    #                              max_length=model.config.decoder.max_length,
    #                              early_stopping=True,
    #                              pad_token_id=processor.tokenizer.pad_token_id,
    #                              eos_token_id=processor.tokenizer.eos_token_id,
    #                              use_cache=True,
    #                              num_beams=1,
    #                              bad_words_ids=[[processor.tokenizer.unk_token_id]],
    #                              return_dict_in_generate=True, )
    #
    #     predictions = []
    #     for seq in processor.tokenizer.batch_decode(outputs.sequences):
    #         seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    #         seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
    #         predictions.append(seq)
    #
    #     scores = []
    #     for pred, answer in zip(predictions, answers):
    #         pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
    #         # NOT NEEDED ANYMORE
    #         # answer = re.sub(r"<.*?>", "", answer, count=1)
    #         answer = answer.replace(processor.tokenizer.eos_token, "")
    #         scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))
    #
    #         if len(scores) == 1:
    #             print(f"Prediction: {pred}")
    #             print(f"    Answer: {answer}")
    #             print(f" Normed ED: {scores[0]}")
    #
    #     print("val_edit_distance", np.mean(scores))
    #
    #     accs.append(sum(scores))
    #
    # scores = {"accuracies": accs, "mean_accuracy": np.mean(accs)}
    # print(scores, f"length : {len(accs)}")
    #
    # print("Mean accuracy:", np.mean(accs))


if __name__ == '__main__':
    main()
