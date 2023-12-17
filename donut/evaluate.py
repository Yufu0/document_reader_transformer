from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
import torch

import re
import json

from tqdm import tqdm
import numpy as np

# from donut import JSONParseEvaluator




def main():
    LOCATION = './donut/pretrained'
    processor = DonutProcessor.from_pretrained(LOCATION)
    model = VisionEncoderDecoderModel.from_pretrained(LOCATION)
    dataset = load_dataset("naver-clova-ix/cord-v2", split="validation")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    output_list = []
    accs = []

    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        # prepare encoder inputs
        pixel_values = processor(sample["image"].convert("RGB"), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        # prepare decoder inputs
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        decoder_input_ids = decoder_input_ids.to(device)

        # autoregressively generate sequence
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.config.decoder.max_length,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        # print(outputs)

        # turn into JSON
        seq = processor.batch_decode(outputs.sequences)[0]
        seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        seq = processor.token2json(seq)

        print(seq)

        ground_truth = json.loads(sample["ground_truth"])
        ground_truth = ground_truth["gt_parse"]
        # evaluator = JSONParseEvaluator()
        # score = evaluator.cal_acc(seq, ground_truth)

        # accs.append(score)
        # output_list.append(seq)

    scores = {"accuracies": accs, "mean_accuracy": np.mean(accs)}
    print(scores, f"length : {len(accs)}")

    print("Mean accuracy:", np.mean(accs))

if __name__ == '__main__':
    main()