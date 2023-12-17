from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import re


def main():
    dataset = load_dataset("hf-internal-testing/example-documents")
    image = dataset['test'][2]['image']
    print(image)

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

    pixel_values = processor(image, return_tensors="pt").pixel_values
    print(pixel_values.shape)

    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    decoder_input_ids = torch.full((1, 1), task_prompt)

    print(decoder_input_ids.shape)
    print(decoder_input_ids)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    outputs = model.generate(pixel_values.to(device),
                             decoder_input_ids=decoder_input_ids.to(device),
                             max_length=model.decoder.config.max_position_embeddings,
                             # early_stopping=True,
                             # pad_token_id=processor.tokenizer.pad_token_id,
                             # eos_token_id=processor.tokenizer.eos_token_id,
                             # use_cache=True,
                             # num_beams=1,
                             # bad_words_ids=[[processor.tokenizer.unk_token_id]],
                             # return_dict_in_generate=True,
                             # output_scores=True,
                             )

    print(outputs)
    print(processor.batch_decode(outputs))

    # sequence = processor.batch_decode(outputs.sequences)[0]
    # print(sequence)
    # sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    # sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    # print(sequence)
    #
    # json_out = processor.token2json(sequence)
    #
    # print(json_out)
    #
    # with open("../test3.json", "w") as f:
    #     f.write(str(json_out))


if __name__ == '__main__':
    main()
