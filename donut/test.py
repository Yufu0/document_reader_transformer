import re

from transformers import DonutProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from datasets import load_dataset
import torch
from tqdm import tqdm


image_size = [1280, 960]
max_length = 768
config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
config.encoder.image_size = image_size  # (height, width)
# update max_length of the decoder (for generation)
config.decoder.max_length = max_length

model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", config=config)

max_length = model.config.decoder.max_length
image_size = model.config.encoder.image_size

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
processor.image_processor.size = image_size[::-1]
processor.image_processor.do_align_long_axis = False

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# load document image
# dataset = load_dataset("hf-internal-testing/example-documents", split="test")
dataset = load_dataset("naver-clova-ix/cord-v2", split="validation")

print(len(dataset))
with torch.no_grad():
    for data in tqdm(dataset[:10]):
        image = data["image"]
        print(image)
        # prepare decoder inputs
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

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

        sequence = processor.batch_decode(outputs)[0]
        print(sequence)
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
        print(processor.token2json(sequence))
        print('\n')