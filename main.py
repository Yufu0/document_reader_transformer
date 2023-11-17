from transformers import VisionEncoderDecoderConfig, AutoConfig, VisionEncoderDecoderModel, AutoModel, RobertaTokenizer, \
    TFBertTokenizer, PreTrainedModel, AutoTokenizer, DonutProcessor
from MyDocParser.DocParser import DocParserConfig, DocParserModel

import torch

if __name__ == '__main__':
    # AutoConfig.register("docparser", DocParserConfig)
    # AutoModel.register(DocParserConfig, DocParserModel)
    #
    # config = VisionEncoderDecoderConfig.from_pretrained("./MyDocParser/")  # fichier config.json
    #
    # model = VisionEncoderDecoderModel(config=config)
    #
    # inputs = torch.ones(1, 3, 100, 100)
    # inputs_ids = torch.ones(10, 512).int()
    #
    # outputs = model(
    #     pixel_values=inputs,
    #     decoder_input_ids=inputs_ids
    # )
    #
    # tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")

    image = torch.ones(3, 500, 500)
    pixel_values = processor(image, return_tensors="pt").pixel_values
    inputs_ids = torch.ones(10, 512).int()

    outputs = model(
        pixel_values=pixel_values,
    )



