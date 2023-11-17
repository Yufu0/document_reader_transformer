from transformers import VisionEncoderDecoderConfig, AutoConfig, VisionEncoderDecoderModel, AutoModel, RobertaTokenizer, \
    TFBertTokenizer, PreTrainedModel, AutoTokenizer
from MyDocParser.DocParser import DocParserConfig, DocParserModel
import numpy as np
import torch

if __name__ == '__main__':
    AutoConfig.register("docparser", DocParserConfig)
    AutoModel.register(DocParserConfig, DocParserModel)

    config = VisionEncoderDecoderConfig.from_pretrained("./MyDocParser/")  # fichier config.json

    model = VisionEncoderDecoderModel(config=config)

    inputs = torch.ones(1, 3, 100, 100)
    inputs_ids = torch.ones(10, 512).int()

    outputs = model(
        pixel_values=inputs,
        decoder_input_ids=inputs_ids
    )

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    #
    # last_hidden_states = outputs.last_hidden_state
    # print(last_hidden_states.shape, last_hidden_states)
    # print(tokenizer.decode(tokenizer.encode("Hello, my dog is cute")))
    # print(len(tokenizer.get_vocab()))


    # model = RobertaModel.from_pretrained("roberta-base")


