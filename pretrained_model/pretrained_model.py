from transformers import VisionEncoderDecoderModel
from transformers import AutoTokenizer, AlbertForMaskedLM

def create_pretrained_model() -> VisionEncoderDecoderModel:
    # initialize a vit-bert from a pretrained ViT and a pretrained BERT model. Note that the cross-attention layers will be randomly initialized
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        "microsoft/swinv2-tiny-patch4-window8-256",
        "albert-base-v2",
    )
    # saving model after fine-tuning
    model.save_pretrained("./model/")
    # load fine-tuned model
    model = VisionEncoderDecoderModel.from_pretrained("./model/")

    return model


# if __name__ == '__main__':
#     model = create_pretrained_model()
#     print(model)

def load_model() -> VisionEncoderDecoderModel:
    # load fine-tuned model
    model = VisionEncoderDecoderModel.from_pretrained("./pretrained_model/model/")
    return model

def save_model(model: VisionEncoderDecoderModel):
    model.save_pretrained("./pretrained_model/model/")