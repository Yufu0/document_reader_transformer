from transformers import VisionEncoderDecoderModel

LOCATION = 'Yufu0/document_reader'

def load_model() -> VisionEncoderDecoderModel:
    # load fine-tuned model
    model = VisionEncoderDecoderModel.from_pretrained(LOCATION)
    return model

def save_model(model: VisionEncoderDecoderModel):
    model.save_pretrained(LOCATION)