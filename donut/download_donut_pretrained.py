from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig

image_size = [1280, 960]
max_length = 768

config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
config.encoder.image_size = image_size # (height, width)
# update max_length of the decoder (for generation)
config.decoder.max_length = max_length

model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", config=config)


LOCATION = './donut/pretrained'
model.save_pretrained(LOCATION)

