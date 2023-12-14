from transformers import VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

LOCATION = './donut/pretrained'
model.save_pretrained(LOCATION)

