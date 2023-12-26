from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig, DonutProcessor


def main():
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

    LOCATION = './donut/pretrained'
    model.save_pretrained(LOCATION)
    processor.save_pretrained(LOCATION)


if __name__ == '__main__':
    main()
