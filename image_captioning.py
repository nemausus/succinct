from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import argparse
import concurrent.futures
import datetime
import logging
import pandas as pd
import time
import torch
import os

# Setup linux:
# sudo apt install python3-pip
# pip3 install transformers torch pandas Pillow

# Setup mac:
# brew install python3
# python3 -m pip install transformers torch pandas Pillow

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_step(image_dir, filename, max_length, num_beams):
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    image_path = os.path.join(image_dir, filename)
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
    except:
        logging.warning(f'Failed to parse image {filename}')
        return [filename, '']

    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [filename, preds[0].strip()]


def parse_arguments():
    parser = argparse.ArgumentParser('Scrape data from medium.')
    parser.add_argument('--image_dir', required=True, help='path to image dir')
    parser.add_argument('--out_file', required=True, help='name of output file')
    parser.add_argument('--max_length', type=int, default='16')
    parser.add_argument('--num_beams', type=int, default='4')
    parser.add_argument('--num_workers', type=int, default='8')
    return parser.parse_args()


def writeToFile(filename, captions):
    df = pd.DataFrame(captions, columns=['name', 'caption'])
    if os.path.exists(filename):
        df.to_csv(filename, index=False, mode='a', header=False)
    else:
        df.to_csv(filename, index=False)


def main():
    args = parse_arguments()
    log_file = datetime.datetime.today().strftime("%m_%d_%H_%M_") + os.path.basename(__file__) + '.log'
    logging.basicConfig(
            format='%(asctime)s %(levelname)s: %(message)s',
            filename=log_file,
            level=logging.INFO)
    images = next(os.walk(args.image_dir), (None, None, []))[2]  # [] if no file
    images.sort()

    processed = set()
    if os.path.exists(args.out_file):
        df = pd.read_csv(args.out_file)
        for index, row in df.iterrows():
            processed.add(row['name'])
    logging.info(f'Already processed: {len(processed)}')

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers)
    futures = []
    for i in images:
        if not i in processed:
            futures.append(executor.submit(
                predict_step, args.image_dir, i,  args.max_length, args.num_beams))
    to_process = len(futures)
    logging.info(f'To process: {to_process}')
    processed_count = 0
    captions = []
    start = time.perf_counter()
    for f in futures:
        res = f.result()
        if res[1]:
            captions.append(f.result())
        processed_count += 1
        if processed_count % 10 == 0:
            writeToFile(args.out_file, captions)
            captions = []
            avg_time = (time.perf_counter() - start) / processed_count
            # estimated_time_remaining
            etr = avg_time * (to_process - processed_count) / 3600
            logging.info(f'Processed {processed_count}/{to_process}, Avg processing time={avg_time:.2f} secs, Estimated time remaining={etr:.2f} hours')


    writeToFile(args.out_file, captions)


if __name__ == "__main__":
    main()
