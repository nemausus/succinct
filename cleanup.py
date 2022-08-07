import argparse
import os
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser('Scrape data from medium.')
    parser.add_argument('--year', required=True, type=int, help='year')
    return parser.parse_args()


def main():
    args = parse_arguments()

    year = args.year
    input_file = f'{year}_medium_data.csv'
    output_file = f'{year}_medium_data_new.csv'
    skipped_file = f'{year}_medium_data_skipped.csv'
    image_dir = f'{year}_images'

    df = pd.read_csv(input_file)
    rows = []
    skipped = []
    for index, row in df.iterrows():
        image = row['image']
        if os.path.exists(f'{image_dir}/{image}'):
            rows.append(row)
        else:
            skipped.append(row)

    fields = ['id', 'url', 'title', 'subtitle', 'first_para', 'image', 'claps',
            'responses', 'reading_time', 'publication', 'date']

    out_df = pd.DataFrame(rows, columns=fields)
    out_df.to_csv(output_file, index=False)

    out_df = pd.DataFrame(skipped, columns=fields)
    out_df.to_csv(skipped_file, index=False)


if __name__ == "__main__":
    main()
