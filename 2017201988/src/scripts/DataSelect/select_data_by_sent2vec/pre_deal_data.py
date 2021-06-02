import re
from os import walk

import pandas as pd
from tqdm import *

img_re_jpg = re.compile(r'<img.*?class="BDE_[a-zA-Z]+".*?src="[^"]*\.jpg".*?>')
img_re_png = re.compile(r'<img.*?class="BDE_[a-zA-Z]+".*?src="[^"]*\.png".*?>')

if __name__ == '__main__':
    raw_data_path = './raw_train_data'
    all_raw_filename = []
    for (dirpath, dirnames, filenames) in walk(raw_data_path):
        all_raw_filename.extend(filenames)

    for raw_filename in all_raw_filename:
        raw_filename_path = raw_data_path + '/' +raw_filename
        raw_data = pd.DataFrame(pd.read_excel(raw_filename_path))

        with open("train_data.txt", 'w+') as f:
            with tqdm(total=raw_data.shape[0]) as pbar:
                for index, row in raw_data.iterrows():
                    one_line = str(row['content_text'])
                    for br_str in one_line.split('<br>'):
                        if len(br_str) == 0:
                            continue
                        for br_dou_ju_str in re.split("[。，]+", br_str):
                            final_str = br_dou_ju_str
                            img_list_jpg = re.findall(img_re_jpg, final_str)
                            if len(img_list_jpg) > 0:
                                for rid_str in img_list_jpg:
                                    final_str = final_str.replace(rid_str, '')
                            img_list_png = re.findall(img_re_png, final_str)
                            if len(img_list_png) > 0:
                                for rid_str in img_list_png:
                                    final_str = final_str.replace(rid_str, '')
                            if len(final_str) < 2:
                                continue
                            f.write(final_str+'\n')
                    pbar.update(1)

