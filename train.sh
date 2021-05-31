bash iwslt14.sh
python source/process_dataset.py --data_path ./dataset/iwslt14.tokenized.de-en --bin_path ./dataset/bin
python source/main.py --data_bin ./dataset/bin
python source/beamsearch.py --bin_path ./dataset/bin --model_path ./source/model/best_model.pkl --src_lang en --trg_lang de