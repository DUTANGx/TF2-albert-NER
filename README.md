# TF2-albert-NER
wrapping albert as tfkeras layer via [bert-for-tf2](https://github.com/kpe/bert-for-tf2), implemented NER task

F1 on MSRA: 95-99%. F1 on Boson: 80-85%.

Features 2019-12-20: Added Boson data(processed as BIO format).

Features 2019-12-16: Added Bi-LSTM.

Features 2019-12-10: Added CRF model training on multiple GPUs.

To use:
1. modify the model_dir which contains the albert weight and config.json.
2. initialize model, if first use, please load_pretrained, otherwise you could directly use load(). The trained weights(in model_dir/training_checkpoints, which will be created automatically while training) will be loaded.
3. follow the script in model_crf to do NER prediction or create a dockerservice then call(in tf2service).   

