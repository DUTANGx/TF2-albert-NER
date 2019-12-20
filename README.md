# TF2-albert-NER
wrapping albert as tfkeras model via bert-for-tf2, implemented NER task

F1 on MSRA: 95%. F1 on Boson: 84%.

Features 2019-12-16: Added Bi-LSTM.

Features 2019-12-10: Added CRF model training on multiple GPUs.

To use:
1. modify the model_dir which contains the albert weight and config.json.
2. initialize model, if first use, please load_pretrained, otherwise you could directly load the trained weights.

