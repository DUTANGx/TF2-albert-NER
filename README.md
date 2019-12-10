# TF2-albert-NER
wrapping albert as keras model via bert-for-tf2, implementing NER task

Features 2019-12-10: Added CRF model training on multiple GPUs.

To use:
1. modify the model_dir which contains the albert weight and config.json.
2. initialize model, if first use, please load_pretrained, otherwise you could directly load the trained weights.

