## Requirements

- Python 3.11
- Git

Python packages (install inside a virtual env):

```bash
pip install -r requirements.txt
```

You also need the base model folder:
./qwen2.5-Coder-0.5B (downloaded from Hugging Face and placed in the project root)

Files

clone_repos.py – clone top N GitHub repos from a CSV (results.csv from SEART)

build_dataset.py – parse Java files, extract methods, deduplicate, build train/test JSONL

utils.py – shared helpers (method extraction, load/save pairs, FIM helpers, debug helpers, analyzing helpers)

trainer.py – fine-tune Qwen2.5-Coder with FIM on the training set

evaluate.py – run inference on the test set and report exact-match accuracy

analyze_prediction.py analyze the model’s prediction results to understand where it performs well or poorly by reporting accuracy, similarity between true and predicted method names, and whether the true-name subtokens appear in the method body. 

Usage

python clone_repos.py --csv results.csv --rows 500 --dest repos

python build_dataset.py \
  --repos-dir repos \
  --max-methods 2000000 \
  --max-unique 50000 \
  --train-file train_dataset.jsonl \
  --test-file  test_dataset.jsonl

python trainer.py \
  --train-file train_dataset.jsonl \
  --model-dir ./qwen2.5-Coder-0.5B \
  --output-dir qwen_methodname_finetune \
  --debug-examples 1

python evaluate.py \
  --model-dir qwen_methodname_finetune \
  --test-file test_dataset.jsonl \
  --predictions-file predictions.jsonl

Run analyze_prediction, you need:
pip install nltk
python -c "import nltk; nltk.download('wordnet')"

python analyze_prediction.py --predictions-file predictions.jsonl