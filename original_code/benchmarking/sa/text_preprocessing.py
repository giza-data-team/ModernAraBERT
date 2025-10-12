import os
import logging
import torch
from datasets import Dataset, DatasetDict
from text_utils import * 
logging.basicConfig(
    filename="SA_Benchmarking.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

def process_dataset(dataset: Dataset, window_size: int, base_dir: str):
    print("Processing and segmenting texts...")
    processed_texts = []
    processed_ids = []
    processed_labels = []

    for example in dataset:
        text = example.get("text", None)
        if text is None or not isinstance(text, str) or text.strip() == "":
            continue 
        doc_id = example.get("id", None)
        label = example.get("label", None)
        if is_arabic_text(text):
            chunks = process_text(text, window_size)
            for chunk in chunks:
                processed_texts.append(chunk)
                processed_ids.append(doc_id)
                processed_labels.append(label)

    print(f"Total processed chunks: {len(processed_texts)}")
    final_dataset = Dataset.from_dict({
        "id": processed_ids,
        "text": processed_texts,
        "label": processed_labels
    })
    split_dataset = final_dataset.train_test_split(test_size=0.4, seed=42)
    test_val_split = split_dataset["test"].train_test_split(test_size=0.5, seed=42)
    dataset_dict = DatasetDict({
        "train": split_dataset["train"],
        "test": test_val_split["train"],
        "validation": test_val_split["test"]
    })
    print("Saving dataset to TXT files...")
    os.makedirs(base_dir, exist_ok=True)
    for split in ["train", "test", "validation"]:
        file_path = os.path.join(base_dir, f"{split}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            for example in dataset_dict[split]:
                if example["label"] is not None:
                    text_line = f"{example['id']}\t{example['text']}\t{example['label']}"
                else:
                    text_line = f"{example['id']}\t{example['text']}"
                f.write(text_line + "\n")
        print(f"Saved {split} split to {file_path}")
    print("Dataset segmentation and splitting complete.")
    print("Files saved: train.txt, test.txt, validation.txt")

def load_sentiment_dataset(file_path, tokenizer, arrow_table, num_labels, max_length=512, dataset_type="default"):
    samples = []
    ratings = []
    
    def is_numeric(s):
        try:
            float(s)
            return True
        except Exception:
            return False

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", maxsplit=2)
            if len(parts) != 3:
                logging.warning(f"Line skipped due to insufficient parts: {line.strip()}")
                continue
            _, text, label = parts
            if text is None or not isinstance(text, str) or text.strip() == "":
                continue
            label = label.strip().replace("|", "").strip()
            if label == "":
                continue

            if dataset_type.lower() == "ajgt":
                if is_numeric(label):
                    rating = int(float(label))
                else:
                    mapping = {"positive": 1, "negative": 0, "pos": 1, "neg": 0}
                    key = label.lower()
                    if key in mapping:
                        rating = mapping[key]
                    else:
                        logging.warning(f"Could not convert AJGT label for sample: {line.strip()}")
                        continue
                sentiment = rating  
            elif dataset_type.lower() == "labr":
                if is_numeric(label):
                    rating = int(float(label))
                else:
                    mapping = {"OBJ": 0, "NEG": 1, "POS": 1, "MIX": 1}
                    key = label.upper()
                    if key in mapping:
                        rating = mapping[key]
                    else:
                        logging.warning(f"Could not convert LABR label for sample: {line.strip()} | Error: {label}")
                        continue
                sentiment = rating
            elif dataset_type.lower() == "astd":
                mapping = {"obj": 0, "pos": 1, "neg": 2, "neu": 3, "neutral": 3}
                key = label.lower()
                if key in mapping:
                    sentiment = mapping[key]
                else:
                    logging.warning(f"Could not convert ASTD label for sample: {line.strip()}")
                    continue
            else:
                if is_numeric(label):
                    rating = int(float(label))
                else:
                    mapping = {"OBJ": 0, "NEUTRAL": 0, "NEG": 1, "POS": 1, "MIX": 1}
                    key = label.upper()
                    if key in mapping:
                        rating = mapping[key]
                    else:
                        logging.warning(f"Could not convert label for sample: {line.strip()}")
                        continue
                sentiment = 0 if rating <= 1 else 1

            if 0 <= sentiment < num_labels:
                samples.append((text, sentiment))
                ratings.append(sentiment)
            else:
                logging.warning(f"Invalid sentiment {sentiment} for text: {text[:30]}...")

    if ratings:
        logging.info(f"Loaded {len(samples)} samples from {file_path}. Label range: min={min(ratings)}, max={max(ratings)}")
    else:
        logging.warning(f"No valid samples loaded from {file_path}.")

    tokenized_samples = []
    for text, label in samples:
        if isinstance(text, list):
            text = " ".join(text)
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_token_type_ids=False
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long).squeeze()
        tokenized_samples.append(item)

    return tokenized_samples

def prepare_astd_benchmark(data_dir, astd_info):
    os.makedirs(data_dir, exist_ok=True)
    main_df = pd.read_csv(
        astd_info["url"],
        sep="\t",
        header=None,
        names=["text", "label"],
        engine="python",
        quoting=csv.QUOTE_NONE
    )
    main_df["id"] = main_df.index.astype(str)
    main_df = main_df[["id", "text", "label"]]
    
    train_ids = pd.read_csv(astd_info["benchmark_train"], header=None, names=["id"], dtype=str)
    train_ids["id"] = train_ids["id"].str.strip().astype(int)
    train_df = pd.merge(train_ids, main_df, on="id", how="left")
    train_df.to_csv(os.path.join(data_dir, "train.txt"), sep="\t", index=False, header=False)

    test_ids = pd.read_csv(astd_info["benchmark_test"], header=None, names=["id"], dtype=str)
    test_ids["id"] = test_ids["id"].str.strip().astype(int)
    test_df = pd.merge(test_ids, main_df, on="id", how="left")
    test_df.to_csv(os.path.join(data_dir, "test.txt"), sep="\t", index=False, header=False)

    val_ids = pd.read_csv(astd_info["benchmark_validation"], header=None, names=["id"], dtype=str)
    val_ids["id"] = val_ids["id"].str.strip().astype(int)
    val_df = pd.merge(val_ids, main_df, on="id", how="left")
    val_df.to_csv(os.path.join(data_dir, "validation.txt"), sep="\t", index=False, header=False)
    print(f"ASTD benchmark files prepared in {data_dir}")

def prepare_labr_benchmark(data_dir, labr_info):
    import pandas as pd
    import os
    os.makedirs(data_dir, exist_ok=True)
    
    main_df = pd.read_csv(
        labr_info["url"],
        sep="\t",
        header=None,
        names=labr_info["column_names"],
        engine="python"
    )
    
    main_df["rating"] = pd.to_numeric(main_df["rating"], errors="coerce")
    main_df = main_df.dropna(subset=["rating"])
    main_df = main_df[main_df["rating"] != 3]
    main_df["label"] = main_df["rating"].apply(lambda x: 1 if x >= 4 else 0)
    
    main_df = main_df.reset_index()  
    main_df["id"] = main_df["index"].astype(int).astype(str)
    main_df["text"] = main_df["review"].astype(str).str.strip()
    main_df = main_df[["id", "text", "label"]]
    
    print("Sample main file IDs (index-based):", main_df["id"].head(5).tolist())
    
    train_ids = pd.read_csv(labr_info["benchmark_train"], header=None, names=["id"], dtype=str)
    train_ids["id"] = train_ids["id"].astype(str).str.strip()
    print("Sample benchmark train IDs:", train_ids["id"].head(5).tolist())
    
    test_ids = pd.read_csv(labr_info["benchmark_test"], header=None, names=["id"], dtype=str)
    test_ids["id"] = test_ids["id"].astype(str).str.strip()
    print("Sample benchmark test IDs:", test_ids["id"].head(5).tolist())
    
    train_df = pd.merge(train_ids, main_df, on="id", how="inner")
    test_df = pd.merge(test_ids, main_df, on="id", how="inner")
    
    train_path = os.path.join(data_dir, "train.txt")
    test_path = os.path.join(data_dir, "test.txt")
    
    train_df.to_csv(train_path, sep="\t", index=False, header=False)
    test_df.to_csv(test_path, sep="\t", index=False, header=False)
    
    print(f"LABR benchmark files prepared in {data_dir}")
    print(f"Train file: {train_path} (rows: {len(train_df)})")
    print(f"Test file: {test_path} (rows: {len(test_df)})")
