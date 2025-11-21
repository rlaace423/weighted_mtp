
import torch
from weighted_mtp.data.dataloader import create_dataloader
from weighted_mtp.data.collators import AlpacaDataCollator
from transformers import PreTrainedTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    unk_token_id = 3
    padding_side = "right"
    
    def __call__(self, text, return_tensors=None, padding=None, truncation=None, max_length=None):
        # Return dummy input_ids and attention_mask
        return {
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1]
        }
    
    def pad(self, encoded_inputs, padding=True, return_tensors="pt"):
        # Simple padding mock
        batch_size = len(encoded_inputs["input_ids"])
        max_len = max(len(ids) for ids in encoded_inputs["input_ids"])
        
        padded_ids = []
        padded_mask = []
        
        for i in range(batch_size):
            ids = encoded_inputs["input_ids"][i]
            mask = encoded_inputs["attention_mask"][i]
            padding_len = max_len - len(ids)
            
            padded_ids.append(ids + [self.pad_token_id] * padding_len)
            padded_mask.append(mask + [0] * padding_len)
            
        return {
            "input_ids": torch.tensor(padded_ids),
            "attention_mask": torch.tensor(padded_mask)
        }

def test_dataloader():
    tokenizer = DummyTokenizer()
    
    # Config mimicking the training setup
    dataset_dir = "/Users/wesley/Desktop/wooshikwon/weighted_mtp/storage/datasets"
    dataset_name = "codecontests"
    
    logger.info("Creating dataloader...")
    dataloader = create_dataloader(
        tokenizer=tokenizer,
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        split="train",
        batch_size=4,
        max_length=512,
        num_workers=0,
        balance_correct=True,
        correct_ratio=0.5,
        seed=42
    )
    
    logger.info("Iterating dataloader...")
    total_correct = 0
    total_samples = 0
    
    for i, batch in enumerate(dataloader):
        if i >= 10: break
        
        is_correct = batch["is_correct"]
        logger.info(f"Batch {i}: is_correct mean = {is_correct.mean().item():.4f}, values = {is_correct.tolist()}")
        
        total_correct += is_correct.sum().item()
        total_samples += is_correct.numel()
        
    logger.info(f"Total mean correct: {total_correct / total_samples:.4f}")

if __name__ == "__main__":
    test_dataloader()
