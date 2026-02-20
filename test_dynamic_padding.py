"""Standalone test to verify dynamic batch padding works correctly."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from functools import partial
from diffuseq.text_datasets import dynamic_collate_fn


# --- Mock tokenizer ---
class MockVocab:
    def __init__(self):
        self.pad_token_id = 3
        self.sep_token_id = 1  # [END]
        # [START]=0, [END]=1, [UNK]=2, [PAD]=3, words=4+

    def encode_token(self, sentences):
        """Mimic dict-based tokenizer: [START] + word_ids + [END]"""
        input_ids = []
        for seq in sentences:
            tokens = [4 + hash(w) % 100 for w in seq.split()]
            input_ids.append([0] + tokens + [1])
        return input_ids


# --- Mock embedding ---
EMBED_DIM = 8

def mock_model_emb(input_ids_tensor):
    """Return random embeddings of shape (seq_len, embed_dim)."""
    return torch.randn(input_ids_tensor.shape[0], EMBED_DIM)


# --- Mock dataset that mimics TextDataset with variable-length samples ---
class MockTextDataset(Dataset):
    def __init__(self, input_ids_list, input_mask_list):
        self.input_ids_list = input_ids_list
        self.input_mask_list = input_mask_list

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        input_ids = self.input_ids_list[idx]
        input_mask = self.input_mask_list[idx]
        with torch.no_grad():
            hidden_state = mock_model_emb(torch.tensor(input_ids))
        arr = np.array(hidden_state, dtype=np.float32)
        out_kwargs = {
            'input_ids': np.array(input_ids),
            'input_mask': np.array(input_mask),
        }
        return arr, out_kwargs


def build_test_data():
    """Create samples with varying lengths using mock merge_and_mask logic."""
    vocab = MockVocab()

    # Simulate sentences of different lengths
    samples = [
        ("hello world", "hi"),                          # short
        ("the quick brown fox jumps", "over the lazy dog"),  # medium
        ("a", "b"),                                     # very short
        ("one two three four five six seven", "eight nine ten eleven twelve"),  # long
    ]

    input_ids_list = []
    input_mask_list = []

    for src_text, trg_text in samples:
        src_tokens = vocab.encode_token([src_text])[0]  # [START, ..., END]
        trg_tokens = vocab.encode_token([trg_text])[0]  # [START, ..., END]

        # Mimic merge_and_mask: remove trailing END, re-append, join with SEP
        end_token = src_tokens[-1]
        src = src_tokens[:-1]
        trg = trg_tokens[:-1]
        src.append(end_token)
        trg.append(end_token)

        merged = src + [vocab.sep_token_id] + trg
        mask = [0] * (len(src) + 1)  # src + sep are masked (condition)
        # Extend mask to full length with default 0 for trg portion
        # (in actual code, unmasked positions are implicitly 0 in the list)

        input_ids_list.append(merged)
        input_mask_list.append(mask)

    return input_ids_list, input_mask_list, vocab


def test_dynamic_padding():
    input_ids_list, input_mask_list, vocab = build_test_data()

    print("=" * 60)
    print("Sample lengths (before padding):")
    for i, ids in enumerate(input_ids_list):
        print(f"  Sample {i}: input_ids len={len(ids)}, input_mask len={len(input_mask_list[i])}")
    print()

    dataset = MockTextDataset(input_ids_list, input_mask_list)
    collate_fn = partial(dynamic_collate_fn, pad_token_id=vocab.pad_token_id)

    # Batch size 2 so we get batches with different max lengths
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    batch_seq_lens = []

    for batch_idx, (arr, kwargs) in enumerate(loader):
        input_ids = kwargs['input_ids']
        input_mask = kwargs['input_mask']
        batch_len = input_ids.shape[1]
        batch_seq_lens.append(batch_len)

        print(f"--- Batch {batch_idx} ---")
        print(f"  arr shape:        {arr.shape}  (batch, seq_len, embed_dim)")
        print(f"  input_ids shape:  {input_ids.shape}")
        print(f"  input_mask shape: {input_mask.shape}")

        for sample_idx in range(input_ids.shape[0]):
            global_idx = batch_idx * 2 + sample_idx
            ids = input_ids[sample_idx]
            mask = input_mask[sample_idx]

            orig_ids_len = len(input_ids_list[global_idx])
            orig_mask_len = len(input_mask_list[global_idx])

            print(f"  Sample {sample_idx} (global idx {global_idx}):")
            print(f"    input_ids:  {ids.tolist()}")
            print(f"    input_mask: {mask.tolist()}")
            print(f"    orig lens:  ids={orig_ids_len}, mask={orig_mask_len}")

            # Assert: outputs are torch tensors
            assert isinstance(arr, torch.Tensor), f"FAIL: arr should be torch.Tensor, got {type(arr)}"
            assert isinstance(ids, torch.Tensor), f"FAIL: input_ids should be torch.Tensor, got {type(ids)}"
            assert isinstance(mask, torch.Tensor), f"FAIL: input_mask should be torch.Tensor, got {type(mask)}"

            # Assert: padding in input_ids uses pad_token_id
            ids_pad = ids[orig_ids_len:]
            if len(ids_pad) > 0:
                assert torch.all(ids_pad == vocab.pad_token_id), \
                    f"FAIL: padding positions should be {vocab.pad_token_id}, got {ids_pad}"
                print(f"    [OK] input_ids padding uses pad_token_id={vocab.pad_token_id}")

            # Assert: input_mask condition portion (0..orig_mask_len) is 0
            assert torch.all(mask[:orig_mask_len] == 0), \
                f"FAIL: condition portion of mask should be 0, got {mask[:orig_mask_len]}"
            print(f"    [OK] Condition portion of mask (0:{orig_mask_len}) is all 0")

            # Assert: input_mask beyond condition portion is 1 (trg + padding)
            mask_rest = mask[orig_mask_len:]
            if len(mask_rest) > 0:
                assert torch.all(mask_rest == 1), \
                    f"FAIL: mask beyond condition should be 1, got {mask_rest}"
                print(f"    [OK] Mask beyond condition ({orig_mask_len}:{batch_len}) is all 1")

            # Assert: embedding padding is zero vectors
            emb_pad = arr[sample_idx, orig_ids_len:]
            if emb_pad.shape[0] > 0:
                assert torch.all(emb_pad == 0), \
                    f"FAIL: embedding padding should be zeros"
                print(f"    [OK] Embedding padding positions are zero vectors")
        print()

    # Assert: different batches have different sequence lengths
    print(f"Batch sequence lengths: {batch_seq_lens}")
    if len(set(batch_seq_lens)) > 1:
        print("[OK] Different batches have different sequence lengths (dynamic padding works!)")
    else:
        print("[WARN] All batches have the same length — try with more varied data")

    print()
    print("All assertions passed!")


if __name__ == "__main__":
    test_dynamic_padding()
