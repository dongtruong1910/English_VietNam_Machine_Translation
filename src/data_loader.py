import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
from configs import cfg


class MTDataset(Dataset):
    def __init__(self, src_path, tgt_path, src_tokenizer_path, tgt_tokenizer_path):
        """Đọc file text và chuyển hóa thành các mảng số ID"""
        # 1. Đọc toàn bộ văn bản vào RAM
        with open(src_path, 'r', encoding='utf-8') as f:
            self.src_texts = [line.strip() for line in f if line.strip()]
        with open(tgt_path, 'r', encoding='utf-8') as f:
            self.tgt_texts = [line.strip() for line in f if line.strip()]

        assert len(self.src_texts) == len(self.tgt_texts), "Lỗi nghiêm trọng: Số lượng câu Anh - Việt không bằng nhau!"

        # 2. Load tokenizers đã được huấn luyện sẵn
        self.src_tokenizer = Tokenizer.from_file(src_tokenizer_path)
        self.tgt_tokenizer = Tokenizer.from_file(tgt_tokenizer_path)

        # 3. Lấy mã số (ID) của các ký hiệu đặc biệt
        self.src_sos = self.src_tokenizer.token_to_id(cfg.SOS_TOKEN)
        self.src_eos = self.src_tokenizer.token_to_id(cfg.EOS_TOKEN)
        self.tgt_sos = self.tgt_tokenizer.token_to_id(cfg.SOS_TOKEN)
        self.tgt_eos = self.tgt_tokenizer.token_to_id(cfg.EOS_TOKEN)

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # Biến chữ thành số
        src_encoded = self.src_tokenizer.encode(src_text).ids
        tgt_encoded = self.tgt_tokenizer.encode(tgt_text).ids

        # Cắt gọt nếu câu quá dài (Chừa lại 2 vị trí cho SOS và EOS)
        src_encoded = src_encoded[:cfg.MAX_SEQ_LEN - 2]
        tgt_encoded = tgt_encoded[:cfg.MAX_SEQ_LEN - 2]

        # Kẹp thẻ <s> vào đầu và </s> vào cuối mỗi câu
        src_ids = [self.src_sos] + src_encoded + [self.src_eos]
        tgt_ids = [self.tgt_sos] + tgt_encoded + [self.tgt_eos]

        # Trả về dưới dạng Tensor của PyTorch
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


class CollateFunc:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        """Gom các câu thành một Batch để đẩy vào GPU"""
        src_batch, tgt_batch = zip(*batch)

        # Chỉ đệm các câu ngắn cho bằng với CÂU DÀI NHẤT TRONG BATCH HIỆN TẠI
        src_padded = pad_sequence(src_batch, padding_value=self.pad_id, batch_first=True)
        tgt_padded = pad_sequence(tgt_batch, padding_value=self.pad_id, batch_first=True)

        return src_padded, tgt_padded


def get_dataloaders():
    print("Đang khởi tạo DataLoaders...")

    # Lấy ID của thẻ <pad>
    tmp_tokenizer = Tokenizer.from_file(os.path.join(cfg.TOKENIZER_DIR, "vocab_en.json"))
    pad_id = tmp_tokenizer.token_to_id(cfg.PAD_TOKEN)

    collate_fn = CollateFunc(pad_id)

    # Khởi tạo Dataset cho Train và Val
    train_dataset = MTDataset(
        src_path=os.path.join(cfg.PROCESSED_DATA_DIR, "train.en.txt"),
        tgt_path=os.path.join(cfg.PROCESSED_DATA_DIR, "train.vi.txt"),
        src_tokenizer_path=os.path.join(cfg.TOKENIZER_DIR, "vocab_en.json"),
        tgt_tokenizer_path=os.path.join(cfg.TOKENIZER_DIR, "vocab_vi.json")
    )

    val_dataset = MTDataset(
        src_path=os.path.join(cfg.PROCESSED_DATA_DIR, "val.en.txt"),
        tgt_path=os.path.join(cfg.PROCESSED_DATA_DIR, "val.vi.txt"),
        src_tokenizer_path=os.path.join(cfg.TOKENIZER_DIR, "vocab_en.json"),
        tgt_tokenizer_path=os.path.join(cfg.TOKENIZER_DIR, "vocab_vi.json")
    )

    # Khởi tạo DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    print(f"-> Hoàn tất! Số lượng batch: Train ({len(train_loader)}), Val ({len(val_loader)})")
    return train_loader, val_loader, pad_id


if __name__ == "__main__":
    t_loader, v_loader, p_id = get_dataloaders()
    for src, tgt in t_loader:
        print(f"Kích thước Tensor của 1 Lô (Batch): {src.shape}")  # Ví dụ: [64, 45] (64 câu, dài nhất 45 chữ)
        break