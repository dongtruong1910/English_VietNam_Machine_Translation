import os
import torch
from tokenizers import Tokenizer

from configs import cfg
from models.transformer import build_transformer
from evaluate import greedy_decode


def load_model_for_inference():
    device = cfg.DEVICE
    print(f"⏳ Đang tải mô hình lên {device.type.upper()}...")

    # 1. Load Tokenizers
    src_tokenizer = Tokenizer.from_file(os.path.join(cfg.TOKENIZER_DIR, "vocab_en.json"))
    tgt_tokenizer = Tokenizer.from_file(os.path.join(cfg.TOKENIZER_DIR, "vocab_vi.json"))

    # 2. Khởi tạo kiến trúc
    model = build_transformer(
        src_vocab_size=cfg.VOCAB_SIZE, tgt_vocab_size=cfg.VOCAB_SIZE,
        src_seq_len=cfg.MAX_SEQ_LEN, tgt_seq_len=cfg.MAX_SEQ_LEN,
        d_model=cfg.D_MODEL, N=cfg.NUM_ENCODER_LAYERS, h=cfg.N_HEADS,
        dropout=cfg.DROPOUT, d_ff=cfg.D_FF
    ).to(device)

    # 3. Bơm trọng số (weights)
    model_path = os.path.join(cfg.MODEL_SAVE_DIR, f"{cfg.MODEL_TYPE}_best.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model tại {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Mô hình đã sẵn sàng!\n")
    print("=" * 50)

    return model, src_tokenizer, tgt_tokenizer, device


def main():
    # Tải mô hình 1 lần duy nhất
    model, src_tokenizer, tgt_tokenizer, device = load_model_for_inference()

    print("🤖 AI DỊCH THUẬT ANH - VIỆT (Gõ 'quit' hoặc 'exit' để thoát)")
    print("=" * 50)

    # Vòng lặp chat liên tục
    while True:
        input_text = input("🇺🇸 Nhập Tiếng Anh: ")

        # Kiểm tra lệnh thoát
        if input_text.strip().lower() in ['quit', 'exit']:
            print("👋 Tạm biệt!")
            break

        if not input_text.strip():
            continue

        # Tiến hành dịch
        try:
            translation = greedy_decode(
                model=model,
                source_text=input_text,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                max_len=cfg.MAX_SEQ_LEN,
                device=device
            )
            print(f"🇻🇳 AI Dịch     : {translation}")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Có lỗi xảy ra: {e}")


if __name__ == "__main__":
    main()