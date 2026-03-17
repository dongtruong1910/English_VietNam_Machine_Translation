import os
import html
import re
import unicodedata
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from configs import cfg


def clean_text(text, is_vietnamese=False):
    # 1. Giải mã kí tự HTML (&apos;, &quot;,...)
    text = html.unescape(text)

    # 2. Xóa các chú thích (Applause), [Music]
    text = re.sub(r'\(.*?\)|\[.*?\]', '', text)

    # 3. Đồng nhất các loại dấu nháy thông minh về dấu nháy chuẩn
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r'[‘’]', "'", text)

    # 4. Chuẩn hóa bảng mã Tiếng Việt về chuẩn NFC
    if is_vietnamese:
        text = unicodedata.normalize('NFC', text)

    # 5. Ép mọi khoảng trắng/tab/enter dị dạng về 1 dấu cách và in thường
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def process_parallel_files(en_raw_path, vi_raw_path, en_out_path, vi_out_path):
    """Hàm xử lý đồng thời 2 file để đảm bảo tính gióng hàng (parallel) giữa câu Anh và câu Việt"""
    if not os.path.exists(en_raw_path) or not os.path.exists(vi_raw_path):
        print(f"Lỗi: Không tìm thấy file {en_raw_path} hoặc {vi_raw_path}")
        return False

    with open(en_raw_path, 'r', encoding='utf-8') as f_en, \
            open(vi_raw_path, 'r', encoding='utf-8') as f_vi, \
            open(en_out_path, 'w', encoding='utf-8') as out_en, \
            open(vi_out_path, 'w', encoding='utf-8') as out_vi:

        # zip() giúp đọc song song từng cặp dòng của 2 file
        for line_en, line_vi in zip(f_en, f_vi):
            clean_en = clean_text(line_en, is_vietnamese=False)
            clean_vi = clean_text(line_vi, is_vietnamese=True)

            # Cả 2 câu đều phải có nội dung thì mới được ghi
            if clean_en and clean_vi:
                out_en.write(clean_en + '\n')
                out_vi.write(clean_vi + '\n')
    return True


def process_and_save_data():
    print("Bắt đầu dọn dẹp dữ liệu (Chế độ Gióng hàng Song song)...")
    os.makedirs(cfg.PROCESSED_DATA_DIR, exist_ok=True)

    file_pairs = [
        (cfg.TRAIN_EN_PATH, cfg.TRAIN_VI_PATH, "train.en.txt", "train.vi.txt"),
        (cfg.VAL_EN_PATH, cfg.VAL_VI_PATH, "val.en.txt", "val.vi.txt"),
        (cfg.TEST_EN_PATH, cfg.TEST_VI_PATH, "test.en.txt", "test.vi.txt")
    ]

    processed_paths = {}
    for en_raw, vi_raw, en_new, vi_new in file_pairs:
        en_out = os.path.join(cfg.PROCESSED_DATA_DIR, en_new)
        vi_out = os.path.join(cfg.PROCESSED_DATA_DIR, vi_new)

        success = process_parallel_files(en_raw, vi_raw, en_out, vi_out)
        if success:
            processed_paths[en_new] = en_out
            processed_paths[vi_new] = vi_out
            print(f"  -> Đã xử lý xong cặp: {en_new} & {vi_new}")

    return processed_paths


def train_tokenizer(file_path, lang_name):
    """Huấn luyện bộ từ điển BPE """
    print(f"\nĐang huấn luyện Tokenizer cho {lang_name}...")

    # mô hình Byte-Pair Encoding
    tokenizer = Tokenizer(BPE(unk_token=cfg.UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()  # Tách từ cơ bản bằng dấu cách

    # Cấu hình Trainer
    trainer = BpeTrainer(
        vocab_size=cfg.VOCAB_SIZE,
        special_tokens=[cfg.PAD_TOKEN, cfg.UNK_TOKEN, cfg.SOS_TOKEN, cfg.EOS_TOKEN]
    )

    # Dạy cho Tokenizer học từ file text
    tokenizer.train(files=[file_path], trainer=trainer)

    # Lưu file JSON ra thư mục
    os.makedirs(cfg.TOKENIZER_DIR, exist_ok=True)
    save_path = os.path.join(cfg.TOKENIZER_DIR, f"vocab_{lang_name}.json")
    tokenizer.save(save_path)
    print(f"  -> Đã lưu Từ điển tại: {save_path}")


if __name__ == "__main__":
    processed_files = process_and_save_data()

    # Bước 2: Huấn luyện Từ điển từ tập TRAIN
    if "train.en.txt" in processed_files and "train.vi.txt" in processed_files:
        train_tokenizer(processed_files["train.en.txt"], "en")
        train_tokenizer(processed_files["train.vi.txt"], "vi")

    print("\nHoàn tất 100% tiền xử lý! Dữ liệu đã sẵn sàng cho DataLoader.")