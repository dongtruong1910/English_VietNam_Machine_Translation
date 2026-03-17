import os
import torch
from tokenizers import Tokenizer
import sacrebleu
from tqdm import tqdm

from configs import cfg
from models.transformer import build_transformer
from preprocess import clean_text


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


def greedy_decode(model, source_text, src_tokenizer, tgt_tokenizer, max_len, device):
    """
    (Greedy Search)
    """
    # Lấy ID của các thẻ đặc biệt
    sos_idx = tgt_tokenizer.token_to_id(cfg.SOS_TOKEN)
    eos_idx = tgt_tokenizer.token_to_id(cfg.EOS_TOKEN)
    pad_idx = src_tokenizer.token_to_id(cfg.PAD_TOKEN)

    # 1. Tiền xử lý & Tokenize câu tiếng Anh
    source_text = clean_text(source_text, is_vietnamese=False)
    src_encoded = src_tokenizer.encode(source_text).ids
    src_encoded = [src_tokenizer.token_to_id(cfg.SOS_TOKEN)] + src_encoded[:max_len - 2] + [
        src_tokenizer.token_to_id(cfg.EOS_TOKEN)]

    # Chuyển thành Tensor: (1, seq_len)
    encoder_input = torch.tensor(src_encoded, dtype=torch.long).unsqueeze(0).to(device)
    src_mask = (encoder_input != pad_idx).unsqueeze(1).unsqueeze(2).to(device)

    model.eval()
    with torch.no_grad():
        # 2. Chạy Encoder (Chỉ chạy 1 lần duy nhất cho mỗi câu)
        encoder_output = model.encode(encoder_input, src_mask)

        # 3. Khởi tạo đầu vào cho Decoder: Bắt đầu bằng thẻ <s>
        decoder_input = torch.tensor([[sos_idx]], dtype=torch.long).to(device)

        # 4. Vòng lặp sinh từ
        for _ in range(max_len):
            tgt_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)

            # Chạy qua Decoder
            decoder_output = model.decode(encoder_output, src_mask, decoder_input, tgt_mask)

            # Chỉ lấy dự đoán của TỪ CUỐI CÙNG vừa được sinh ra
            prob = model.project(decoder_output[:, -1, :])

            # Chọn từ có xác suất cao nhất
            _, next_word = torch.max(prob, dim=1)

            # Ghép từ mới vào chuỗi đầu vào của Decoder để dự đoán từ tiếp theo
            decoder_input = torch.cat(
                [decoder_input, torch.tensor([[next_word.item()]], dtype=torch.long).to(device)], dim=1
            )

            # Nếu gặp thẻ kết thúc </s> thì dừng lại
            if next_word.item() == eos_idx:
                break

    # Lấy toàn bộ mảng ID đã sinh ra (bỏ qua thẻ <s> ở đầu)
    generated_ids = decoder_input.squeeze(0).tolist()

    # Giải mã ID thành chữ, skip_special_tokens=True sẽ tự động xóa <s>, </s>, <pad>
    translated_text = tgt_tokenizer.decode(generated_ids, skip_special_tokens=True)
    return translated_text


def evaluate_model():
    device = cfg.DEVICE
    print(f" Bắt đầu đánh giá trên: {device.type.upper()}")

    # 1. Load Tokenizers
    src_tokenizer = Tokenizer.from_file(os.path.join(cfg.TOKENIZER_DIR, "vocab_en.json"))
    tgt_tokenizer = Tokenizer.from_file(os.path.join(cfg.TOKENIZER_DIR, "vocab_vi.json"))

    # 2. Khởi tạo Model
    model = build_transformer(
        src_vocab_size=cfg.VOCAB_SIZE, tgt_vocab_size=cfg.VOCAB_SIZE,
        src_seq_len=cfg.MAX_SEQ_LEN, tgt_seq_len=cfg.MAX_SEQ_LEN,
        d_model=cfg.D_MODEL, N=cfg.NUM_ENCODER_LAYERS, h=cfg.N_HEADS,
        dropout=cfg.DROPOUT, d_ff=cfg.D_FF
    ).to(device)

    # 3. Load Model
    best_model_path = os.path.join(cfg.MODEL_SAVE_DIR, f"{cfg.MODEL_TYPE}_best.pth")

    if not os.path.exists(best_model_path):
        print(f" Không tìm thấy model tại {best_model_path}. Vui lòng train xong rồi mới chạy file này!")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(" Đã load thành công tệp weights (Best Model).")

    # 4. Đọc dữ liệu tập TEST
    en_test_path = cfg.TEST_EN_PATH
    vi_test_path = cfg.TEST_VI_PATH

    with open(en_test_path, 'r', encoding='utf-8') as f:
        src_sentences = [line.strip() for line in f if line.strip()]
    with open(vi_test_path, 'r', encoding='utf-8') as f:
        tgt_references = [line.strip() for line in f if line.strip()]

    # Để đảm bảo an toàn, làm sạch references giống hệt lúc preprocess
    tgt_references = [clean_text(vi, is_vietnamese=True) for vi in tgt_references]

    assert len(src_sentences) == len(tgt_references), "Lỗi: Số câu Anh - Việt trong tập Test không khớp!"

    print(f"Tổng số câu test: {len(src_sentences)}")

    # 5. Dịch máy
    predictions = []

    for src in tqdm(src_sentences, desc="Đang dịch (Greedy Search)"):
        pred = greedy_decode(model, src, src_tokenizer, tgt_tokenizer, cfg.MAX_SEQ_LEN, device)
        predictions.append(pred)

    # 6. Tính điểm BLEU
    bleu = sacrebleu.corpus_bleu(predictions, [tgt_references])

    print("\n" + "=" * 50)
    print(f"🏆 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (tst2013)")
    print("=" * 50)
    print(f" Điểm BLEU Score: {bleu.score:.2f}")
    print(f" Chi tiết n-grams: {bleu.counts}")
    print("=" * 50)

    print("\n--- IN THỬ VÀI CÂU DỊCH ---")
    for i in range(5):
        print(f"🇺🇸 Tiếng Anh: {src_sentences[i]}")
        print(f"🇻🇳 Thực tế  : {tgt_references[i]}")
        print(f"🤖 AI Dịch  : {predictions[i]}")
        print("-" * 30)


if __name__ == "__main__":
    evaluate_model()