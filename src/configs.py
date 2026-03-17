import torch
import os


class Config:
    # 1. ĐƯỜNG DẪN THƯ MỤC & FILE
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
    PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
    TOKENIZER_DIR = os.path.join(ROOT_DIR, "data", "tokenizers")
    MODEL_SAVE_DIR = os.path.join(ROOT_DIR, "saved_models")

    # File dữ liệu thô
    TRAIN_EN_PATH = os.path.join(RAW_DATA_DIR, "train.en.txt")
    TRAIN_VI_PATH = os.path.join(RAW_DATA_DIR, "train.vi.txt")
    VAL_EN_PATH = os.path.join(RAW_DATA_DIR, "tst2012.en.txt")
    VAL_VI_PATH = os.path.join(RAW_DATA_DIR, "tst2012.vi.txt")
    TEST_EN_PATH = os.path.join(RAW_DATA_DIR, "tst2013.en.txt")
    TEST_VI_PATH = os.path.join(RAW_DATA_DIR, "tst2013.vi.txt")

    # 2. THÔNG SỐ TIỀN XỬ LÝ & TOKENIZER
    VOCAB_SIZE = 10000
    MAX_SEQ_LEN = 128  # Cắt hoặc đệm các câu về đúng 128 từ

    # Các token đặc biệt bắt buộc phải có
    PAD_TOKEN = "<pad>"  # Dùng để đệm cho bằng MAX_SEQ_LEN
    UNK_TOKEN = "<unk>"  # Từ lạ chưa gặp bao giờ
    SOS_TOKEN = "<s>"  # Start of Sentence (Bắt đầu dịch)
    EOS_TOKEN = "</s>"  # End of Sentence (Kết thúc dịch)


    # 3. KIẾN TRÚC MÔ HÌNH (Tối ưu cho RTX 4060 8GB)
    D_MODEL = 512
    N_HEADS = 8  # Số đầu Attention

    # 4 LỚP
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    D_FF = 2048  # Kích thước mạng nơ-ron ẩn bên trong FeedForward
    DROPOUT = 0.1

    # 4. THÔNG SỐ HUẤN LUYỆN (Training)
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 20
    PATIENCE = 3


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    MODEL_TYPE = "baseline"



cfg = Config()