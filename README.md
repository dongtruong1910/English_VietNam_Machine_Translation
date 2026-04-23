# Neural Machine Translation Anh-Việt (Transformer)

Đây là dự án Neural Machine Translation end-to-end cho bài toán English -> Vietnamese, được xây dựng bằng PyTorch. Dự án bao gồm:

- custom Transformer implementation (encoder-decoder)
- BPE tokenization cho cả hai ngôn ngữ
- đầy đủ pipeline training và evaluation
- CLI inference
- FastAPI + HTML web demo


## Mục lục

- [1. Tổng quan dự án](#1-project-overview)
- [2. Cấu trúc repository](#2-repository-structure)
- [3. Data pipeline](#3-data-pipeline)
- [4. Model và chi tiết training](#4-model-and-training-details)
- [5. Cài đặt](#5-installation)


<a id="1-project-overview"></a>
## 1. Tổng quan dự án

Pipeline:

1. Raw parallel corpus (EN-VI)
2. Text cleaning và normalization
3. Training BPE tokenizer (tách riêng EN và VI)
4. Xây dựng Dataset và DataLoader
5. Transformer training với mixed precision
6. Evaluation với SacreBLEU
7. Deployment qua CLI hoặc FastAPI web app

Mục tiêu cốt lõi:

- Xây dựng Transformer từ low-level blocks thay vì dùng high-level wrapper.
- Giữ preprocessing và tokenization có thể tái lập (reproducible).
- Hỗ trợ cả offline evaluation và online translation demo.

<a id="2-repository-structure"></a>
## 2. Cấu trúc repository

```text
.
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- tokenizers/
|-- models/
|   |-- blocks.py
|   `-- transformer.py
|-- results/
|-- saved_models/
|-- src/
|   |-- api.py
|   |-- configs.py
|   |-- data_loader.py
|   |-- evaluate.py
|   |-- inference.py
|   |-- index.html
|   |-- preprocess.py
|   `-- train.py
|-- utils/
|   |-- count_para.py
|   `-- visualize.py
|-- requirements.txt
`-- README.md
```

<a id="3-data-pipeline"></a>
## 3. Data pipeline

### 3.1 Các file raw cần có

Dự án yêu cầu các file song ngữ sau trong `data/raw`:

- `train.en.txt`, `train.vi.txt`
- `tst2012.en.txt`, `tst2012.vi.txt` (validation)
- `tst2013.en.txt`, `tst2013.vi.txt` (test)

### 3.2 Preprocessing (`src/preprocess.py`)

- HTML entity unescape
- loại bỏ các ghi chú trong ngoặc tròn/ngoặc vuông
- quote normalization
- Vietnamese Unicode normalization (NFC)
- lowercase và làm sạch khoảng trắng
- line-by-line parallel filtering để giữ đúng EN-VI alignment

Output files được ghi vào `data/processed`.

### 3.3 Kích thước split đã xử lý 

- Train: 133,163 sentence pairs
- Validation: 1,553 sentence pairs
- Test: 1,268 sentence pairs

Tokenizer files được lưu tại:

- `data/tokenizers/vocab_en.json`
- `data/tokenizers/vocab_vi.json`

### 3.4 Nguồn dataset và link chính thức

Dự án này sử dụng dữ liệu tác vụ IWSLT'15 English-Vietnamese (miền TED talks).

- IWSLT 2015 official page: https://sites.google.com/site/iwsltevaluation2015/
- IWSLT 2015 data page (official "provided/permissible" list): https://sites.google.com/site/iwsltevaluation2015/data-provided
- Stanford NMT preprocessed IWSLT'15 EN-VI split: https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/


## 4. Model và chi tiết training

### 4.1 Kiến trúc

Được cài đặt trong `models/blocks.py` và `models/transformer.py`:

- Encoder-decoder Transformer
- Multi-head attention
- Feed-forward network
- Residual connections + layer normalization
- Sinusoidal positional encoding
- Xavier uniform initialization
- Projection sang target vocabulary

Attention sử dụng `torch.nn.functional.scaled_dot_product_attention`.

### 4.2 Hyperparameters mặc định (`src/configs.py`)

| Parameter | Value |
|---|---:|
| `VOCAB_SIZE` | 10000 |
| `MAX_SEQ_LEN` | 128 |
| `D_MODEL` | 512 |
| `N_HEADS` | 8 |
| `NUM_ENCODER_LAYERS` | 4 |
| `NUM_DECODER_LAYERS` | 4 |
| `D_FF` | 2048 |
| `DROPOUT` | 0.1 |
| `BATCH_SIZE` | 32 |
| `LEARNING_RATE` | 2e-4 |
| `NUM_EPOCHS` | 20 |
| `PATIENCE` | 3 |

### 4.3 Chiến lược training (`src/train.py`)

- Loss: CrossEntropy với label smoothing = 0.1
- Optimizer: Adam (`eps=1e-9`)
- Scheduler: ReduceLROnPlateau
- Mixed precision training (`autocast` + `GradScaler`)
- Auto-resume từ latest epoch checkpoint
- Early stopping dựa trên validation loss

Checkpoints được lưu trong `saved_models`:

- `baseline_epoch_*.pth`
- `baseline_best.pth`

<a id="5-installation"></a>
## 5. Cài đặt

Chạy lệnh từ thư mục gốc của repository.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Với API và plotting utilities, cài thêm các package:

```powershell
pip install fastapi uvicorn pydantic matplotlib
```

