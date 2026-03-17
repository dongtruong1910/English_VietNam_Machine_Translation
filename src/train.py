import os
import glob
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast  # ✅ Cú pháp AMP chuẩn mới
from tqdm import tqdm

from configs import cfg
from data_loader import get_dataloaders
from models.transformer import build_transformer


def causal_mask(size):
    """Mặt nạ che tương lai cho Decoder"""
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


def train_model():
    device = cfg.DEVICE
    print(f"🔥 Bắt đầu huấn luyện trên thiết bị: {device.type.upper()} 🔥")

    train_loader, val_loader, pad_id = get_dataloaders()

    model = build_transformer(
        src_vocab_size=cfg.VOCAB_SIZE, tgt_vocab_size=cfg.VOCAB_SIZE,
        src_seq_len=cfg.MAX_SEQ_LEN, tgt_seq_len=cfg.MAX_SEQ_LEN,
        d_model=cfg.D_MODEL, N=cfg.NUM_ENCODER_LAYERS, h=cfg.N_HEADS,
        dropout=cfg.DROPOUT, d_ff=cfg.D_FF
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1).to(device)

    # Cú pháp khởi tạo Scaler mới, chỉ định rõ thiết bị
    scaler = GradScaler(device='cuda')

    # Thêm LR Scheduler: Tự động giảm LR nếu val_loss đi ngang
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)

    # ==========================================
    # 1. AUTO-RESUME: TỰ ĐỘNG TÌM CHECKPOINT MỚI NHẤT
    # ==========================================
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0

    checkpoint_files = glob.glob(os.path.join(cfg.MODEL_SAVE_DIR, f"{cfg.MODEL_TYPE}_epoch_*.pth"))

    if checkpoint_files:
        # Tự động lấy file có số epoch cao nhất, không cần đổi tên thủ công
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_epoch_')[-1].split('.pth')[0]))
        print(f"🔄 Đã tìm thấy Checkpoint mới nhất: {os.path.basename(latest_checkpoint)}")

        checkpoint = torch.load(latest_checkpoint, map_location=device)

        # Bơm lại trí nhớ
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        # Bơm lại cả Scaler để quá trình huấn luyện tiếp tục chính xác
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']

        print(f"▶️ Bắt đầu chạy tiếp từ Epoch {start_epoch + 1}")
    else:
        print("▶️ Không tìm thấy Checkpoint. Bắt đầu train từ con số 0...")

    # ==========================================
    # VÒNG LẶP HUẤN LUYỆN CHÍNH
    # ==========================================
    for epoch in range(start_epoch, cfg.NUM_EPOCHS):

        # ---TRAIN ---
        model.train()
        total_train_loss = 0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}/{cfg.NUM_EPOCHS} [Train]")

        for batch in train_loop:
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)

            decoder_input = tgt[:, :-1]
            label = tgt[:, 1:]

            src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2).to(device)
            tgt_mask = (decoder_input != pad_id).unsqueeze(1).unsqueeze(2).to(device)
            tgt_mask = tgt_mask & causal_mask(decoder_input.size(1)).to(device)

            optimizer.zero_grad()

            # Khai báo dtype=torch.float16
            with autocast(device_type='cuda', dtype=torch.float16):
                encoder_output = model.encode(src, src_mask)
                decoder_output = model.decode(encoder_output, src_mask, decoder_input, tgt_mask)
                proj_output = model.project(decoder_output)

                loss = loss_fn(proj_output.reshape(-1, cfg.VOCAB_SIZE), label.reshape(-1))

            total_train_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        total_val_loss = 0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch + 1:02d}/{cfg.NUM_EPOCHS} [Val]")

        with torch.no_grad():
            for batch in val_loop:
                src, tgt = batch
                src, tgt = src.to(device), tgt.to(device)

                decoder_input = tgt[:, :-1]
                label = tgt[:, 1:]

                src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2).to(device)
                tgt_mask = (decoder_input != pad_id).unsqueeze(1).unsqueeze(2).to(device)
                tgt_mask = tgt_mask & causal_mask(decoder_input.size(1)).to(device)

                with autocast(device_type='cuda', dtype=torch.float16):
                    encoder_output = model.encode(src, src_mask)
                    decoder_output = model.decode(encoder_output, src_mask, decoder_input, tgt_mask)
                    proj_output = model.project(decoder_output)

                    val_loss = loss_fn(proj_output.reshape(-1, cfg.VOCAB_SIZE), label.reshape(-1))

                total_val_loss += val_loss.item()
                val_loop.set_postfix(val_loss=val_loss.item())

        avg_val_loss = total_val_loss / len(val_loader)

        # Ép LR Scheduler theo dõi val loss
        scheduler.step(avg_val_loss)

        print(f"\n👉 TỔNG KẾT EPOCH {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- LƯU TRỮ VÀ EARLY STOPPING ---

        # Dọn dẹp file cũ, lưu file mới
        old_epoch_path = os.path.join(cfg.MODEL_SAVE_DIR, f"{cfg.MODEL_TYPE}_epoch_{epoch}.pth")
        if os.path.exists(old_epoch_path):
            os.remove(old_epoch_path)

        new_epoch_path = os.path.join(cfg.MODEL_SAVE_DIR, f"{cfg.MODEL_TYPE}_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss
        }, new_epoch_path)

        # Lưu Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            best_model_path = os.path.join(cfg.MODEL_SAVE_DIR, f"{cfg.MODEL_TYPE}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f" TUYỆT VỜI! Đã lưu Best Model mới với kỷ lục Val Loss: {best_val_loss:.4f}\n")
        else:
            patience_counter += 1
            #  Sử dụng tham số từ Config
            print(f" CẢNH BÁO: Val Loss không giảm. Lần cảnh cáo: {patience_counter}/{cfg.PATIENCE}\n")

            if patience_counter >= cfg.PATIENCE:
                print(f" KÍCH HOẠT EARLY STOPPING! Đã dừng huấn luyện tại Epoch {epoch + 1}.")
                break


if __name__ == '__main__':
    train_model()