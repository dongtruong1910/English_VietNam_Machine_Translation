import os
import matplotlib.pyplot as plt


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    results_dir = os.path.join(root_dir, "results")

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(results_dir, exist_ok=True)

    # 2. Dữ liệu thực tế
    epochs = list(range(1, 14))
    train_loss = [4.2231, 3.8672, 3.5250, 3.1113, 2.9867, 2.8888, 2.8068, 2.7369, 2.6751, 2.6187, 2.5686, 2.5227,
                  2.3826]
    val_loss = [4.5423, 4.0153, 3.6387, 3.3487, 3.2922, 3.2665, 3.2523, 3.2289, 3.2346, 3.2198, 3.2265, 3.2362, 3.2363]

    # 3. Vẽ đồ thị
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2)
    plt.axvline(x=10, color='g', linestyle='--', linewidth=2, label='Best Model (Epoch 10)')  # Kỷ lục là ở Epoch 10

    plt.title('Training and Validation Loss (Baseline Transformer)', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (Cross Entropy)', fontsize=12)

    # Ép trục X hiển thị tất cả các số nguyên từ 1 đến 13
    plt.xticks(epochs)

    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 4. LƯU ẢNH TRƯỚC KHI HIỂN THỊ
    save_path = os.path.join(results_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"✅ Đã lưu đồ thị độ phân giải cao tại: {save_path}")

    # 5. Hiển thị lên màn hình
    plt.show()


if __name__ == "__main__":
    main()