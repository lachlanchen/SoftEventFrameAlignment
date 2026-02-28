[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Soft Event-Frame Alignment

> Kho mã cho bài báo **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white)
![Research Code](https://img.shields.io/badge/Type-Research%20Code-0A7EA4)
![License](https://img.shields.io/badge/License-Apache%202.0-2EA043)

## 🔎 Tổng quan

Kho này chứa mã nghiên cứu cho bài toán căn chỉnh luồng camera sự kiện và camera khung hình bằng một biểu diễn neural ngầm thống nhất (INR/INN).

### Pipeline chính (đường dẫn chuẩn)

| Bước | Thành phần | Mục đích |
|---|---|---|
| 1 | `softalign/data_processing.py` | Đọc luồng AEDAT4, chuẩn hóa dữ liệu, tạo mẫu điểm |
| 2 | `softalign/implicit_model.py` | Định nghĩa hàm ngầm dùng chung + tham số căn chỉnh có thể học |
| 3 | `softalign/training.py` + `main.py` | Tối ưu đồng thời tái tạo event/frame |
| 4 | `evaluation.py` | Trực quan hóa căn chỉnh và báo cáo loss |

Ngoài luồng chính, kho còn có các script độc lập/thử nghiệm cho biến thể chỉ-event, chỉ-frame, sparse và INR dựa trên đạo hàm.

## ✨ Tính năng

- Căn chỉnh event-frame với các tham số kiểu affine có thể học: `scale`, `shift_x`, `shift_y`, `shift_t`.
- Hàm ngầm dùng chung `F(x, y, t)` được dùng bởi cả nhánh event và frame.
- Mô hình hóa event qua đạo hàm thời gian sai phân hữu hạn với `threshold` và `dt` có thể học.
- Nạp dữ liệu AEDAT4 (`dv_processing`) và workflow với `.npy` đã xử lý.
- Có sẵn đầu ra trực quan hóa cho đường cong loss, diễn tiến tham số, và lớp phủ căn chỉnh.
- Hỗ trợ CUDA khi khả dụng (`torch.cuda.is_available()`, không thì dùng CPU).

## 🗂️ Cấu trúc dự án

```text
SoftEventFrameAlignment/
├── README.md
├── LICENSE
├── main.py                          # Điểm vào huấn luyện event-frame chuẩn
├── evaluation.py                    # Điểm vào đánh giá/trực quan hóa chuẩn
├── softalign/
│   ├── __init__.py
│   ├── data_processing.py           # Nạp AEDAT4 + chuẩn hóa + lấy mẫu điểm frame
│   ├── implicit_model.py            # MLP + tham số căn chỉnh
│   └── training.py                  # Wrapper dataset + vòng lặp tối ưu
├── event_inn.py                     # Thử nghiệm INR chỉ-event
├── frame_inn.py                     # Thử nghiệm INR chỉ-frame
├── event_deri.py                    # Biến thể event đạo hàm/log-đạo hàm
├── event_deri_2.py                  # Biến thể đạo hàm với regularization zero bổ sung
├── event_inn_sparse.py              # INR event sparse (phụ thuộc PyG/torch_scatter)
├── softalign_standalone.py          # Quy trình căn chỉnh đơn khối all-in-one
├── data/                            # Mảng đã xử lý (được tạo hoặc đã commit)
├── checkpoints/                     # Checkpoint và đồ thị huấn luyện chính
├── event_inn_results/               # Đầu ra thử nghiệm đã tạo
├── frame_inn_results/               # Đầu ra thử nghiệm đã tạo
├── event_drivative_results/         # Đầu ra thử nghiệm đã tạo
├── alignment_results_*/             # Các lần chạy căn chỉnh có timestamp
├── yuqing/                          # Tệp AEDAT4 mẫu
└── i18n/                            # Vị trí file bản dịch
```

## 📋 Yêu cầu trước

- Khuyến nghị Python 3.10+.
- Ví dụ shell bên dưới dùng Linux/macOS (điều chỉnh cho Windows nếu cần).
- Tùy chọn nhưng khuyến nghị: GPU hỗ trợ CUDA để tăng tốc huấn luyện.
- Đọc AEDAT4 cần `dv_processing` và các phụ thuộc hệ thống tương thích.

## ⚙️ Cài đặt

Hiện chưa có `requirements.txt` hoặc `pyproject.toml`, nên cài phụ thuộc thủ công.

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python tqdm torch
pip install dv-processing
```

Phụ thuộc tùy chọn cho thí nghiệm sparse (`event_inn_sparse.py`):

```bash
pip install scikit-learn torch-geometric torch-scatter
```

## 🚀 Bắt đầu nhanh

### 1. Huấn luyện mô hình căn chỉnh chính

```bash
python main.py \
  --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 \
  --duration 10 \
  --num_epochs 1000 \
  --lr 1e-3 \
  --reprocess
```

Đầu ra được ghi vào:
- `data/` (`events.npy`, `frames_timestamps.npy`, `frame_points.npy`, `sample_frame.png`)
- `checkpoints/` (`model_epoch_*.pt`, `model_final.pt`, `loss_curves.png`, `parameter_history.png`)

### 2. Đánh giá checkpoint đã huấn luyện

`evaluation.py` import `EventFrameAlignmentModel` bằng `from implicit_model import ...`, trong khi module đang dùng nằm ở `softalign/implicit_model.py`. Cách chạy thực tế từ thư mục gốc repo:

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

Lệnh này tạo các kết quả trực quan/chỉ số như `evaluation/alignment_visualization.png` và `evaluation/evaluation_results.txt`.

## 🧪 Cách dùng

### Pipeline chính (`main.py`)

```bash
python main.py --help
```

Tùy chọn quan trọng:
- `--filepath`: đường dẫn đầu vào AEDAT4.
- `--duration`: số giây đọc từ bản ghi.
- `--data_dir`: thư mục đầu ra dữ liệu đã xử lý.
- `--checkpoint_dir`: thư mục đầu ra checkpoint.
- `--num_epochs`, `--lr`, `--batch_size`, `--lambda_reg`.
- `--device`: `cuda` hoặc `cpu`.
- `--reprocess`: ép tái tạo dữ liệu.

### Script thử nghiệm

INR chỉ-event:
```bash
python event_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_event
# hoặc
python event_inn.py --events_file data/events.npy --output_dir output_event
```

INR chỉ-frame:
```bash
python frame_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_frame --grayscale
# hoặc
python frame_inn.py --video_file path/to/video.mp4 --output_dir output_frame
# hoặc
python frame_inn.py --image_dir path/to/images --image_pattern "*.png" --output_dir output_frame
```

Biến thể đạo hàm:
```bash
python event_deri.py --events_file data/events.npy --output_dir output_deri
python event_deri_2.py --events_file data/events.npy --output_dir output_deri2
```

Biến thể sparse:
```bash
python event_inn_sparse.py --events_file data/events.npy --output_dir output_sparse
```

Căn chỉnh đơn khối standalone:
```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

## 🧩 Ghi chú cấu hình

Giá trị mặc định mô hình chính trong `softalign/implicit_model.py`:

| Tham số | Mặc định |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

Tiền xử lý dữ liệu trong `softalign/data_processing.py` hiện áp dụng một sai lệch tổng hợp (synthetic misalignment) lên event trước khi huấn luyện. Đây là chủ đích trong mã hiện tại và cần được tính đến khi diễn giải các chỉ số.

## 🧠 Mô tả toán học

Mạng ngầm dùng chung mô hình hóa `F(x, y, t)`.

Nhánh frame:
- Dự đoán trực tiếp đáp ứng giống cường độ bằng `F(x, y, t)`.

Nhánh event:
1. Áp dụng biến đổi event có thể học:
   - `x' = x / scale + shift_x`
   - `y' = y / scale + shift_y`
   - `t' = t + shift_t`
2. Xấp xỉ đạo hàm theo thời gian:
   - `dF/dt ≈ (F(x', y', t' + dt) - F(x', y', t')) / dt`
3. Tạo đáp ứng event:
   - `sigmoid(dF/dt - threshold)`

Mục tiêu huấn luyện kết hợp:
- Event MSE loss
- Frame MSE loss
- Regularization trên các tham số căn chỉnh

## 🧾 Ví dụ

Chỉ dùng mảng đã tiền xử lý:

```bash
python main.py --data_dir data --checkpoint_dir checkpoints --num_epochs 200 --lr 5e-4
```

Ép chạy CPU:

```bash
python main.py --device cpu --num_epochs 50 --reprocess
```

Đánh giá vào thư mục tùy chỉnh:

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ Ghi chú phát triển

- Kho hiện thiên về script (chưa có metadata đóng gói).
- Artifact sinh ra có trong repo (checkpoints/results), nên dung lượng đĩa có thể lớn.
- `readme-file.md` và `project-structure.txt` chứa giả định cấu trúc cũ và chưa khớp hoàn toàn với layout root hiện tại.
- Hiện chưa phát hiện test tự động hoặc workflow CI.

## 🧯 Khắc phục sự cố

- `ModuleNotFoundError: No module named 'implicit_model'` trong `evaluation.py`:
  - Chạy với `PYTHONPATH=softalign` như bên trên.
- Vấn đề cài/chạy `dv_processing`:
  - Xác nhận nền tảng và phiên bản Python của bạn được wheel/lib `dv-processing` hỗ trợ.
- CUDA OOM:
  - Giảm `--batch_size`, giảm `--max_events` (nơi có hỗ trợ), hoặc chạy với `--device cpu`.
- Thiếu tệp AEDAT4:
  - Kiểm tra đường dẫn trong `yuqing/` hoặc dùng bản ghi của bạn qua `--filepath` / `--aedat4_file`.

## 🗺️ Lộ trình

- Thêm file môi trường có thể tái lập (`requirements.txt` hoặc `pyproject.toml`).
- Thống nhất đường dẫn import cho đánh giá và chạy package/module.
- Thêm test tự động cho xử lý dữ liệu và kiểm tra forward/huấn luyện mô hình.
- Thêm CI cho linting và smoke test.
- Thêm README đa ngôn ngữ trong `i18n/`.

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón.

1. Fork repository.
2. Tạo nhánh tính năng.
3. Thực hiện thay đổi tập trung, có tài liệu rõ ràng.
4. Gửi pull request kèm chi tiết tái lập và mẫu đầu ra khi phù hợp.

Nếu bạn định làm thay đổi lớn (luồng huấn luyện mới, refactor, hoặc thay đổi phụ thuộc), hãy mở issue trước để thống nhất hướng đi.

## 📚 Trích dẫn

Nếu bạn dùng kho này trong nghiên cứu, hãy trích dẫn bài báo liên quan và/hoặc repository.

Metadata hiện có từ nội dung README hiện tại:
- Tiêu đề bài báo: *Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation*

Placeholder BibTeX cho repository (cập nhật tác giả/URL khi cần):

```bibtex
@misc{soft-event-frame-alignment,
  title        = {Soft Event-Frame Alignment},
  note         = {Code repository for "Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation"},
  year         = {2025},
  howpublished = {GitHub repository}
}
```

## 🙏 Lời cảm ơn

- Hệ sinh thái công cụ event-camera, bao gồm `dv-processing`.
- PyTorch và các thư viện Python khoa học được dùng xuyên suốt thí nghiệm.

## 💡 Hỗ trợ

Phần hỗ trợ/tài trợ hiện chưa có trong metadata repository. Nếu bạn muốn thêm, hãy bổ sung liên kết ở lần cập nhật tiếp theo và chúng sẽ được giữ lại trong các bản sửa sau.

## 📄 Giấy phép

Phát hành theo Apache License 2.0. Xem [LICENSE](LICENSE).
