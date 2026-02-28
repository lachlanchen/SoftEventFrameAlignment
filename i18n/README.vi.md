[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Soft Event-Frame Alignment

> Kho này là mã nguồn cho bài báo **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white&style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white&style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white&style=flat-square)
![Research Code](https://img.shields.io/badge/Type-Research%20Code-0A7EA4?style=flat-square)
![License](https://img.shields.io/badge/License-Apache%202.0-2EA043?style=flat-square)

| Focus | Details |
| --- | --- |
| 🎯 Mục tiêu | Căn chỉnh luồng camera sự kiện và luồng camera khung hình với một biểu diễn ngầm (implicit neural representation) thống nhất |
| ⚙️ Pipeline chính | `main.py` (train) + `evaluation.py` (analyze/visualize) |
| 🧪 Các script biến thể | `event_inn.py`, `frame_inn.py`, `event_deri*.py`, `event_inn_sparse.py` |

</div>

## Mục lục

- [🔎 Tổng quan](#-tổng-quan)
- [✨ Tính năng](#-tính-năng)
- [🗂️ Cấu trúc dự án](#-cấu-trúc-dự-án)
- [📋 Điều kiện tiên quyết](#-điều-kiện-tiên-quyết)
- [⚙️ Cài đặt](#-cài-đặt)
- [🚀 Khởi tạo nhanh](#-khởi-tạo-nhanh)
- [🧪 Sử dụng](#-sử-dụng)
- [🧩 Ghi chú cấu hình](#-ghi-chú-cấu-hình)
- [🧠 Công thức toán học](#-công-thức-toán-học)
- [🧾 Ví dụ](#-ví-dụ)
- [🛠️ Ghi chú phát triển](#-ghi-chú-phát-triển)
- [🧯 Xử lý sự cố](#-xử-lý-sự-cố)
- [🗺️ Lộ trình](#-lộ-trình)
- [🤝 Đóng góp](#-đóng-góp)
- [❤️ Support](#-support)
- [📄 Giấy phép](#-giấy-phép)

## 🔎 Tổng quan

Kho này chứa mã nghiên cứu để căn chỉnh luồng camera sự kiện và luồng camera khung hình bằng một biểu diễn ngầm (INR/INN) thống nhất.

### Đường dẫn chính (Canonical Path)

| Bước | Thành phần | Mục đích |
|---|---|---|
| 1 | `softalign/data_processing.py` | Đọc các luồng AEDAT4, chuẩn hóa dữ liệu, tạo mẫu điểm |
| 2 | `softalign/implicit_model.py` | Định nghĩa hàm ngầm dùng chung + các tham số căn chỉnh có thể học |
| 3 | `softalign/training.py` + `main.py` | Tối ưu đồng thời tái tạo event/frame |
| 4 | `evaluation.py` | Trực quan hóa kết quả căn chỉnh và báo cáo loss |

Ngoài đường dẫn chính, repository còn có các script độc lập và thử nghiệm cho các biến thể chỉ event, chỉ frame, sparse, và biến thể INR dựa trên đạo hàm.

## ✨ Tính năng

- Căn chỉnh event-frame với các tham số dạng affine có thể học: `scale`, `shift_x`, `shift_y`, `shift_t`.
- Hàm ngầm dùng chung `F(x, y, t)` được cả nhánh event và frame cùng sử dụng.
- Mô hình event qua đạo hàm thời gian sai phân hữu hạn với `threshold` và `dt` có thể học.
- Nạp AEDAT4 (`dv_processing`) cộng với quy trình `.npy` đã tiền xử lý.
- Có sẵn các đầu ra trực quan hóa cho đồ thị loss, sự tiến triển tham số và lớp phủ căn chỉnh.
- Hỗ trợ CUDA khi có sẵn (`torch.cuda.is_available()` sẽ fallback về CPU).
- Script dataset để kiểm tra nhanh và gỡ lỗi nhẹ:
  - `read_events.py`
  - `read_frames.py`
  - `reader.py`, `reader_norm.py`, `simple_count.py`

## 🗂️ Cấu trúc dự án

```text
SoftEventFrameAlignment/
├── README.md
├── LICENSE
├── main.py                              # Canonical event-frame training entrypoint
├── evaluation.py                        # Canonical evaluation/visualization entrypoint
├── softalign/
│   ├── __init__.py
│   ├── data_processing.py               # AEDAT4 loading + normalization + frame point sampling
│   ├── implicit_model.py                # MLP + alignment parameters
│   └── training.py                      # Dataset wrapper + optimization loop
├── event_inn.py                         # Event-only INR experiment
├── frame_inn.py                         # Frame-only INR experiment
├── event_deri.py                        # Derivative/log-derivative event variant
├── event_deri_2.py                      # Derivative variant with extra zero regularization
├── event_inn_sparse.py                  # Sparse event INR (PyG/torch_scatter dependencies)
├── softalign_standalone.py              # Monolithic all-in-one alignment workflow
├── softalign_old.py                     # Legacy implementation retained for comparison
├── read_events.py                       # Event I/O and basic preprocessing checks
├── read_frames.py                       # Frame/video extraction and shape checks
├── reader.py                            # AEDAT reader utility
├── reader_norm.py                       # Reader with normalization helpers
├── simple_count.py                      # Lightweight event-count utility
├── data/                                # Processed arrays (generated or checked in)
├── checkpoints/                         # Main training checkpoints and plots
├── event_inn_results/                   # Generated experiment outputs
├── frame_inn_results/                   # Generated experiment outputs
├── event_drivative_results/             # Generated experiment outputs
├── alignment_results_*/                 # Timestamped alignment runs
├── yuqing/                              # Sample AEDAT4 files
├── events_processed.csv                  # Event summary artifact
├── frames_processed.csv                  # Frame summary artifact
└── i18n/                                # Translated README files
```

## 📋 Điều kiện tiên quyết

- Python 3.10+ được khuyến nghị.
- Ví dụ shell dưới đây dùng Linux/macOS (hãy chỉnh cho Windows nếu cần).
- Tùy chọn nhưng khuyến nghị: GPU hỗ trợ CUDA để tăng tốc huấn luyện.
- Việc đọc AEDAT4 cần `dv_processing` và các phụ thuộc hệ thống tương thích.

## ⚙️ Cài đặt

Hiện chưa có `requirements.txt` hay `pyproject.toml`, nên cần cài dependencies thủ công.

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python tqdm torch
pip install dv-processing
```

Các phụ thuộc tùy chọn cho thí nghiệm sparse (`event_inn_sparse.py`):

```bash
pip install scikit-learn torch-geometric torch-scatter
```

Giả định: khả dụng của CUDA quyết định hành vi thiết bị mặc định, nhưng toàn bộ lệnh bên dưới cũng hỗ trợ chỉ định `--device cpu`.

## 🚀 Khởi tạo nhanh

### 1. Huấn luyện mô hình căn chỉnh chính

```bash
python main.py \
  --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 \
  --duration 10 \
  --num_epochs 1000 \
  --lr 1e-3 \
  --reprocess
```

Kết quả đầu ra được ghi vào:
- `data/` (`events.npy`, `frames_timestamps.npy`, `frame_points.npy`, `sample_frame.png`)
- `checkpoints/` (`model_epoch_*.pt`, `model_final.pt`, `loss_curves.png`, `parameter_history.png`)

### 2. Đánh giá checkpoint đã huấn luyện

`evaluation.py` import `EventFrameAlignmentModel` qua `from implicit_model import ...`, trong khi module đang chạy thực tế nằm ở `softalign/implicit_model.py`. Một cách gọi thực tế từ root repo là:

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

Lệnh này tạo các trực quan/đánh giá như `evaluation/alignment_visualization.png` và `evaluation/evaluation_results.txt`.

## 🧪 Sử dụng

### Pipeline chính (`main.py`)

```bash
python main.py --help
```

Các tùy chọn chính:
- `--filepath`: đường dẫn input AEDAT4.
- `--duration`: thời gian đọc (giây) từ bản ghi.
- `--data_dir`: thư mục đầu ra dữ liệu đã xử lý.
- `--checkpoint_dir`: thư mục output checkpoint.
- `--num_epochs`, `--lr`, `--batch_size`, `--lambda_reg`.
- `--device`: `cuda` hoặc `cpu`.
- `--reprocess`: ép tạo lại dữ liệu.

### Script thí nghiệm

INR chỉ event:
```bash
python event_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_event
# hoặc
python event_inn.py --events_file data/events.npy --output_dir output_event
```

INR chỉ frame:
```bash
python frame_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_frame --grayscale
# hoặc
python frame_inn.py --video_file path/to/video.mp4 --output_dir output_frame
# hoặc
python frame_inn.py --image_dir path/to/images --image_pattern "*.png" --output_dir output_frame
```

Các biến thể đạo hàm:
```bash
python event_deri.py --events_file data/events.npy --output_dir output_deri
python event_deri_2.py --events_file data/events.npy --output_dir output_deri2
```

Sparse variant:
```bash
python event_inn_sparse.py --events_file data/events.npy --output_dir output_sparse
```

Standalone monolithic alignment:
```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

Readers bổ sung:
```bash
python read_events.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python read_frames.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python simple_count.py --input data/events.npy
```

## 🧩 Ghi chú cấu hình

Mặc định mô hình chính trong `softalign/implicit_model.py`:

| Tham số | Mặc định |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

Quy trình tiền xử lý dữ liệu trong `softalign/data_processing.py` hiện đang áp dụng một sự sai lệch tổng hợp cho event trước khi huấn luyện. Đây là thiết kế có chủ đích trong mã hiện tại và nên được cân nhắc khi diễn giải các chỉ số.

Training scripts cũng đưa ra các tùy chọn chỉnh được qua CLI flags (ví dụ `--hidden_dim`, `--num_layers`, và `--batch_size`) hữu ích cho ablation.

## 🧠 Công thức toán học

Mạng ngầm dùng chung mô hình hóa `F(x, y, t)`.

Nhánh frame:
- Dự đoán trực tiếp phản hồi giống cường độ bằng `F(x, y, t)`.

Nhánh event:
1. Áp dụng biến đổi event có thể học:
   - `x' = x / scale + shift_x`
   - `y' = y / scale + shift_y`
   - `t' = t + shift_t`
2. Xấp xỉ đạo hàm theo thời gian:
   - `dF/dt ≈ (F(x', y', t' + dt) - F(x', y', t')) / dt`
3. Sinh phản hồi event:
   - `sigmoid(dF/dt - threshold)`

Mục tiêu huấn luyện kết hợp:
- Event MSE loss
- Frame MSE loss
- Regularization cho các tham số căn chỉnh

## 🧾 Ví dụ

Chỉ dùng các mảng đã tiền xử lý:

```bash
python main.py --data_dir data --checkpoint_dir checkpoints --num_epochs 200 --lr 5e-4
```

Ép chạy CPU:

```bash
python main.py --device cpu --num_epochs 50 --reprocess
```

Đánh giá vào thư mục tùy biến:

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ Ghi chú phát triển

- Repo tập trung vào script (chưa có metadata đóng gói).
- Các artifact sinh ra nằm trong repo (checkpoints/results), nên dung lượng đĩa có thể lớn.
- `readme-file.md` và `project-structure.txt` có giả định cấu trúc legacy và chưa hoàn toàn khớp với layout root hiện tại.
- Hiện chưa phát hiện tests tự động hay workflow CI.

## 🧯 Xử lý sự cố

- `ModuleNotFoundError: No module named 'implicit_model'` trong `evaluation.py`:
  - Chạy với `PYTHONPATH=softalign` như đã nêu ở trên.
- Sự cố cài đặt/chạy `dv_processing`:
  - Kiểm tra nền tảng và phiên bản Python có được hỗ trợ bởi wheel/lib của `dv-processing`.
- CUDA OOM:
  - Giảm `--batch_size`, giảm `--max_events` (nếu có), hoặc chạy với `--device cpu`.
- Thiếu file AEDAT4:
  - Kiểm tra đường dẫn trong `yuqing/` hoặc đưa bản ghi của bạn qua `--filepath` / `--aedat4_file`.
- Dữ liệu không khớp giữa các lần chạy:
  - Chạy lại với `--reprocess` khi thay đổi giả định tiền xử lý.

## 🗺️ Lộ trình

- Thêm file môi trường tái hiện được (`requirements.txt` hoặc `pyproject.toml`).
- Thống nhất đường dẫn import cho evaluation và chạy package/module.
- Thêm automated tests cho kiểm tra tiền xử lý dữ liệu và kiểm tra forward/training của mô hình.
- Thêm CI cho linting và smoke tests.
- Thêm README đa ngôn ngữ trong `i18n/`.

## 🤝 Đóng góp

Các đóng góp đều được chào đón.

Luồng đề xuất:
1. Tạo issue hoặc nhánh tập trung mô tả thay đổi thí nghiệm.
2. Giữ script đồng nhất (đặc biệt imports, quy ước CLI, và tên thư mục output).
3. Bảo toàn hành vi artifact hiện có trừ khi đổi tên/thay đổi versioning.

## 📄 Giấy phép

Repository này được phát hành theo giấy phép Apache License 2.0. Xem [`LICENSE`](LICENSE) để đọc đầy đủ.

Giả định: nếu repository này được dùng trong các bài báo, hãy bổ sung chi tiết trích dẫn tại đây và trong ghi chú phát hành khi cần.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
