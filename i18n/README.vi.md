[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Soft Event-Frame Alignment

*Mã nghiên cứu tối thiểu để học căn chỉnh không gian và thời gian giữa mẫu camera sự kiện và camera khung hình bằng một biểu diễn nơ-ron ẩn chung.*

[![CI](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml/badge.svg)](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-2EA043)](../LICENSE)
[![Sponsor](https://img.shields.io/badge/GitHub-Sponsor-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/lachlanchen)

Đây là bản triển khai công khai, có thể clone đầu tiên của **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**. Bản phát hành gồm mô hình chuẩn, tiền xử lý AEDAT4, huấn luyện, đánh giá, siêu dữ liệu phụ thuộc, CI và kiểm tra CPU tổng hợp có tính xác định.

Kho không phát hành bản ghi, khung hình có thể nhận dạng, mảng dẫn xuất hay checkpoint đã huấn luyện. Chỉ dùng dữ liệu mà bạn có quyền xử lý.

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=kofi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## Kiến trúc

```mermaid
flowchart LR
    A[Rights-cleared AEDAT4] --> B[Normalize x, y and seconds]
    B --> C[Event samples]
    B --> D[Frame-point samples]
    C --> E[Learnable spatial/time transform]
    E --> F[Shared implicit field F(x,y,t)]
    D --> F
    F --> G[Reconstruction losses]
    G --> H[Checkpoint + bounded diagnostics]
```

Nhánh sự kiện biến đổi tọa độ và xấp xỉ đạo hàm theo thời gian của MLP dùng chung. Nhánh khung hình đánh giá trực tiếp cùng một trường. Tham số tỉ lệ và bước đạo hàm luôn dương để tránh phép chia cho không.

## Nội dung bản phát hành công khai

| Đường dẫn | Mục đích |
| --- | --- |
| `softalign/implicit_model.py` | MLP ẩn dùng chung và tham số căn chỉnh có thể học |
| `softalign/training.py` | Bộ bọc dữ liệu có kiểm tra và vòng huấn luyện chung |
| `softalign/data_processing.py` | Đầu vào AEDAT4 tùy chọn và tiền xử lý theo giây |
| `softalign/synthetic.py` | Dữ liệu tổng hợp xác định do dự án tạo |
| `main.py` / `evaluation.py` | CLI huấn luyện và chẩn đoán |
| `examples/` / `tests/` | Kiểm tra đầu-cuối và kiểm thử lõi |

Thử nghiệm cũ, bản ghi gốc, dữ liệu cục bộ và checkpoint lịch sử được chủ ý loại khỏi bản này.

## Cài đặt và xác minh

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,viz]'
python -m unittest discover -s tests -v
python examples/synthetic_smoke.py --output-dir .smoke-output --epochs 8
```

Kiểm tra tạo checkpoint, biểu đồ và `evaluation_results.json`. Nó xác minh đường ống chạy với giá trị hữu hạn, không chứng minh độ chính xác căn chỉnh.

## Chạy với bản ghi AEDAT4 của bạn

```bash
python -m pip install -e '.[aedat,viz]'
python main.py --filepath /path/to/your-recording.aedat4 --reprocess --data_dir data --checkpoint_dir checkpoints
python evaluation.py --model_path checkpoints/model_final.pt --data_dir data --output_dir evaluation --device cpu
```

Tiền xử lý ghi schema `softalign-processed/v1` và đổi dấu thời gian AEDAT từ micro giây sang giây. Nhiễu lệch tổng hợp mặc định tắt, chỉ bật bằng `--synthetic-misalignment`. Mảng cũ không có siêu dữ liệu đơn vị tương thích sẽ bị từ chối.

Bộ sự kiện lớn nằm trong bộ nhớ CPU; chỉ batch được lấy mẫu mới chuyển sang thiết bị đích. Hãy bắt đầu với `--device cpu` rồi dùng CUDA sau khi kiểm tra bộ nhớ.

## Hợp đồng dữ liệu

| Tệp | Ý nghĩa |
| --- | --- |
| `events.npy` | `N × 4`: x/y chuẩn hóa, thời gian theo giây, mục tiêu sự kiện |
| `frame_points.npy` | `M × 4`: x/y chuẩn hóa, thời gian theo giây, cường độ |
| `preprocessing.json` | Schema, đơn vị, seed, cờ biến đổi và số lượng |
| `model_final.pt` | Trạng thái, tham số và kiến trúc mô hình |
| `evaluation_results.json` | MSE tái dựng trên số mẫu đầu vào có giới hạn |

Kết quả là chẩn đoán trên mảng được cung cấp, không phải chỉ số hold-out, benchmark hay bằng chứng hiệu chuẩn cảm biến vật lý.

## Xác minh và giới hạn nghiên cứu

- CI chạy kiểm thử CPU xác định và chu trình huấn luyện/đánh giá ngắn.
- Hệ thống kiểm tra hình dạng, dữ liệu rỗng, giá trị hữu hạn, đơn vị thời gian và thiết bị.
- Hàm mất mát hiện tại là đường cơ sở nghiên cứu; nghiên cứu thực cần ground truth, đánh giá tách biệt, ablation và phân tích bất định.
- Không đính kèm bản ghi riêng tư vào GitHub Issues công khai.

## Trích dẫn

Nếu dùng trong nghiên cứu, hãy trích dẫn kho. GitHub đọc [CITATION.cff](../CITATION.cff) và hiển thị **Cite this repository**.

```bibtex
@software{chen_soft_event_frame_alignment_2026,
  author = {Chen, Lachlan},
  title = {Soft Event-Frame Alignment: Unified Implicit Neural Representation Research Code},
  year = {2026},
  url = {https://github.com/lachlanchen/SoftEventFrameAlignment}
}
```

## Trạng thái

Phiên bản `0.1.0` là bản nghiên cứu tối thiểu. Chúng tôi hoan nghênh lỗi có thể tái hiện bằng kiểm tra tổng hợp hoặc mẫu nhỏ đã được cấp quyền.

## Giấy phép

Apache License 2.0. Xem [LICENSE](../LICENSE).
