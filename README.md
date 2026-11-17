# Grounded Vision Assistant

## 1. Đề bài

Xây dựng website nhận một ảnh JPEG/PNG và câu mô tả bằng tiếng Việt hoặc tiếng Anh. Ví dụ, với ảnh có ba chiếc cốc và câu “chiếc cốc trắng bên trái bàn phím”, hệ thống cần khoanh đúng chiếc cốc và hiển thị vùng ảnh làm bằng chứng.

**Sản phẩm bàn giao** gồm website, API inference, hai phương pháp benchmark, tập đánh giá tiếng Việt, báo cáo kết quả và video demo.

## 2. Hợp đồng API

- `GET /health` trả về trạng thái dịch vụ.
- `POST /ground` nhận ảnh và câu mô tả dạng multipart. Giới hạn dự kiến 10 MB mỗi ảnh.
- Kết quả gồm `status`, `bbox_xyxy`, `image_width`, `image_height`, `backend` và `latency_ms`.
- `found` trả về đúng một khung hợp lệ. `ambiguous` dùng khi có nhiều ứng viên phù hợp. `not_found` dùng khi không tìm được ứng viên. Hai trạng thái sau trả về khung null.
- Chuẩn hóa hướng EXIF trước khi inference. Dùng tọa độ pixel `[x1,y1,x2,y2]` trên ảnh đã chuẩn hóa, với cạnh phải và cạnh dưới là giới hạn ngoài của vùng cắt.
- Giao diện vẽ box đúng tỷ lệ khi ảnh được thay đổi kích thước và hiển thị vùng ảnh cắt khi tìm thấy vật.

## 3. Hai cách triển khai

1. **Direct VLM grounding.** Qwen2.5-VL-3B nhận ảnh và câu. Xử lý phản hồi, quy đổi từ không gian tọa độ model sang không gian ảnh, xác thực box và ánh xạ trạng thái.
2. **Proposal-based grounding.** Grounding DINO nhận cụm tên vật, sinh box ứng viên và lọc box trùng. VLM nhìn ảnh có đánh số các box và dùng câu đầy đủ để chọn, kể cả màu, vị trí và quan hệ giữa các vật. Ghi riêng lỗi bộ sinh ứng viên và lỗi bước chọn ứng viên.

Dùng cùng danh sách ảnh và câu cho hai cách. Trên RTX 4060 Laptop 8 GB, bắt đầu với một ảnh mỗi lượt và đo VRAM trước khi tăng độ phân giải hoặc chạy đồng thời hai model.

## 4. Dữ liệu

| Gói | Nội dung | Thư mục |
|---|---|---|
| COCO train2014 | 82.783 ảnh dùng chung | data/raw/coco/train2014 |
| COCO annotations train/val | Annotation bổ sung để đối chiếu | data/raw/coco/annotations |
| RefCOCO | Câu tham chiếu và box chuẩn, dùng split UNC | data/raw/refer/refcoco |
| RefCOCOg | Câu tham chiếu dài hơn, dùng split Google | data/raw/refer/refcocog |
| Tập tiếng Việt tự gán nhãn | Vật có mặt, mơ hồ và vật vắng mặt | data/vietnamese |

Downloader lấy toàn bộ các archive công khai trên. Ảnh COCO val2014 và test2014 nằm ngoài pipeline này. Giữ khoảng 30 GB trống trên D nếu lưu cả ZIP và bản giải nén. Tập tiếng Việt là phần tự làm, mục tiêu ban đầu khoảng 100 câu.

Giữ nguyên split từ annotation. Ghi rõ tập phát triển và tập đo cuối. Nếu giới hạn số mẫu, lưu seed, ID mẫu và số lượng theo split. Tách tập tiếng Việt theo ảnh. Đọc pickle chỉ từ nguồn bạn tin cậy.

## 5. Benchmark và cách phân tích

- **Grounding Acc@0.5** tính tỷ lệ mẫu có vật đích mà box dự đoán đạt IoU từ 0,5 trở lên. Mẫu có vật đích nhưng hệ thống từ chối vẫn tính sai.
- **Proposal Recall@0.5** tính tỷ lệ có ít nhất một box ứng viên đạt IoU từ 0,5 trước khi VLM chọn.
- **False positive rate** đo tỷ lệ mẫu vật vắng mặt bị trả về found.
- **Ambiguity recall** đo tỷ lệ mẫu mơ hồ được phát hiện đúng.
- **Latency p50/p95 và VRAM đỉnh** đo cho từng backend, ghi cấu hình máy, độ phân giải, precision và warm-up.
- Lưu kết quả từng mẫu trong outputs. Báo cáo lỗi định vị, chọn sai vật, vật vắng mặt và câu mơ hồ. Ví dụ, nếu proposal recall cao mà Acc thấp thì kiểm tra bước chọn ứng viên trước.

## 6. Thứ tự làm

1. `prepare_dataset.py` và `metrics.py` đọc annotation, nối ảnh-câu-box, kiểm thử phép IoU. Đổi box COCO `[x,y,w,h]` sang `[x,y,x+w,y+h]`.
2. `schemas.py`, `image_io.py`, `vlm.py` tạo hợp đồng output, chuẩn hóa ảnh, chạy baseline trên một ảnh.
3. `detector.py`, `pipeline.py`, `benchmark.py` sinh ứng viên, xử lý ba trạng thái và chạy hai backend trên cùng manifest.
4. `api.py` và `web/` nối upload, inference, vẽ box và crop.
5. Hoàn tất test, thử các ca khó, ghi kết quả thực tế và quay demo.

## 7. Nghiệm thu

- Downloader có thể chạy lại và kiểm tra file bắt buộc cùng số lượng ảnh.
- Web xử lý ảnh đúng, ảnh lỗi và câu rỗng. Demo có ca tìm đúng theo quan hệ không gian, câu mơ hồ và vật vắng mặt.
- Hai phương pháp dùng cùng tập đánh giá. Có lệnh tái lập, cấu hình, dự đoán theo từng mẫu và bảng metric.
- Báo cáo nêu giới hạn và phân tích ca sai. CV ghi kết quả đã đo.

## 8. Trạng thái và lệnh tải

Bộ cài tạo cấu trúc, cấu hình, README và downloader. Mã inference, API, benchmark và tests được tạo dưới dạng khung để triển khai theo mục 6. Danh sách dependencies là kế hoạch ban đầu; chốt phiên bản khi kiểm thử CUDA và model.

Từ thư mục gốc, chạy lại downloader:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_datasets.ps1
```

## 9. Nguồn

- [REFER và dữ liệu gốc](https://github.com/lichengunc/refer/tree/master/data)
- [COCO download](https://cocodataset.org/#download)
- [Qwen2.5-VL](https://github.com/QwenLM-corp/Qwen2.5-VL)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)

## 10. Cài đặt và chạy bản tham chiếu

Môi trường đã kiểm tra: Windows 11, Python 3.11.9. Từ thư mục gốc:

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH="src"
python -m uvicorn grounded_vision.api:app --host 127.0.0.1 --port 8000
```

Mở `http://127.0.0.1:8000`. Model được nạp lười ở yêu cầu đầu tiên. `GET /health`
chỉ kiểm tra API, không tải model. Lỗi tải model (503), inference (500), parse output
(502), ảnh đầu vào (400) và validation (422) tách khỏi ba trạng thái nghiệp vụ.

Qwen2.5-VL trả tọa độ theo kích thước pixel thực của ảnh (không dùng không gian 0..1000
của một số recipe Qwen2-VL cũ). Grounding DINO trả box pixel sau
`post_process_grounded_object_detection(..., target_sizes=[(height,width)])`.
Hai adapter chạy batch size 1. Ngưỡng proposal mặc định `box=0.25`, `text=0.20` là
điểm khởi đầu từ API model, **chưa được tune trên test split**; không dùng số benchmark
cuối cho đến khi chốt ngưỡng trên development split.

## 11. Chuẩn bị dữ liệu

Preview nhanh không ghi file, sau đó xuất manifest tái lập:

```powershell
python scripts/prepare_dataset.py --dataset refcoco --split val --limit 3 --seed 42 --preview 1
python scripts/prepare_dataset.py --dataset refcoco --split val --limit 100 --seed 42 --output data/manifests/refcoco-val-100.jsonl
python scripts/prepare_dataset.py --dataset refcocog --split test --limit 100 --seed 42 --output data/manifests/refcocog-test-100.jsonl
```

Loader chỉ mở các pickle có tên cố định bên trong thư mục REFER đã cấu hình. Mỗi dòng
JSONL có `id`, dataset/split, ID ảnh/annotation, đường dẫn ảnh tương đối, query,
`gold_status`, `gold_bbox_xyxy`, và cờ `is_test_data`. Manifest tiếng Việt là tùy chọn
ở `data/vietnamese/annotations.jsonl`; loader báo rõ nếu chưa có. Dữ liệu tự tạo phải
đặt `is_test_data: true`.

## 12. Test và benchmark

```powershell
$env:PYTHONPATH="src"
python -m pytest -q
python scripts/benchmark.py --manifest data/manifests/refcoco-val-smoke.jsonl --backend proposal --limit 1
python scripts/benchmark.py --manifest data/manifests/refcoco-val-100.jsonl --backend both --split val
```

Benchmark ghi prediction từng mẫu và summary theo dataset/split/backend vào `outputs/`.
Mẫu lỗi kỹ thuật có `technical_error` và bị loại khỏi mẫu số metric; số lỗi luôn được
công bố. Grounding Acc@0.5 tính trên mẫu gold `found`; proposal recall được đo trước
bước chọn; false-positive/ambiguity recall chỉ xuất hiện khi manifest có nhãn tương
ứng. Latency dùng p50/p95. CUDA peak là `max_memory_allocated`; trường này null nếu
PyTorch không có CUDA. Cấu hình model, precision, Python, máy, seed và warm-up được lưu
cùng báo cáo. Hiện `--warmup` mới ghi metadata; nên chạy một manifest warm-up riêng
trước lượt đo chính để không trộn thời gian tải model.

## 13. Trạng thái kiểm chứng tại máy phát triển

- `12 passed`: conversion, IoU, percentile, trạng thái direct/proposal, proposal miss,
  selector, query rỗng, schema multipart và ảnh hỏng.
- Đã nối thật RefCOCO `ref_id=0`, `ann_id=1719310`, `image_id=581857`; ảnh JPEG tồn tại,
  kích thước 427×640, gold box `[103.93,299.99,238.15,477.41]`.
- Đã xuất `data/manifests/refcoco-val-smoke.jsonl` từ ảnh thật.
- API thật trả health JSON và trang web trả HTTP 200 trên localhost.
- Grounding DINO tiny đã chạy thật trên ảnh COCO 559760, query `front giraffe`, trả
  `[54.46,62.04,335.66,386.96]`; gold `[56.67,63.65,304.30,383.64]`, IoU xấp xỉ 0.87.
- Máy có RTX 4060 Laptop 8188 MiB nhưng PyTorch hiện là `2.15.0+cpu`; vì vậy chưa thể
  đo CUDA VRAM hay xác minh Qwen2.5-VL-3B trên GPU. Direct inference và bước Qwen chọn
  proposal chưa được tuyên bố là đã chạy thật. Cài đúng wheel PyTorch CUDA tương thích
  driver rồi chạy smoke benchmark để hoàn tất hai bước này.
- Tập tiếng Việt tự gán nhãn chưa tồn tại, nên chưa đo false-positive và ambiguity recall.

## 14. Tự viết lại project

Học theo thứ tự phụ thuộc dưới đây. Với mỗi file, hãy viết test nhỏ trước khi mở phần
implementation tham chiếu.

| File | Nhận → trả | Ai gọi | Viết trước / tự kiểm tra |
|---|---|---|---|
| `metrics.py` | hai box → IoU; xywh → xyxy | data, benchmark | `coco_xywh_to_xyxy`; hai box lệch nửa phải cho 1/7 |
| `prepare_dataset.py` | REFER+COCO → JSONL | người chạy | join ID; preview phải trỏ tới ảnh tồn tại |
| `schemas.py` | dữ liệu Python → contract | pipeline/API | enum; thử `found` mà box null phải lỗi |
| `image_io.py` | bytes → RGB đúng hướng | API | decode; tạo JPEG có EXIF rotate |
| `vlm.py` | ảnh+query → status/box/index | pipeline | parser JSON; thử output có prose và JSON |
| `detector.py` | ảnh+cụm vật → proposal boxes | pipeline | prompt; kiểm tra box nằm trong ảnh |
| `pipeline.py` | adapters → result chuẩn | API/benchmark | direct trước; detector rỗng phải `not_found` |
| `api.py` | multipart HTTP → JSON | web | `/health`; gửi PNG 8×6 bằng fake backend |
| `web/*` | file+query+JSON → overlay/crop | trình duyệt | scale box; resize cửa sổ không làm lệch box |
| `benchmark.py` | manifest+pipeline → predictions/summary | người chạy | một fake prediction IoU=1 phải Acc=1 |

### Trace một mẫu thật

**Cụm 1 — annotation → manifest**

- Code: `load_refer()` đọc `refs(unc).p`, tìm `ann_id=1719310` trong `instances.json`.
- Console Output: `image_id=581857`, ảnh `(427, 640)`.
- Internal State: COCO xywh `[103.93,299.99,134.22,177.42]` → xyxy
  `[103.93,299.99,238.15,477.41]`.
- Explanation: cạnh phải/dưới là giới hạn ngoài nên chỉ cộng width/height, không trừ 1.

**Cụm 2 — manifest → inference → API**

- Code: benchmark mở ảnh, chuẩn hóa EXIF, gọi `GroundingPipeline.run`; API gọi cùng hàm.
- Console Output: Không có (Qwen cho mẫu này chưa được chạy thật trên môi trường CPU-only).
- Internal State: minh họa `status=found`, `bbox_xyxy=<model output>`, kích thước 427×640.
- Explanation: pipeline clip/validate box; lỗi model đi theo exception/HTTP code riêng.

**Cụm 3 — prediction → IoU**

- Code: `box_iou(prediction, gold_bbox_xyxy)`.
- Console Output: Không có cho mẫu 581857. Với smoke Grounding DINO ảnh 559760, IoU
  tính từ số thực tế xấp xỉ `0.87`.
- Internal State: intersection/union dùng diện tích hình chữ nhật half-open.
- Explanation: đạt Acc@0.5 khi IoU ≥ 0.5; từ chối mẫu gold found vẫn tính sai.

### Bài tập cuối

Chỉ nhìn README, JSONL và test: tự viết lại `metrics.py`, `schemas.py`, `image_io.py`, rồi
fake adapters và `pipeline.py`. Làm test xanh trước. Sau đó tự dựng API/web và benchmark,
chạy trên ba mẫu. Cuối cùng mới mở code tham chiếu, lập bảng khác biệt theo: contract,
tọa độ/EXIF, lỗi kỹ thuật, tính tái lập và cách tính mẫu số metric.
