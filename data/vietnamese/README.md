# Tập đánh giá tiếng Việt

Tạo `annotations.jsonl`. Loader chấp nhận tên dễ gán nhãn `sample_id`, `status`,
`bbox_xyxy` và chuyển rõ ràng sang schema benchmark `id`, `gold_status`,
`gold_bbox_xyxy`. Mỗi dòng còn cần `image_path`, `query`, `split`; `dataset` mặc định
là `vietnamese`. Dùng box pixel cho `found`, và null cho `ambiguous`/`not_found`.

```json
{"sample_id":"vi-fixture-001","image_path":"data/vietnamese/fixtures/desk.jpg","query":"chiếc cốc trắng bên trái bàn phím","status":"found","bbox_xyxy":[10,20,90,130],"split":"fixture","is_test_data":true}
```

Fixture tự tạo phải có `is_test_data:true` và không được báo như metric dữ liệu thật.
