# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Hoang Trung Khai
- MSSV: 2A202602947
- Email: khaihoang004@gmail.com
- Link repo (fork): https://github.com/khaihoang004/K4-L2L3-DAY23-HoangTrungKhai-2A202602947-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): Lấy từ lệnh này sau commit cuối và ghi cùng link repo trên LMS; không thể ghi hash của chính commit hiện hành vào file này.

## Tóm tắt kết quả

- `fusion_mode = compare`, `frames = [0, 198]` (199 frame mỗi mode), `segment = training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `seed = 0`.
- `detection.precision = 0.970093`, `detection.recall = 0.700405`, `tp/fp/fn = 519/16/222`.
- `tracking.lidar`: `rmse = 0.150323 m`, `matches = 502`, `sum_sq_err = 11.343648 m²`, `ghost_track_frames = 0`, `missed_gt_frames = 239`, `mean_confirmed_tracks = 2.522613`.
- `tracking.fused`: `rmse = 0.135867 m`, `matches = 502`, `sum_sq_err = 9.266810 m²`, `ghost_track_frames = 0`, `missed_gt_frames = 239`, `mean_confirmed_tracks = 2.522613`.
- Fused giảm RMSE `0.014456 m` (khoảng 9.6%) trên cùng 502 cặp ghép. Hai mode đều có 0 ghost và 239 GT bị bỏ sót, nên khác biệt RMSE phản ánh sai số vị trí của các cặp đã ghép, không phải việc mode fused theo dõi thêm xe. Với `matches = 502` và `det_tp = 519`, coverage theo rubric là khoảng `0.9672`; precision track là `1.0` ở cả hai mode. Camera trong thí nghiệm dùng tâm hộp 2D từ nhãn GT có thêm nhiễu, nên kết quả chưa chứng minh chất lượng của một camera detector độc lập.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. LiDAR đo vị trí 3D trong hệ cảm biến: `z` có 3 phần tử, `R` là ma trận 3×3 với phương sai vị trí theo mét vuông. Camera đo tâm hộp 2D `[u, v]`: `z` có 2 phần tử, `R` là ma trận 2×2 với phương sai theo pixel vuông. EKF dùng `h(x)` và Jacobian `H` tương ứng với từng cảm biến để đưa cùng trạng thái 6D vào không gian đo.
2. Mahalanobis chuẩn hóa sai lệch `γ = z − h(x)` bằng `S = HPHᵀ + R`, nên xét cả bất định của track và cảm biến. Cổng χ² theo số chiều đo loại cặp không hợp lý trước bước chọn cặp gần nhất; cặp ngoài FOV bị loại trước cả khi tính projection.
3. Pipeline là **track-then-fuse**. Trong mỗi frame, `fusion-run-lab` dự báo từng track một lần, gán và cập nhật LiDAR, rồi gán và cập nhật camera trên cùng danh sách track ở mode `fused`. Log ghi một record `(mode, frame)` để đối chiếu mode LiDAR và fused; log không ghi từng phép EKF riêng lẻ.
4. Lệch extrinsic hoặc intrinsic làm `h(x)` dự báo sai pixel, khiến innovation `γ` và Mahalanobis tăng. Thí nghiệm yaw extrinsic 0°/1°/3°/5° bên dưới đo được innovation ứng viên tốt nhất trung bình 15.0/50.6/101.5/106.7 pixel. Tỷ lệ track không còn ứng viên camera qua cổng χ² tăng từ 1.2% lên 97.4% ngay tại 1°. Một số update vẫn qua cổng và có thể kéo trạng thái lệch: RMSE fused ở 1° tăng từ 0.1359 m lên 0.1738 m. RMSE tại 5° giảm còn 0.1598 m so với 1° vì số update và tập cặp qua cổng thay đổi; không thể suy ra RMSE luôn tăng theo độ lệch.
5. Khi `meas_list` rỗng vẫn phải biết lượt hiện tại là LiDAR hay camera. Lượt LiDAR cần tính miss cho track trong FOV, xét xóa, và có thể tạo track từ đo chưa ghép. Lượt camera chỉ hiệu chỉnh EKF cho cặp đã ghép, không đổi score và không tạo hay xóa track. Vì vậy `associate_and_update` luôn chuyển `sensor` cho `manage_tracks`, kể cả ở frame rỗng.
6. Track mới có state `initialized`, score `1/window`. Mỗi LiDAR hit cộng `1/window` (tối đa 1); hit của track chưa confirmed chuyển sang tentative, rồi confirmed khi score **lớn hơn** `confirmed_threshold`. LiDAR miss trong FOV trừ `1/window`; state confirmed được giữ sau miss. Xóa khi `Pxx` hoặc `Pyy` **lớn hơn** `max_P`, hoặc confirmed có score **nhỏ hơn** `delete_threshold`, hoặc track chưa confirmed có score **nhỏ hơn hoặc bằng** 0.

## Bonus (không bắt buộc)

### Trực quan hoá track và camera update (+3)

Notebook `student/bonus/visualize_fusion.ipynb` phát lại cùng segment, weights và seed 0; hai ảnh chọn nộp là `student/bonus/frame_131_bev.png` và `frame_134_bev.png`. Ở frame 131 có 3 cặp track–GT trong cả hai mode, tổng sai số bình phương 3D giảm từ 0.1230 xuống 0.0581 m². Trong ô phóng to của `frame_131_bev.png`, GT `8EFRSwEXBf9P-SzIDRzx0A` ghép với track #10: sai số bình phương của mode chỉ LiDAR là 0.0896 m², của mode fused là 0.0120 m². Mũi tên từ vị trí fused trước camera đến sau camera cho thấy hiệu chỉnh **trong frame 131**; chênh lệch giữa hai mode còn gồm ảnh hưởng camera ở các frame trước. Notebook đã đối chiếu matches và sai số của các frame 80, 83, 131, 134 với `grade_run_lidar.log` và `grade_run_fused.log` khi tạo ảnh.

### Độ nhạy với lệch extrinsic camera (+4)

Script `student/bonus/calibration_experiment.py` chạy frame 0–198 với cùng detection LiDAR và cùng pixel camera có nhiễu seed 0 cho bốn tracker riêng. Chỉ ma trận extrinsic camera thay đổi: `T_sensor_to_vehicle = Rz(góc lệch quanh trục đứng của xe) @ T_gốc`. Dòng 0° được kiểm tra khớp từng frame với log fused của lần chấm điểm; mọi kết quả thô lưu trong `student/bonus/calibration_results.json`.

| Lệch yaw | Innovation ứng viên tốt nhất trung bình | Track bị cổng χ² chặn | Camera updates | RMSE fused | Matches / ghost / miss |
|---:|---:|---:|---:|---:|---:|
| 0° | 15.0 px | 6/490 (1.2%) | 509 | 0.1359 m | 502 / 0 / 239 |
| 1° | 50.6 px | 480/493 (97.4%) | 20 | 0.1738 m | 502 / 0 / 239 |
| 3° | 101.5 px | 487/497 (98.0%) | 17 | 0.1736 m | 502 / 0 / 239 |
| 5° | 106.7 px | 471/500 (94.2%) | 42 | 0.1598 m | 502 / 0 / 239 |

Mỗi *ứng viên tốt nhất* là phép đo camera có Mahalanobis nhỏ nhất cho một track confirmed đang trong FOV và có ít nhất một phép đo FRONT. “Bị chặn” nghĩa là track đó không có **bất kỳ** ứng viên nào qua cổng χ²; đây là thống kê trước ghép tham lam, không phải số camera update bị từ chối sau ghép. Ở 1° và 3°, innovation tăng mạnh và gần như toàn bộ track được khảo sát không qua cổng, nhưng các update hiếm còn được chấp nhận vẫn có thể làm RMSE xấu đi. Ở 5°, tập update khác đi nên RMSE thấp hơn 1°/3° dù innovation trung bình cao hơn. Vòng đời track do LiDAR quyết định nên matches, ghost và miss giữ nguyên trong bốn mức thử này. Dữ liệu camera là tâm hộp GT 2D có nhiễu; phép thử đo độ nhạy của mô hình fusion với calibration sai, không đo hiệu quả detector camera độc lập.

### CVAT (+3)

Đã export track từ segment `10072231702153043603_5725_000_5745_000` bằng `fusion_lab.export_cvat.export_tracks_json`; file bằng chứng là `student/bonus/cvat_tracks_10072231702153043603_5725_000_5745_000_with_camera_labels.json`. Script `prepare_cvat.py` phát lại tracker; `package_cvat_ghost.py` tạo 13 ảnh BEV và annotation MOT 1.1 cho CVAT từ các frame 143–155. Khi kiểm tra task CVAT ở ảnh `000004.jpg` (Waymo frame 146), ảnh hiển thị **0 GT vehicle hợp lệ** nhưng vẫn có **1 rectangle track**. Ảnh chụp giao diện đã import là `student/bonus/cvat_ghost_frame146.png`; `student/bonus/cvat_ghost_review.json` đối chiếu Waymo frame 146–153 với track nền tảng #8 (ID 9 trong MOT; CVAT hiển thị ID nội bộ của task). Đây là ghost confirmed có thật trên segment thứ hai. Segment mặc định không có ghost confirmed hoặc đổi ID, nên minh hoạ CVAT dùng segment khác với lần chạy chấm điểm; artifacts chấm điểm vẫn là segment mặc định. Các ảnh và ZIP dùng để import nằm trong `data/cvat_bonus/` và không đưa lên Git.

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): OpenAI Codex.
- Dùng cho phần nào (hàm, câu hỏi, debug): Triển khai các hàm Part E–H; soạn giải thích thuật toán, notebook trực quan, script thí nghiệm calibration và script chuẩn bị CVAT; kiểm tra test và số liệu.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): `pytest student/tests -q` đạt 128 passed; đối chiếu công thức EKF, Mahalanobis, projection và điều kiện lifecycle với hướng dẫn kỹ thuật. Chạy segment Waymo mặc định đủ frame 0–198 bằng `--fusion compare --seed 0`; đối chiếu `metrics.json` với các log JSONL. Notebook đối chiếu bốn frame đã vẽ với log; script calibration đối chiếu baseline 0° với log fused ở cả 199 frame.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: giữ nguyên code có sẵn
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm số liệu thực nghiệm và khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
