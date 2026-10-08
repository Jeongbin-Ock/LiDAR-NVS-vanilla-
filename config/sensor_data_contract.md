# LiDAR 센서 변경 시 데이터·렌더링 설정

이 문서는 현재 소스와 KITTI-360 공식 데이터, LiDAR4D 전처리 및 제조사 문서를
대조한 결과다. `nuscene_train.yaml`의 학습 기본값은 변경하지 않았다.

## 물리 센서와 현재 비교용 config

| 항목 | HDL-64E 계열의 명목 사양 | 현재 KITTI-360 inference config |
|---|---|---|
| laser 채널 수 | 64 | panorama의 행 수 64, 레이저당 한 행(데이터에서 복원한 레이저 각도) |
| 수직 FOV | 26.8°, 대표 사양 +2°~−24.8° | 레이저 각도 +1.937°~−24.999° (benchmark: +2°~−24.9°) |
| 수평 FOV | 360° | 360° |
| panorama 폭 | 채널별 발사 주기·회전율·return 방식·투영 격자에 따라 달라짐 | 2178 |
| 유효거리 | 반사율 약 10% 포장도로 50m, 약 80% 차량·수목 120m | GT return <80m |

현재 기본 config는 64×2178 calibrated 레이저 투영(`hdl64e_kitti360_calibrated`)이다. 별도
`kitti360_inference_lidar4d_benchmark.yaml`의 `66×1030`,
`(fov_up, fov)=(2.0,26.9)`, 80m는 공식 LiDAR4D 전처리 설정과 일치한다.
VFOV 아래쪽의 0.1° 차이와 66행은 비교용 투영 격자다.
센서가 66개 beam을 가졌다는 의미는 아니다. 명목 FOV의 양 끝과 공장 보정된
64개 beam 방향도 구분해야 한다.

제조사의 120m는 모든 표면에서 보장되는 거리가 아니다. 거리 성능은 반사율,
입사각과 환경 조건에 의존한다. 현재 평가에 사용한 40개 target raw 스캔을
직접 검사한 결과, 최대 range는 80.00317m이고 80m 이상 포인트는 합계 3개,
120m 이상 포인트는 0개였다. 이것은 해당 40개 스캔의 관측 결과이며 데이터셋
전체의 절대 cutoff나 센서의 최대 거리를 입증하지 않는다.

## Ring와 GT panorama

nuScenes raw 파일의 다섯 번째 값은 ring ID다. 현재 nuScenes 로더는 XYZI와
ring을 분리한 후, `data.ring_to_elevation_deg`로 ring→row를 계산한다.
`Camera.row_to_theta`와 renderer도 같은 beam 방향을 사용한다.

KITTI-360 공개 Velodyne `.bin`에는 float32 `[x,y,z,intensity]`만 있다.
Ring ID, 포인트별 firing time, factory beam calibration은 열로 제공되지 않는다.
다운로드한 `calibration/`에도 per-laser `db.xml`은 없다. 그러나 2026-10-06
검사 결과, 레이저 ID와 레이저별 수직 보정값은 데이터에서 정확히 복원된다.

- 각 return은 Velodyne 보정 모델(ROS driver `rawdata.cc`)로 계산되었다.
  따라서 한 레이저의 점은 센서 z축 둘레의 원뿔 `z = ρ·tanδ_k + V_k`
  (`ρ = hypot(x, y)`) 위에 있다.
- `.bin`은 레이저 순서로 저장되어 있다(최상단부터, 각 레이저는 +x에서
  시작). 이 순서로 레이저를 나누고 레이저마다 직선을 맞췄다.
  - 9개 drive의 722개 스캔에서 최대 잔차는 3 µm(float32 한계)다.
  - δ는 +1.937°~−24.999°다.
  - V는 상단 블록(0~31) 0.198~0.209 m, 하단 블록(32~63) 0.116~0.126 m다.
- 이 표(`src/dataloader/kitti360_hdl64e.py`)로 각 점을 가장 잘 맞는 원뿔에
  할당하면 저장 순서가 필요 없다. 저장 순서 분할은 1,102개 중 3개 스캔에서
  한 레이저를 둘로 나누지만, 원뿔 할당은 모든 스캔에서 정확하다.

수직 offset 때문에 센서 원점 기준 elevation은 한 레이저 안에서도 거리에 따라
변한다(40~80 m +0.24°, 2.5~5 m +1.54°). 그래서 아래의 uniform 방식처럼
elevation으로 row를 정하면 레이저가 섞인다. 기본 projection
`hdl64e_kitti360_calibrated`는 nuScenes의 ring→row처럼 레이저→row를 쓴다.
상/하단 블록은 각자의 원점에서 렌더한다([kitti360_inference.md](kitti360_inference.md)).

별도 KITTI benchmark 모드(`panorama_projection=uniform`)는 sensor 좌표의
XYZ로 다음처럼 구면 투영한다:

```text
elevation = atan2(z, sqrt(x²+y²))       # 아래 row 식에는 degree 사용
azimuth   = atan2(y, x)                # 아래 column 식에는 radian 사용
row       = round((2.0-elevation)/26.9 * 66)
column    = round((π-azimuth)/(2π) * 1030)
range     = sqrt(x²+y²+z²)
```

같은 픽셀에 여러 점이 들어오면 가장 가까운 return의 range와 intensity를
사용한다. 0~80m의 유효 포인트만 GT에 반영하고 나머지 픽셀은 no-return이다.
실제 스캔의 결과는 공식 LiDAR4D converter와 픽셀 단위로 일치함을 검사했다.
이는 uniform-angle panorama이며 원본 ring별 range image와 동일하지 않다.

`data.ring_to_elevation_deg: null`은 raw 파일에 ring 열이 없음을 반영한다.
calibrated projection은 이 값 대신 위의 레이저별 δ_k 표를 row 각도로 쓴다.

## 센서 채널에 맞춘 높이·폭

높이를 채널 수에 맞추려면 **H=64**다. 폭은 beam 수와 Hz만으로 결정할 수 없다.
레이저의 실제 발사율이나 수평 sampling step이 추가로 필요하다. 근사 평균식은:

```text
W_average ≈ point_rate / (beam_count * rotation_hz)
```

예를 들어 제조사 문서의 약 1.3M points/s를 대입하면 10Hz에서 약 2031,
실측 9.57118Hz에서 약 2122다. 1.333M points/s를 대입하면 각각 약 2083,
2176이다. 이 숫자는 point-rate 예산을 beam에 균등하게 나눈 추정치이며
KITTI-360 장치의 정확한 ring당 발사 수가 아니다. 초기 HDL-64E 제조사 manual
Appendix B는 upper/lower laser block의 sampling이 서로 다른 예도 제시한다.
브로셔의 단일 각 해상도를 현재 회전율에서 모든 laser의 firing 간격으로
치환해서도 안 된다.

따라서 uniform rasterization을 위한 **64×2048**은 약 2k columns를 선택한
실용적인 근사 설계다. 이를 실제 센서의 정확한 native image 크기로 표기하면
안 된다. 40개 raw target의 평균 유효 포인트 수는 115,818.875개이며, 이를 64로
나눈 약 1810도 no-return이 제거된 평균 유효 return 수라 native W를 결정하지
못한다. nuScenes의 약 1080±10 points/ring도 해당 dataset의 sampling 정보를
확인한 값이며 beam 수와 회전율만으로 얻은 숫자가 아니다.

공식 LiDAR4D/GS-LiDAR 격자로 비교하려면 별도 benchmark config의 66×1030을
선택한다. 현재 기본 64×2178 실험은 다른 geometry/protocol로 기록한다.
ray 방향과 GT/renderer에는 같은 레이저 각도표를 적용하고, 블록별 원점도 같이 쓴다.

사용자가 제공한 [HDL-64E S2 Rev-B data sheet](https://hypertech.co.il/wp-content/uploads/2015/12/HDL-64E-Data-Sheet.pdf)는
수평 angular resolution을 **0.08°**로 명시한다. 그 간격을 panorama 격자의
기준으로 선택하면 `W=360/0.08=4500`이다. 위의 2048은 평균 point-rate에서
고른 근사 격자였으며, 이 문서의 0.08° 격자를 의미하지 않았다. 문서는 각 해상도를
어느 RPM에서 측정했는지 명시하지 않으므로 이를 현재 KITTI 장치의 정확한
ring당 저장 sample 수로 단정하지 않는다.

시간을 이용한 식은 `W=T_sweep/Δt_same_laser`다. `T_sweep/360`은 seconds/degree로
시간당 각도 관계만 주며 sample 수는 주지 않는다. nuScenes HDL-32E의 경우
추가적인 firing-cycle 값 46.08μs를 알기 때문에 `0.05s/46.08μs≈1085`를 계산할
수 있다. [HDL-32E 제조사 manual](https://ustc-flicar.github.io/pdf/MANUAL_USERS_HDL32E.pdf)

추가로 확인한 [HDL-64E S2/S2.1 제조사 manual, Appendix G/I](https://www.termocam.it/pdf/manuale-HDL-64E.pdf)는
S2 FW V48의 같은 laser에 대한 packet timestamp 이전 시간(지연값)이
288→240→192→144→96→48μs로 변함을 제시한다. 실제 발사 시각은
`packet_timestamp - 지연값`이다. 따라서 해당 모델/firmware의 64-beam group 반복 주기는
**48μs**다. Upper/lower의 laser가 pair로 동시에 발사되므로 32개 pair가
한 주기에 64개의 측정을 만든다. 이는 단순한 photon 왕복 시간이나 모든 laser를
동시에 발사한다는 뜻이 아니라, 동일 채널의 다음 발사까지의 주기다.

Appendix I의 수치도 이와 일치한다: 10Hz에서 laser당 약 2083 samples/revolution,
0.1728°; 5Hz에서는 약 4167 samples/revolution, 0.0864°. 데이터시트의 0.08°를
현재 약 9.571Hz의 고정 실제 sampling 간격으로 적용하는 것보다, firing-cycle을
사용하는 계산이 이 모델의 sampling에 직접 대응한다.

```text
KITTI drive0000 mean sweep time = 1/9.571177803406085 = 0.104480349s
S2 FW V48 cycle = 48μs
W_mean = 0.104480349/0.000048 = 2176.674
fixed image W = 2177
pixel angle = 360/2177 = 0.165365°
```

따라서 이 모델/firmware를 전제로 센서 주기에 맞춘 uniform panorama는
**64×2177**이다. KITTI-360의 공개 소개는 HDL-64E라고만 명시하며 정확한
unit firmware는 확인되지 않았으므로 이 숫자는 S2 FW V48 조건을 명시한다.
앞의 2048은 편의적 평균 근사 격자, 4500은 0.08° 규격을 그대로 rasterize한
격자이며, 2177은 현재 회전율과 문서화된 cycle에서 얻은 폭이다.

### 2026-10-06 원본 sweep 및 센서 버전 재검사

9개 공개 train/val 시퀀스의 `.bin` 76,715개를 모두 검사했다. float32 XYZI
한 점은 16 bytes이므로 원본 점 개수는 `file_size / 16`이다. 평균 118,229.84점,
중앙값 119,730점, 최소 60,858점, 최대 124,792점이다. 현재 평가 target 40개는
평균 115,818.875점, 최소 107,136점, 최대 121,309점이다. 이 수치는 모델의
ego-radius 제외, 거리/FOV 투영, voxelization 이전에 파일에 저장된 점 수다.

주신 S2 Rev-B PDF의 0.08°는 문서에 실제로 있지만 회전율 조건이 생략돼 있다.
현재 drive 0000에서 64×4500개의 측정이 매 sweep 생성되려면 약 2.7565M
measurements/s가 필요하므로 약 1.333M/s single-return 동작과 일치하지 않는다.
같은 회전율에서 48μs 주기의 가능한 측정 수는 약 139,307점이다. 반사가 없는
발사와 공개 point cloud의 저장/필터링 정책 때문에 유효 XYZ 수는 가능한
측정 수보다 작을 수 있으며, 파일의 점 수를 64로 나눠 native W로 삼지 않는다.

초기 HDL-64E manual의 약 1M/s, S2/S2.1 manual의 약 1.333M/s,
[S3 2013 Rev-H manual](https://s3-us-west-1.amazonaws.com/ptab-filings%2FIPR2018-00255%2F2051)의
single-return 약 1.333M/s, [S3 2018 Rev-G data sheet](https://mapping-solutions.com/downloads/data/pdf/P006.pdf)의
single 약 1.3M/s / dual 약 2.2M/s를 비교했다. S2.1 manual의 확장 수직 FOV는
+2°~−29.5°로 공개 KITTI 계열의 약 26.8°와 다르다. 가장 가까운 동작은
S2 FW V48 또는 같은 single-return sampling을 가진 S3다. KITTI-360 논문은
HDL-64E라고만 명시하므로 유효 점 수로 S2와 S3를 구분하거나 펌웨어를 확정할
수 없다. S3의 신형 브로셔 최대 rate를 2013 KITTI 장치에 그대로 적용하지 않는다.

40개 target의 원본 XYZ 순서에서 거리>20m, 연속 azimuth 차이 0.04°~0.215°인
쌍을 선택해 검사한 조건부 평균 간격은 0.16534064°이고, 이에 대한 근사 폭은
2177.32다. 개별 간격은 주로 0.09°와 0.18°로 나타난다. 이 결과는 48μs와
실측 회전율의 약 2177열을 지지하지만 ring ID/발사 timestamp를 복원한 것은
아니며 누락 return과 센서 보정에 의한 영향이 남아 있는 진단값이다.

10Hz용 0.1728°를 선택하면 `360/0.1728≈2083.33`이므로 2084열도 합리적인
명목 10Hz 격자다. 하지만 실제 9.57118Hz에 맞추면 각도 간격 역시
`0.1728×9.57118/10≈0.16538995°`로 바뀐다. 따라서 현재 저장 데이터에 맞춘
센서 주기 근사 격자는 2177열이다. 현재 기본 config는 사용자 요청에 따라
W=2178을 쓰며(calibrated 레이저 투영), 66×1030 격자는 별도 benchmark config에
보존한다.

재현 스크립트, 전체 파일별 CSV, 버전별 문서 비교 및 원점 복구 검증은
`outputs/kitti360_sensor_audit_20261006_114546/`에 저장했다.

수평 mapping은 현 KITTI panorama 방향에 맞춰 실제 XYZ azimuth로 계산한다:

```text
azimuth_deg = degrees(atan2(y,x))
col = round((180-azimuth_deg)/(360/W)) % W
```

Modulo는 360° 경계의 wrap을 위한 것이다. 현 LiDAR4D 비교 projector는 공식
코드와 일치하도록 경계 밖 열을 제외하는 기존 동작을 유지한다.

실측 검증은 모델 eval 폴더의 `diagnostics/sensor_audit_20261005_183816/`에
기록했다. 40개 target, range<80m, benchmark VFOV를 유지한 경우 픽셀당 한 점을
남긴 후 삭제되는 추가점 비율은 66×1030에서 56.43%, 64×1030에서 57.08%,
64×2048에서 18.12%, 64×4500에서 5.69%다. 64×4500에 명목 VFOV 26.8°를 쓰면
5.58%다. 이는 동일 픽셀 중복이며 ring ID가 없어 beam 간 중복과 동일 beam의
수평 중복을 완전히 분리하는 수치는 아니다.

## `nuscene_train.yaml`에서 확인·변경할 항목

| 항목 | 현재 nuScenes 설정/의미 | 센서·데이터셋 변경 시 처리 |
|---|---|---|
| `data.dataset_name` | `nuscenes`; variant에 따라 `nuscenes_dynamic_lidar4d` | 학습용 loader를 등록한 뒤 새 이름 지정 |
| `data.dataroot` | `/data1/nuScenes` | 새 데이터 루트 |
| `data.version`, `test_version`, `train_split`, `eval_split`, `test_split` | nuScenes release/split 이름 | 새 dataset의 시퀀스·구간 split 정의 |
| `data.vfov`, `hfov` | `[-30.67,10.67]`, `[-180,180]` | 센서 FOV 및 비교 프로토콜 |
| `data.image_height`, `image_width` | 32×1085 | 채널/투영 격자 및 수평 sampling에 맞게 선택 |
| `data.ring_to_elevation_deg`, `p2g.ring_to_elevation_deg` | data가 p2g의 32-entry table을 참조 | 새 센서의 ring·calibration에 맞는 table; KITTI uniform projection은 null |
| `data.coordinate_frame` | `ego_vehicle` (과거 checkpoint는 sensor frame 가능) | 물리 축 convention 유지, loader의 extrinsic/pose 방향 변환 구현 |
| `data.window_us`, `sample_gap_us` | 1s window, 0.1s 선택 간격 | 원본 timestamp로 선택; nuScenes의 50ms sweep 상수 재사용 금지 |
| `data.pair_mode`, `pair_kf_stride` | nuScenes keyframe pairing | KITTI에는 동일한 keyframe 정의가 없으므로 timestamp/sweep pairing으로 재설계 |
| `data.gt_middle_count` | 중간 GT 3개 | 새 window에서 유효한 중간 스캔 수와 pose 정책 확인 |
| `data.dynamic_window.*` | variant별 0.2~1.2s 등 physical interval | 물리 초 단위 유지; exclusion 및 sampling은 새 dataset에 맞춤 |
| `data.train_excluded_scenes` | nuScenes scene name | KITTI 시퀀스/구간을 사용해 leakage 없는 split 정의 |
| `data.compact_val.enable/path` | nuScenes 전용 200-window manifest | disable하거나 KITTI용 manifest 새로 작성 |
| `data.mode`, `bbox_json_path(s)` | bbox 또는 bbox-free nvs; nuScenes tracking JSON | bbox-free는 nvs; bbox 사용 시 XML/track/frame annotation adapter 필요 |
| `p2g.ego_radius`, `data.ego_radius` | 2.5m 입력 self-point 제외 | 차량·센서 설치 위치에 맞는 mask 검토 |
| range policy | base YAML에 공통 max-depth 항목 없음 | point cutoff, GT, loss, metrics, backprojection의 거리 정책을 함께 정의 |
| `train.batch_size`, `grad_accum_steps`, `data.num_workers` | 현재 training throughput 설정 | 증가한 point/token 수로 메모리·시간을 측정한 뒤 조절 |

`data.max_depth_m`은 현재 KITTI inference GT projector에서 읽는 항목이다.
nuScenes training `Camera.from_nuscenes`에는 이 거리 cutoff가 연결돼 있지 않다.
평가 스크립트의 `EVAL_MAX_DEPTH_M=80`도 별도로 사용되므로 YAML에 120만 적어서
학습·GT·point metric·depth metric의 거리가 모두 120m로 바뀌지는 않는다.

Config 외에 point 포맷·강도·좌표 처리도 필요하다. nuScenes의 `/255` intensity
정규화를 이미 0~1인 KITTI intensity에 적용하면 안 된다. KITTI IMU의
forward/right/down을 모델의 forward/left/up으로 변환하고, sparse pose를 직접
사용/이전값 유지/보간 중 어떤 방식으로 처리하는지 명시해야 한다.

현재 KITTI ego 원점은 제공된 GPS/IMU 위치다. 축 방향만 FRD에서 FLU로
변환하며 다른 데이터셋의 설치 위치에 맞추는 추가 translation은 사용하지 않는다.
원래 [-0.1,+0.1] 평가의 120개 frame instance 중 110개는 direct pose,
10개 입력은 이전 pose 유지다. 독립 camera-world 경로와 비교한 LiDAR world
pose의 최대 위치 차이는 0.643mm이며, 실제 점의 ref↔camera 왕복 오차는
최대 0.023mm였다. 이는 제공된 calibration/pose와의 일치 검사이며 누락 pose의
현재 sweep 실제 위치를 새로 측정한 결과가 아니다.

2026-10-06 사용자 요청에 따라 임시 nuScenes 원점 정규화 옵션, reference
calibration 파일 및 pose/camera에 적용하던 추가 원점 변환을 제거했다. 과거
실험 결과는 당시 좌표 기준을 기록한 채 보존한다.

센서가 바뀌어도 `p2g.gs_params`, attention layer/head 수, adaptation rank 등을
beam 수에 비례해 바꾸지 않는다. 특히 `p2g.gs_params.shs:32`는 intensity/raydrop
SH coefficient 개수이며 LiDAR의 32 channels가 아니다. XYZI input 계약,
meter 단위의 `g2p.scale_factor:1.0`, 초 단위 time reference, m/s velocity 단위는
유지한다. Voxel 크기, 범위별 token 통계, batch 크기와 학습률은 성능·메모리를
측정한 후 조절할 hyperparameter다.

## 저장 경로

`out_dir: null`은 checkpoint 부모 폴더를 선택한다:

```text
logs/<model>/lidar4d_kitti360_eval/
logs/<model>/lidar4d_nuscenes_eval/
```

명시적인 CLI `out_dir`이 있으면 그 경로를 사용한다. 여러 offset 결과를 보존할
때는 이 폴더 아래 run별 child directory를 지정한다. 과거 outputs 결과는 원래
provenance를 유지하며 모델의 eval 디렉토리 `runs/`에서 링크로 접근할 수 있다.

## 자료

- [KITTI-360 공식 센서 소개](https://www.cvlibs.net/datasets/kitti-360/)
- [KITTI-360 공식 raw point loader](https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/viewer/kitti360Viewer3DRaw.py)
- [Velodyne HDL-64E S2 제조사 manual: Specifications / Appendix B](https://manualzz.com/doc/1706698/velodyne-hdl-64e-s2-lidar-sensor-user%E2%80%99s-manual)
- [Velodyne HDL-64E S2 제조사 data sheet](https://www.termocam.it/pdf/scheda-tecnica-HDL-64E.pdf)
- [LiDAR4D GT panorama 설정](https://github.com/ispc-lab/LiDAR4D/blob/main/data/preprocess/generate_rangeview.py)
- [nuScenes 공식 센서·포맷 설명](https://www.nuscenes.org/nuscenes)
