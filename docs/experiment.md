# Controlled augmentation experiment

Research question: Does stronger data augmentation improve YOLO11n pedestrian
detection performance on the Penn-Fudan dataset?

Both conditions start independently from the same COCO `yolo11n.pt`, with the
same split, seed, resolution, epochs, optimizer, learning rate, weight decay,
batch size, and hardware. Only the following augmentation parameters change:

| Parameter | Baseline default | Strong augmentation |
|---|---:|---:|
| hsv_h | 0.015 | 0.03 |
| hsv_s | 0.7 | 0.8 |
| hsv_v | 0.4 | 0.5 |
| degrees | 0 | 10 |
| translate | 0.1 | 0.2 |
| scale | 0.5 | 0.7 |
| shear | 0 | 2 |

Mosaic remains 1.0, fliplr remains 0.5, mixup remains 0, and close_mosaic
remains 10. All other augmentation defaults remain unchanged. The baseline
already includes augmentation; the second condition strengthens color and
geometric variation. More distortion may improve generalization or harm
localization; one seeded split cannot establish statistical significance.

Reference: https://docs.ultralytics.com/guides/yolo-data-augmentation/
