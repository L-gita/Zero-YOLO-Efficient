# 颗粒版（默认 auto）vs 原版基线

数据：C:\Users\12983\Documents\repro\demo_mosaic\default_vs_base2\metrics.csv；图例：base=原版 Zero-DCE，def=颗粒版默认（s=0.20、不碰色度、自动阴影补偿）

## 按集合汇总（均值）

| 集合 | n | 指标 | 原版 | 颗粒版 | 变化 |
| --- | --- | --- | --- | --- | --- |
| eval15 | 15 | bright | 80.8464 | 79.6687 | -1.5% |
| eval15 | 15 | luma_noise_flat | 5.1520 | 4.1827 | -18.8% |
| eval15 | 15 | chroma_noise_flat | 5.5374 | 6.0858 | +9.9% |
| eval15 | 15 | blockiness | 1.0017 | 0.9670 | -3.5% |
| eval15 | 15 | detail_std8 | 8.0167 | 6.9703 | -13.1% |
| eval15 | 15 | cos_struct | 0.9201 | 0.9185 | -0.2% |
| eval15 | 15 | artificial_edges | 38.1367 | 64.3089 | +68.6% |
| eval15 | 15 | shadow_mean | 70.2861 | 68.9358 | -1.9% |
| LIME | 10 | bright | 100.8948 | 100.1739 | -0.7% |
| LIME | 10 | luma_noise_flat | 5.5540 | 4.8619 | -12.5% |
| LIME | 10 | chroma_noise_flat | 2.3560 | 2.5638 | +8.8% |
| LIME | 10 | blockiness | 1.0212 | 0.9752 | -4.5% |
| LIME | 10 | detail_std8 | 11.5145 | 10.9726 | -4.7% |
| LIME | 10 | cos_struct | 0.8526 | 0.8581 | +0.6% |
| LIME | 10 | artificial_edges | 10.8011 | 13.7202 | +27.0% |
| LIME | 10 | shadow_mean | 65.5011 | 64.4785 | -1.6% |
| DICM | 16 | bright | 110.2596 | 109.4471 | -0.7% |
| DICM | 16 | luma_noise_flat | 4.9747 | 4.0720 | -18.1% |
| DICM | 16 | chroma_noise_flat | 1.7501 | 1.6723 | -4.4% |
| DICM | 16 | blockiness | 0.9904 | 0.9923 | +0.2% |
| DICM | 16 | detail_std8 | 10.3035 | 9.5671 | -7.1% |
| DICM | 16 | cos_struct | 0.8312 | 0.8403 | +1.1% |
| DICM | 16 | artificial_edges | 7.9930 | 9.1300 | +14.2% |
| DICM | 16 | shadow_mean | 52.3592 | 50.9377 | -2.7% |
| extreme | 6 | bright | 103.8238 | 102.8997 | -0.9% |
| extreme | 6 | luma_noise_flat | 3.3754 | 2.7453 | -18.7% |
| extreme | 6 | chroma_noise_flat | 1.9538 | 2.3594 | +20.8% |
| extreme | 6 | blockiness | 1.0072 | 1.0071 | -0.0% |
| extreme | 6 | detail_std8 | 4.5379 | 3.8535 | -15.1% |
| extreme | 6 | cos_struct | 0.9184 | 0.9053 | -1.4% |
| extreme | 6 | artificial_edges | 22.4946 | 44.0496 | +95.8% |
| extreme | 6 | shadow_mean | 90.2305 | 89.1534 | -1.2% |

> 注意：`artificial_edges` 在压亮度噪声后会虚高（原先被亮度噪声掩盖的色度跳变被计入），`detail_std8` 含噪声成分、压噪后必然下降；这两个指标不能单独当结论。
