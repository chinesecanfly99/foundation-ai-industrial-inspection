# 演示样例来源

- `demo/*.png`：本项目 `generate_synthetic` 生成的合成垫圈与划痕，只用于软件联调。不是工业效果证据。
- `bottle/*.png`：MVTec AD bottle 的测试图片，原始作者为 MVTec Software GmbH，许可 **CC BY-NC-SA 4.0**，仅作非商业课程研究演示。文件名保留缺陷类型与原图编号，来源见下表。测试图片不用于模型拟合或选阈值。

| 样例文件 | 数据根目录内路径 |
| --- | --- |
| good_000.png | bottle/test/good/000.png |
| broken_large_000.png | bottle/test/broken_large/000.png |
| broken_small_000.png | bottle/test/broken_small/000.png |
| contamination_000.png | bottle/test/contamination/000.png |

[原始数据集](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) · [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) · [本次镜像](https://huggingface.co/datasets/foersben/mvtec-ad)

模型评估使用全部83张test图；这些样例以各目录首个文件固定选出，不按模型得分筛选。
