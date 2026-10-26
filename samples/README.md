# 三类真实演示样例与来源

本目录包含 **75 张真实演示图片**：瓶口 20 张、金属螺母 25 张、瓷砖 30 张。另外保留 `demo/` 的 6 张合成联调图片，合成图片不作为工业检测效果证据。

## 图片分别是什么

| 产品与目录 | 原始缺陷名称 | 中文含义 | 演示数量 |
| --- | --- | --- | ---: |
| 瓶口 `bottle/` | `good` | 正常完整瓶口 | 5 |
| 瓶口 `bottle/` | `broken_large` | 较大破损 | 5 |
| 瓶口 `bottle/` | `broken_small` | 较小缺口 | 5 |
| 瓶口 `bottle/` | `contamination` | 污染或异物 | 5 |
| 金属螺母 `metal_nut/` | `good` | 正常螺母 | 5 |
| 金属螺母 `metal_nut/` | `bent` | 弯折、变形 | 5 |
| 金属螺母 `metal_nut/` | `color` | 颜色异常 | 5 |
| 金属螺母 `metal_nut/` | `flip` | 翻面、朝向异常 | 5 |
| 金属螺母 `metal_nut/` | `scratch` | 表面划痕 | 5 |
| 瓷砖 `tile/` | `good` | 正常表面纹理 | 5 |
| 瓷砖 `tile/` | `crack` | 裂纹 | 5 |
| 瓷砖 `tile/` | `glue_strip` | 胶条、胶痕 | 5 |
| 瓷砖 `tile/` | `gray_stroke` | 灰色涂痕 | 5 |
| 瓷砖 `tile/` | `oil` | 油污 | 5 |
| 瓷砖 `tile/` | `rough` | 粗糙区域 | 5 |

这些名称来自数据集的参考标签，不代表模型能直接分类出上述缺陷原因。模型输出异常分数、位置与规则判断；是否检出要以实际预测结果为准。

`demo/synthetic_good_0/1/2.png` 是亮度、噪声略有区别的合成正常垫圈；`demo/synthetic_scratch_0/1/2.png` 分别在圆环、中心、外部背景添加红色划痕，用于验证热图和规则流程。它们集中用于开发测试。

## 如何选出样例

每个 `test/<缺陷名称>/` 目录按文件名排序，固定复制前 5 张；不按模型分数、成功率或视觉效果挑选。原有四张瓶口 `*_000.png` 保留。样例仍属于原测试集，只是复制到界面展示目录，不计为额外数据、不参与训练，也不用于校准阈值或调整规则。

文件名如 `metal_nut/scratch_000.png` 对应 `data/raw/mvtec_ad/metal_nut/test/scratch/000.png`。不同类别、缺陷目录都可各自从 `000` 编号，同一编号不等于同一张图片。完整源路径与文件 SHA256 见本目录 `samples.json`，每个产品目录也有自己的 `samples.json`。

重新生成样例：

```powershell
python scripts/prepare_samples.py --root data/raw/mvtec_ad --output samples --categories bottle metal_nut tile --per-type 5
```

脚本保留已有文件；如同名文件内容与数据源不同会报错，避免静默覆盖。调整为更少样例时，旧图片不会自动删除。

## 完整数据与划分

演示只选出部分测试图片，训练与评估使用各类别完整数据。以下计数均指原始图像，不包含缺陷分割掩码：

| 类别 | 正常训练 | 正常校准 | 独立测试：正常 / 缺陷 | 图像总数 | 演示图片 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 瓶口 `bottle` | 167 | 42 | 20 / 63 | 292 | 20 |
| 金属螺母 `metal_nut` | 176 | 44 | 22 / 93 | 335 | 25 |
| 瓷砖 `tile` | 184 | 46 | 33 / 84 | 347 | 30 |
| 合计 | 527 | 132 | 75 / 240 | 974 | 75 |

使用固定种子 `42`，仅将原 `train/good` 按内容组划出 20% 正常校准图片，原 `test` 保持独立。三个类别内容组计数分别为 292、335、347，当前清单无完全相同像素内容的重复图像；这不保证没有视觉相似图片或同一实物的不同照片。清单位于 `data/manifests/<类别>.csv`，统计与来源在同名 `.summary.json`。

## 下载与完整性

原始作者：**MVTec Software GmbH**。来源为 [MVTec AD 官方介绍](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)，本次使用 [foersben/mvtec-ad 镜像](https://huggingface.co/datasets/foersben/mvtec-ad)，固定版本 `c75b39616f84db43677bcc8228caaafaf5096d7f`。许可为 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，本项目仅作非商业课程研究与演示，保留来源并遵守相同方式共享要求。

泛化下载命令（在项目根目录运行，每次一个类别）：

```powershell
python scripts/download_mvtec.py --category metal_nut
python scripts/download_mvtec.py --category tile
```

`python scripts/download_bottle.py` 保留旧入口，等价于 `python scripts/download_mvtec.py --category bottle`。下载器支持上述三个类别，自动跟随目录分页，固定镜像版本，限制路径边界，使用超时重试与临时文件原子替换。已存在且校验匹配的文件直接复用。

2026-10-26 已核对三类完整镜像文件：瓶口 357 个、金属螺母 430 个、瓷砖 433 个（含图像、掩码及每类两个许可说明文本）。所有图像和掩码按**镜像 LFS SHA256 与文件长度**校验；镜像中的 `license.txt`、`readme.txt` 不采用 LFS，按 **Git blob SHA1 与长度**校验。每类的详细记录在 `data/raw/mvtec_ad/<类别>/source.json`，旧根目录的瓶口 `source.json` 保留。

上述校验验证镜像文件传输与缓存一致性，**不等于核对原始发布站整包归档校验和**。
