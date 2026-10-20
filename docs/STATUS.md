# 项目状态

状态日期：2026-10-20。已在当前Windows电脑完成环境、真实bottle数据、两模型拟合和83张独立测试。数据完整性与可移植性修复后，同配置复验完成，检测指标完全一致；没有根据已见测试结果调参。本机初稿完成不代表整个六周研究完成。

## 阶段概览

| 阶段 | 初稿状态 | 完成依据/下一项 |
| --- | --- | --- |
| 01 范围与验收 | 规划已完成；课程约束待确认 | 题目、边界、验收与风险；rubric 未提供 |
| 02 环境与首个闭环 | 本机CPU已通过；最终33项测试通过，pip依赖检查通过 | `results/environment.json`、`results/tests.xml` |
| 03 数据与划分 | 本机已完成数据获取、QA与清单 | 292张图，167训练/42校准/83测试；无精确内容重复 |
| 04 基线与模型 | 两模型已用167正常图拟合、42正常图校准并保存 | `models/bottle_pixel/`、`models/bottle_patch/` |
| 05 对照与稳健性 | 两模型完整83图对照及同配置复验完成；扰动/消融未完成 | pixel/patch metrics、predictions和13例FN清单 |
| 06 规则与界面 | 规则关键测试及本机浏览器9项功能检查通过；业务效果待标注 | `results/ui_smoke.json`、`ui_environment.json`与规则测试 |
| 07 新电脑与离线验收 | 待执行 | 独立电脑、断网、中文路径与连续操作 |
| 08 报告与提交 | 报告初稿已建立；定稿待实验 | report_draft、课程格式、演练 |

## 本次证据索引

已读取本次环境、数据清单与两模型结果；测试按实际JUnit记录。UI、新电脑、断网和业务有效性分别记录。

| 证据路径 | 应说明的内容 | 当前状态 |
| --- | --- | --- |
| `results/environment.json` | Python3.11.1、torch2.6.0+cpu、torchvision0.21.0+cpu、Gradio5.49.1；Windows build22631 | 本机CPU通过，CUDA不可用 |
| `results/tests.xml` | 数据/模型/规则/输入关键检查 | 33项通过，0失败/错误/跳过，14.826秒 |
| `results/ui_smoke.json` | 浏览器9项检查：真实上传、热图、JSON/CSV下载、规则、质量分支、移动布局 | passed=true，2026-10-20 11:50（UTC+8）完成；存在非阻断进度提示404 |
| `results/ui_environment.json` | Gradio配套Web依赖实际版本 | 已核对；依赖锁文件同步 |
| `data/manifests/bottle.csv`、`bottle.summary.json` | 镜像版本、清单与292张图支持数/哈希 | 已完成；167/42/83，测试20正常/63异常 |
| `results/bottle_pixel/metrics.json` | 完整83张test，冻结正常阈值 | AUROC0.9675、Recall0.7619；TN20/FP0/FN15/TP48 |
| `results/bottle_patch/metrics.json` | 完整83张test，冻结正常阈值 | AUROC0.9921、Recall0.7937；TN20/FP0/FN13/TP50 |

CNN模型排序能力较好，但冻结阈值仍漏掉13/63张异常图；像素模型漏掉15张。20张正常测试图均未误报仅描述本测试集，不保证真实产线零误报。初稿像素指标为128×128协议，不能当作官方全分辨率或AUPRO结果。最终CPU热身后推理p50/p95：像素105.33/117.95毫秒，CNN118.06/130.30毫秒；一次热身调用分别117.42与2,008.79毫秒。不是浏览器端到端耗时。

已生成 `results/examples/bottle_examples.png`（四类各首图原图/GT/热图）、逐图report及 `results/bottle_patch/false_negatives.csv`（全部13漏检）。完整性复验检查292图、63掩码，并在metrics记录模型卡/评估代码哈希。

浏览器检查覆盖DOM与空图友好错误、合成正常/异常图上传与热图/JSON、CSV对应最新图片、规则展开、黑图进入无法判定、真实bottle正常/破损两图上传与热图/JSON，以及390px视窗无整页横向溢出。图片实际解码检查通过，page_errors为空。仍记录两次Gradio `upload_progress?upload_id=undefined` 的404及请求中止日志；它们未阻断已验收上传/检测/下载，不能写成“控制台零错误”。这属于功能验收，不是业务处置准确率或长时间压力测试。

数据经 `foersben/mvtec-ad` 镜像 revision `c75b39616f84db43677bcc8228caaafaf5096d7f` 取得，357文件、156,570,806字节；已校验镜像LFS SHA256/大小，未校验原发布者完整归档哈希。清单SHA256为 `e91d0a6b4c17ebc1e9fa9be41dbd64e94278011067b73411d6f1e2805274217a`。

## 明确保留的未完成项

- 当前课程原始 rubric、最终篇幅/时长/截止日期、成员认领。
- 第二台电脑独立复演、断网演示和可选 GPU 实测。
- 规则处置协议、盲于模型输出的独立标注与 D00/D01 对照。
- 正式扰动/预算消融、必要的多 seed、充分失败案例分析。
- 生产线真实样本、真实质量标准与外部有效性验证。
- 最终报告与 PPT、全员演练、冻结提交包。

本次已完成真实bottle完整测试，阈值只由正常校准集取得，CNN使用官方ResNet18预训练权重；没有将合成联调结果混入正式成绩。同配置完整性复验只检查一致性，没有使用测试成绩选择新参数。
