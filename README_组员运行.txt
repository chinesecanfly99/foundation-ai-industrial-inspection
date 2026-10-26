# 工业异常检测项目｜组员运行说明

项目：Foundation of AI 小组作业——工业异常定位与可解释质量决策  
三类扩展说明日期：2026-10-26；首轮单瓶口交接日期：2026-10-20。
当前版本：[v0.2.0](https://github.com/chinesecanfly99/foundation-ai-industrial-inspection/releases/tag/v0.2.0)，源码已同步，模型资产可直接下载；包含三类各两个模型与合成联调模型。v0.1.0为历史瓶口版本。

## 先看这几句

**三类演示需要本轮更新源码与对应模型资产。** 先完整解压，再安装依赖，然后启动网页；已有资产不需要重新训练或下载完整数据。历史 v0.1.0 转发包与远程模型附件不能代替本轮三类资产。

先读 [README_演示.md](README_演示.md)，了解产品、样例、模型对照与导出用途。正常使用先选产品，再选中文样例；三类为瓶口、金属螺母、瓷砖表面，合成样例默认隐藏。

- Windows 路线已在原电脑验证；组员新电脑仍需按下文验收。
- 使用 64 位 Python 3.11，默认 CPU。演示无需配置 CUDA、显卡或 Anaconda。
- 首次安装需要联网。依赖装好后，包内模型和骨干缓存可用于本地推理。
- 网页只在运行它的电脑上打开；每位组员的 `127.0.0.1` 指向自己的电脑。
- 保持启动窗口打开；关闭窗口或按 Ctrl+C 后网页服务会停止。

## 推荐：通过 GitHub 获取与更新

仓库地址：**https://github.com/chinesecanfly99/foundation-ai-industrial-inspection**

这是公开仓库。任何人均可直接查看、克隆源码和下载 Release 附件，无需邀请；直接推送代码或参与仓库管理仍需仓库所有者授予相应权限。

已安装 Git 的同学，在自己准备存放项目的文件夹打开终端：

```powershell
git clone https://github.com/chinesecanfly99/foundation-ai-industrial-inspection.git industrial-inspection
cd industrial-inspection
```

本轮三类源码已同步到仓库，含 `src/industrial_inspection/catalog.py`、三类选择页面和 `README_演示.md`。从 [v0.2.0 Release](https://github.com/chinesecanfly99/foundation-ai-industrial-inspection/releases/tag/v0.2.0) 下载 [demo-assets-v0.2.0.zip](https://github.com/chinesecanfly99/foundation-ai-industrial-inspection/releases/download/v0.2.0/demo-assets-v0.2.0.zip)（也可接收同名本地转发包），将ZIP全部内容（包括 `models`、`.cache`、`samples`、`configs` 与资产清单）解压合并到项目根目录，不要多套一层同名文件夹，再按下文安装。

[v0.1.0历史版本](https://github.com/chinesecanfly99/foundation-ai-industrial-inspection/releases/tag/v0.1.0) 的旧附件只有瓶口与合成联调，不含螺母和瓷砖；本轮请使用v0.2.0。需要自行重建时按第七节执行。

不装 Git 也可下载源码ZIP或接收本轮更新源码，再另外解压对应模型资产。普通仓库源码ZIP不自带模型；它与历史完整交接包不同。

今后更新代码，在项目目录执行：

```powershell
git pull --ff-only
```

有自己的未提交修改时先保存并提交，不要强行覆盖。新增模型版本另下对应附件。以下“一”描述完整交接包的布局，GitHub获取方式可直接跳到“二、Windows首次安装”。

## 一、收到的文件怎么用

源码与三类资产合并到自己的目录，例如 `D:\AIGroup\industrial-inspection`。实际项目结构应包含：

**项目根目录**指同时包含 `pyproject.toml`、`app`、`src` 和启动脚本的文件夹。GitHub下载源码ZIP可能解压成 `foundation-ai-industrial-inspection-main`，以这个实际文件夹为准；资产解压后应直接出现 `<项目根>/models/bottle_patch`、`<项目根>/.cache/torch`、`<项目根>/samples/tile`，不能变成 `<项目根>/demo-assets-v0.2.0/models`。

```text
industrial-inspection/              ← 实际项目根目录，文件夹名称可不同
    pyproject.toml
    README_组员运行.md
    README_组员运行.txt
    README_演示.md
    首次安装_Windows.cmd
    setup_windows.ps1
    start_windows.cmd
    app/
    src/
    configs/
    requirements/
    models/                        ← 6个真实模型＋1个合成联调模型
    .cache/torch/hub/checkpoints/   ← 官方ResNet18骨干缓存
    samples/                       ← 可直接演示的图片
    docs/                          ← 执行规划、分工、报告初稿
    results/                       ← 如另附则为原电脑记录，非自己电脑新测结果
```

历史完整交接包若另带 `可选_Git历史/industrial-inspection.bundle`，只作旧快照备用，不属于本轮模型资产，也不保证包含三类更新。

不要直接在微信预览或压缩软件内部双击运行。保留 `models`、`samples` 和以点开头的 `.cache` 文件夹；Mac 的 Finder 默认可能隐藏 `.cache`。

模型资产包不带原电脑的 `.venv` 和完整原始工业数据；每台电脑新建环境，只有重新训练或完整评估时才需下载数据。旧 `可选_Git历史` bundle 只是历史快照，不保证含本轮三类修改。

### 资产完整性

本轮 `demo-assets-v0.2.0.zip` 为105,178,715字节，SHA256为：

```text
faa82d2228abb7b2f2803e14ed89d5e286a13758c5886d7b81e2bcf482f07d98
```

在ZIP所在文件夹运行 `Get-FileHash -Algorithm SHA256 .\demo-assets-v0.2.0.zip` 可核对下载是否完整。解压后的 `DEMO_ASSETS_MANIFEST.json` 列出每个资产文件的长度和SHA256。原电脑已校验CRC、106个文件哈希与重复构建一致性，并在禁网隔离目录验证7个模型后端推理；这不代替你的电脑验收。

## 二、Windows 首次安装

### 1. 准备 Python 3.11（64位）

已装 Python 的同学先在 PowerShell 或终端输入：

```powershell
py -3.11 --version
```

能显示 `Python 3.11.x` 即可。仅显示3.12或3.13不等于已装3.11。

没装的同学从 [Python官网](https://www.python.org/downloads/windows/) 安装64位 Python 3.11。使用传统安装器时保留 Python Launcher，并勾选加入 PATH；官方说明见 [Python 3.11 Windows安装文档](https://docs.python.org/3.11/using/windows.html)。安装后重新打开终端再检查上面的命令。

### 2. 一次性安装项目依赖

进入解压后的 `industrial-inspection` 文件夹，双击：

**`首次安装_Windows.cmd`**

该入口调用 `setup_windows.ps1`，创建本机 `.venv`、安装 CPU 版依赖、安装本项目并准备合成联调样例。等终端出现 `Ready. Run start_windows.cmd ...`，确认过程没有报错。

也可以在该文件夹的地址栏输入 `powershell` 并回车，然后运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

这个命令仅对本次安装进程设置执行策略，不需要永久更改电脑策略。安装中途失败时保留报错，排除网络或 Python 问题后重新运行同一入口；安装依赖需要一些时间。

如果 `py` 不存在，但 `python --version` 已确认是 **3.11.x**，可先在项目文件夹运行：

```powershell
python -m venv .venv
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

现有 `.venv` 仅限这台电脑自己创建的环境，不能拿另一台电脑的环境文件夹替代。

## 三、Windows 每次启动

进入 `industrial-inspection`，双击：

**`start_windows.cmd`**

保持终端窗口打开，在浏览器地址栏输入：

**http://127.0.0.1:7860**

以后无需重复安装；直接启动即可。如果老师或组员打开不了页面，先检查运行服务的窗口是否仍在、是否有报错，以及浏览器是否在同一台电脑。

停止运行：回到启动窗口按 **Ctrl+C**；如出现确认提示按 Y。不要随意结束不认识的后台进程。

## 四、怎么完成一次演示

1. 在“产品类别”选择瓶口，再选正常样例，默认主模型为 `bottle_patch`。
2. 点击“开始检测”，等待热图和建议。首次载入主模型可能较慢。
3. 换破损样例再次检测，对比位置、分数与面积；也展示实际漏检/困难案例。
4. 切换金属螺母和瓷砖，分别展示正常及缺陷图，确认模型分别为 `metal_nut_patch`、`tile_patch`。
5. 同一图片可切换该类像素基线对照；比较各自分数与各自阈值，不能只比较分数大小。
6. 展开逐条规则，下载 JSON/CSV，并核对产品、模型、图片标识对应当前输入。

CSV 当前一次一图一行，适合 Excel 对比；JSON 含详细事实、规则轨迹和版本/校验信息。两者不含热图图片，热图用组件下载入口保存。完整批量测试表由 `evaluate` 生成。更完整讲解见 [README_演示.md](README_演示.md)。

合成样例默认隐藏，开启高级选项后使用 `synthetic_pixel`，仅联调。系统不自动命名缺陷，菜单名称来自数据集标签。输入质量不合格可显示“无法判定”。不保证任意正常图通过或缺陷图拒收，分数不是概率，质量规则未经业务验证。上传自有图片应选择对应类别与相近拍摄条件。

## 五、常见问题

| 现象 | 处理方式 |
| --- | --- |
| 双击启动提示先安装 | 先运行“首次安装_Windows.cmd”；检查项目内是否生成 `.venv\Scripts\python.exe` |
| `py` 不是命令或找不到3.11 | 安装 Python 3.11及Launcher，或使用上文已确认版本的 `python -m venv .venv` 替代步骤 |
| 下载依赖超时、连接失败 | 保留报错，检查网络后重试安装；不要随意混换 torch/torchvision 版本 |
| 页面打不开 | 等启动完成，确认窗口未关闭，并使用启动日志中的端口；本项目不需要登录外部服务 |
| 7860已被占用 | 如果自己的项目已在运行，直接用它；否则按下方命令换7861，不结束无关进程 |
| 找不到模型或想下载骨干权重 | 确认完整解压，并保留 `models` 和 `.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`；不要仅转发源码文件夹。若设置过 `TORCH_HOME`，将它指向本项目 `.cache/torch`，或取消该变量后重新启动 |
| 提示缺包，但明明装过 | 使用项目 `.venv` 的解释器或启动脚本；不要混用系统 Python、其他项目环境 |
| DLL载入失败或WinError126 | 保存完整错误交给环境负责人。若报错确认缺少微软VC++运行库，再按 [微软官方文档](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) 安装匹配的x64运行库 |
| 黑图显示“无法判定” | 属于输入质量门控的预期行为；换清晰、曝光正常的图片 |
| 控制台偶见 `upload_progress ... undefined` 的404 | 原电脑验收记录过该进度提示警告；若图片、热图或下载失败，不要仅凭此表格判断成功，应保存错误排查 |

换端口时，在项目文件夹打开终端：

```powershell
.\.venv\Scripts\python.exe -m app.app --host 127.0.0.1 --port 7861
```

随后打开 **http://127.0.0.1:7861**。

## 六、Mac 同学

先确认：

```bash
python3.11 --version
uname -m
```

项目提供以下实验安装/启动入口，默认CPU：

```bash
cd "/自己的解压路径/industrial-inspection"
bash setup_macos.sh
bash start_macos.command
```

浏览器同样打开 **http://127.0.0.1:7860**。

**当前未在真实 Mac 上验收。** 现有 torch 2.6.0 安装路线主要面向 Apple Silicon（`arm64`）；Intel Mac（`x86_64`）不能直接假定该版本有可用安装包，需要环境负责人另外确认兼容组合，发行文件见 [PyTorch 2.6.0](https://pypi.org/project/torch/2.6.0/#files)。不要复制 Windows 的 `.venv`，也不要把 Windows 依赖锁文件用于 Mac。安装出错时提供芯片架构、Python版本和完整报错，由组内统一处理。

## 七、做代码、规划或实验的同学看哪里

| 工作 | 入口 |
| --- | --- |
| 整体执行规划 | `docs/README.md` |
| 已完成项和未完成项 | `docs/STATUS.md` |
| 团队分工与六周排期 | `docs/06_课程映射与团队排期.md` |
| 报告初稿 | `docs/report_draft.md` |
| 数据、检测、规则和评估 | `src/industrial_inspection/` |
| 界面 | `app/app.py` |
| 规则参数 | `configs/quality_rules*.json`，按模型来源自动匹配类别 |
| 三类扩展规划 | `docs/10_三类扩展执行规划.md` |
| 演示操作与报告说明 | `README_演示.md` |

三类当前结果以 `results/three_products/summary.json` 与各类评估目录为准。`results/bottle_pixel/` 和 `results/bottle_patch/` 是2026-10-20首轮历史；原电脑记录不是当前组员电脑新测成绩。安装脚本会生成本机环境记录和合成联调数据；复验用新输出目录，保留正式资产原件。

需要重新训练或完整测试时，从项目根目录激活环境，并分别下载三个类别。下例会更新指定清单和模型；请先保留正式版本：

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/download_bottle.py
python scripts/download_mvtec.py --category metal_nut
python scripts/download_mvtec.py --category tile
foreach ($product in @("bottle", "metal_nut", "tile")) {
    python -m industrial_inspection prepare --data-root data/raw/mvtec_ad --category $product --output "data/manifests/$product.csv"
    python -m industrial_inspection train --manifest "data/manifests/$product.csv" --method pixel --model-dir "models/${product}_pixel"
    python -m industrial_inspection train --manifest "data/manifests/$product.csv" --method patch --model-dir "models/${product}_patch"
    python -m industrial_inspection evaluate --manifest "data/manifests/$product.csv" --model-dir "models/${product}_pixel" --output "results/my_run/${product}_pixel"
    python -m industrial_inspection evaluate --manifest "data/manifests/$product.csv" --model-dir "models/${product}_patch" --output "results/my_run/${product}_patch"
}
python scripts/prepare_samples.py --categories bottle metal_nut tile --per-type 5
```

三个类别分别获取，下载大小以脚本记录为准。数据来自固定 revision 公开镜像，图像/掩码核对镜像LFS哈希和大小，不等于原发布者完整归档校验。原数据采用 [CC BY-NC-SA 4.0](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)，用于非商业课程研究。每类仅使用正常图拟合/校准，不依据独立异常测试结果调整阈值或规则。

## 八、可选：恢复 Git 历史后继续开发

**通过 GitHub 克隆的同学可跳过本节，克隆目录已经有Git历史。** 本节仅适用于另收到历史完整交接包及 `可选_Git历史/industrial-inspection.bundle` 的同学；本轮模型资产包不含Git历史。历史bundle不保证有三类更新，不能用它替代本轮源码。

已安装 Git 的同学，在演示项目 `industrial-inspection` 文件夹打开 PowerShell，恢复到旁边的新目录：

```powershell
git clone -b main "..\可选_Git历史\industrial-inspection.bundle" "..\industrial-inspection-git"
```

Git历史不包含模型、骨干和环境。要在恢复目录中演示，可从当前完整包复制模型与骨干，然后在新目录重新安装环境：

```powershell
Copy-Item -LiteralPath .\models -Destination ..\industrial-inspection-git -Recurse
Copy-Item -LiteralPath .\.cache -Destination ..\industrial-inspection-git -Recurse
```

不要向Git提交 `.venv`、完整数据或大型权重。主要远程仓库已建立，离线bundle仅作为备用快照；继续开发建议从上面的GitHub地址克隆，组内再安排分支和合并。

## 九、每位组员安装后的验收

- [ ] 包已完整解压，模型和 `.cache` 存在。
- [ ] 运行的是本机创建的Python3.11环境。
- [ ] 网页可打开，瓶口、螺母、瓷砖的正常与缺陷样例均能生成热图。
- [ ] 产品切换后模型/样例匹配该类，旧输入与报告不残留；合成样例默认隐藏。
- [ ] 规则记录可展开，JSON/CSV能下载且对应当前图片。
- [ ] 记录自己的系统、芯片/Python版本、运行是否成功；失败时保留完整报错。

本轮本机93项自动测试通过（0失败/错误/跳过，25.244秒），证据为 `results/three_products/tests.xml`。本机真实浏览器17项功能检查通过（`results/three_products/ui/ui_smoke.json`），覆盖三类正常/缺陷、主模型与基线、热图和下载、切换及错误恢复、合成联调和390px窄屏。详情见 `docs/STATUS.md`；首轮2026-10-20的33项测试、9项浏览器检查仅代表旧版，不替代各电脑验收。独立业务标注、扰动/消融、新电脑/Mac验证、最终报告和PPT仍待继续完成。
