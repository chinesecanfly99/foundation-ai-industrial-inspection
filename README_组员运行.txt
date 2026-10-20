# 工业异常检测项目｜组员运行说明

项目：Foundation of AI 小组作业——工业异常定位与可解释质量决策  
交接日期：2026-10-20  
原型版本：v0.1.0；本交接包补充组员运行说明与安装入口。

## 先看这几句

**完整转发包：先完整解压，再安装依赖，然后启动网页，模型已在包内。GitHub 克隆：获取源码后另下模型附件。两种方式演示都不需要重新训练或下载完整数据。**

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

随后在浏览器打开 **https://github.com/chinesecanfly99/foundation-ai-industrial-inspection/releases/tag/v0.1.0**，下载 `demo-assets-v0.1.0.zip`，将其中的 `models`、`.cache` 和 `samples` 解压到刚克隆的 `industrial-inspection` 根目录。确认不是多套一层同名文件夹，再按下文安装和启动。

不装 Git 也可在仓库页面选择 Code → Download ZIP，再另外解压模型附件。仓库下载的源码ZIP不自带模型；它与前面单独转发的完整交接包不同。

今后更新代码，在项目目录执行：

```powershell
git pull --ff-only
```

有自己的未提交修改时先保存并提交，不要强行覆盖。新增模型版本另下对应附件。以下“一”描述完整交接包的布局，GitHub获取方式可直接跳到“二、Windows首次安装”。

## 一、收到的文件怎么用

完整解压 ZIP 到自己的目录，例如 `D:\AIGroup\`。解压后应看到：

```text
README_组员运行.md
README_组员运行.txt
industrial-inspection/              ← 实际项目目录
    首次安装_Windows.cmd
    setup_windows.ps1
    start_windows.cmd
    app/
    src/
    configs/
    requirements/
    models/                        ← 3个已拟合模型
    .cache/torch/hub/checkpoints/   ← 官方ResNet18骨干缓存
    samples/                       ← 可直接演示的图片
    docs/                          ← 执行规划、分工、报告初稿
    results/                       ← 原电脑的实验与验收记录
可选_Git历史/
    industrial-inspection.bundle   ← 继续开发时恢复Git历史
```

不要直接在微信预览或压缩软件内部双击运行。保留 `models`、`samples` 和以点开头的 `.cache` 文件夹；Mac 的 Finder 默认可能隐藏 `.cache`。

本包不带原电脑的 `.venv`，安装脚本会为每台电脑新建环境。完整的292张原始工业图片也不在包里，只有重新评估或训练时才需下载。

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

1. 在“演示样例”选择 `MVTec bottle / good_000`，模型应为 `bottle_patch`。
2. 点击“开始检测”，等待热图和质量建议。第一次载入 CNN 可能比后续慢。
3. 换 `broken_large_000` 再检测，对比异常位置、分数与面积。
4. 展开“逐条规则与完整记录”，查看采用了哪条规则。
5. 点击“下载 JSON 报告”或“下载 CSV 摘要”，检查下载内容是否对应刚才的图片。
6. 可上传自己的 PNG/JPEG；模型目前只对相近拍摄条件的瓶类图片有研究依据。

合成样例会自动选 `synthetic_pixel`，仅用于联调。固定真实样例中，正常图应显示“通过”，大破损图应显示“拒收建议”；这只是演示检查，不是对任意图片的保证。

演示前先用包内样例确认流程，再尝试其他图片。检测分数不是概率；质量规则仍是教学策略。

## 五、常见问题

| 现象 | 处理方式 |
| --- | --- |
| 双击启动提示先安装 | 先运行“首次安装_Windows.cmd”；检查项目内是否生成 `.venv\Scripts\python.exe` |
| `py` 不是命令或找不到3.11 | 安装 Python 3.11及Launcher，或使用上文已确认版本的 `python -m venv .venv` 替代步骤 |
| 下载依赖超时、连接失败 | 保留报错，检查网络后重试安装；不要随意混换 torch/torchvision 版本 |
| 页面打不开 | 等启动完成，确认窗口未关闭，并使用启动日志中的端口；本项目不需要登录外部服务 |
| 7860已被占用 | 如果自己的项目已在运行，直接用它；否则按下方命令换7861，不结束无关进程 |
| 找不到模型或想下载骨干权重 | 确认完整解压，并保留 `models` 和 `.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`；不要仅转发源码文件夹 |
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
| 规则参数 | `configs/quality_rules.json` |

查看真实测试结果用 `results/bottle_pixel/` 与 `results/bottle_patch/`。这些是原电脑历史结果，不是当前组员电脑新测出的成绩。安装脚本会生成本机环境记录和合成联调数据；正式复验请使用新的输出目录，保留原始ZIP。

只有需要重跑完整83张测试时，才在项目目录下载数据并建立清单：

```powershell
.\.venv\Scripts\python.exe scripts/download_bottle.py
.\.venv\Scripts\python.exe -m industrial_inspection prepare --data-root data/raw/mvtec_ad --category bottle --output data/manifests/bottle.csv
.\.venv\Scripts\python.exe -m industrial_inspection evaluate --manifest data/manifests/bottle.csv --model-dir models/bottle_patch --output results/my_run/bottle_patch
```

下载约157 MB，只获取 MVTec AD bottle。数据来自公开镜像，已记录原始发布方、版本和镜像哈希；原数据采用 [CC BY-NC-SA 4.0](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)，用于非商业课程研究。模型阈值不应根据独立测试结果反复调整。

## 八、可选：恢复 Git 历史后继续开发

**通过 GitHub 克隆的同学可跳过本节，克隆目录已经有Git历史。** 本节仅供使用完整转发包但暂时无法连接GitHub的同学：包中源码目录没有 `.git`，离线历史另放在 `可选_Git历史/industrial-inspection.bundle`。

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
- [ ] 网页可打开，正常和大破损样例能生成热图。
- [ ] 规则记录可展开，JSON/CSV能下载且对应当前图片。
- [ ] 记录自己的系统、芯片/Python版本、运行是否成功；失败时保留完整报错。

本包已在原电脑完成33项自动测试与9项浏览器功能检查；这不替代每台组员电脑的上述验收。当前仍为课程项目初稿，独立业务标注、扰动/消融、新电脑与Mac验证、最终报告和PPT仍待小组继续完成。
