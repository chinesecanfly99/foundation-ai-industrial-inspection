# 在 PyCharm 中运行

1. 在 PyCharm 选择 **Open / 打开**，选择项目文件夹 `F:\Codex_file\Foudationai.groupwork`。
2. 项目解释器使用 **Python 3.11 (Foudationai.groupwork)**，路径为 `F:\Codex_file\Foudationai.groupwork\.venv\Scripts\python.exe`。如果首次打开提示缺少解释器，在 **Settings → Project → Python Interpreter** 中添加这个现有解释器。
3. 右上角选择 **Industrial Inspection Local**，点击绿色运行按钮。出现本地地址后，打开 <http://127.0.0.1:7861/>，即可选择样例或上传图片检测。
4. 结束时点击 PyCharm 运行窗口中的红色 **停止** 按钮。下次继续使用同一运行配置。

## 三类版本怎么演示

代码更新后，先停止当前服务，再用同一配置重新启动。页面先选**瓶口、金属螺母、瓷砖表面**，再选中文样例；默认使用该类特征记忆主模型，也可切换像素基线对照。合成垫圈默认隐藏，只用于高级联调。完整步骤、样例含义和 CSV/JSON 用途见 [README_演示.md](README_演示.md)。

每类需要自己的模型与阈值，本轮共6个真实模型和1个合成模型。本机文件与三类资产通过 `deliverables/demo-assets-v0.2.0.zip` 交接；GitHub v0.1.0 旧附件不含螺母、瓷砖，也不能由本机更新推断远程已同步。不要只复制源码后假定模型都在。

另一个运行配置 **Env Check** 会检查当前环境，不会启动网页服务。

先前启动的演示使用端口 **7860**；PyCharm 配置使用 **7861**，便于区分两个实例。在 PyCharm 停止服务不会停止原来的 7860 演示。

F 盘原有的 `F:\PyTorch\venv` 是 GPU 环境，目前缺少本项目的网页依赖。此次复用已经验证能运行项目的 `.venv`，保留原有 GPU 环境。不要把项目的 CPU 依赖锁定文件直接安装到原有 GPU 环境。

项目代码位于 `app` 和 `src`，其中 `src` 已设为源码根目录。虚拟环境、模型和大体积数据目录已排除索引，仍然保留在磁盘上并可正常读取。
