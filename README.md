# GPT-2 Medium E2E LoRA

GPT-2 Medium 的 Base、Q/K/V 独立 LoRA、全参数微调（FFT）训练与 E2E 评测工作台。参考平行目录的 small 项目，前端与后端分离，所有代码位于 src。

## 环境与启动

在 PowerShell 中运行根目录启动脚本（也可双击 run.bat），它会定位项目并使用已有 conda `lora` 环境：

```powershell
.\run.bat
# 可选：指定端口
.\run.bat --server.port 8502
```

脚本不自动安装依赖；首次使用请按下方命令准备环境。无需提前激活 lora。

统一使用名称全小写的 Conda **lora** 环境。禁用用户 site-packages，保证依赖来自该环境：

```powershell
conda activate lora
python -s -m pip install --no-user -r requirements.txt
python -s -m streamlit run src/app.py
```

本机也可直接使用已验证的 Python 路径：

```powershell
& 'C:\ProgramData\anaconda3\envs\lora\python.exe' -s -m streamlit run src/app.py
```

如果 ProgramData 安装权限不足，在管理员终端执行安装，仍安装到 lora，不能改装到用户目录或 base。CUDA 版本的 torch 已单独安装；requirements 中 torch 版本不指定 CUDA wheel 索引。

首次运行需要获取官方 `openai-community/gpt2-medium` 和 `GEM/e2e_nlg`。模型及数据使用 Hugging Face 缓存；项目模型扫描只扫描本项目内目录，Base 独立使用模型标识。

## 独立启动 TensorBoard

```powershell
.\run_tensorboard.bat
# 可选：指定端口
.\run_tensorboard.bat --port 6007
```

默认打开地址为 http://127.0.0.1:6006；脚本仅汇总 medium 的正式 LoRA / FFT 训练日志，不包括冒烟。使用 conda lora，不安装依赖。

## 前端

- **训练**：LoRA / FFT，超参数、精度和少量样本设置；只有 Q rank、K rank、V rank 三个模型级设置，各自独立调整，统一应用到所有 Transformer 层，无需逐层配置。rank=0 没有 adapter 分支；全部冻结时可测试 Base，但不能启动无参数训练。
- **测试 / benchmark**：仅扫描 medium 内有效模型和 checkpoint；支持多个模型与一个、多个或全部指标，任务按顺序执行，避免抢占 GPU。
- **历史测试记录**：从磁盘恢复记录，按类型、状态、日期 / 模型文本、指标筛选；查看和下载全部明细，使用已有预测重新评分。
- **任务进度**：后台状态、进度、日志、显存 / 温度 / GPU 占用率、耗时和预计完成时间。刷新或关闭前端不结束后台任务。
- 选择有训练记录的模型后，自动准备该运行的本地 TensorBoard 服务，“打开训练 TensorBoard”按钮直接跳转；保存的训练数据摘要也可以在页面中展开查看。

## 待训练池与批量训练

在训练页配置一组 LoRA / FFT 参数，填写可选任务名称并点击“加入待训练池”。每项独立保存完整参数快照，继续改表单不会修改此前入池的配置。可混合添加多组 LoRA 和 FFT。

展开某项后点击“开始这一项”，或点击“全部开始”提交当前所有待训练项。后台按添加顺序逐个运行，上一项结束后才启动下一项；某项失败会保留错误和日志并继续下一项。已有训练或测试批次运行时，启动会提示等待，池中参数仍保留。

池保存于 `outputs/training_pool`，刷新页面、重启前端后仍存在；点击“刷新训练池状态”查看状态，运行进度与日志在“任务进度”页查看。已提交项不能重复启动或移除；需重新训练时重新加入一项。尚未提交的项可以移除。“全部开始”后新加入的配置留在待训练池，等待下一次启动。

## 停止与断点续训

在“任务进度”页点击“停止当前批次”，会取消整个批次，后续 seed 不再启动。LoRA / FFT 优先在当前 step 结束后保存完整 checkpoint 再退出；90 秒内未能退出则强制停止进程树，此时只能使用先前完整保存的断点。测试任务直接停止。关闭 Streamlit 或在前端终端按 Ctrl+C 不会停止独立后台任务。

选择已停止或失败的训练，页面会列出此运行的完整 checkpoint，默认选最新一个，点击“断点续训”即可恢复。没有 checkpoint 的未开始任务需要重新加入待训练池。续训只恢复所选任务，不会自动继续此前取消的其他 seed。

恢复包含模型 / adapter 权重、优化器、学习率调度器、随机状态和 Trainer 步数。使用原训练参数与原输出目录，epochs 表示原计划总轮数；旧日志继续保留，历史状态记录在 attempts.json。续训不会从零开始，但未保存的 step 需要重跑。当前已停止的真实 FFT seed 42 可从 checkpoint-8382（epoch 2）继续；此功能添加后未自动启动正式续训。

“任务进度”中的两块日志分别标为训练内部日志（run.log）和后台进程日志（process.log）；后者包含前者的重复输出，并不表示重新训练。

## 默认参数与显存配置

LoRA 论文参数见 [参数来源.md](参数来源.md)。默认 Q=4、K=0、V=4，alpha=32；适配器分支按 alpha/r 缩放。Q、K、V 可以分别使用不同 rank 或关闭，例如 Q=8、K=0、V=4；同一投影在所有层使用同一个 rank，各层分别学习自己的 adapter 权重。当前 Hugging Face GPT-2 Small 和 Medium 都将 QKV 存在合并的 c_attn 中；项目在该合并层上分别添加 Q/K/V 分支，基础模型权重被冻结。

论文有效 batch=8，默认 micro batch=8 / accumulation=1；验证 batch=4、seed=110、fp32、最大长度 512、关闭梯度检查点、日志间隔 100。LoRA dropout=0.1 仅用于 adapter 分支，原生 GPT-2 dropout 关闭。默认值来自论文表 11 和官方 E2E 示例命令。显存不足时可显式改 micro batch=1、accumulation=8、eval batch=1；精度和最大长度为单独可调的硬件设置，实际配置会保存。不要把冒烟的 warmup=0、max_steps=1 当作完整训练设置。

本机 RTX 4070 Laptop 8GB 的真实模型验证使用 BF16；首次 FP16 的非有限梯度记录已保留。训练中若 loss 或 grad_norm 非有限则标记失败；保存最终模型前还会检查是否发生权重更新。

FFT 默认参考 Prefix-Tuning 的 E2E FT：学习率 5e-5、5 epochs、单卡 batch=10、梯度累积=1、验证 batch=10、seed=101、warmup=0、weight decay=0、label smoothing=0、原生 dropout=0.1、fp32、最大长度 1024、关闭梯度检查点。论文报告值与作者代码补充值的区别见参数来源.md；warmup=0 为启动路径与框架默认值推断。8GB 适配保留有效 batch=10，使用 bf16 和梯度检查点。

2026-10-14 已用真实 GPT-2 Medium 验证新 FFT 默认：仅覆盖训练样本 10、验证样本 2、max_steps=1、logging_steps=1、离线缓存与 smoke 标记；训练 loss=3.291145、验证 loss=3.710723，所有参数可训练且权重确实更新，checkpoint / 最终模型保存成功。运行位于 `outputs/smoke/gpt2-e2e-full-ft/20261014_130742_374057_lr5e-05_e5_b10_ga1_s101_d881fd5f`。这是流程验证，不是完整 E2E 指标复现。前端新 FFT 默认值及 8GB 有效 batch=10 的 AppTest 检查通过。

## 后端命令

各脚本顶部的 DEFAULTS 为默认入口，公共值位于 src/config.py。通过项目内部 JSON 文件覆盖参数：

```powershell
python -s src/train_lora.py --config outputs/my_lora_config.json
python -s src/train_fft.py --config outputs/my_fft_config.json
python -s src/test_e2e.py --config outputs/my_test_config.json
python -s src/run_benchmarks.py --config outputs/my_benchmark_config.json
```

无 --config 时使用默认参数。CLI 每次新建运行目录；--run-dir 仅供已排队任务使用，禁止覆盖完成记录。

LoRA 覆盖配置示例（ranks 只有 Q、K、V 三项，不使用逐层数组）：

```json
{
  "ranks": {"q": 8, "k": 0, "v": 4},
  "batch_size": 1,
  "gradient_accumulation": 8,
  "eval_batch_size": 1,
  "precision": "bf16"
}
```

评测配置示例：

```json
{
  "model": {"type": "base", "path": "openai-community/gpt2-medium"},
  "split": "test",
  "samples": 100,
  "metrics": ["BLEU", "CIDEr"]
}
```

重新评分配置可加 `"predictions": "outputs/benchmarks/<运行>/predictions.jsonl"`。新运行会保存预测副本、源配置路径和全部新评分结果，不加载模型或重新生成。

## 输出与持久化

```text
outputs/
  gpt2-e2e-lora/<时间戳_参数摘要>/
  gpt2-e2e-full-ft/<时间戳_参数摘要>/
  tests/<时间戳_模型与生成参数>/
  benchmarks/<时间戳_模型与生成参数>/
  jobs/       # 后台批次与进程日志
  resources/  # WordNet 等评测资源
  smoke/      # 微型模型和冒烟 fixture
```

训练运行包含 config.json、environment.json、data_summary.json、parameters.json、status.json、run.log、training.jsonl、train_log.json、gpu.jsonl、tensorboard、checkpoints、final 和 result.json。

评测运行包含配置、状态、预测 JSONL、每项指标的 scores.json 或 error.json、sample_scores.jsonl、metrics.json / CSV、gpu.jsonl 及耗时。每条预测生成后立刻 flush，评分失败仍保留预测与已经成功的分数。全部参考答案与评分采用的 target 分开保存。

GPU 数据为 **整张设备** 的占用率（%）、核心温度（°C）、显存已用 / 总量（MiB）、功耗（W）、核心 / SM / 显存频率（MHz）、核心电压（mV）、功耗上限、性能状态及限速标记，不是当前进程专属用量。当前驱动无法返回电压与功耗上限时记录 null；电压查询失败单独记录原因，其他采样仍保留。缺失指标记录原因。采样器独立放在 src/utils/gpu_monitor.py，由每个任务管理生命周期。

耗时估算使用同配置、同 GPU 的成功记录均值，排除标记为冒烟的记录；无可比数据时显示暂无估算，运行进度产生后可按 steps / 样本更新。

## 评分口径

延续 small：pycocoevalcap BLEU-1…4、ROUGE-L、CIDEr，NLTK NIST-5、METEOR。选择 BLEU 会保留四种 n-gram 分数。NIST 是语料级指标；逐样本栏明确为 null。CIDEr 的逐样本分数依赖本次语料文档频率，不能拿一个样本单独评分后拼接替代整体评分。

当前采用 small 的按数据行、单 target 评分口径，并另存原始完整 references；它不是论文官方多参考评测脚本。分数不能直接当作论文表 3 的同口径成绩。

METEOR 需要 WordNet，可将已有 wordnet.zip 放到 `outputs/resources/nltk_data/corpora/wordnet.zip`。正常网络下也会尝试 NLTK 官方下载；代理安全检查失败时保留错误，不关闭安全检查。当前本机已将 small 的资源复制到 medium 内，运行时不读取 small。

## 逐步冒烟

按下面顺序运行，每步通过再执行下一步；详细结果见 [实施记录.md](实施记录.md)。

```powershell
python -s src/smoke.py runtime
python -s src/smoke.py models
python -s src/smoke.py lora
python -s src/smoke.py fft
python -s src/smoke.py evaluation
python -s src/smoke.py frontend
python -s src/smoke.py training-pool
python -s src/smoke.py resume
python -s src/smoke.py stop
python -s src/smoke.py formal-records
python -s src/smoke.py timestamps
python -s src/smoke.py medium-base
python -s src/smoke.py medium-lora
python -s src/smoke.py medium-fft
python -s src/smoke.py guards
```

微型模型用于快速验证算法与集成；medium-* 使用真正的 24 层 GPT-2 Medium，训练只跑 1 step，评测只取 2 个样本。完整训练、完整测试集和多个随机种子的论文实验尚需单独启动。

本项目使用自定义 QKV adapter 格式 `medium-qkv-lora-v1`，通过本项目加载器重载。它不是可直接传给 PEFT 的 adapter；检查点的原模型基础权重来自其 adapter_config.json 中记录的模型。

新训练只接受模型级 QKV rank 对象；旧版逐层数组格式仅保留加载兼容，已有实验记录和模型不删除、不覆盖。

## 版本管理

独立仓库：[piiiiiix/gpt2-medium-e2e-lora](https://github.com/piiiiiix/gpt2-medium-e2e-lora)。每个已验证 pipeline 单独提交并推送。新的冒烟运行直接写入 outputs/smoke 下对应分类。notebooks 与 outputs 被 .gitignore 忽略；本地模型和实验日志仍完整保存，不随代码上传。

## 代码阅读与格式

所有 Python 文件、类和具名函数均有中文职责说明；关键配置和流程另有必要的中文注释。学习率加减按钮每次调整 `0.000001`（`1e-6`），可直接输入更小数值，最低为 `1e-9`。表单中的修改在点击启动训练后提交。

使用 `pyproject.toml` 统一 Ruff 格式，行宽上限 100，长表达式和字符串按语义换行。开发工具同样装在 conda `lora` 中：

```powershell
python -s -m pip install --no-user ruff==0.17.0
python -s -m ruff format src
python -s -m ruff check src
```

VSCode 可安装 Ruff 扩展并选择其作为 Python 格式化器；检查规则从项目配置读取。

## 正式记录与需求维护

模型列表、历史测试记录及任务进度默认排除 outputs/smoke 与配置标记 smoke=true 的内容。冒烟文件仍完整保留，内部冒烟测试可显式访问。每次用户提出的修改须先整理并补入需求文档.md，同时更新使用说明与实施记录，再提交推送。

续训耗时记录兼容 Python 3.10 与 PowerShell 的不同小数位数 ISO 时间戳，包括五位与七位；统一补齐或截断至六位微秒精度。无法解析或倒序的历史时间会跳过耗时统计并保留警告，不影响断点恢复。修复升级后，刷新前端并重新点击“断点续训”即可；此前失败的启动没有删除 checkpoint。
