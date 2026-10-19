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

首次运行需要获取官方 `openai-community/gpt2-medium`，模型使用 Hugging Face 缓存。新训练/评测默认 `lora/e2e-original`，原始数据与官方评分器按固定提交及 SHA256 下载到 `outputs/resources`。GEM 仅保留为显式历史对照选项。项目模型扫描只扫描本项目内目录，Base 独立使用模型标识。

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
- **任务列表**：统一训练与测试 / benchmark 记录，按创建时间倒序（旧记录兼容目录时间，缺失时回退首次开始 / 开始 / 结束时间，显示来源及香港时间）；使用任务类型、模型类型（base / FFT / LoRA）、状态三组切换按钮组合筛选，支持重置。tests 与 benchmarks 均归入 benchmark；中断和进程异常退出归入失败，详情保留原始状态。
- 每条任务独立折叠，展开后加载配置、结果、下载、重新评分、进度、日志、GPU 曲线和断点续训。默认收起，可同时展开多条，每 3 秒刷新状态。
- 对应训练任务详情提供“打开 TensorBoard”，点击后准备服务，再点击“进入 TensorBoard”跳转。无日志时显示原因，失败训练的已有日志仍可查看。评测配置页不再启动训练 TensorBoard。

## 待训练池与批量训练

在训练页配置一组 LoRA / FFT 参数，填写可选任务名称并点击“加入待训练池”。每项独立保存完整参数快照，继续改表单不会修改此前入池的配置。可混合添加多组 LoRA 和 FFT。

展开某项后点击“开始这一项”，或点击“全部开始”提交当前所有待训练项。后台按添加顺序逐个运行，上一项结束后才启动下一项；某项失败会保留错误和日志并继续下一项。已有训练或测试批次运行时，启动会提示等待，池中参数仍保留。

池保存于 `outputs/training_pool`，刷新页面、重启前端后仍存在；点击“刷新训练池状态”查看状态，运行进度与日志在“任务列表”页查看。已提交项不能重复启动或移除；需重新训练时重新加入一项。尚未提交的项可以移除。“全部开始”后新加入的配置留在待训练池，等待下一次启动。

## 停止与断点续训

在“任务列表”页点击“停止当前批次”，会取消整个批次，后续 seed 不再启动。LoRA / FFT 优先在当前 step 结束后保存完整 checkpoint 再退出；90 秒内未能退出则强制停止进程树，此时只能使用先前完整保存的断点。测试任务直接停止。关闭 Streamlit 或在前端终端按 Ctrl+C 不会停止独立后台任务。

选择已停止或失败的训练，页面会列出此运行的完整 checkpoint，默认选最新一个，点击“断点续训”即可恢复。没有 checkpoint 的未开始任务需要重新加入待训练池。续训只恢复所选任务，不会自动继续此前取消的其他 seed。

恢复包含模型 / adapter 权重、优化器、学习率调度器、随机状态和 Trainer 步数。使用原训练参数与原输出目录，epochs 表示原计划总轮数；旧日志继续保留，历史状态记录在 attempts.json。续训不会从零开始，但未保存的 step 需要重跑。当前已停止的真实 FFT seed 42 可从 checkpoint-8382（epoch 2）继续；此功能添加后未自动启动正式续训。

“任务列表”中的两块日志分别标为训练内部日志（run.log）和后台进程日志（process.log）；后者包含前者的重复输出，并不表示重新训练。

## 默认参数与显存配置

LoRA 论文参数见 [参数来源.md](参数来源.md)。默认 Q=4、K=0、V=4，alpha=32；适配器分支按 alpha/r 缩放。Q、K、V 可以分别使用不同 rank 或关闭，例如 Q=8、K=0、V=4；同一投影在所有层使用同一个 rank，各层分别学习自己的 adapter 权重。当前 Hugging Face GPT-2 Small 和 Medium 都将 QKV 存在合并的 c_attn 中；项目在该合并层上分别添加 Q/K/V 分支，基础模型权重被冻结。

论文有效 batch=8，默认 micro batch=8 / accumulation=1；验证 batch=4、seed=110、fp32、最大长度 512、关闭梯度检查点、日志间隔 100。LoRA dropout=0.1 仅用于 adapter 分支，原生 GPT-2 dropout 关闭。默认值来自论文表 11 和官方 E2E 示例命令。显存不足时可显式改 micro batch=1、accumulation=8、eval batch=1；精度和最大长度为单独可调的硬件设置，实际配置会保存。不要把冒烟的 warmup=0、max_steps=1 当作完整训练设置。

本机 RTX 4070 Laptop 8GB 的真实模型验证使用 BF16；首次 FP16 的非有限梯度记录已保留。训练中若 loss 或 grad_norm 非有限则标记失败；保存最终模型前还会检查是否发生权重更新。

FFT 默认参考 Prefix-Tuning 的 E2E FT：学习率 5e-5、5 epochs、单卡 batch=10、梯度累积=1、验证 batch=10、seed=101、warmup=0、weight decay=0、label smoothing=0、原生 dropout=0.1、fp32、最大长度 1024、关闭梯度检查点。论文报告值与作者代码补充值的区别见参数来源.md；warmup=0 已由作者仓库 training_args.py 与启动路径确认。8GB 适配保留有效 batch=10，使用 bf16 和梯度检查点。

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

E2E 测试与 benchmark 默认最多生成 64 个新 token，对齐官方 LoRA E2E 示例的 `--eval_len 64`。前端与 CLI 共用此默认值，可通过“最大新 token 数”或配置字段 `max_new_tokens` 调整；已有配置和历史结果保留原值。

默认 `e2e_official`：直接运行作者链接的 E2E 评分器，Perl MTEval 计算 BLEU/NIST，PTB 分词后由 Java METEOR 1.5 / COCO 计算 METEOR、ROUGE-L、CIDEr，使用全部参考。BLEU 以 `BLEU-4` 保存（0–1），不再伪造 BLEU-1…3；BLEU、METEOR、ROUGE-L 展示成论文百分尺度时乘100，NIST/CIDEr 不乘。BLEU/NIST 是语料级分数，不提供伪造的逐句值；METEOR 总分也不拿句均值代替。CIDEr 的文档频率在整次评测内计算。显式 `legacy_single` 保留旧的8项单参考评分，仅供历史对照。

当前采用 small 的按数据行、单 target 评分口径，并另存原始完整 references；它不是论文官方多参考评测脚本。分数不能直接当作论文表 3 的同口径成绩。

仅 `legacy_single` 的 NLTK METEOR 会自动准备 WordNet；默认官方 Java METEOR 不使用这套实现。评分模块优先复用项目内资源；缺失时通过系统代理从 NLTK 官方固定 HTTPS 地址直接下载到 `outputs/resources/nltk_data/corpora/wordnet.zip`，保留默认 TLS 校验，不调用 `nltk.download()`，不修改 NLTK 安全开关。

下载先写临时文件，通过 ZIP CRC 与必需文件检查后原子替换目标文件，并实际加载语料、查询验证；失败会清理临时文件并保存评分错误。已有 ZIP 也会检查，损坏时需移走后重试。也支持手动放入官方 ZIP 或解压后的 `corpora/wordnet` 目录。资源位于被 Git 忽略的 outputs 中，迁移代码不会自动带上，运行时不读取 small 项目。WordNet 准备仍在 METEOR 评分阶段执行。

## 逐步冒烟

按下面顺序运行，每步通过再执行下一步；详细结果见 [实施记录.md](实施记录.md)。

```powershell
python -s src/smoke.py runtime
python -s src/smoke.py models
python -s src/smoke.py lora
python -s src/smoke.py fft
python -s src/smoke.py evaluation
python -s src/smoke.py frontend
python -s src/smoke.py task-list
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

模型列表及任务列表默认排除 outputs/smoke 与配置标记 smoke=true 的内容。冒烟文件仍完整保留，内部冒烟测试可显式访问。每次用户提出的修改须先整理并补入需求文档.md，同时更新使用说明与实施记录，再提交推送。

续训耗时记录兼容 Python 3.10 与 PowerShell 的不同小数位数 ISO 时间戳，包括五位与七位；统一补齐或截断至六位微秒精度。无法解析或倒序的历史时间会跳过耗时统计并保留警告，不影响断点恢复。修复升级后，刷新前端并重新点击“断点续训”即可；此前失败的启动没有删除 checkpoint。
# Benchmark 对比图与数据

8 项指标的五组横评图位于 `outputs/pics`，包含 PNG、SVG、汇总 CSV、数据来源及绘图脚本；运行 `python -s outputs/pics/plot_comparison.py` 可重新生成。Git 纳入整个 `outputs/pics` 与 `outputs/benchmarks` 下所有 CSV，其余实验输出仍忽略。


## 与原文的数据一致性分析

报告：[GPT-2 Medium 数据一致性分析](analysis/GPT2_Medium_数据一致性分析报告.md)。包含 seed 42/43/44 三次均值、全部五次记录均值、512/1024 同 seed 对照、原文表 3 对比、固定预测的参考集合/评分实现消融，以及数据来源哈希。GEM 清洗数据和单参考评分与原文不同，当前分数不能直接当作原文复现结果；原始 benchmark 保持不变。

运行 `python analysis/seed_variability.py` 可生成八项指标的五组纵向柱状图（Base、LoRA 3、LoRA 5、FT 3、FT 5），柱高为均值，误差线为 ±1 样本标准差，散点对应各次完整测试运行。图位于 `analysis/figures/variability_*.png`（另有 SVG 与总览）；`analysis/variability_summary.csv` 保留原始尺度的均值、SD、CV，`analysis/variability_runs.csv` 可追溯每个散点。Base 单次 SD 不可估计；五记录组仅四个不同 seed，包含同 seed 长度对照。运行 `python analysis/build_report.py` 会同步更新完整报告。Small 旧六次 FT 与当前 Medium 的测试分割及生成设置不同，CV 仅作描述对照。


## 第 8 节设置修正（2026-10-17）

新训练默认分别采用 `lora_e2e` / `prefix_ft` 输入协议，显式保存 Adam betas、epsilon、梯度裁剪和作者版 AdamW。LoRA epsilon=1e-6、裁剪关闭；FFT epsilon=1e-8、裁剪=1.0。原有学习率、epoch、batch、warmup、Q/V rank 和 dropout 默认值保持各自论文/作者来源。完整对应表及实现边界见 [参数来源.md](参数来源.md#2026-10-17按分析报告第-8-节落实训练与解码设置)。

训练页可设置协议、优化器、Adam 参数和梯度裁剪；8GB 档仍只是显存适配。测试页“生成协议”默认 auto：新 LoRA 使用论文表值（beam=10、length penalty=0.9、上限64、EOS628）；新 FFT 使用 Prefix-Tuning FT 档（beam=5、length penalty=1.0、上限100）。勾选“覆盖协议的生成参数”后才覆盖数值；CLI 用 null 表示跟随协议默认值。实际参数写入 `generation_config.json`。

新模型和 checkpoint 旁保存 `training_protocol.json`，旧模型没有标记时自动沿用 MR/Text 格式。历史排队任务和断点续训缺少新字段时使用旧默认值；新建训练配置使用新默认值。已有权重不会因修改代码而变成新协议训练的权重。

以上为 2026-10-17 的历史实施范围：当时默认数据仍是 GEM、评分为单参考。2026-10-18 更新已替换新任务默认数据与评分，见下节；历史报告和记录保持原样。

```powershell
python -s src/verify_protocols.py
# CLI 可在 JSON 中设置 protocol、optimizer、adam_epsilon、max_grad_norm 等
python -s src/train_lora.py --config <项目内配置.json>
python -s src/train_fft.py --config <项目内配置.json>
```


## 2026-10-18：原始 E2E 数据与官方评测

家里拉取后先准备依赖（使用现有 `lora` 环境）：

```powershell
conda run -n lora python -m pip install -r requirements.txt
# Java 已在 PATH / JAVA_HOME 中时不用 --install-java；该选项仅下载 Windows x64 便携 JRE 到项目内。
conda run -n lora python src/prepare_paper.py --install-java
```

官方评分还需要 Perl（Windows 可安装 Strawberry Perl，并让 `perl -v` 可执行）。`prepare_paper.py` 验证数据校验和、评分器 Python 导入、Java 与 Perl；缺依赖明确失败，不回退为 NLTK 近似评分。数据/评分器来自固定提交，下载保留 TLS 校验；`local_files_only=true` 要求资源已准备好。便携运行时、下载和冒烟产物不提交 Git。

数据下载到 `outputs/resources/paper_e2e/{train,validation,test}.txt` 后持续缓存，训练/测试校验通过即复用，缺失或损坏才重新下载。清单 SHA-256 必须按远端原始字节（LF 换行）计算，不能使用 Windows Git 检出后 CRLF 文件的哈希；下载校验错误会显示预期/实际哈希及字节数。

若旧任务日志出现 `PermissionError: [WinError 5]`，且路径是 `progress.json.tmp -> progress.json`，表示更新本地进度文件时替换失败。当前实现使用每次唯一的临时文件，并对 PermissionError 最多尝试 8 次（总等待 1.4 秒），保留原子替换；持续占用或真实权限问题仍明确报错，保留旧 JSON。更新代码后需重启界面，重新启动失败任务，已运行的 Python 进程不会自动载入修复；历史失败日志也不会被改写。可运行 `python src/verify_file_io.py` 独立验证短暂占用、持续占用、并发写入和 Windows 实际文件句柄占用，无需模型或联网。

- **数据**：训练 42,061 行、验证损失 4,672 行；生成评测按原始 context 分组，validation=547 组、test=630 组，test 共4,693条参考。完全保留原始 context/目标空格，不再用 GEM 字段转换代替原始输入。没有另做随机划分。
- **LoRA 训练**：`source_training=true` 使用作者目标 token 损失与分母 `mask.sum()+1e-4`；训练平滑0.1、验证不平滑；固定长度补0、尾批 Random(911) 补样、作者分布式 sampler 的单卡顺序与训练 epoch 从1计数；Q/V 共享同一次 adapter dropout。按论文 D.3 每 epoch 选最佳；保留原先已对齐的优化器及超参数。
- **生成**：`lora_paper/lora_readme` 使用移植的作者 beam 候选逻辑，只在生成历史内禁重复 ngram；EOS=[50256,628]，长度惩罚分母只计生成长度。论文档0.9、README档0.8。FFT 保持 Prefix FT 独立协议，不能套 LoRA beam。
- **评测**：默认 test、全部样本、`e2e_official`。会保存 `paper_sources.json`、`scoring_source.json`、`generation_config.json`、`coverage.json`、官方原始分数与日志以及可供官方 CLI 直接评分的 references/hypotheses 文本。
- **完整性**：生成及重评分都核对目标 split 的数量、顺序、输入、target 和完整 references；613条不能冒充完整测试。`completed` 表示所配置样本范围完成，`coverage.full_split` 说明是不是全量。进度文件使用独立临时文件，并对 Windows 短暂文件锁重试；持续拒绝访问仍保留失败状态。

**正式对照需要重新训练。** 旧 GEM 权重不会因为换测试集变成原始数据训练模型。新训练保持默认原始数据，完整跑5轮，再选 final 测试；不要恢复旧 GEM checkpoint。旧记录与旧分析报告不覆盖，重评分也不补生成。Base 是未微调对照，不是论文 FT 基线。

论文报告3个训练种子的均值，但没有公开完整种子组合。训练页可填 `110,111,112`（项目约定，不声称是作者组合），串行启动/入池；留空只跑下方单个 Seed。完成3个模型的完整官方 test 后运行：

```powershell
conda run -n lora python src/summarize_paper.py outputs/benchmarks/RUN1 outputs/benchmarks/RUN2 outputs/benchmarks/RUN3
```

汇总器要求3个不同训练种子、原始数据、完整630组、相同训练/生成/评分配置，输出均值及样本标准差。不会混入部分评测、GEM、不同长度/参数或旧评分记录。

验证命令：`python src/verify_paper.py`、`python src/verify_protocols.py`、`python src/smoke.py medium-lora`。`verify_paper.py` 会取得固定校验的作者参考源码；测试 BPE、分组、损失、作者 beam 原循环一致性和文件锁重试。真实 Medium 冒烟使用少量样本/1步，只证明链路可运行，不代表论文性能复现。HF 模型内核、现代 PyTorch/CUDA 与作者旧软件栈存在浮点差异；不承诺逐位相同权重，也不凭冒烟宣称达到70.4 BLEU。

终止符核对补充：官方 `gpt2_beam.py` 的 argparse 使用 `action="append", default=[50256]`，所以 README 传入628后实际为 `[50256,628]`；新LoRA生成档已修正，前文只写628的历史描述不再代表当前设置。
