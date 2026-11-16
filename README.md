# SCAR

**State · Control · Action · Resource**：面向 Python 程序的宏观编译器。

```text
静态源码 + 必要动态绑定 + 硬件库存
→ 有类型、可展开的程序/资源图
→ 结构化简、成本区间及资源计划
→ C++20 / 算子 ABI1
→ 原入口、guard 和完整区域回退
```

当前优先完成三阶段核心开发，暂停新增性能实验。受支持的 CPU float64 二维数组、复合循环和事务区域已贯通；通用覆盖仍未完成，完整入口及兼容/I/O 仍保留 Python。
LeWM、Sunfish、Gray–Scott、Cirq 都是外部 testcase，分析规则不按业务名称选择优化。

## 使用现有环境

```bash
conda activate lewm
python -m scar --help
```

使用已有环境，不升级 PyTorch/CUDA。安装入口需要时可用 `python -m pip install --no-deps -e .`。以下命令的输出目录应为新目录。

## 三阶段主链

先用已有形式前端逐层处理源码，再建立统一图。动态目标不确定时，可按原 CALL 位置作有限观察；不逐次展开循环或保存 Tensor 数值。

```bash
scar frontend /path/to/program --jobs 8 --out FRONTEND
scar observe-call-bindings --all-sources --catalog FRONTEND/catalog --out CALLS -- python ENTRY.py [arguments]
scar build --entry ENTRY.py --catalog FRONTEND/catalog --call-bindings CALLS --out PROGRAM
scar plan-build --build PROGRAM --shape ROWS COLS --memory-budget-bytes BYTES --out DOMAINS
scar compile-build --domains DOMAINS --stateful-interval-plans REGION_TO_INTERVAL_PLAN.json --out COMPILED
```

`observe-call-bindings` 是必要动态绑定的可选步骤；纯静态 build 可省略 `--call-bindings`。shape/预算必须来自声明或观测，不凭模型名称猜测。
`--stateful-interval-plans` 按精确 region ID 选择已有区间计划；也可使用 `--calibration` 等已准入模型输入。未决计划保留 NO_OP，`--plan-only` 只保存计划。

`plan-build` 自动保存正文、caller 接口、局部内容 epoch、生命期和共享循环 carry。`compile-build` 独立重放后消费，并校验同源硬件域的 URI 别名；未知成本不冒充零时长。标量与区间计划共用原 C++ lowering。

查看第一阶段的图：`scar view-graph --build PROGRAM --out GRAPH_VIEW`，随后打开 `GRAPH_VIEW/index.html`。支持拖动、缩放、双击展开/收起、关系筛选、原记录查看和本地布局保存；程序图与原硬件库存分别浏览，离线使用，不修改 IR。默认有导出范围限制，可用 `--depth`、`--max-parts`、`--max-nodes` 调整，未导出的边界会保留标记。已有任务图也可用 `--schedule SCHEDULE.json` 查看。

已有来源核验的 standalone 串行区域成本可继续进入局部时间规划与 C++ 试验：

```bash
scar project-activity-time --costs COSTS/result.json --out TIME
scar plan-closed-windows --activity-time TIME/result.json --out WINDOWS
scar prepare-window-rewrite --windows WINDOWS/result.json --out REWRITE
scar compile-window-trial --rewrite REWRITE/result.json --out TRIAL
```

这条路径保留未知残差、原存储生命期及外部边界；自动发现同形状、独立 raw-copy 循环，生成真正的替换任务图，重接其就绪/时间/硬件关系，再由同一证书生成 C++20/ABI1。仅在唯一可试验窗口时自动定位，否则在准备替换图时给 `--window`。融合减少循环控制，保留原数据复制；新成本重新标为 UNKNOWN，默认 NO_OP，不构成全图选优或部署。各命令均支持 `--check`；旧 `compile-window-trial --windows` 保留为直接方案试验接口。

替换区域还可接源忠实的规范化组件成本与有限模型选择：

```bash
scar calibrate-copy-window --rewrite REWRITE/result.json --out COPY_COST --repetitions 7 --warmup 2
scar choose-copy-window --calibration COPY_COST/observations.json --out CHOICE
scar compile-copy-choice --choice CHOICE/result.json --out CHOICE_TRIAL
```

原完整窗口与融合窗口交替采样，各自一个计时区间；不累加旧单项成本，不运行完整目标来搜索变体。选择器按这两个固定 CPU 组件的 median 选择，平局保留原窗口；C++ 构建端消费选择并绑定对应子产物。经验范围和未证明的上下文差异保留，默认仍 NO_OP，仅显式试验，不据此宣称整程序收益或全局最优。这三个命令也支持 `--check`。

原源循环的私有 workspace 生命期外提可使用：

```bash
scar prepare-workspace --loop-build SOURCE_LOOP/result.json --out WORKSPACE
scar compile-workspace --preparation WORKSPACE/result.json --out WORKSPACE_NATIVE
scar export-workspace-loop --native WORKSPACE_NATIVE --out WORKSPACE_RUNTIME
scar run-stateful-loop --bundle WORKSPACE_RUNTIME --admission-sha256 EXPORT_SHA --report RUN.json --validate-only --exclusive-inputs -- ENTRY.py
scar project-workspace-time --native WORKSPACE_NATIVE --out WORKSPACE_TIME
scar project-workspace-demands --time-schedule WORKSPACE_TIME/result.json --out WORKSPACE_DEMANDS
scar bind-time-resources --time-schedule WORKSPACE_DEMANDS/result.json --out WORKSPACE_RESOURCES
scar cost-activities --resources WORKSPACE_RESOURCES/result.json --out WORKSPACE_COSTS
scar view-graph --time-schedule WORKSPACE_TIME/result.json --out WORKSPACE_VIEW
```

准备端重放原 F4 循环/单步证明，检查本轮所有私有读取均由本轮写入覆盖；图保留一份 LOOP_REGION 正文，外层持有固定容量，每轮内容仍有独立版本。C++ 保留单步编译边界及每轮 guard、计算、FP/CPU 还原和提交，只移动私有分配/释放。导出命令返回 `expected_admission_sha256`，作为上面 `EXPORT_SHA`；运行端只读取 compact 合同、动态库和原程序源码，不加载分析器或大图。省略 `--validate-only` 时仍执行原程序，显式试验失败只续跑未提交后缀。`prepare-workspace --recipe RECIPE --trip-count N` 仅生成固定输入的显式调用组合，不授权替换未经证明的原调用循环，也不能导出为原入口替换包。减少分配请求不保证更快；当前均未自动启用。

`project-workspace-demands` 当前只细化单线程 C++ 源路径：成功 bind 后才推导选定 CPU，provider 使后续位置知识失效，restore 后的 COMMIT 不沿用选定 CPU。成本命令默认保留 UNKNOWN；可追加 `--domain DOMAIN --candidate ID --calibration PRIMITIVE_JSON`，严格匹配来源、CPU、形状与布局后接入条件组件。外层 pool 计一次、正文按迭代计费，未知残差与完整时长仍保留，不把组件和当成整程序时间。

边界组件可用 `scar calibrate-workspace-boundary --time-schedule WORKSPACE_DEMANDS/result.json --out BOUNDARY --repetitions 7 --warmup 2` 采集，再给成本命令追加 `--boundary-calibration BOUNDARY/observations.json`。微型探针保留实际检查、分配、绑核和恢复片段，跳过目标 DAG、COMMIT 和 provider；每份保留逐迭代原切点及未映射间隙，计时开销与实现差异仍为残差。

已有完整有限模型证书可通过 `scar compile-pipeline-refinement --certificate REFINEMENT/certificate.json --out COMPILED` 构建 C++20 并导出 compact 包，再用 `run-caller-pipeline --bundle COMPILED/bundle --admission-sha256 EXPORT_SHA --validate-only --exclusive-inputs -- ENTRY.py` 验证。当前只落实计算调用的 CPU lane 顺序与原依赖，沿用原初始化、内存和 join 保护；模型分配时间、释放时间及 makespan 不作为原生运行保证，默认仍 NO_OP。

闭合数组 caller 可通过 `compile-caller-pipeline --pipeline-workers N --pipeline-dispatch ready` 使用原数据依赖和 pool 复用约束，按实际完成情况派发后继；显式 CPU 分配不代表最快方案。对应 `run-caller-pipeline --validate-only --exclusive-inputs` 只验证执行，不采集性能或授权部署。

追加 `--pipeline-memory-order dependencies` 时，只在原 RAW 已证明旧值全部消费者结束后复用，避免为省槽位新增等待；预算不足明确拒绝。默认 `source` 保持按源码串行生命周期规划。两种容量模型均不等同最短执行时间，联合 makespan 选择仍需成本和执行证书。

## 交付与非计时验证

已有 compact stateful C++ 产物和 DIRECT/COMPOSITE whole-loop 产物使用同一个最终交付核心。准备与运行保持同一原环境；payload/slot SHA 来自 prepare 输出。

```bash
scar prepare-stateful-delivery --pipeline READY_PIPELINE.json --out CAPSULE -- ENTRY.py [arguments]
scar run-stateful-delivery --capsule CAPSULE --slot CAPSULE/validation.slot --payload-sha256 PAYLOAD_SHA --slot-sha256 SLOT_SHA --report CAPSULE/run.json --exclusive-inputs --validate-only -- ENTRY.py [arguments]
```

whole-loop 对应 `prepare-loop-delivery` / `run-loop-delivery`。省略 `--validate-only` 保留原执行。当前交付槽为 UNMEASURED，不使用旧性能收据授权新路径，也不导出 RELEASE。
已有 native ABI、源/状态/形状/预算 guard 与失败后的整区域回退保留；运行端无需 compiler、IR、planner 或 g++。

原 F4 状态循环现可用 `compile-stateful-loop --help` 构建；省略 `--step` 时，从选定原 CALL 自动发现唯一状态区域。再以 `run-stateful-loop --help` 接入原入口，复用既有单步 ABI1，逐步提交；部分成功后只恢复未提交的迭代。当前仅支持固定参数、每轮一个普通 global step CALL 的封闭 profile，显式验证使用 `--validate-only --exclusive-inputs`，默认仍原执行。

## 模型与开发状态

图用共享正文、强类型记录、ID/FactRef 引用和分片表存储；循环和函数可以继续展开，静态模板、动态调用、逻辑内容与物理 storage 分开。
硬件库存保留完整平台，存在、可访问、后端支持和实际使用分别表达。当前单区域 CPU 方案选择尚非全程序联合调度。

- [模型](docs/MODEL.md)：实体、端口、状态、版本、循环和证据。
- [架构](docs/ARCHITECTURE.md)：分层组件与交接。
- [当前状态](docs/STATUS.md)：实际验收、范围与未完成项。
- [开发与查询](docs/DEVELOPMENT.md)：形式前端、接口和工作区约定；更多命令见 `scar --help`。

真实 testcase 在项目外；`tests/workloads` 仅保留小型编译回归。大产物留在 `artifacts/current`，精简验收报告在 `artifacts/baseline`。只维护当前工作区源码，版本历史由 Git 保存。
