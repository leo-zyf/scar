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

闭合数组 caller 可通过 `compile-caller-pipeline --pipeline-workers N --pipeline-dispatch ready` 使用原数据依赖和 pool 复用约束，按实际完成情况派发后继；显式 CPU 分配不代表最快方案。对应 `run-caller-pipeline --validate-only --exclusive-inputs` 只验证执行，不采集性能或授权部署。

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
