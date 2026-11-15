# 当前状态

三阶段已贯通受约束的 CPU 数组 pipeline、共享循环和硬件 SIMD 布局。
通用宏观编译器尚未完成；LeWM 未改，本机无可用 GPU。

| 阶段 | 已实现并核验 | 主要缺口 |
|---|---|---|
| 建图 | F0–F4、共享可展开函数/循环、原 CALL 参数/返回/world、局部动态字段；有界 CALL 观察自动接入统一软件/硬件图 | 完整环境/内容版本/effects、一般控制流与原生实现接线 |
| 硬件化简 | 跨区域/跨迭代融合、payload/生命周期规划、有限 CPU 域；有来源的 SSE2 依赖/吞吐代理与 two-address 活跃区间布局 | 整体执行成本、并行扩展、缓存/NUMA、部分分区、COPY/驻留/重算/调度联合选择 |
| C++20 | ABI1 算子、原生 host/pool、loop ABI1；v2 layout 合同、packed 多链与标量尾部、ELF ISA/栈审计 | 一般控制流、真实传输/异步、外部效果与完整程序交付 |

**Verified：最新硬件布局主链**

- 原 F4 自动识别 iterator/world 回边和 lexical carry，数组不是原图中的显式 phi。循环正文保留一份，400 次不 unroll。一次入口采集取得 shape/layout/count 与真实嵌套 callee CodeID，不保存数组数值。
- 使用独立规范 SSE2 原语，在固定 CPU/编译器合同下测依赖/吞吐代理与匹配的整数 finite guard。原始样本、ISA/context、单位和 FactReference 均可核；没有执行多个目标版本进行排名。初版 floating-compare guard 校准已排除，未用于所选布局。
- 根据 DAG last-use、first-operand two-address 复用和行宽，选择 2 lane × 3 chains、tile 6。抽象预算为 5 个共享常量 + 2 reserve + 3×3 活跃值 = 16 XMM；它不是实际寄存器分配或无 spill 证明。只编译这一布局。
- C++20 一次 FFI 完成整个循环，最终数组独立新建；payload 8,405,000 bytes、无中间数组 workspace。12 MiB 可行，4 MiB 在编译前 NO_OP。CPU 8 核仍是显式未排名配置。
- 实际 ELF 含 packed add/sub/mul/div；静态审计未见 XMM 栈访问或动态 RSP 减法，最大静态 RSP 调整 200B。该范围排除库调用、CPU mask 分配、完整线程栈及 RSS，不宣称无 spill。旧 v1 源码仍逐字节重放；新布局 artifact 为 v2，loop ABI 保持 1。

**Verified：干净 A/B，两种计时边界**

| 同一外部 CPU 科学程序，1025²、400 步 | baseline median | native median | 加速比 |
|---|---:|---:|---:|
| 已加载模块后的 main 调用，包含 CLI 解析、RNG 和文件保存 | 4.637 s | 0.181 s | 25.618× |
| 独立进程启动至退出，包含导入、artifact 加载/验证与文件 IO | 4.103 s | 2.036 s | 2.015× |

两组各自 warmup 1、配对重复 3 次；不是可相加的分项测量。每次运行 1 native/0 fallback。最终状态、stdout、npy 字节一致；输入修改重算、调用方修改输出、次数变化、零/负次数别名及嵌套 callee 副作用回退均核验。关闭 trial 后正常选用通过。未观测中间每次迭代输出；同步无外部并发写及 FP flags 不作为程序输出属于外部合同。

独立进程暴露当前交付瓶颈：loader 仍导入编译/规划模块并重放模型、profile 及 ISA 审计。不能把 main 的 25.618× 当作命令端到端加速；下一层须分离构建期证明与部署期必要守卫。

这是特定科学程序对原始执行的实测，既不是 LeWM 加速，也不是全局最优证明。历史 flat/scalar-temporal 测量有不同边界，不能当作本轮受控版本排名。原始 NO_OP 始终保留。

**Verified：动静结合并自动装配**

原入口 main 的 19 个 CALL 由 F4/bytecode 静态映射，0.619 秒的小规模原程序采集运行观察到 19 个实际目标；该时间含进程启动和原程序工作，不是纯采集开销。只用目标 code 的局部 CALL，不设全局事件/profile，不抓 operands 或数组内容。已把静态依赖移出 collector：fresh 导入约 1.06→0.34 秒，analysis 模块变为 0；原 plan/join 字节不变。

`build --call-bindings RUN_DIR` 自动保留 plan/raw/join 收据并挂入公开 `BuildSession.view`，独立读回可查 19 条 CALL_TARGET/MAY。相同 CodeID 的观察仍有不同 ProviderID，Invocation 数仍为 0；CALL 不能证明 entry/completion、native effects 或未来分派。这 19 个 main 位点不是旧报告中 19 个外部未连接边的同一分母。

软件图与保留 CPU/host memory/SSD/NVMe 图已接；资源边仍是带条件的 MAY。库存中的 accelerator 不代表当前 GPU 可用，实际 placement/I/O/transfer 及内容/effects 字段继续开放。旧统一图因规则更新保留为历史收据，未重签为当前验收。

下一步仍按层交接：
1. 建图：补观察目标到原库正文或 native schema，以及参数/receiver/内容与效果依赖。
2. 硬件规划：补实际执行、并行和内存层级成本，联合分块、放置、ready/release、COPY 与调度；未知保持字段前沿。
3. C++ 后端：先分离部署运行时与编译/规划，再消费已准入计划扩展一般控制流及真实传输/异步；未实现的任务继续拒绝。

最新真实产物：`artifacts/current/experiments/loop-simd-execution-v1/`。
版本化验收入口：`artifacts/baseline/formal-frontend-results.json`；历史 pipeline/loop 结果保留原范围。
