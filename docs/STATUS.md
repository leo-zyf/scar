# 当前状态

SCAR 三阶段仍未全部完成。当前贯通真实 Gray–Scott 的状态方法建模、统一执行图、C++20 降低及原入口自动装配；Sunfish/Cirq 暂只有基线，没有 LeWM/GPU 加速结论。

## 已验证的主链

原 F4 自动提取一个状态方法：54 个 Expr、24 次求值、2 次写入，保原代码/端口/world 来源。输入类型是运行时守卫的候选约束。规则不按 workload、函数或字段名称选择；原程序未修改。

公共 `ResourceSchedule/v2` 将逻辑值、物化版本和物理 slot 分开。真实图有 20 个任务、8 个逻辑值、12 个物化版本、6 个 slot：分配不代表值就绪；覆盖旧 slot 前所有消费者必须 happens-before；外部 COMMIT 在完整 success fence 后执行。最后写回与 private shadow 共用一个逻辑值，各自保物化 ready 时间。v1 序列化保持不变。

统一 recipe 引用同一份正文/来源及原硬件库存，没有展开 5000 步。完整重放后经原 C++20 backend 降低，build ID 与 library SHA 和此前实测版本相同，未引入新数学实现。recipe 94,322B；源区域含内联 provenance 约 2.8MB。

物理预算：workspace 8,454,304B + 外部数组 4,243,600B = 12,697,904B，声明预算 32MiB。CPU 0–3 / 4 线程显式指定、未排名；内存页 UNBOUND。通用 makespan solver 已接 v2，按 4 个 workspace slot 计费，不重复计算 12 个物化版本。真实逐任务成本缺失，仍返回 UNRESOLVED，不补零；未证明联合硬件或全局时间最优。

轻量运行包 40,550B，复用原库字节；无需 catalog、完整图、planner 或 g++。原入口自然 import 后装配类方法；真实 20 步 u/v、time、dt 逐位相同，native 20 / fallback 0，未导入 compiler/codegen/analysis/IR/optimization。仍依赖原应用源码、Python/NumPy 和系统共享库，不是独立 C++ 主程序。

## 性能证据分开报告

均为真实 Gray–Scott、512×512、5000 步、float64。原版 native library 1 线程；候选区域 4 线程。保留原积分、ghosts、dt、日志和 PNG；不减少步骤或降低精度，无诊断函数 trace/profiler。原 baseline 最后 dt=0 的行为保留。

| 范围 | 预热 / 正式配对 | 原版 | native | 比率 |
|---|---|---:|---:|---:|
| 进程内完整积分，中位数 | 1 / 3 | 32.949516s | 8.881688s | 3.709826× |
| 独立进程候选自动入口，中位数 | 1 / 3 | 50.793170s | 9.729574s | 5.220493× |
| 独立进程正常成本门，一次观察 | 0 / 1 | 50.341570s | 9.780156s | 5.147318× |

独立进程候选原始样本：baseline `[50.290012, 50.893130, 50.793170]`；native `[9.729525, 9.729574, 9.779819]`。stdev 分别 0.323242s、0.029023s；完整启动、导入、加载、守卫、应用 I/O 和报告 I/O 均计入。

候选正式 15000 次 native、0 fallback；正常成本门另 5000 次 native、0 fallback。新入口的报告字段、日志、PNG 一致，明确排除原程序输出的观察者 `wall_seconds` 计时字段；新轻量边界另做 20 步完整数组逐位核对。此前进程内完整积分已逐位比较最终 u/v。不同范围不混算，单次正常入口不冒充三次统计。

可确认实现了整方法一次 FFI、写入屏障间融合、跨屏障物化保留、固定 buffer 复用及 4 线程并行。尚未做各机制的消融，不能分别归因加速贡献。5.22× 与 3.71× 使用相同库、不同测量边界，不代表新增数学优化。

正常路径最初由 DECLARED_MODEL bootstrap 选择：候选完整进程观测 + 独立冷 import/metadata gate 组件，max 样本 0.011253s，不是理论上界。真实正常入口完成后，单对完整时钟另立 OBSERVED 总成本事实与 metadata replay；未把候选标签改成正常观测，也未宣称新的 observed profile 已再次运行。精确 argv/pin/CPU/内存范围变化会拒绝旧成本。

## 当前边界与下一步

- 建图：还需完整搜索、多进程任务、一般 alias/content/effects 和 caller 循环状态连接；catalog 冷加载约 81s、首次源准备约 11s，复用查询 4.33s，仍需加速及压缩共享来源。
- 化简与硬件规划：当前是选定实现的有序任务图；缺有出处的逐任务与全部边界成本、自动 worker/分块/NUMA/放置、CPU/GPU/内存/存储联合调度。stage 估计不冒充完整 native/Python 调用成本。
- 交付：尚未支持一般控制、复杂 dtype、跨进程及异步设备后端、入口自身定义目标的装配和独立 C++ 宿主。所有可能别名的调用覆盖未证明，运行报告保留这一限制。

CPU 主测试组：Sunfish 单核搜索 7.047s；Gray–Scott 如上；Cirq 22 qubit / 8 进程 620.142s。旧单 stencil 完整 A/B 0.894566× 继续拒绝。自建 temporal_reaction 的约 10× 仅为回归，不能归给这三个程序。

当前证据在 `artifacts/current/experiments/real-cpu-suite-v1/gray-scott/stateful-recipe-current/` 和 `stateful-auto-v1/`；版本化摘要见 `artifacts/baseline/formal-frontend-results.json`、`cpu-workloads.json`。
