# 当前状态

主测试组为项目外的 **Sunfish、Gray–Scott、Cirq**。三个真实负载已重新运行；尚无这三个程序的可信加速结果，三个阶段均未完成。本轮未修改原程序，没有 LeWM/GPU 测量。

## Observed / Verified：指定的三个 CPU 程序

| 程序 | 实际配置 | 本轮原版运行 | 优化状态 |
|---|---|---:|---|
| Sunfish | startpos，`go movetime 10000`，单核搜索 | 7.047s，平均 0.991 核 | 仅基线；自然返回 bestmove，未做固定工作量 A/B |
| Gray–Scott | 512×512，5000 步，原版 PDE integrate，float64 | 初次完整运行 50.69s；配对测量见下 | C++ stencil 区域正确，但成本门拒绝 |
| Cirq | 22 qubit、depth 20、12 rounds、8 circuits / 8 进程、complex128 | 620.142s，105600 gates | 仅基线；实际 8 个 worker，峰值采样约 803% CPU |

Sunfish 返回 `bestmove d2d4 ponder d7d5`，定时搜索自然提前返回，因此历史运行秒数不能作为固定工作量加速比较。Cirq 最大 norm error 为 1.22e-14；与历史同配置结果的 norm/Z0 标量一致，不代表已逐元素比较完整 state vector。进程 RSS 求和峰值约 4.74GiB，可能重复计算共享页，采样间隔 2 秒。

## Verified / Rejected：真实 Gray–Scott 完整 A/B

原源码及适配器哈希未变。静态图自动发现的 stencil 区域已有通用 C++20 实现；本次用显式安装的候选路径验证，外层 Euler 更新、ghosts、5000 步循环、dt、stdout、PNG 输出继续执行原程序。原版 native library 线程数为 1；候选区域显式使用 4 个 CPU（0–3），内存页未绑定。

预热 1 对、正式 3 对，交替 AB/BA；未开启函数级 trace 或 profiler。计时包含原 integrate 的输出，不是独立进程自动部署时间。

| 秒 | 原版 | C++ 区域版本 |
|---|---:|---:|
| 原始样本 | 49.182733 / 33.752529 / 32.215041 | 38.763393 / 37.730605 / 23.272260 |
| median | 33.752529 | 37.730605 |
| min–max | 32.215041–49.182733 | 23.272260–38.763393 |
| stdev | 9.384008 | 8.661078 |

**0.894566×，not profitable，NO_OP。** 候选中位数增加约 11.8%；样本波动较大，未筛样或重跑获取正结果。现有通用 `select_measured` 根据同一原始样本返回成本拒绝，保留原执行。

区域、单步、20 步 trajectory、完整 integrate 的 u/v 状态逐位、dt、dump 次数、stdout 和 PNG 字节比较通过；正式候选共 30000 次 native、0 fallback。此结论验证了区域正确性与该配置的成本拒绝，没有证明全程序已编译或硬件安排最优。

## 三阶段实际边界与下一步

| 阶段 | 已有能力 | 真实测试组暴露的主要缺口 |
|---|---|---|
| 建图 | F0–F4、共享 function/loop、调用/参数/返回/state 端口、硬件库存、有限动态绑定 | Sunfish 旧 capture 主要覆盖启动协议，不能代表完整搜索；Cirq 小配置单进程 capture 未闭合 8 worker 任务依赖；通用 alias/content/effect 仍未闭合 |
| 化简与硬件规划 | 区域规则、依赖/容量计划、受约束 execution recipe、成本选择与 NO_OP | 真实循环整体的数据/状态/边界写入及中间值生命周期；宏观操作成本；联合 worker/分块/放置/搬运/同步规划 |
| C++20 交付 | 受约束 real float64 数组区域、共享点式递推、源/ABI/资源守卫及 fallback | 一般递归/字典/生成器/IO、complex128 耦合算子、跨进程任务与全程序宿主交付 |

下一步用同一主测试组按层补实际调用与状态/进程接口，再形成更大的合法替换区域，最后让硬件计划决定调度和交付。保持循环正文共享，不展开上万次迭代，不逐次保存 tensor 值。局部算子继续交给 GCC/LLVM、成熟库或外部算子后端。

先前约 **10.256×** 来自自建 `temporal_reaction` 点式递推，仅保留为受约束回归案例，不能归给上述三个程序、LeWM 或通用宏观编译器。统一 recipe 的既有贯通结果仍有效，其实测成本模型和全程序推广尚未完成。

本轮原始结果：`artifacts/current/experiments/real-cpu-suite-v1/summary.json`；版本化配置及摘要：`artifacts/baseline/cpu-workloads.json`、`artifacts/baseline/formal-frontend-results.json`。
