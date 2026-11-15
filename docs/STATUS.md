# 当前状态

主测试组为项目外的 **Sunfish、Gray–Scott、Cirq**。真实 Gray–Scott 已贯通有状态方法的 F4 提取、图规划、C++20 交付和完整积分 A/B，受测范围加速 **3.710×**。Sunfish/Cirq 暂只有基线；三个阶段均未全部完成，未修改原程序，没有 LeWM/GPU 加速结论。

## Observed / Verified：指定的三个 CPU 程序

| 程序 | 实际配置 | 本轮原版运行 | 优化状态 |
|---|---|---:|---|
| Sunfish | startpos，`go movetime 10000`，单核搜索 | 7.047s，平均 0.991 核 | 仅基线；自然返回 bestmove，未做固定工作量 A/B |
| Gray–Scott | 512×512，5000 步，原版 PDE integrate，float64 | 新配对基线中位数 32.950s | 整方法版本 8.882s，3.710×；旧单 stencil 版本仍拒绝 |
| Cirq | 22 qubit、depth 20、12 rounds、8 circuits / 8 进程、complex128 | 620.142s，105600 gates | 仅基线；实际 8 个 worker，峰值采样约 803% CPU |

Sunfish 返回 `bestmove d2d4 ponder d7d5`，定时搜索自然提前返回，因此历史运行秒数不能作为固定工作量加速比较。Cirq 最大 norm error 为 1.22e-14；与历史同配置结果的 norm/Z0 标量一致，不代表已逐元素比较完整 state vector。进程 RSS 求和峰值约 4.74GiB，可能重复计算共享页，采样间隔 2 秒。

## Implemented / Verified：真实有状态方法闭环

原 F4/UnitFacts 自动发现唯一可接受的有状态方法，机械提取 54 个 Expr、24 次求值、2 次有序写入、2 个数组及 5 个标量字段输入，保原 SourceRef/CodeID/world。接收者、字段和函数名称都不参与规则选择；静态分析未执行目标。

规划区分 slice alias、已计算的不可变临时值和 call-local storage epoch。跨写入仍被消费的旧计算值物化保留；stencil RHS 完成后才写 private shadow，成功后提交原 ROI。11 个阶段共享 4 个 payload slot；workspace 8,454,304B，外部数组 4,243,600B，总 region payload 12,697,904B，预算 32MiB。CPU 0–3 来自原硬件库存，4 线程显式指定且未排名；内存页 UNBOUND。不是进程 RSS、物理驻留或全局最优声明。

后端 C++20 / stateful ABI v1 按该计划执行，一次原方法调用对应一次 native 调用；`np.power` 复用原 NumPy native ufunc，不换成乘法。该 provider 精确绑定 NumPy 2.5.2、加载二进制和实验性 capsule ABI；没有改全局环境或重写算子数学。

完整原 `integrate` 保留外层 5000 步、ghosts、dt、stdout、PNG。预热 1 对、交替 AB/BA 正式 3 对，原版 library 线程数 1，候选区域 4；无诊断函数 trace/profiler，计时包含构造和原程序输出，排除进程启动/候选加载。

| 秒 | 原版 | 整方法 C++ |
|---|---:|---:|
| samples | 32.150804 / 33.303012 / 32.949516 | 8.908676 / 8.872576 / 8.881688 |
| median | 32.949516 | 8.881688 |
| min–max | 32.150804–33.303012 | 8.872576–8.908676 |
| stdev | 0.590266 | 0.018773 |

**3.709826×，TRANSFORM_UNDER_CONTRACT。** 三次正式运行共 15000 次 native、0 fallback；区域、20 步 trajectory、完整积分的 u/v 逐位、数组对象身份、dt、dump、stdout 和 PNG 字节均一致。原 baseline 最后 dt=0 的行为保留。更改标量或调用者写数组后重新读取；替换嵌套 callee 回原执行；数值失败在外部零写时拒绝。

成本门以完整 A/B 样本选择后，非 trial 的正常守卫入口另验证 20 步、native 20 / fallback 0、状态及时间逐位一致。没有将此 20 步验证冒充第二组完整 A/B，也没有取得独立进程自动部署资格。完整程序仍由 Python 宿主执行，只有发现的方法区域下放 C++。

实际 catalog 冷加载约 81 秒，首次 source 准备约 11 秒，复用查询约 4.33 秒；不能仅报后者。带内联来源的 region JSON 约 2.8MB，未按调用次数复制，来源引用的进一步共享仍需改进。

## Rejected：此前只替换单 stencil 的完整 A/B

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
| 建图 | F0–F4、共享 function/loop、硬件库存、有限动态绑定；实际有状态方法的读/写/版本/原调用来源 | 完整搜索及多进程任务依赖；通用 alias/content/effect 闭合；catalog 冷加载和共享来源存储 |
| 化简与硬件规划 | 有序写入间融合、跨 barrier 物化、payload liveness/slot 复用、资源绑定、完整 A/B 成本门 | 将 state epoch/外部 ROI commit/成功 fence 接入共享 recipe；宏观操作成本与联合 worker/分块/放置/搬运/同步时序 |
| C++20 交付 | 受约束数组及状态方法、原 NumPy provider、源码/ABI/资源守卫与事务 fallback | 一般控制/复杂 dtype/跨进程后端、正式自动入口、完整独立 C++ 宿主 |

下一步将新区域引用接到共同宏观 recipe，明确每 epoch 的值、独立 storage slot、成功屏障与外部提交，不能把多次修改 storage 冒充旧单写者 fresh-output/EXPORT。随后补有来源的任务与部署成本、真实搜索/进程依赖。保持循环正文共享、不保存逐次 tensor 值，局部算子继续交给 GCC/LLVM、成熟库或外部算子后端。

先前约 **10.256×** 来自自建 `temporal_reaction` 点式递推，仅保留为受约束回归案例，不能归给上述三个程序、LeWM 或通用宏观编译器。统一 recipe 的既有贯通结果仍有效，其实测成本模型和全程序推广尚未完成。

当前原始结果：`artifacts/current/experiments/real-cpu-suite-v1/current-results.json`、`gray-scott/stateful-integration.json`；版本化配置及摘要：`artifacts/baseline/cpu-workloads.json`、`artifacts/baseline/formal-frontend-results.json`。
