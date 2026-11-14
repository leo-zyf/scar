# 当前状态

三阶段已贯通受约束的多函数 CPU 数组 pipeline 与共享时间循环：原 F4 图自动拆解→跨区域/跨迭代化简与内存规划→C++20 执行。
整程序建模与通用宏观编译器尚未完成；LeWM 未改，本机无可用 GPU。

| 阶段 | 已实现并核验 | 主要缺口 |
|---|---|---|
| 建图 | F0–F4、保调用 scope 的共享正文查询、原 CALL 路由、自动数组 pipeline 与单 carry range 循环、有限动态绑定与输入字段 | 完整环境/内容版本/effects、一般控制流和原生边界接线 |
| 硬件化简 | 跨区域完整 elementwise 融合、跨迭代元素局部递推、表示/payload 选择、原语成本与有限 CPU 域、固定 pool 生命周期规划 | SIMD/依赖延迟成本、部分分区、放置/COPY/驻留/重算/调度联合优化；整体时间模型 |
| C++20 | ABI1 算子、单 kernel、资源计划 v2、多区域原生宿主与 pool、独立 loop ABI1/真实循环 | 一般循环/分支、真实 COPY/异步、外部效果与完整程序交付 |

**Verified：最新共享循环主链**

`compile-loop` 从原 F4 自动识别 iterator/world 回边与 lexical carry，不伪造数组 phi；正文保留一份，400 次迭代不展开。一次原入口采集耗时 0.316 秒，取得 shape/layout、精确 count 及真实 body/leaf CodeID，未保存数组数值；采集前缀主动停止，不作为完整运行或性能数据。

图证明各元素间无依赖、输出形状保持后，C++20 每个元素局部执行完整时间递推，一次 FFI、无中间数组，独立最终输出 8,405,000 bytes。12 MiB 增量 payload 预算可行，4 MiB 在编译前 NO_OP；不证明整体时间最优。

新外部科学程序 1025²/400 步，warmup 1、配对重复 3 次：完整原入口 baseline median **4.665 s**，native **0.766 s**，**6.091×**。最终状态、stdout、npy 字节一致，每次运行 1 native/0 fallback；关闭 trial 后正常选用通过。零/负次数返回原输入别名，次数变化及嵌套 callee 副作用回退原循环一次。未观测中间每次迭代输出，不宣称全 Python 效果等价。

生成码核查发现新 temporal 热循环使用标量算术，历史 flat kernel 有跨元素 packed SIMD。两次历史测量不是受控版本排名；减少材料化不能推出时间更快。下一层硬件规划必须同时考虑 SIMD、算术依赖链与调度成本，不能只最小化 payload。当前 8 CPU 显式未排名，flags 不作为程序输出属于外部合同。

7 项相关检查通过（2.37 秒）。最新验收目录 `artifacts/current/experiments/loop-temporal-execution-v1/`；下方旧 pipeline 测量保留原范围，不能当作 LeWM 或所有程序的加速比。

双文件入口静态 F4 另核验通过（27.61 秒，RSS 85,316 KiB），统一 `build --target` 用 3.38 秒接入保留硬件图；main 有界展开覆盖 329 操作，755 条资源边均为条件 MAY。CPU/host memory/SSD/NVMe 引用可查，保留库存中的 accelerator 不代表本机当前可用 GPU。19 个外部 CALL 接线、内容/效果及实际 placement/I/O/transfer 仍开放，不能把工作流成功当作整程序闭合。

**Verified：最新多区域主链**

- 新增完整 elementwise 融合：沿原调用的 SSA 值边替换 leaf 输入，将四调用展开为 17 个 raw Expr，再由既有优化器得到 16 个 Expr。原 F4/leaf→raw 证书与 raw→优化图映射分别核验，原 pipeline 与四个 callee guards 保留；没有按名字或 storage 合并数据。
- `compile-pipeline` 默认先比较原 pool 与合法融合表示，再判预算。1025² 输入的原 pool 要 25,215,000 bytes；融合只保留独立最终输出 8,405,000 bytes，无 workspace，在 12 MiB 预算内达到该 dense/fresh-final payload 下界。4 MiB 不够最终输出，编译前返回 NO_OP。
- 选择由图合同和精确 payload 推导，没有编译多个目标版本后试跑排名。内存表示最优不等于时间或物理最优，CPU 8 核仍为显式未排名配置。slice、开放效果、逃逸与不通向最终值的原 call 不越过融合边界，必要时保原 host/pool。
- 项目外构造的 split-reaction 科学数组测试程序，经正式前端 25 层自动建图，41.401 秒完成；没有运行目标来建静态图。原 F4 自动提取四个函数调用及完整参数路由，没有手写 pipeline wiring 或业务名规则。
- 函数/循环原正文仍共享。新增 pipeline 序列化共享 leaf plans；调用引用 plan ID，不复制函数体。浮点常量由原 F4 指向的已准入 F2 字段读取，修复了稀疏常量事实表造成的缺口，没有重新解析 AST 选择表达式。
- 两次局部进入观测取得实际 1025² float64 输入与四个实际 callee CodeID；不保存输入数值，关闭局部事件后原程序继续。运行边界逐调用核实际函数对象/code/defaults，callee 改变时执行当前原函数一次。
- 固定顺序内存模型明确 producer、consumer、ready 和 last-use。三个中间逻辑值映射至两个 workspace 槽；同一槽承载不同逻辑值不代表内容相同。只有 last-use 严格早于下一 producer 才可复用，当前输入不能被覆盖，最终输出独立新建。
- 保原 pool 路径的 workspace 16,810,000 bytes，加最终输出 8,405,000 bytes，共 25,215,000 bytes；24 MiB 在该固定 pool 模型内不可行。预算仅指增量 payload，不含输入、allocator 元数据/对齐开销、线程 scratch 或 RSS，也不证明物理最优。
- 保原调用时，C++ 宿主执行四次 ABI1 调度并管理 workspace；融合时只执行单个 ABI1 kernel。Python 都只做边界检查和一次 FFI。pool 的逻辑释放允许复用，物理槽在调用退出时释放。
- 单算子 leaf 的 standalone 不盈利判断保留，只在显式 pipeline component 上下文准入。CPU 8 核为显式未排名配置；旧 leaf Python.run 的成本不能冒充新宿主时间模型。

**Verified：干净 A/B 与实际选用**

| 范围 | baseline median | optimized median | 加速比 |
|---|---:|---:|---:|
| 融合区域，warmup 3、交替配对 9 次 | 9.971 ms | 1.004 ms | 9.929× |
| 完整科学测试程序，1025²、400 步、warmup 1、交替配对 3 次 | 3.856 s | 0.185 s | 20.806× |

每轮 400 次原生调用、零回退；20 步演化轨迹、最终状态、stdout、npy 文件字节一致。
区域验收还覆盖输入原地变化、调用方修改输出、dtype/非有限值回退、callee 替换及一次 fallback。
使用测量报告、关闭 trial 后，已另核验正常选择模式确实执行原生版本。
这是跨函数融合、并行、dispatch 与内存改进的组合收益；没有隔离各因素贡献。两行是不同测量范围，不代表优化器预测了这些时间。
它不是 LeWM 加速，也不是所有程序的通用加速比，外部 Python 驱动仍保留。

96 项相关检查通过（8.51 秒）。本轮 wheel 的 1009 项源码/资产逐字节一致，旧 core、原 host 与新 fusion 实现指纹全部核对；未升级 PyTorch/CUDA。

已有进展保留在版本化收据：调用 scope 的内部条件查询不证明内容保持；有限 CPU 模型最优不等于物理最优。
此前 Gray–Scott 完整短程序约 1.052×，耗时下降 4.99% 未达门槛，仍 NO_OP，不能用新测试程序结果替换该结论。

后续三层继续有序交接：
1. 建图：从共享单 carry 循环扩到一般控制结构，补入口到原生/IO 的依赖；保输入、callee、版本与效果边界。
2. 硬件规划：先补 SIMD/依赖链与 native 执行成本，再联合 materialization、COPY/ready/release、驻留/重算及执行次序；不猜时间。
3. C++ 后端：消费上层明确计划扩展向量化/分块与真实传输/异步，未实现的任务继续拒绝。

最新真实产物：`artifacts/current/experiments/pipeline-fusion-execution-v1/`；原 host/pool 验收保留在 `pipeline-memory-execution-v1/`。
精简版本化验收入口：`artifacts/baseline/formal-frontend-results.json`。
