# 当前状态

三阶段已贯通受约束的多函数 CPU 数组 pipeline：原 F4 图自动拆解→内存生命周期优化→C++20 宿主执行。
整程序建模与通用宏观编译器尚未完成；LeWM 未改，本机无可用 GPU。

| 阶段 | 已实现并核验 | 主要缺口 |
|---|---|---|
| 建图 | F0–F4、保调用 scope 的共享正文查询、原 CALL 参数/返回路由、自动数组 pipeline、有限动态绑定与输入字段 | 完整环境/内容版本/effects、一般控制流和原生边界接线 |
| 硬件化简 | 有来源的原语成本、有限 CPU 实现选择、固定顺序下 workspace 生命周期与精确槽位复用规划 | 跨区域融合、放置/COPY/驻留/重算/调度联合优化；新宿主总时间模型 |
| C++20 | ABI1 算子、资源计划 v2、单区域 timed 执行、多区域原生宿主与 buffer pool、一次入口 FFI | 一般循环/分支、真实 COPY/异步、外部效果与完整程序交付 |

**Verified：最新多区域主链**

- 项目外构造的 split-reaction 科学数组测试程序，经正式前端 25 层自动建图，41.401 秒完成；没有运行目标来建静态图。原 F4 自动提取四个函数调用及完整参数路由，没有手写 pipeline wiring 或业务名规则。
- 函数/循环原正文仍共享。新增 pipeline 序列化共享 leaf plans；调用引用 plan ID，不复制函数体。浮点常量由原 F4 指向的已准入 F2 字段读取，修复了稀疏常量事实表造成的缺口，没有重新解析 AST 选择表达式。
- 两次局部进入观测取得实际 1025² float64 输入与四个实际 callee CodeID；不保存输入数值，关闭局部事件后原程序继续。运行边界逐调用核实际函数对象/code/defaults，callee 改变时执行当前原函数一次。
- 固定顺序内存模型明确 producer、consumer、ready 和 last-use。三个中间逻辑值映射至两个 workspace 槽；同一槽承载不同逻辑值不代表内容相同。只有 last-use 严格早于下一 producer 才可复用，当前输入不能被覆盖，最终输出独立新建。
- workspace 16,810,000 bytes，加最终输出 8,405,000 bytes，共 25,215,000 bytes，满足 32 MiB 预算。24 MiB 在该固定 pool 模型内不可行，已在编译前拒绝。预算仅指增量 payload，不含输入、allocator 元数据/对齐开销、线程 scratch 或 RSS，也不证明物理最优。
- C++ 宿主执行完整四次 ABI1 调度并管理 workspace；Python 只做边界检查和一次 FFI，不逐算子调用 Python leaf.run。逻辑释放允许后续复用，物理槽在本次调用退出时统一释放。
- 单算子 leaf 的 standalone 不盈利判断保留，只在显式 pipeline component 上下文准入。CPU 8 核为显式未排名配置；旧 leaf Python.run 的成本不能冒充新宿主时间模型。

**Verified：干净 A/B 与实际选用**

| 范围 | baseline median | optimized median | 加速比 |
|---|---:|---:|---:|
| 四调用区域，warmup 3、交替配对 9 次 | 9.361 ms | 1.770 ms | 5.289× |
| 完整科学测试程序，1025²、400 步、warmup 1、交替配对 3 次 | 3.984 s | 1.470 s | 2.709× |

每轮 400 次原生调用、零回退；20 步演化轨迹、最终状态、stdout、npy 文件字节一致。
区域验收还覆盖输入原地变化、调用方修改输出、dtype/非有限值回退、callee 替换及一次 fallback。
使用测量报告、关闭 trial 后，已另核验正常选择模式确实执行原生版本。
这是函数 pipeline 的组合收益，包含算子内部融合、并行、dispatch 与内存改进；没有隔离各因素贡献。
它不是 LeWM 加速，也不是所有程序的通用加速比，外部 Python 驱动仍保留。

90 项相关检查通过（7.77 秒）。wheel 漏打包两个旧原生分析源的问题已修复，未升级 PyTorch/CUDA。
新宿主的测量后改动仅加强 build/load 的 CPU placement 收据校验；执行热路径和生成的 C++ 未变，两个实现指纹及机械差异证据单列保存。

已有进展保留在版本化收据：调用 scope 的内部条件查询不证明内容保持；有限 CPU 模型最优不等于物理最优。
此前 Gray–Scott 完整短程序约 1.052×，耗时下降 4.99% 未达门槛，仍 NO_OP，不能用新测试程序结果替换该结论。

下一层先在原图上做跨区域化简，再扩硬件规划：
1. 用已有逻辑值/效果/调用合同证明可融合或复用的子图，保留原输入、callee guard 和效果边界。
2. 把这些子图变换与 materialization 位置、COPY/ready/release、驻留/重算及执行次序联合建模；补新宿主成本，不猜时间。
3. C++ 后端按上层计划增加真实传输与异步能力，未实现的任务继续拒绝。

真实产物：`artifacts/current/experiments/pipeline-memory-execution-v1/`。
精简版本化验收入口：`artifacts/baseline/formal-frontend-results.json`。
