# 当前状态

统一上层 execution recipe 已贯通一个受约束的 CPU 共享循环：语义边界→任务/资源计划→估价→C++20→自动入口实测。
通用宏观编译器尚未完成；本轮没有 LeWM/GPU 测量，未改 LeWM。

| 阶段 | 已实现并核验 | 主要缺口 |
|---|---|---|
| 建图 | F0–F4、共享 function/loop、CALL/参数/返回/world、原端口引用；资源独立的 loop 语义边界；全部 target 与条件化 cache/SMT/NUMA 引用 | 通用 alias/content/effects、递推/迭代距离、一般控制流闭合 |
| 上层规划 | 统一 recipe 的 control、value/materialization、ready/ownership、资源/预算与后端能力；成本和降低共用 recipe ID | 真实宏观操作成本、自动 worker/分块/放置、全程序跨 CPU/GPU/内存/存储的联合时序 |
| C++20 | 共享公开准入、recipe 精确重放、能力/编译器/ABI/layout 比对；原生循环、普通模块自然导入和轻量部署 | 一般控制/异步/设备后端、正式 main 挂接、完整 standalone C++ |

**Verified：本轮主链**

`scar recipe plan-loop/estimate/lower` 已实际执行。recipe 56,190B，引用组件各保存一次；400 步仍只有一份共享正文、3 个宏观任务和 12 个原 target 资源引用，另附条件化 CPU hierarchy。语义边界不随 CPU 数/布局改变；recipe ID 还绑定资源及后端实现、编译上下文、ABI/layout。源、边界、任务图或能力改变须重放，不能沿用原身份。

分配只准备存储；完整 temporal compute 完成才令最终数组 ready，export 持有逃逸输出。原数据/状态/carry/顺序依赖保留，未选资源不被程序当前用法裁剪。新增输出 payload 为 8,405,000B，workspace=0，显式预算 12MiB；不包括输入、线程 scratch、分配器及 RSS，不声称物理 cache/NUMA 驻留。

估价 JSON、降低收据与导出的 native 身份已逐项接通。新编译器/ABI/实现/layout 必须等于 recipe 的 capability record。合法性已从 codegen 私有函数提到 compiler.loop_admission。此次重新编译一次；模板、flags、数学表达式未改，生成 library 与原 kernel 逐字节相同。本轮验证的是上层计划交接，不能归因为新增局部加速。

**Verified：真实运行**

区域验证通过输入变化重算、输出修改隔离、零/负次数别名、次数变化与嵌套 callee 状态回退，输出逐位一致，获得 TRANSFORM_UNDER_CONTRACT。随后导出 32,739B 核心包，经原未改入口自然导入执行；无需 testcase 手写 install。

独立进程父时钟，包括启动/导入/成本门/装配/原程序 IO/报告 IO；1025² float64、400 步、8 个显式未排名 CPU，warmup 1、配对重复 3：

| | median | min | max | stdev |
|---|---:|---:|---:|---:|
| 原程序 | 4.053302s | 4.037117s | 4.071071s | 0.016983s |
| 原生路径 | 0.395195s | 0.388790s | 0.398225s | 0.004817s |

本 scope **10.256×**；每次 1 native/0 fallback，stdout/最终 npy 字节一致。seconds.txt 是时间测量输出。不是 LeWM 加速、全局最优或旧版本排名。

**明确前沿**

recipe 时间估价仍 UNKNOWN：尚无真实多 worker temporal profile，不从旧 single-step 成本或总调用时间拆造子任务时长。部署 A/B 的启动模型仍显式使用候选时间＋0.1s metadata surcharge 假设，不冒充实测上界；正常成本门路径另测，观测不回写模型。全别名调用覆盖仍未证明，cost_scope_verified 保守 false。新主链 4 项聚焦检查通过（3.14s），另外现有 tiny native/source-admission 回归已通过。

下一步按层推进：
1. 保留 source/state/loop/Q 与原图关系，统一共享分析及 PassResult/失效协议；不重做低层编译器。
2. 为宏观操作接入有来源的时间、workspace、搬运/同步成本；联合资源容量、数据 ready、memory lifetime 求 makespan，再扩 worker/分块/CPU-GPU/存储规划。LLVM/GCC 和算子库承担局部实现。
3. 让正式入口、异步/设备后端消费同一 recipe；扩 source==main 原型及 C++ 宿主 IO，保持原路径 fallback。

最新产物：artifacts/current/experiments/macro-recipe-v1/。
版本化验收入口：artifacts/baseline/formal-frontend-results.json。三个阶段均未宣告全部完成。
