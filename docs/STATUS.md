# 当前状态

三阶段已贯通受约束的 CPU 数组 pipeline、共享循环、硬件 SIMD 布局和自动导入交付。
通用宏观编译器尚未完成；LeWM 未改，本轮没有 LeWM/GPU 测量。

| 阶段 | 已实现并核验 | 主要缺口 |
|---|---|---|
| 建图 | F0–F4、共享函数/循环、原 CALL/参数/返回/world、软件/硬件资源图；静态导入归属、原 module 绑定锚点；CPU cache/SMT/NUMA 条件化子图 | 子图接入统一 planner；完整环境/内容版本/effects、native schema、一般控制流与别名覆盖 |
| 硬件化简 | 跨区域/跨迭代融合、payload/生命周期、SSE2 依赖/吞吐代理与活跃区间布局；实际 CPU/分配需求门控；temporal worker 成本/选择适配层 | 独立多 worker 校准及自动绑定执行；分层 PassManager、并行/缓存/NUMA、跨设备/存储联合调度 |
| C++20 | 算子/loop ABI1、host/pool 与 packed 多链；普通目标模块自动装配；完整原 main module 挂接原型 | main 原型尚未接 CLI；custom loader/未知 alias、一般控制/传输/异步与完整 standalone C++ |

**Verified：最新闭环**

`scar inspect-entry` 复用静态导入分析，自然找到 entry→目标 source；实测 0.48s、峰值 RSS 33,976KiB，读取 2 文件/1,359B。静态候选不证明实际 loader 或初始化完成。

`scar run` 的轻量 dispatcher 不初始化编译器。scoped 普通 source loader 在目标模块执行完成后安装已有源码/环境守卫，让自然 from-import 获得包装函数；退出撤销装配、保留应用主动重绑，逃逸包装引用转为原函数。无需 testcase 手写 install，不修改应用源码、不全局 profile/trace、不采数组值。构建端 IR/profile/ELF 证明仍由原 export 准入；32,894B 核心 bundle 和原 C++ library 未改变。

正常成本路径核 entry 源版本、逐字 argv、runner 实现、launch policy、完整 native IDs/admission、原复用次数，以及实际合同的 CPU 和新增输出 payload。原手工接入或测量 bypass 的计时不能冒充正常自动路径。无计划或成本/资源不符走 NO_OP，不加载 native。预算明确排除输入、allocator/alignment、线程 scratch 与 RSS，不是整进程内存上限。

**Verified：真实独立进程 A/B**

外部 CPU 科学程序，1025² float64、400 步、8 个显式未排名 CPU；warmup 1、配对重复 3 次。父进程计时包括启动、导入、成本门、装配、原程序 IO 和执行报告 IO：

| | median | min | max | stdev |
|---|---:|---:|---:|---:|
| 原程序 | 4.195692s | 4.136185s | 4.222197s | 0.044049s |
| 自动原生路径 | 0.398857s | 0.387545s | 0.408249s | 0.010367s |

本 scope 加速 **10.519×**。每次 1 native/0 fallback；stdout 与最终 npy 字节一致，原 entry/physics 字节未改。seconds.txt 是测量输出，其数值不要求相同。不是 LeWM 加速，也不是全局最优或与旧 scope 的版本排名。

部署计划使用显式 DECLARED_MODEL：候选测量时间加 0.1s metadata surcharge 假设，不冒充实测上界。随后正常成本门路径单独完成上述 A/B；观测没有回写或改签模型。已安装边界计数与所有别名的覆盖度分开，当前 boundary_calls_observation_complete / cost_scope_verified 保守 false。真实正例与预算不足 NO_OP 均运行；7 项相关检查通过（1.10s）。

本轮又完成三个独立交接：完整 module 的静态绑定锚点及局部挂接原型、worker 选择适配层、CPU 层次子图，各有聚焦核验；尚未将前两项接成新业务路径。CPU 查询实测 0.0724s，读取 6,242 原始字节，得到 86 个去重 cache 节点、64B cache line、2 个 35MiB L3；sidecar 207,520B。容量不当可用预算，原 target/当前 sysfs 的连接带条件，不冒充 cache traffic 或同机证明。

**下一层交接，四人并行**

1. 统一共享分析/合法性、PassResult 与分析失效协议，按作用域维护原图→变换图→执行 recipe；参见 ARCHITECTURE.md 的拟定五层。
2. 接独立 temporal 多 worker 校准→每 CPU 组成本→binding→现有 C++，再扩 cache/NUMA/进程。模型同时保留 GPU/设备内存/存储及链接的放置、搬运、驻留与同步域；LLVM/GCC 仅局部降低，不替代宏观规划。
3. 接 main 绑定锚点与原型到正式入口，核原代码/实际 predecessor/readiness；再扩 C++ 宿主 IO/异步及可用算子后端。成本和代码须消费同一执行 recipe。

最新真实产物：artifacts/current/experiments/automatic-entry-v1/。
版本化验收入口：artifacts/baseline/formal-frontend-results.json。三阶段均未宣告全部完成。
