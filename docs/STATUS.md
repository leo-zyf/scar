# 当前状态

三阶段已贯通受约束的 CPU 数组 pipeline、共享循环、硬件 SIMD 布局及轻量部署。
通用宏观编译器尚未完成；LeWM 未改，本机无可用 GPU。

| 阶段 | 已实现并核验 | 主要缺口 |
|---|---|---|
| 建图 | F0–F4、共享函数/循环、原 CALL 参数/返回/world；局部动态字段自动接入统一软件/硬件图 | 完整环境/内容版本/effects、原生 schema 与一般控制流闭合 |
| 硬件化简 | 跨区域/跨迭代融合、payload/生命周期规划、SSE2 依赖/吞吐代理与活跃区间布局；部署 scope/count 成本门控 | 全边界组件成本、并行/缓存/NUMA、COPY/驻留/重算/放置联合调度 |
| C++20 | 算子/loop ABI1、原生 host/pool 与 packed 多链；构建准入与轻量 runtime 分离 | 自动整程序接入、一般控制/传输/异步、完整 standalone C++ 交付 |

**Verified：最新部署主链**

`export-loop-runtime` 在构建端完整重放原 strategy/layout/profile/ISA/source 证明，导出 runtime 合同、admission 收据和原 library；三个核心文件共 **32,894 bytes**。library 与上轮逐字节相同，没有重编旗舰 kernel。额外 export-report 只供构建诊断，运行 loader 不读取/信任它。

`scar.runtime` 仅核调用者明确提供的 admission pin、包字节、ABI/table 与 live process/CPU。安装时核原应用源码和分层函数 globals/builtins，调用时保 count/array/owner/fresh-output/CPU/fenv/回退守卫。运行不导入 compiler/optimization/analysis/codegen/IR，不规划、不读 profile/target/template、不启动 objdump。哈希不是签名，producer receipt 必须由调用者信任。

真实隔离检验把三个文件复制到新目录，禁止原 proof 档案/模板读取、编译器族导入及子进程，仍执行同一 native kernel：1 native/0 fallback，输出逐位一致。仍需原应用源码、Python/NumPy 及系统共享库；不是独立 C++ 程序。

**Verified：实际 A/B，范围分开**

同一外部 CPU 科学程序，1025²、400 步、8 个显式未排名 CPU；每组 warmup 1、配对重复 3 次：

| 计时范围 | baseline median | native median | 加速比 |
|---|---:|---:|---:|
| 模块加载后的 main，含参数解析、RNG、文件保存 | 3.862 s | 0.176 s | 21.995× |
| 独立进程启动至退出，含导入、轻量加载、安装与文件 IO | 4.155 s | 0.410 s | 10.130× |

每次运行 1 native/0 fallback，最终状态、stdout 与 npy 字节一致。输入修改重算、调用方修改输出、零/负次数别名、次数变化及嵌套 callee 状态回退均核验。未观察中间每次迭代输出。

Fresh child 直接计时的“模块导入＋load”median 约 29.7ms，“源模块＋安装守卫”约 1.4ms；这些组合区间不是纯 dlopen、每调用成本或两个 scope 差值。各组原始样本/min/max/stdev 均保留。不是 LeWM 加速或全局最优证明；历史 25.618×/2.015×保留各自边界，不用于受控版本排名。

**Verified：成本门控与正确性边界**

`plan-deployment` 可在 native 加载前回放有来源的 COMPONENTS 或 COMPLETE_SCOPE 成本。真实两个 scope 各自计数 1 的记录通过成本门；scope 改变、未经观测的计数 2、预算不足均 NO_OP，不外推或猜 0。正向仅 LOAD_NATIVE_CANDIDATE，不授权跳过执行合同。组件临界复用次数是显式模型推导，不冒充完整观测。

同步、无外部并发输入写及 FP status flags 不作为程序输出仍是外部合同。另记录并修复旧 range guard 的 baseline defect：函数使用创建时保留的 builtin 字典，不能以之后 globals['__builtins__'] 的值代替；该修复不归因为性能收益，旧原始测量未改签。

4 项相关检查通过（0.97s），另有 legacy guard/native 回归通过（1.79s）。图仍保条件 MAY、未确定 effects/content/真实 placement 等字段，不能把局部准入当作整程序闭合。

下一步按层交接：
1. 建图：从实际目标继续接 library/native schema、参数/default/receiver/内容版本与效果。
2. 硬件规划：补直接组件及并行/内存层级成本，联合 partition/placement/COPY/ready/release；未知保持前沿。
3. 交付：补自动入口装配和可移植执行合同，逐步扩一般控制流与 C++ 宿主 IO/异步任务。

最新真实产物：`artifacts/current/experiments/loop-runtime-delivery-v1/`。
版本化验收入口：`artifacts/baseline/formal-frontend-results.json`；目标持续开发，三个阶段均未宣告全部完成。
