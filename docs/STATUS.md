# 当前状态

当前优先完成三阶段核心开发；暂停新增性能实验。已在受支持的 CPU 数组、循环和事务区域贯通主链，**通用覆盖尚未完成**。不能把正确回退、单个区域编译或历史性能记录当作整个系统已经完成。

最新主链：`compile-caller-pipeline --pipeline-workers N` 自动发布原 caller 的参数替换/数据图，重放 pool 生命周期，生成 RAW/WAR/WAW、实际 CPU 与静态 wave 计划，再交既有 ABI1 算子/C++20 host。每个 leaf 单核，wave 内 CPU 不重叠；全部 worker join 后才能推进或回退。语义依赖图没有硬化 wave 的额外 barrier。

真实 fork/join：3 CALL、5值、41条公共关系、10个资源任务；前两步 CPU0/1，第三步等待两者。冷读、native、浮点溢出回退、舍入/affinity恢复，以及普通入口 compact 交付均已非计时验证。耗时全为 UNKNOWN，wave 只是合法执行预览，默认仍 NO_OP；没有最优性能或加速比结论。报告在 `artifacts/current/experiments/pipeline-waves-v1/`。

此前也已贯通标准 ndarray C-order copy 的内容转发与独立物化，报告在 `array-copy-v1/`。规则/runtime SHA 更新后的旧 receipt 仅保留历史证据，不能授权当前实现；这些封闭区域的正常完成合同不承诺内部 allocator/异常序列一致。

## 分层顺序

| 阶段 | 顺序 | 当前主链 |
|---|---|---|
| 建图 | 源码/版本 → token/语法/作用域 → 局部端口 → 跨层调用/状态/循环 → 动态及硬件证据 | 原 F4、共享正文、caller 接口、内容 epoch、生命期及条件循环 carry；硬件保持独立原引用 |
| 化简 | 原图准入 → 结构规则/合法性 → 硬件需求与成本 → 选择 → 重放 | 不变量外提/融合的现有后端；有限 CPU 域、成本区间、支配证明与明确未决项 |
| 交付 | 已选计划准入 → C++20/算子 ABI1 → compact 包/guard/回退 → 行为验证 | 标量与区间选择共用 C1 lowering；函数、whole-loop、stateful 分别保留原合同 |

每层消费上游原引用，保存范围、来源、条件及未完成项。支持范围内的接口贯通不等于任意 Python 都能独立交付为 C++。

## 本轮核心实现

### 建图进入主规划命令

`plan-build` 自动保存有序正文、caller/内联接口、输入生命期和共享循环的状态转移；`compile-build` 独立重放后再消费。单独的 inspect 不再是唯一入口。

真实两单元图保留 4235 个原操作、51425 个端点、55018 条规范关系；基础硬件快照有 204 个资源、477 条连接。资源存在、可访问、后端支持及实际使用分别表达，库存不只保留被使用的 CPU。

- 有序正文 92 节点/317 条关系；caller 接口 43 条路由、7 条输入路径、2 个独立内联 Frame，360 条投影边双向可查。
- 生命期为 7 个输入、4 个局部内容 epoch、25 次直接使用、13 条条件关系。局部 last use、外部对象生命期和正常出口所需内容分开。
- 自动连接 1 个共享循环、2 层原 CALL、1 条 world carry/backedge：2 个数组条件 carry、1 个字段有 caller 重写路径，其余4个标量保留不变性边界。18 条定义/匹配路径关系双向可查，58 个 effect 前沿具体保留。
- world 不是数组内容版本，Frame 不是运行时 InvocationID，引用路径相同不证明跨调用内容相同；循环不逐次展开，不授权释放外部存储或无条件外提。

上述表均为分片 typed 记录及 ID/FactRef 引用，读回重放原源码和接口，未保存 Tensor 数值。

### 区间规划进入 compile-build

106 个硬件派生 CPU 方案与原执行全部保留。已有完整成本为49项（含 NO_OP），58项仍没有完整成本。固定组件模型中，未知项以非负区间表示，**不填零成本 profile**。

从原辅助 component FactRefs 重算必需工作下界：31个未知方案可严格排除，27个仍可能优于已知方案。通用区间 solver 另排除48个更贵的已知方案，完整107项仍可追溯；当前结果为 UNRESOLVED / NO_OP，不宣称物理或全程序最优。

`compile-build --stateful-interval-plans MAP.json` 已实际走通：独立读回主图，验证旧/新域的原材料 URI 别名与全部107个 context 等价，再消费 interval plan。命中该路由后不重复跑标量成本生成或标量 choice。

C0 独立重建原 facts 和整个区间选择；选中分支还需重放完整 cost、原数值/效果图、CPU、shape 和预算，再进入与原标量选择共用的 C++20 recipe lowering。当前真实区间主链验证了未决回退；新选中分支完成静态接口检查，共用 C1 的正例通过实际 native 检查，不能混称区间选优已经在真实数据完成。

CPU/cache 足迹也已有结构化推导：共享缓存域、SMT、按 storage 合并的访问区间及未知对齐 phase 的 cache-line 上下界；这些几何不能直接当命中率、驻留、带宽或耗时。

### 统一最终交付

已有 compact DIRECT/COMPOSITE whole-loop 与 stateful 事务区域接入同一 final-entry 核心。沿用原 C++20、算子/状态 ABI1、ELF、原来源证明及 guard，不新增数学后端。

- whole-loop 普通入口：显式非计时验证 native1；输出与原入口一致，深层 callee 改绑时整循环回退一次。
- Gray–Scott 普通入口：512网格/4步，显式验证 native4；返回、u/v/dt、数组对象绑定和生成文件与原执行一致，改绑时整方法回退一次。
- 两类默认均 NO_OP；禁止导入 compiler/analysis/IR/planner 的独立进程仍可运行 compact 包。
- 当前成本槽是 UNMEASURED，只允许显式验证，不导出 RELEASE，不用旧正性能记录给新路径授权。完整入口及兼容/I/O 仍由 Python 保留。

报告分别在 `artifacts/current/experiments/whole-loop-delivery-current-v1/`、`stateful-final-delivery-v1/`；旧的单 profile 验收仅作历史材料。

## 接下来仍需开发

1. 建图：已能在闭合 caller 内替换形参内容根并连接共享 leaf；继续把有写入的条件 state-path 推进为跨调用逻辑值/版本，扩大 callback、对象协议与外部算子效果摘要。相同引用仍不证明跨激活内容相同。
2. 化简：已有纯数组多区域 CPU waves 与 pool hazard 约束；继续补联合内存/放置/就绪调度、NUMA、多进程和成本事实。当前固定 pool/显式 CPU 预览不是全程序 makespan 优化，未支持 MPI 或 GPU offload。
3. 交付：扩大可降低的数据类型、控制/对象和外部接口，统一更多已准入 artifact。当前不是任意 Python 的独立 C++ 主程序；Sunfish/Cirq 未获得完整编译闭环，当前也没有 GPU/LeWM 新验证。

当前首先补核心能力与接口，之后才恢复新路径的性能准入。历史 A/B、失败模型和诊断材料保持原样，详见既有 baseline JSON；不将它们改写为当前版本结果。
