# 当前状态

当前按 G1→G6 落实时间模型：G1 同一 IR 的事件/活动/占用/约束 → G2 真实资源域映射 → G3 有来源的成本区间 → G4 放置/并发/内存/时间联合规划 → G5 C++ 执行证书 → G6 原入口正确性与 clean A/B。以下旧 L2–L4 结果只是已有有限声明模型的证据，不代表 G2–G6 已完成；未提交的旧 C++ solver-order 试作暂不扩展。

G1 已验：v3 时间字段及 `scar time-schedule` 接入现有 IR，v2 字段与哈希保持不变。CPU pipeline 为 16 tasks/41 events/5 owned slots；真实 Gray–Scott 为 20/54/4；状态循环保留一份 9-task 正文、27 events 及 distance=1 依赖。三份保存图均通过独立来源重放，Gray 原 F4 主链另经32.69s冷读确认；7项定向检查通过。耗时/服务需求未知项保留 UNKNOWN，旧求解器不得丢弃新约束；没有新目标执行、标定、加速比或部署授权。验收：`artifacts/baseline/g1-semantic-time.json`。

G2 资源接线已验，尚有显式未决字段：`bind-time-resources` 将每个占用 ID 接入原 snapshot；pipeline 的5个计算活动精确绑定原线程，其余11个需求单位仍未知，候选域保留56线程/28共享物理核。Gray/状态循环没有借用所选计算 CPU 冒充宿主活动的绑定。三源分别5/4/2份 owned slot 各计一次，HOST 候选按 CPU_ALLOCATE 合同推出，原预算保留；未证明页驻留、可用 RAM 或吞吐。三源独立冷读及9项定向检查通过，报告：`artifacts/baseline/g2-temporal-resources.json`。接下来补 G3 的活动资源需求和有来源的真实成本区间，再接联合求解；G4–G6 未完成，无本轮新加速比。

当前优先完成三阶段核心开发；暂停新增性能实验。已在受支持的 CPU 数组、循环和事务区域贯通主链，**通用覆盖尚未完成**。不能把正确回退、单个区域编译或历史性能记录当作整个系统已经完成。

当前严格按层推进：L1 原图/资源输入 → L2 成本与约束投影 → L3 联合求解 → L4 执行证书 → L5 C++ → L6 原入口。L2 源码主链已验：16项时长和11项控制占用默认 UNKNOWN，2560B增量pool与external分账，矛盾shape拒绝，禁止solver/codegen。L3 `solve-pipeline-temporal` 已重放显式有限域：声明模型的两布局得21/13 ticks，五CALL顺序例得13并匹配下界；UNKNOWN case保留并阻止闭域选择，独立域收据防裁剪。上述ticks均非测量。报告在 `pipeline-solver-order-v1/l2-acceptance.json`、`l3-acceptance.json`。L4以后仍须逐层验收，试做文件不算完成。

L4 已验：`refine-pipeline-temporal` 消费整个已闭合原域，输出 CPU0 的 C→A→E 与 CPU1 的 B→D；未知域拒绝，篡改顺序拒绝。证书保原引用，不复制大正文，只兑现计算顺序，不声称完整控制时间线或原生13ticks。后续进入 L5，再验原生执行是否真正服从证书，见 `l4-acceptance.json`。

最新主链：`compile-caller-pipeline --pipeline-workers N --pipeline-dispatch ready --pipeline-memory-order dependencies` 将原 RAW 图、逻辑值生命周期和内存分配一起规划。仅当原 RAW 已证明旧 producer 和所有 consumers 严格先于新 writer 时复用，禁止覆盖当前调用输入；因此不为复用新增先后关系。原 source-order pool 和新 partial-order pool 使用不同 schema，均独立重放来源、形状、分配、完整搜索与预算。

真实源码用例的增量 payload 从1536B变为2048B，解除无关 B→C 等待；1536B预算下新策略被正确拒绝。冷读、C++20、compact 和原入口均通过，正常/输入变化/只读别名结果逐位一致，溢出保留私有结果并回退一次。原 host 的独立 ABI 握手确认 B 未结束时 C 已启动；该诊断不构成数学/source 准入或性能证据。报告在 `artifacts/current/experiments/pipeline-partial-order-v1/`。

此前的 ready 执行已去掉整 wave barrier，完整浮点环境和调用方 affinity 恢复通过，见 `pipeline-ready-v1/`。CPU lane 仍是显式未排名分配；新 coloring 只证明 RAW-only 固定池模型内最省容量，不证明最快。成本 UNKNOWN、默认 NO_OP。

`compile-stateful-loop` 现在可省略 `--step`：沿原 CALL/body/CodeID 自动找到唯一状态区域，再独立重放共享循环和参数接线。自动发现至 C++ repeat adapter、compact 和原入口四步已验证，报告在 `stateful-loop-auto-v1/`。写入数组显式携带 `(iteration, local_epoch)` 内容，固定引用不当成固定内容；此前已验证完成1步后失败只恢复剩余3步，不重跑已提交前缀，见 `stateful-loop-v1/`。

该 profile 限定每轮一个普通 global CALL、固定参数和无中间观察；保留每步分配、线程组和提交边界，尚未外提 workspace/team。成本 UNKNOWN、默认 NO_OP，无新性能或最优性结论。

此前的 `compile-caller-pipeline --pipeline-workers N` 自动发布原 caller 的参数替换/数据图，重放 pool 生命周期，生成 RAW/WAR/WAW、实际 CPU 与静态 wave 计划，再交既有 ABI1 算子/C++20 host。每个 leaf 单核，wave 内 CPU 不重叠；全部 worker join 后才能推进或回退。语义依赖图没有硬化 wave 的额外 barrier。

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
- 自动连接 1 个共享循环、2 层原 CALL、1 条 world carry/backedge：2 个数组条件 carry、1 个字段有 caller 重写路径，其余4个标量保留不变性边界。18 条定义/匹配路径关系双向可查。最新原 CALL 窗口以配对控制栈和互斥分支证据排除4个无关操作，effect 前沿58→55；剩余未知效果不自动清除。主链已持久化并独立冷读，未重跑目标程序。
- world 不是数组内容版本，Frame 不是运行时 InvocationID，引用路径相同不证明跨调用内容相同；循环不逐次展开，不授权释放外部存储或无条件外提。

上述表均为分片 typed 记录及 ID/FactRef 引用，读回重放原源码和接口，未保存 Tensor 数值。

stateful 类型推导改为固定两遍原 F4：先收集数组读写要求，再严格建图，修复先出现 `state * factor` 时错误锁定 scalar 的问题。原 Gray–Scott 区域重建 digest 一致，不修改 testcase。

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

1. 建图：已连接封闭状态循环的参数化内容递推；继续补有写入的条件 state-path、callback、对象协议与外部算子效果摘要，再扩大原循环替换证明。相同引用仍不证明跨激活内容相同。
2. 化简：已有两种有明确等待/容量权衡的 pool 模型。现有 `resource_solver` 已支持 v2 图和最短 makespan，但 ready 主链仍缺完整 task 成本、精确预算映射及 solver 顺序到 C++ 的执行证书；必须先接通这些，再进行联合内存/放置/时间选择。NUMA、多进程、MPI 和 GPU offload 尚未完成。
3. 交付：扩大可降低的数据类型、控制/对象和外部接口，统一更多已准入 artifact。当前不是任意 Python 的独立 C++ 主程序；Sunfish/Cirq 未获得完整编译闭环，当前也没有 GPU/LeWM 新验证。

当前首先补核心能力与接口，之后才恢复新路径的性能准入。历史 A/B、失败模型和诊断材料保持原样，详见既有 baseline JSON；不将它们改写为当前版本结果。
