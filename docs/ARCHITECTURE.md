# SCAR 宏观编译器架构

三个阶段共享原始身份与证据，分层开发：建模建图 → 结构/资源规划 → C++20交付。完整类型见[MODEL](MODEL.md)，实际范围见[STATUS](STATUS.md)。

SCAR 前端从源码、运行绑定和目标硬件构建程序 IR，记录算子的提供方、重载、数据端口、效果和资源要求。中端化简图并决定批次、数据位置、搬运、工作区和资源时序。C++20 / ABI v1 后端生成执行这些计划所需的调用编排、同步和回退代码。现成的外部提供方负责数值内核。缺少满足合同的提供方入口时保留原执行。

## 1. 前端与统一图

形式前端按材料依赖推进：F0源码字节/版本 → F1词法 → F2语法 → F3作用域/绑定/声明 → F4操作、值、world及控制端口。AST只用于语言前端，后续使用有类型图。
局部函数、循环和区域正文共享；跨模块调用通过原actual/formal、返回、world和匹配Frame接线。静态不能确定的目标可由有限动态CodeID选择补充，观察结果不代表未执行分支也已证明。

`trace --execution-evidence --torch-call-scopes` 在项目源码或显式采集计划的原生 CALL 上打开 RecordFunction，由同帧、同位置、同目标的 CPython `C_RETURN` 或 `C_RAISE` 关闭。`import-capture` 使用原始调用索引、ETO 父子关系及 Kineto RF ID 接线。一次调用可包含多个直接观测到的原生算子，多次执行同一源码位置各有独立身份。调用范围证据尚未证明实参映射、状态效果或设备完成。

`plan-aten-capture` 先冷重放 F4 字节码位置、实参和状态边，再解码已绑定的提供方端口。原 F4 路径保存共享正文中的引用序号。每个正常返回的原始 CALL 对应一个规划操作，可保留多个已观测的直接算子。原状态顺序作为前驱约束。源码引用和原生描述相符时，仍要证明包装器映射、内容与就绪，才能加入物理数据边。CPU 参与、内存放置或可比较成本缺失时保留原执行。

公共接口为 `ProgramView` / `ProgramDependencyGraph`：`get`、`original`、`expand`、`incoming/outgoing` 与来源解释。Operation、Value、StateLocation、Region、Resource是核心实体；Use/control等辅助端点有明确role，Contract与Measurement保留独立来源。
Control、数据依赖、状态版本、别名和资源映射是不同关系，不能将world carry当数组内容phi，也不能把相同storage或引用路径当相同逻辑内容。

`plan-build`在同一个原 `BuildSession` 自动接入有序数组profile：

- 原源码重放的正文、显式读写和局部内容epoch。
- 精确原CALL选择的caller/内联ABI与Frame链。
- 输入直接使用、局部last use、正常出口必需版本。
- 原共享循环/header/backedge与受条件约束的state-path carry。

实现为 `compiler/ordered_planning_graph.py`；C0消费时独立重建原source/body/activation/lifetime/loop各层，逐typed表及公共图关系核验。未知效果保留具体field/frontier，不能自动当纯或dead。

`source_selected_call_window` 在无回边的 wrapper 内，用原 CALL/RETURN 配对栈及同次条件求值的互斥分支证据界定效果窗口；它只排除无法与选定 CALL 共现的操作，不把保留效果改成 must/pure。typed 窗口随 loop-transfer 分片保存并冷读重推。

硬件由独立 `TargetHardwareGraph` 保全平台库存和连接，软件图只引用它。资源存在、访问约束、后端能力、实际使用分别表达；只保留可由程序/优化后端管理的资源抽象，不把不可控制的OS内部实现展开为优化节点。

## 2. 优化与资源计划

先建立输入、状态、输出与效果合同，再选择可替换区域。现有结构规则包括循环不变量切分、融合/物化选择、buffer及有序事务计划；重复外观本身不能授权复用。

`plan-build`保存原region/proof与硬件域；数据保持原FactRef、源digest、shape、预算、ABI和guard条件。执行recipe显式表示分配、计算、复制、就绪/屏障及提交；这些任务不是源码语义图的替代品。

联合规划的目标是最短完工时间：先确认语义依赖与可替换边界，再确定实现/存储/资源域，随后约束开始时间、容量和数据就绪，最后把选定时间表落实为可验证执行证书。现有 `resource_solver` 可搜索 v2 task 图；完整 task 成本、外层存储布局域和 solver-order 到 runtime 的桥接仍需接通，UNKNOWN 时长不得补零。

成本接口分为完整点成本与区间。`stateful_cost_bounds`从已核验必需component重算下界，未知项保留非负无穷区间；`interval_choices`核同contract/scope/tick，严格支配才排除，未决项不消失。它是显式模型的比较，不是物理执行时间定理。
`compile-build --stateful-interval-plans`通过严格域等价与原fact重放接入这一路，旧的标量选择继续受支持。

CPU/cache需求使用原CpuCore/CacheDomain及storage-scoped访问区间，保留SMT、对齐未知和层级。访问足迹不等于峰值驻留或缓存失效率；指令调度模型也不能替代这一层。联合放置和跨设备调度仍需继续开发。

闭合数组 caller 已接入 `caller_pipeline_values → pipeline_schedule → pipeline_parallel_v1`：复用原参数/返回和共享 leaf，先推导数据依赖及 pool 的全部旧消费者 WAR，再生成有界 CPU waves。现有 v2 typed task graph 保留分配、就绪、成功 fence、释放及 fresh export；每任务使用原硬件线程引用。时间保持 UNKNOWN，独立 exact solver 尚未替它建立物理最优性。

可选的 `pipeline_ready_schedule → pipeline_ready_v1` 使用相同 RAW/WAR/WAW 和固定 pool，在每 CPU lane 上扫描已就绪任务；CPU 互斥是容量约束，不变成永久语义边。完成即通知真正后继，最终 export/回退前才统一 join；失败停止领取，保留所有运行中任务的存储。原 caller 编译与 compact runtime 共用既有 ABI1，未添加数值后端，也不把可执行计划等同最快计划。

`pipeline_partial_order_memory` 从原 RAW 传递闭包构造逻辑值干涉图，以有界精确 coloring 求固定槽容量和。复用须由旧 producer 与全部 consumers 到新 writer 的严格 RAW happens-before 证明；source-index last-use 仅作诊断。该模型保留并行自由度，可能比 source-order pool 占用更多 payload；完整、搜索耗尽和预算不可能分开记录。统一 memory reader 按独立 schema 重放，ready/wave/C++ 准入核对对应完整状态，原 ABI、模板和私有输出回退不变。

### 编译器与数据库的图接口

1. **先核对图改写的合法性。** 借鉴编译器的分层降级和 SSA 的定义引用链。原图保留源码、调用和效果身份。改写以区域端口为边界，需要核对返回值、别名、状态、异常和数值合同。存储复用沿数据图检查旧版本的全部读取，控制与效果边约束重排。验收时由原图独立重放改写证据。 [MLIR 降级合法性](https://mlir.llvm.org/docs/DialectConversion/)、[缓冲化](https://mlir.llvm.org/docs/Bufferization/)、[副作用](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/)。

2. **从出口要求倒推物理计划。** 借鉴 Cascades，只有已证明符合相同区域合同的逻辑改写才共用候选记录。物理候选按目标设备、布局、位置和就绪要求向上游传播约束。拷贝、布局转换和同步显式进入任务图，时间表同时核对关键路径和资源容量。成本证据保留作用域和不确定性，再将提名方案送入实测。当前 ATen 可在同一数据区域比较分组与就绪调度，跨区域共享候选记录及跨设备属性传播待接入。 [Cascades](https://15721.courses.cs.cmu.edu/spring2019/papers/22-optimizer1/graefe-ieee1995.pdf)。

3. **用执行器兑现任务图。** 借鉴数据库按数据就绪和 NUMA 局部性调度管线的做法。证书给出数据依赖、任务所在资源、存储和发布出口。C++20 适配器调用现有算子，消费者等所读版本真正完成后运行。显式的汇合、物化或外部效果形成执行边界。当前 ATen 支持同 NUMA 和有页证据的跨 NUMA CPU 通道，主机与加速器之间的完成事件、分批流水和有界运行时调整仍需验收。 [Morsel-Driven Parallelism](https://db.in.tum.de/~leis/papers/morsels.pdf)。

4. **把版本和可见性贯穿规划及执行。** 借鉴 MVCC 的快照和提交。计划绑定可核对的图版本、读写集合和发布条件。区域内结果先由自身消费者读取，出口守卫完成后向原调用方发布，旧版本在最后一个读取者完成后可回收。当前 ATen 具备只读输入、私有结果和最后读取释放。后续写区域须有显式的冲突验证与提交协议。 [PostgreSQL 快照隔离](https://www.postgresql.org/docs/current/transaction-iso.html)。

## 3. C++20与运行交付

固定网格 Tensor 循环与 `torch.export` 的 CPU 形状特化共用 `ProgramGraph`、物理问题记录、CMake 生成和独立验收。固定网格图含外层回边、共享步体与可展开 RHS，步体未测成本保持 UNKNOWN。原生稀疏流通过既有证书保留 ABI，并复用 CMake 生成和构建。跨设备中转可按已资格化的固定端点复制能力展开多跳队列活动和存储租约，目前仅验了模型。

C0先验证选择及其源码/成本/CPU/shape/预算条件。原标量与区间选择共用 `selected_stateful_lowering`，再经原 `ExecutionRecipe` / task graph进入既有C++20代码生成；算子与stateful ABI均为v1。接口未知时保留原执行。

数值算子的交付条件是具备现成的原生提供方入口。建图保全原调用及重载，规划按目标设备、dtype、布局、别名与副作用合同核对可用入口，并纳入调用、同步、临时存储和跨设备搬运成本。代码生成核验所选提供方的版本和链接符号，再生成 C++20 调用、调度、数据搬运及必要的守卫。融合等改写也按这个规则交付，缺少等价的原生入口时可继续编排已核验的原算子，或者保留原执行。

稀疏求解以原 SciPy 调用为源码绑定和行为对照。IR 保留 SuperLU 符号状态和每轮 CSR 数值输入，C++20 产物通过原生 ABI 调用同版本 SuperLU 和 OpenBLAS。独立流入口的运行依赖为包内原生库与系统 C/C++ 运行库。兼容 Python 入口另行保留首次比对及回退。独立的 `cpu-cost` 可选组使用 `scipy.optimize` 拟合成本与求线性规划界限。

`link-aten-provider` 在构建时核对已安装的 LibTorch 修订和完整算子 schema，通过 C++20 编译及链接验证实际重载，并输出可直接引用的类型化调用。编译期断言核对生成头里的声明。产物保留目标库的路径及 SHA256，当前范围是本机安装的分发包。内核放置、调用上下文、端口映射、效果和收益仍需另行证明。

`link-aten-capture --call-attempt N` 可从导入模型冷重放原始调用范围，直接提取其中实际观测到的算子声明。相同重载共享类型化 C++ 入口，每次原始调用和算子事件仍各自保存。也可用 `--eto-node '#/nodes/N'` 选定原始 ETO 事件。

`discover-aten-regions` 一次盘点同次采集的原生调用，将同一源码身份和算子序列的多次激活归组，再核对完整函数和显式包装器合同。唯一闭合区域可投影为现成提供方调用。投影保留原绑定，进一步使用须通过成本和原入口验证。

`compile-aten-entry` 重放发现、源码和目标图摘要，以源码参数名接收校准样本。默认交付是独立 C++20 区域，直接链接已核验的 LibTorch CPU 库。可以交付的串行路径仍由原生库执行。`--entry-function` 运行单独的原 Python 差分实验。

`compile-aten-entry --compare-ready` 在成本模型选定的同一放置上验证分组和就绪候选。原生串行基线用于区域准入。完整入口按轮启动原程序和候选，使用同一源码、参数、守卫协议及 CPU 掩码，核对输出和实际调用覆盖。挑战者须在完整进程边界达到相对原程序和已入选方案的增益门槛。候选试验记录和最终入口选择都可独立重放。

三核分组路径沿用相邻窗口。规划从 ISA 兼容的物理核心中提名有序代表组，按各核心服务时间、交接和 Tensor 容量比较分段时间表。跨 NUMA 组合要求页采样完整，并沿原 Tensor 依赖使用该线程和页位置上下文的实测耗时。跨节点首选与可行同节点方案的成本采样范围重叠时，保留同节点方案及排序未决证据。C++20 固定两条工作线程并在分段完成后统一释放到期句柄。

并发校准按合法窗口组合实测派发至汇合的间隔，并沿 Tensor 依赖测量同次回放中的串行后继算子。规划器保留实际汇合后与独立消费者校准的观测范围。页采样、几何与字节检查在计时外执行。内存控制器带宽与并发执行顺序的独立归因保留为未测项。

`compile-aten-graph --ready-dag` 和 `plan-aten-ready` 接入三核路径。前端保留原 Tensor 生产和消费端口，并从完整数据图证明存在多个可同时就绪的操作。校准在各核测量区域内每个合格算子的服务时间。中端按依赖和成本派发算子，证书区分数据边与核内顺序。ABI 2 枚举各核的可达完成前缀，按可能同时启动的算子核算张量存储上界。枚举超预算时使用全保留上界。C++20 在最后一个实际消费者结束后释放中间句柄，并在返回前汇合。原分组在存在合格窗口时作为独立配对基线。

`plan-aten-ready --joint-order` 在已绑定的同 NUMA 核组内联合求算子位置与核内顺序，优化目标为采样服务成本下的完工时间加派发开销。按拓扑次序重算时间表，再用所有可达完成前缀验证存储上界。新次序通过原序就绪、串行及可用分组基线的配对准入后，由独立 C++20 执行器运行。图证书保留次序，执行器启动前核对排列及依赖。模型的下界仅对应固定核组和成本范围。

跨 NUMA 就绪调度使用两个步骤。先用独立服务时间提名固定核分配，再沿原图按该分配重放，测量已包含输入页位置影响的算子服务时间。证书把每条读取边与其样本页位置、消费者 CPU 和 NUMA 节点相连，底层仍是同一可访问的主机存储。页采样和内容核验在计时外进行。并发收益由选中计划的区域和整入口运行验证。

可选的输入副本规划保留同一逻辑值的两种物理读取。直接读取沿用原边。副本路线为目标核加入一次显式复制，核内消费者改接副本，其余消费者继续读取原存储。证书计入拷贝和保留到返回的页对齐容量。执行器在目标核调用 LibTorch `copy_` 首次写入匿名页，校准单独核对目标驻留及等价。收益门槛同时检查关键路径估计与完整区域配对实测。

`plan-aten-ready --try-retained-input-replica` 还可评估跨调用保留同一个私有副本。目标核校验输入几何和完整字节，未命中时释放旧副本再复制。时间表为热命中编排校验和消费者，并在入口预留副本容量。首调用单独实测，回收期以其余调用全部命中为前提。工作负载真实命中率仍待入口观测，采用前需验证完整入口。

校准在计时外对每份 Tensor 存储最多查询 32 页，将采样节点绑定到原硬件图和消费者依赖。对于最多八种非重叠窗口组合的完整集合，按各组合重放操作与后继数据流，分别测量执行位置对应的算子成本。规划依据组合成本、交接时间和容量选择时间表。未知页保留主线程基准成本，组合超出采样预算时沿用独立窗口成本并保留范围说明。

旧数组融合、状态区域和显式 SSE2 试验仍含自产数值循环。这些路径保留为早期实验，正式交付需要迁移到等价的现成算子入口并重新验证。

运行产物按原合同区分：普通数组caller、DIRECT/COMPOSITE whole-loop、stateful事务区域。不同profile不互相强制转换，不因表面输入相似而借用证明。
`final_entry_capsule`为loop/stateful提供同一个最终准入核心，profile adapter复用原compact loader、ELF与guard。构建侧固定payload和源码ledger，4KiB slot保成本/验证角色，运行端不重放庞大源码图或benchmark原始证据。

当前只导出UNMEASURED验证slot，默认NO_OP，显式验证仍保完整区域回退。源码/默认值/实际callee、observer、输入metadata/bounds及必要状态分别守卫；私有结果不提前发布。入口及兼容/I/O仍由Python执行，尚非任意程序的独立C++主程序。

状态循环另走 `stateful_loop_discovery → stateful_loop_frontend → stateful_loop_schedule → compile-stateful-loop → stateful_loop_runtime`。由原 CALL/body/完整 CodeID 自动发现唯一 step，独立证明原 F4 一轮的 CALL/控制/None 返回，再存一份 step 和 `(i,e)` 内容递推；降低前再次重放，复用既有 stateful/repeat ABI1。compact trial 显式准入，`completed=k` 后只执行原 Python 后缀，并按原参数布局重新查找每轮 callee；保留每步分配与提交，不假定整循环可以整体回退。默认入口不导入 NumPy/guard、不加载 ELF；安装 wrapper 不提前求值延迟注解。

## 存储与并行

源码单元、共享正文、关系、状态及资源材料各自分片，typed记录用ID与FactRef连接。manifest是引用目录，不能仅靠其hash证明语义；reader按原材料和当前规则重推。按需展开不复制每次调用或循环迭代。
前端与独立子图可按单元并行，共享有界worker预算；动态采集只保必要事件，离线分析负责接线。模型推导、观测、成本、合法性与部署权限分开记录，避免把采集干扰当目标性能。
