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

## 3. C++20与运行交付

C0先验证选择及其源码/成本/CPU/shape/预算条件。原标量与区间选择共用 `selected_stateful_lowering`，再经原 `ExecutionRecipe` / task graph进入既有C++20代码生成；算子与stateful ABI均为v1。接口未知时保留原执行。

稀疏求解以原 SciPy 调用为源码绑定和行为对照。IR 保留 SuperLU 符号状态和每轮 CSR 数值输入，C++20 产物通过原生 ABI 调用同版本 SuperLU 和 OpenBLAS。独立流入口的运行依赖为包内原生库与系统 C/C++ 运行库。兼容 Python 入口另行保留首次比对及回退。独立的 `cpu-cost` 可选组使用 `scipy.optimize` 拟合成本与求线性规划界限。

`link-aten-provider` 在构建时核对已安装的 LibTorch 修订和完整算子 schema，通过 C++20 编译及链接验证实际重载，并输出可直接引用的类型化调用。编译期断言核对生成头里的声明。产物保留目标库的路径及 SHA256，当前范围是本机安装的分发包。内核放置、调用上下文、端口映射、效果和收益仍需另行证明。

`link-aten-capture --call-attempt N` 可从导入模型冷重放原始调用范围，直接提取其中实际观测到的算子声明。相同重载共享类型化 C++ 入口，每次原始调用和算子事件仍各自保存。也可用 `--eto-node '#/nodes/N'` 选定原始 ETO 事件。

`discover-aten-regions` 一次盘点同次采集的原生调用，将同一源码身份和算子序列的多次激活归组，再核对完整函数和显式包装器合同。唯一闭合区域可投影为现成提供方调用。投影保留原绑定，进一步使用须通过成本和原入口验证。

`compile-aten-entry` 重放发现、源码和目标图摘要，以源码参数名接收校准样本。较宽的 CPU 域从拓扑提名有序核心对，按两侧算子服务时间、交接和峰值 Tensor 容量计算时间表。整段配对验证只执行模型选中的放置。交付具备回退的 C++20 桥接。完整入口 A/B 的两侧共享所选执行掩码。

三核路径要求三个相邻、数据独立且效果合同允许重叠的调用。规划从 ISA 兼容的物理核心中提名有序代表组，按各核心服务时间、交接和 Tensor 容量比较分段时间表。跨 NUMA 组合要求页采样完整，并沿原 Tensor 依赖使用该线程和页位置上下文的实测耗时。跨节点首选与可行同节点方案的成本采样范围重叠时，保留同节点方案及排序未决证据。C++20 固定两条工作线程并在分段完成后统一释放到期句柄。

并发校准按合法窗口组合实测派发至汇合的间隔，并沿 Tensor 依赖测量同次回放中的串行后继算子。规划器保留实际汇合后与独立消费者校准的观测范围。页采样、几何与字节检查在计时外执行。内存控制器带宽与并发执行顺序的独立归因保留为未测项。

校准在计时外对每份 Tensor 存储最多查询 32 页，将采样节点绑定到原硬件图和消费者依赖。对于最多八种非重叠窗口组合的完整集合，按各组合重放操作与后继数据流，分别测量执行位置对应的算子成本。规划依据组合成本、交接时间和容量选择时间表。未知页保留主线程基准成本，组合超出采样预算时沿用独立窗口成本并保留范围说明。

旧数组融合、状态区域和显式 SSE2 试验仍含自产数值循环。该部分需要迁移到现成算子调用，再进入按此边界验收的交付路径。循环调度、ABI 适配、分配和数据搬运属于 SCAR 的代码生成职责。

运行产物按原合同区分：普通数组caller、DIRECT/COMPOSITE whole-loop、stateful事务区域。不同profile不互相强制转换，不因表面输入相似而借用证明。
`final_entry_capsule`为loop/stateful提供同一个最终准入核心，profile adapter复用原compact loader、ELF与guard。构建侧固定payload和源码ledger，4KiB slot保成本/验证角色，运行端不重放庞大源码图或benchmark原始证据。

当前只导出UNMEASURED验证slot，默认NO_OP，显式验证仍保完整区域回退。源码/默认值/实际callee、observer、输入metadata/bounds及必要状态分别守卫；私有结果不提前发布。入口及兼容/I/O仍由Python执行，尚非任意程序的独立C++主程序。

状态循环另走 `stateful_loop_discovery → stateful_loop_frontend → stateful_loop_schedule → compile-stateful-loop → stateful_loop_runtime`。由原 CALL/body/完整 CodeID 自动发现唯一 step，独立证明原 F4 一轮的 CALL/控制/None 返回，再存一份 step 和 `(i,e)` 内容递推；降低前再次重放，复用既有 stateful/repeat ABI1。compact trial 显式准入，`completed=k` 后只执行原 Python 后缀，并按原参数布局重新查找每轮 callee；保留每步分配与提交，不假定整循环可以整体回退。默认入口不导入 NumPy/guard、不加载 ELF；安装 wrapper 不提前求值延迟注解。

## 存储与并行

源码单元、共享正文、关系、状态及资源材料各自分片，typed记录用ID与FactRef连接。manifest是引用目录，不能仅靠其hash证明语义；reader按原材料和当前规则重推。按需展开不复制每次调用或循环迭代。
前端与独立子图可按单元并行，共享有界worker预算；动态采集只保必要事件，离线分析负责接线。模型推导、观测、成本、合法性与部署权限分开记录，避免把采集干扰当目标性能。
