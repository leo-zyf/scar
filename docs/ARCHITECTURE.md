# SCAR 宏观编译器架构

三个阶段共享原始身份与证据，分层开发：建模建图 → 结构/资源规划 → C++20交付。完整类型见[MODEL](MODEL.md)，实际范围见[STATUS](STATUS.md)。

## 1. 前端与统一图

形式前端按材料依赖推进：F0源码字节/版本 → F1词法 → F2语法 → F3作用域/绑定/声明 → F4操作、值、world及控制端口。AST只用于语言前端，后续使用有类型图。
局部函数、循环和区域正文共享；跨模块调用通过原actual/formal、返回、world和匹配Frame接线。静态不能确定的目标可由有限动态CodeID选择补充，观察结果不代表未执行分支也已证明。

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

成本接口分为完整点成本与区间。`stateful_cost_bounds`从已核验必需component重算下界，未知项保留非负无穷区间；`interval_choices`核同contract/scope/tick，严格支配才排除，未决项不消失。它是显式模型的比较，不是物理执行时间定理。
`compile-build --stateful-interval-plans`通过严格域等价与原fact重放接入这一路，旧的标量选择继续受支持。

CPU/cache需求使用原CpuCore/CacheDomain及storage-scoped访问区间，保留SMT、对齐未知和层级。访问足迹不等于峰值驻留或缓存失效率；指令调度模型也不能替代这一层。多区域异步/跨设备联合调度仍需继续开发。

闭合数组 caller 已接入 `caller_pipeline_values → pipeline_schedule → pipeline_parallel_v1`：复用原参数/返回和共享 leaf，先推导数据依赖及 pool 的全部旧消费者 WAR，再生成有界 CPU waves。现有 v2 typed task graph 保留分配、就绪、成功 fence、释放及 fresh export；每任务使用原硬件线程引用。时间保持 UNKNOWN，独立 exact solver 尚未替它建立物理最优性。

## 3. C++20与运行交付

C0先验证选择及其源码/成本/CPU/shape/预算条件。原标量与区间选择共用 `selected_stateful_lowering`，再经原 `ExecutionRecipe` / task graph进入既有C++20代码生成；算子与stateful ABI均为v1。接口未知时保留原执行。

运行产物按原合同区分：普通数组caller、DIRECT/COMPOSITE whole-loop、stateful事务区域。不同profile不互相强制转换，不因表面输入相似而借用证明。
`final_entry_capsule`为loop/stateful提供同一个最终准入核心，profile adapter复用原compact loader、ELF与guard。构建侧固定payload和源码ledger，4KiB slot保成本/验证角色，运行端不重放庞大源码图或benchmark原始证据。

当前只导出UNMEASURED验证slot，默认NO_OP，显式验证仍保完整区域回退。源码/默认值/实际callee、observer、输入metadata/bounds及必要状态分别守卫；私有结果不提前发布。入口及兼容/I/O仍由Python执行，尚非任意程序的独立C++主程序。

状态循环另走 `stateful_loop_frontend → stateful_loop_schedule → compile-stateful-loop → stateful_loop_runtime`。先证明原 F4 一轮的 CALL/控制/None 返回，再存一份 step 和 `(i,e)` 内容递推，独立重放后复用既有 stateful/repeat ABI1。compact trial 显式准入，`completed=k` 后只执行原 Python 后缀，并按原参数布局重新查找每轮 callee；保留每步分配与提交，不假定整循环可以整体回退。默认入口不导入 NumPy/guard、不加载 ELF；安装 wrapper 不提前求值延迟注解。

## 存储与并行

源码单元、共享正文、关系、状态及资源材料各自分片，typed记录用ID与FactRef连接。manifest是引用目录，不能仅靠其hash证明语义；reader按原材料和当前规则重推。按需展开不复制每次调用或循环迭代。
前端与独立子图可按单元并行，共享有界worker预算；动态采集只保必要事件，离线分析负责接线。模型推导、观测、成本、合法性与部署权限分开记录，避免把采集干扰当目标性能。
