# 开发与用法

使用现有 `conda activate lewm`，只在 `~/AAA/scar` 的 `main` 开发。
不创建 worktree、日期版源码或历史实验副本；正常提交 Git，不 reset。
不修改外部 LeWM、数据/权重、CUDA/driver 或整套环境。删除实现前先查引用。

三阶段并行：主代理维护交接契约/CLI及集成；三个agent分别负责前端与建图、图化简与资源规划、C++交付，
禁止同时编辑共享契约。先提交可用接口，再以真实外部科学程序的完整区域共同验收，不等待全图全部字段闭合。
当前共同契约为 `scar.compiler.model` 的 CPU array-region profile；完整源码图仍由原前端持有。
验收顺序：A 契约/ID与计划校验 → B 各阶段独立可执行 → C 真实科学区域贯通 → D 正确性/清洁A/B与NO_OP决策。
结果按 Observed/Implemented/Verified/Rejected 区分；不以区域加速代替整程序加速，也不以局部闭环代替第一阶段完成。

本轮贯通顺序为源码作用域与效果 → 化简后继图与成本 → 同版本的 C++20 区域交付。

| 阶段 | 本轮验收 | 下一验收 |
| --- | --- | --- |
| 建图 | 保留原 F4 接口及 `assert` 的执行分支。`compile-region --catalog C --assertion-policy omit-proven-debug` 仅在能证明条件及可达消息无额外可观察行为时使用原图的省略分支，并明确记录允许省略的 `AssertionError`。默认 `preserve`。 | 用外部 CPU 科学程序的实际入口绑定循环活入值、循环携带值和效果，记录尚未接通的调用及资源。 |
| 化简与计划 | `model-pipeline-cost --macro-preparation M --out COST` 消费实际后继图与优化后 lowering，保留同一合同及目标硬件引用。成本缺失时保持 `KEEP`。 | 按同一受保护入口分离 guard、调度、分配及执行成本，用独立输入验证冻结参数，再用时序与容量约束选择合法放置。 |
| 交付 | `plan-semantic-region --macro-preparation M --forward-model COST/result.json --compile-validation --out NATIVE` 核对同一候选后生成 C++20/ABI 1。`validate_call` 可在 `KEEP` 时显式隔离比较，普通调用保留原函数。 | 将通过成本资格和时序校验的计划用于真实入口，验收输出、可见状态和回退，再做清洁 A/B。 |

断言政策和五调用 CPU 数组区域分别通过验收。真实 Gray-Scott 入口同时重建了源码图、投影了 `_euler`
的 20 个内部任务，并用原驱动对比了原生试跑的状态轨迹、日志和输出文件。阶段门以各自的证据独立验收，
完整入口成本仍须校准，运行选择保持 `KEEP`。记录和 C++ 路径见 `artifacts/baseline/three-stage-fusion-binding.json`。

真实入口继续沿原 `entry-calls → source-summary → plan-controlled-region → deliver-controlled-stateful` 推进。
`plan-controlled-region --entry-calls CALLS --source-summary SUMMARY --ordered-graph GRAPH` 独立重放原调用、
状态生命期和共享循环证据。数组内容跨迭代传递、调用方改写的字段、只找到来源的活入值分别保留。
循环状态依赖证据约束相应范围的变换。完整运行时效果和跨迭代成本仍须验证。
`deliver-controlled-stateful` 复验同一计划、原生代码和调用包装器，显式验证时保留原驱动及可见行为对照。
`stateful-lifetime-training` 分开标定首调用与后续调用。新留出中的完整方法点估计误差降至 18.76%，
实测收益低于原 5% 门槛。下一关按固定范围定位首调用中的亲和性、分配及执行成本，再比较合法放置的时序。报告见
`artifacts/baseline/three-stage-real-entry.json`。

下一轮资源验收从原调用点、固定 C++ 任务和实际运行三个关口交接。限定调用点的 NumPy 附加采集
以对象身份核对公开和 C 扩展导出，`scar.analysis.numpy_ufunc_graph` 将候选操作及库文件接回原 F4。
`plan-controlled-region` 为现有串行顺序重放时序关系及槽生命周期，生成 `native-resource-certificate.json`。
证书使用事件顺序和字节上界，实耗时间仍由成本层求证。`deliver-controlled-stateful --validate --validate-resources`
在原驱动核对正常调用和移除计划 CPU 后的回退。诊断入口 `observe-stateful-entry-phases` 单独编译探针，
直接记录原 ELF 的亲和性及数组分配 API。下一关继续区分扫描、运算及提交阶段的时长，再比较改变资源安排的收益。
报告见 `artifacts/baseline/three-stage-resource-entry.json`。

循环改写沿 `ordered-graph → session-capacity → joint-capacity → C++ 验证` 推进。共享循环中的可变数据继续逐步读取。
容量方案在首个通过守卫的步骤申请私有槽，外层退出时释放。13 处私有读取均由本步写入覆盖，输入快照每步照常复制。
`joint-capacity` 计算两种模式的分配请求、峰值载荷和跨步占用。对于单区域完整点成本直接比较，等价时选原模式。
对于成本未知的区域，`--validation-mode CAPACITY_ONLY_DIRTY --export BUNDLE` 仅导出同一问题中的候选供显式试跑。
Gray-Scott 的四步短入口中，容量版把每步创建的四个槽改为整轮创建一次，整进程中位数为 1.158 s，原 Python 为 1.062 s。
下一关以源码初始化覆盖和顺序效果证明为前提，尝试减少快照复制，再核对完整调用成本。证据见
`artifacts/baseline/three-stage-loop-rewrite.json`。

单步快照化简先由 `scar.analysis.workspace_step_snapshot` 复验原读写版本和提交矩形，
再由 `plan-stateful-snapshot` 发布删去快照与重定向读取的改写、逻辑字节和固定容量。
`compile-workspace-borrowed-roots` 消费同一份计划。候选在提交前借读当前输入，完整写入私有提交区域，原有限值扫描和回退保留。
Gray-Scott 259×259 证明可省一个输入根。另一个根交给原 NumPy provider，仍保留快照。
真实 420 步入口已通过逐步、最终状态和输出验收。下一关用独立标定验证完整入口的跨配置成本，并完善 provider 的效果边界。
记录见 `artifacts/baseline/three-stage-snapshot-rewrite.json`。

入口选择先由 `snapshot-entry-calibration` 冻结两个训练配置和一个留出配置。训练后锁定预测，留出独立验收状态、输出和完整进程耗时。
`plan-snapshot-launch` 冷验原图、C++ 产物和成本资格。`run-snapshot-selection` 用发布的校验和检查本次参数、环境和资源，才采用获准的原生路径。
Gray-Scott 的 577 步留出通过验收，另两种 CLI 配置已验证源码守卫适用性。详细记录见 `artifacts/baseline/three-stage-snapshot-selection.json`。

### 稀疏区域的三阶段交接

1. 建图。`entry-calls` 将有界运行时调用绑定到原 F4 源体。`plan_csr_region` 自动识别 CSR 更新区间，并核对原语句顺序和状态端口。scikit-fem `ex01.py` 的求解调用已接到实际 SuperLU 入口，库内部作为算子边界。
2. 化简。原矩阵复制及调用方状态继续沿原图传递。闭合的 CSR 更新区间消去索引及对角线临时数组，并保留 SciPy 首次类型查询的状态写入。计划引用实测单核成本和目标硬件图，对单次调用计入预编译产物加载成本。
3. 交付。用 `python -m scar.compiler.plan_csr_region --calls CALLS --out OUT` 生成 `C++20 / ABI 1` 产物及选择收据。用 `python -m scar.runtime.csr_region_entry --selection OUT/selection.json --selection-sha256 SHA --exclusive-inputs --study -- ENTRY` 验证替换与回退。`--native` 和 `--cost` 可绑定已生成的 C++ 产物与同版本实测成本。普通运行按选择保留原入口。

当前适用域为 SciPy 1.18.1 的标准 `float64` CSR，要求每行有唯一的对角项。最终比较用独立的非插桩进程，核对解向量、稀疏结构和入口输出，分别报告区域和入口成本。

scikit-fem `ex01.py` 的热更新区间达到 1.529×。完整入口由 0.949 s 增至 1.091 s，默认决策为 `KEEP`。试跑每次完成一次 C++ 更新，解向量、CSR 数组和标准输出均通过对照。记录见 `artifacts/baseline/three-stage-sparse-entry.json`。

重复调用场景先合并 F4 循环和有界计数证据，再用 `--scope-cost COST` 将一次加载、首调和后续调用接入同一资源计划。`--qualify-selection SELECTION --validation VALIDATION --out OUT` 核对同一源图、产物、入口参数、准确调用次数和清洁进程结果。未取得可确认收益时保留原执行。`ex01` 派生的 512 次真实参数扫描已完成以上验收，详见 `artifacts/baseline/three-stage-csr-repeated-scope.json`。

新版 CSR 产物在构建期封存 Python 区域字节码和提供方代码指纹，首次有效调用才加载 C++20 / ABI 1 内核。旧 CSR 产物使用新 builder 重建。

`--topology-scope SCOPE` 复验 F4 循环外参数、原复制和后继消费的连接。索引内容可能被别名修改，所以 C++ 首次构造索引计划，后续对 `indptr`、`indices` 和边界行做精确比较，并按本轮数值更新。计划受 `--topology-budget` 限制，在入口作用域退出时释放。

`--topology-cost COST` 在原执行、逐次 C++ 校验、复用索引计划之间比较。同一 CPU 线程的加载、索引构造、更新和释放分别计时，持有的索引元数据提供显式内存需求。整程资格验收还核对实际命中次数、数值结果和完整进程耗时。

512 次更新的区域模型选出复用索引计划。原执行、逐次 C++、复用 C++ 的总成本点估计依次为 563.2、537.2、519.6 ms。完整入口中位数从 8.636 s 降到 8.582 s，样本范围重叠，发布结果保持 `KEEP`。详细记录见 `artifacts/baseline/three-stage-csr-topology-hoist.json`。下一轮按阶段检查稀疏求解器的结构分析、分解和求解边界及其状态依赖，再确定可复用的工作与实际成本。

当前主链纠偏（2026-11-17）：按“语义化简→工作/传输/存储需求→解析组织→构造时序→独立证书→C++”推进；默认不做整版本试跑排名，也不把全局CP-SAT作为每个程序的必经入口。原Python/NO_OP必须参与同边界成本判断；已有OIR保留实现和来源，但不先固定五CALL的ready骨架。

传统编译器方法以本机GCC13.3为基线；2026-11-17的`-Q --help=optimizers`查询与保留的O2/O3结果逐字相同；本次继续只读核验21份既有源码/dump收据（200546B）及所引用的研究JSON，SHA和大小全部相符，未重复编译或运行目标。另直接复核13.3 release的LICM、向量成本与pass顺序源码。`-O3`是在既有分析上按序运行变换和目标成本判断，采用启发式；不保证全局最快，默认也不启用多线程。以下是SCAR的外延设计，不能作为已完成声明：

| 顺序 | GCC机制 | SCAR对应及当前边界 |
| --- | --- | --- |
| 1 统一依赖事实 | GIMPLE、SSA、虚拟内存use/def、alias、函数摘要 | 复用ProgramView及值/状态版本、控制、effects；补跨层摘要，不能以同对象或相同采样值证明不变 |
| 2 确定合法作用域 | LICM沿操作数/内存依赖求最外可移循环，并保执行条件 | 沿region端口与callee摘要求最大安全作用域；循环显式区分live-in、carried、exit。已有数组循环profile，通用嵌套region尚缺 |
| 3 改写并更新分析 | 按序pass和有界清理；维护SSA/CFG等性质 | 已有闭合fusion的实际后继图、revision校验和use-def重算；仍缺跨循环/跨region的统一失效与重算，不把metadata缓存当作它 |
| 4 计算适用域/收益 | target hooks估算正文、准备、余数和检查成本；求向量化阈值 | 从后继图推导算术、逻辑访问、分配、搬运及guard/launch数量，再接同边界组件模型。已有部分，完整上下文资格仍未闭合 |
| 5 组织资源与时间 | 目标相关lowering及调度 | 扩到CPU/GPU/显式存储/链路日历，联合考虑并行、搬运和生命周期。已有构造式planner与独立verifier；真实异构全入口未贯通 |
| 6 生成实现 | RTL及机器后端 | SCAR输出已有C++20/ABI1或算子库调用，交GCC/库做SIMD等底层优化；源码数值契约继续约束lowering |

串行固定上下文可算`ΔC(N)=Δsetup+Δguard+Δalloc+Δmove+N·Δbody+Δtail(N)`，由`ΔC<0`求规模阈值；并行/异步用依赖和容量日历计算makespan，不能把所有服务时间直接相加。系数必须注明目标、compiler/flags、shape/layout和边界；GCC相对成本单位不是纳秒，逻辑读写字节也不是DRAM流量。少量组件标定用于建立/校验模型；不默认整版本黑盒竞速。下界及可行时序分别报告，未达到下界不声称最优。

当前落实顺序：①保留已贯通的array/composite源码effects、控制与移动证明；②从实际后继图重建计算/搬运/分配需求及生命期，父子摘要不重复收费；③冻结真实科学区域同调用边界、数据前驱、allocator和CPU上下文的成本模型并独立验证；④按依赖与容量构造时序，将同份证书交现有C++20后端。外提会延长值的存活期，并行会引入启动/同步及带宽竞争，均在②③重新计入；合法外提不直接等于收益。验收材料为后继图/需求、合格成本或明确残差、独立时序证书和实际lowering，不能只保存“建议优化”。前三层未闭合时原路径仍保留，不通过增加backend绕过。

`macro-pipeline` schema 2 将结构候选和实际选择分开保存。保序整数折叠、copy 转发、CSE 和 DCE 先在既有 Expr DAG 上执行。浮点运算保留到运行时，可能产生异常的死标量运算会阻止删除，整数真除法保留 Python 语义。融合候选复用原 F4 来源、公开端口及效果合同，独立发布 `candidate_revision` 和 `candidate_demand`，后继图与实际优化后 lowering 各自重算需求。成本未知时实际选择为 `KEEP`。下一层把后继图的逻辑工作量和存储生命期交给资源模型，按执行器、内存域和搬运链路推导时间与容量约束，C++ 后端仍只消费已验证的方案。见 `artifacts/baseline/strict-fp-and-structural-fusion.json`。

内部需求现已接主入口：`plan-controlled-region` schema4按原`native_stage_context.versioned_schedule`发布`native-task-problem/projection/plan`，独立check同源重推。`controlled_native_time`复用原时间IR绑定caller/native参与lane，原时间记录保留，同端点同类型关系只建一次；`controlled_native_tasks`把task、order、material readiness、scalar及slot overwrite映为共同Problem实际前驱，完整recipe和原20任务仍可展开。固定pool预算预留一次，避免共同IR按logical data edge重复租用物理槽；原图的12个版本及生命周期保在同一typed schedule。无样本时照常发布20个UNKNOWN操作，有joint样本时按原连续范围共同计费；不可分摊整调用或组成本，不能从FFI差值填wrapper。`native_group_costs`重算原clock的精确均值并保迁移残项，当前12操作仍UNRESOLVED。真实完整冷读及依赖/预算反例已验，未运行目标或改C++。scope仅固定native recipe；下一步先补实际entry adapter/trial/guard/packing、caller及数据/allocator上下文的完整成本，再用独立holdout验证新冻结模型。不能用此内部展开授权原ELF重排、物理独占或完整程序交付。报告：`artifacts/baseline/controlled-native-task-mainline.json`。

`calibrate-stateful-entry`现可prepare→按SHA显式run原ELF→model→cold check；canonical binder仅从原源码compile/CodeID重建普通对象和函数，不exec目标module，fallback为fatal。成功call与真实install共享guard/adapter/artifact代码，完整调用一起计时、不加组件medians、不插内部clock。新model/profile官方reader可接原future holdout；资格必须匹配requested entry protocol，不能跨compact/trialFalse。实测已证同协议仍不足：固定buffer refill的919245ns模型对真实新receiver/演化数据序列误差46.8%，仍拒；first/later是诊断，不作新模型系数或删样本依据。新训练必须关联原值/存储生命期和前驱上下文，不能取holdout值回填。future policy gain的舍入顺序缺陷另记baseline defect；其修复须新freeze，旧原始capture仅对应原producer，不补写资格。见`artifacts/baseline/matched-entry-cost-mainline.json`。

已有C实证：未知alias生成带检查的双路径；有`restrict`契约时省去该检查，成本阈值随之改变。两次常量返回均可折叠，但volatile状态写仍保留。因此Python常量传播与删除import须分别证明值和初始化effects，单次动态观测不能替代。原始材料：`artifacts/baseline/gcc-o3-study.json`；依据：[GCC13.3选项](https://gcc.gnu.org/onlinedocs/gcc-13.3.0/gcc/Optimize-Options.html)、[依赖/alias](https://gcc.gnu.org/onlinedocs/gccint/Alias-analysis.html)、[LICM源码](https://github.com/gcc-mirror/gcc/blob/releases/gcc-13.3.0/gcc/tree-ssa-loop-im.cc)、[成本源码](https://github.com/gcc-mirror/gcc/blob/releases/gcc-13.3.0/gcc/tree-vect-loop.cc)、[pass顺序](https://github.com/gcc-mirror/gcc/blob/releases/gcc-13.3.0/gcc/passes.def)。

跨层摘要补充实证：保留三层`noinline`函数，GCC13.3仍沿调用图传播const摘要，把稳定调用移到循环外；同一指针指向的内容每轮被写时，读取留在循环内；未知effects的外部调用也保留。仅编译、未运行或测加速，见`artifacts/baseline/gcc-ipa-scope.json`。借鉴[IPA](https://gcc.gnu.org/onlinedocs/gccint/IPA.html)的“局部摘要→调用边参数替换→跨过程传播→改写”，不要求展开全部函数。SCAR现已实现条件MAY依赖的SCC组合，但effects/alias/termination并未闭合；当前LICM仍仅支持严格array-carry域。`macro_pass.PassSpec`目前仅单fusion使用，不能称通用pass管理器；现有后继图的use-def重算也不能代替跨region分析失效。逻辑工作量弱支配不等于CFG支配或时间收益。

源码移动证明分为三个连接：①由原CALL/Frame、实形参/environment/world及经过的调用，界定候选原位置到目标位置的完整窗口；②将该窗口内和回边上的reads/writes与alias查询组合为no-clobber结论，单次region局部epoch与不同input ID均不证明物理不相交；③由支配/控制路径给出must-execute或可安全提前执行的事实，交现有零trip、FP/失败和fresh-output guards消费。MAY依赖为空不能充当pure，guard并集不能充当must-execute。上述结果必须绑定原revision/region；改写后重推受影响分析，再从后继图重建需求和成本。缓存已验证材料只是避免重复读验，不能替代这些证明。以下记录其分步落实；当前有界array/composite已接通，任意Python域仍未闭合。

三个连接现已在既有array-carry域接通：`source_motion_window`查询原F4共享loop的完整normal窗口、进入body后的匹配recipe控制路径及单独的DATA反向slice；异常排除逐项保留，normal事实不冒充异常/终止证明。`array_motion_legality`将原loop/CALL/actual/world与body精确绑定，沿匹配RETURN/CALL把actual追至唯一原lexical READ；不跨primitive、不按变量名猜测，不把引用传递当内容身份。真实首probe发现direct-ID比较误报未知，补原15条DATA边后`fixed`经5边接通；只读输入互alias允许，写入/额外opaque CALL仍阻挡。data slice预算缺口只影响data字段，不擦除已知control/effect。

`read_invariant_loop_motion`现发布schema2及canonical source interface、完整window和逐root legality；lowering附加shape/authority不能改变canonical接口。`invariant_build`在规划前保留未闭合域的NO_OP，C0/导出独立冷重推来源；native interface只携source-motion Ref，plan保存约2KB的实际optimized-node binding，发射时重验优化DAG及每PRE节点formal依赖，不套原node IDs。低层无来源micro仍仅conditional，不能导出替换caller。真实4次native与原函数逐位一致，旧fallback保持；真实科学2循环9窗口中7个条件必经、2个有可绕过路径，未由此授pure/复用。三项metadata检查及独立cold check通过，不是性能验收。原科学stateful ELF/模型未改；旧temporal产物因generator版本需重建，不绕过pin。下一层将该窗口接口接入已准入跨函数routes与可闭合的effect摘要，再重建同边界成本与资源日历；不把此窄域当通用MemorySSA或整程序编译完成。材料：`artifacts/baseline/motion-legality-mainline.json`。

上述跨函数连接已在原COMPOSITE路径实现。`array_call_effects`由既有ArrayRegion/ExecutionPlan和PipelinePlan重算条件effects，callee摘要按source/body共享，逐CALL保参数、返回、层次及静态multiplicity；读取中间结果与其input provenance分开。pipeline read_formals是callee边界的传递footprint，不是wrapper新增load；父子summary不能再相加成访问量或时间。输入返回/view不能假fresh，input-write/read已知事实仍保留；frees/allocatorcallback/失败顺序按scope保UNKNOWN。`composite_call_effect_binding`复用原SourceFunctionComponents/namespace resolver/signature binder，逐原F4 CALL核actual/formal/environment/world及全部global路径，不造callee或运行目标。

`read_composite_invariant_loop_motion`发布独立`scar.source-composite-loop-motion/v1`，typed proof与direct严格分开，纯材料读取仍不授F4；source authority来自完整rediscovery。`array_motion_legality`显式区分两个proof key，消费effect/source binding并核outer CALL原端口；C0重算typed材料及effects，actual optimized DAG保持独立binding。原composite build/plan_only/export必须消费证书，未解保NO_OP；export先cold replay再load，并核原ELF bytes。两caller facade仅检查loader已消费的短Ref/绑定，不每call读source graph；无证书micro只可直接run。已有exact-copy合同的标量表达复用cpp规则，未新设backend。3源unit/5函数/4调用/4守卫的真实原链已编译，5次native（其中普通入口1次）验证一致，普通入口结果bytes SHA/stdout/stderr相同，深层callee变化回退；counter伪造即使重算全部报告ID仍拒绝。集成脚本一度误用tuple/list直接比较，改为原canonical hash后复用已完成ELF，不为此重编或复跑性能。direct源重放/生成另metadata验证，无新加速比。下一步集中真实CPU科学区域同边界cost和可控硬件条件；任意Python/callee闭合、递归effects以及完整程序C++仍未完成。材料：`artifacts/baseline/composite-effect-mainline.json`。

GCC源码定位以固定`releases/gcc-13.3.0`版本及函数名为准：`tree-ssa-loop-im.cc`的`determine_max_movement`同时约束最外作用域与执行保留；`tree-vect-loop.cc`的`vect_estimate_min_profitable_iters`计alias/alignment/trip检查及准备/主体/收尾成本，计算收益阈值；`passes.def`在循环改写间穿插清理，并要求if-conversion后紧接vectorize。SCAR合理外延为同一图上的可控资源、存储和链路需求及容量日历；GCC目标相对成本不能直接换算完整Python调用耗时，局部最便宜实现也不保证联合时序最快。资源计划改变布局、并发或调用边界时，要使相应成本适用域失效，不能沿用旧组件表。

该结构工作的首段已实现：`array_motion.derive_array_motion/verify_array_motion`在既有ArrayRegion上统一依赖与placement，不创造新SSA/CFG；body和interface改动必须重推，常量算术或view阻挡的根不获移动。`read_invariant_loop_motion/verify_invariant_loop_motion`通过原F4完整重发现附source CALL/参数/lexical写/control证据；`invariant_build`自动发布`source-motion.json`。CPU计划保完整分析与未履行Q，匿名成本图用相同placement且不把source IDs放入primitive key；temporal生成器对实际optimized region另derive并verify，parallel复用。原F4/runtime/外部写入条件仍分开，此查询不是通用MemorySSA证明。当前按原样例贯通并完成4次native正确性验证；下一段集中实现沿原CALL路由的可组合依赖/读写摘要，再接一般移动证明，不新增后端。验收：`artifacts/baseline/unified-motion-mainline.json`。

可组合摘要已落地：`scar source-summary --program PROGRAM --entry-calls ENTRY_JSON --out FRESH_JSON`，`--check JSON`独立由原ProgramBuild重推。局部原F4有限port种子/bitset传播与条件目标SCC组合共用原图；选择条件影响、callee/receiver/key、未知call已知参数来源不丢失，effects/termination/未来dispatch仍未闭合。derived集合用`dependency_ref`与共享`dependency_sets/unknown_facts`；`resolve_dependency_set`读取，展示用`expand_interprocedural_dependencies`无损展开，分析集合ID不是张量LogicalValueID。`plan-controlled-region --source-summary JSON --entry-calls ENTRY_JSON`冷绑定source/native，并将原port→既有semantic operation→native stage的引用与digest纳入同一Problem；prep schema3明确拒旧producer/schema。真实入口8body/49CALL、80来源投影冷验已通过，不授pure/变换或计时资格。原CLI+既有prep独立check126.2s仅为那次离线分析耗时，不是后续prepare的对照样本。材料：`artifacts/baseline/source-summary-mainline.json`。

该重复已在单次owner事务内合并：`AdmissionContext.program/stateful`及原session持有的`EntryAdmission`只能由实际冷重放产生，禁止调用方布尔或序列化凭证跳过；每个完整ProgramBuild引用持独立session，命中/退出复核源码、provider、文件版本与live环境。原FactRef即使logical URI不是文件，也必须由原`FactMaterialResolver`核完整receipt、locator与physical/catalog链；不猜路径、不造文件、不跳过未知。真实prepare和独立check各2个session、entry/witness/native各1次，展开摘要核心SHA相同。新增`stateful_region_effects`从既有有序region导出局部epoch、18访问及RAW/WAR/WAW，54Expr+26action的80投影均接原semantic/native及Problem；原序不删、不同input和逻辑不交区间不授物理disjoint。三项定向metadata测试、真实禁目标/native/编译的冷验通过；带计数prepare68.20s/check68.15s，不构成A/B或新加速比。成本仍UNKNOWN/NO_OP，下一步是上述三个移动证明连接，不把复用和访问投影当成通用外提完成。材料：`artifacts/baseline/scoped-admission-mainline.json`。

下一轮集中顺序：flags、shape、正文已确认一致；先处理已识别的独立函数/真实阶段组、wrapper+trial、前驱数据及分配/触页条件，冻结可转用成本模型，再接规模阈值和同边界原路径比较；随后扩展嵌套region改写和时序。旧预测超阈值仍拒绝，不用GCC启发式替它授权，也不新增优化后端。

入口动静接线现已实测：原短driver的有界CALL目标采集接入既有ProgramBuild，`scar entry-calls --program PROGRAM --out FRESH_JSON`生成入口条件投影，`--check JSON`独立冷验；共享body、loop/递归引用不unroll。当前2个入口可达单元、10条件接口、8共享body、66原typed参数/环境/world路由；按path/SHA/CodeID自动见证同一native源。旧303条是236个lazy视图、18个depth预算、17个effect、13个resource、19个材料/覆盖问题，不能当303个语义失败。构造协议及外部Python源码仍未接全。

同frame阶段组：`scar calibrate-native-groups --native ARTIFACT_JSON --out FRESH`先冻结design；`--run --design DESIGN --sha256 SHA`才编译、核一组canonical输出并采样；`--check OBSERVATIONS`只读重放。原完整run及真实前驱/scalar/slot版本、commit/逆序RAII都保留，仅组边界加clock，未生成saved-port代值。CPU0/259例2warm+7样本中，body组mean0.540264ms、预包装FFI窗口mean0.760542ms；是固定canonical输入、带插桩的联合窗口，无A/B或整方法预测。使用样本均值及精确均值分数，不加组件medians；两窗差只叫联合未归因耗时。`check-cost-context --native-collection RESULT --group-observations OBS --out JSON`机械核7阶段GCC模型字段相等、2阶段UNKNOWN，保数据/缓存/插桩残项。

`plan-controlled-region --entry-calls JSON --native-group-observations OBS --compiler-feedback RESULT`已将同源见证和组上下文接到原source action/expr、时序及Problem身份；有实测component而无完整方法profile时仍保whole-call cost UNKNOWN/NO_OP。下一层集中补同边界caller/数据/allocator条件下的整方法模型，第一阶段则补构造及外部源码协议；不从GCC字段相等推导纳秒相等。验收：`artifacts/baseline/entry-native-frame-context.json`。

GCC反馈已接主链：`scar native-optimization --native ARTIFACT_JSON --stage-capture CAPTURE_JSON --out FRESH`只重编一份已保留C++，沿原compiler/flags/headers加`-fsave-optimization-record`；不加载分析ELF。`--check FRESH/result.json`只读重推原source/body行区间与诊断投影，禁止compiler/native仍通过；自洽篡改span并重算projection也被拒绝。raw GCC记录和派生投影均gzip保存，正文按JSON pointer引用；地址型pass ID只在该record内有效。`scar gcc-cost-features --collection RESULT --out FRESH_JSON`提取受支持GCC13.3的相对成本字段；它们不是ns，不合并不同版本，不授予计时资格。无诊断不证明死工作。

`plan-controlled-region --compiler-feedback RESULT`已把同manifest的9个非COMMIT阶段接回原action/expr及Problem参数；两个COMMIT未纳入此source-map profile，原图/效果/依赖不改。preparation现为schema2，独立check明确拒绝旧schema1。`--cost-use EXPLICIT_DECLARED_ANALYSIS`保原条件分析；`--cost-use QUALIFIED_TARGET_TIMING --cost-qualification RECEIPT`必须冷重放原model/holdout/policy并绑定实际native源码和plan。旧1,423,927ns分析仍可生成条件计划；同一数值请求合格时间用途时追加missing_terms，实际选择0，保持原政策及NO_OP。GCC注释不能解除这些义务。报告：`artifacts/baseline/gcc-lowering-feedback.json`。

首条pass现已贯通：`pipeline_region_pass`投影原CALL/GlobalPort及conditional Q；`macro_pass`复用已有成本gate与OIR，记录requires/selection/invalidates；`pipeline_fusion_delivery`交已有无worker/绑核的融合C++；`macro_pipeline`全源重放及受控模块接入。真实8×8受保护调用成本判KEEP，不加载库；正声明模型仅验REWRITE链。`model-pipeline-cost`已继续接通原source occurrence需求、有效仿射系数与完整入口成本，缺证据不补零。`macro-pipeline`现另外发布`semantic/revision.json`、before/after图、分析及实际融合body；冷读重放选中OIR和原F4，原图/公共边界/Q/UNKNOWN effects不变，重算use-def、源码顺序及静态last-use。旧revision或另一region的分析、错误body引用均拒绝；成本、物理寿命、调度与native绑定仍列未解，不拿旧五CALL计划套新composite。无通用分析缓存，未重写整程序。材料：`artifacts/baseline/semantic-region-revision.json`。下一层先从后继图重建实现需求与同边界成本，再重建硬件时序及交付绑定。

后继图主入口：`scar plan-semantic-region --macro-preparation PREPARATION --out FRESH`冷重放原F4/实际after图，区分原eager、raw composite、optimized lowering三种需求，再接既有共同Problem和构造式planner。`--logical-analysis`只做显式逻辑分析；`--compile-validation`仅构建已有融合C++验证候选，不借逻辑计划授物理/部署资格；`--check FRESH/result.json`独立核新图/需求/硬件引用/计划/生成CPP及native字节，不打开ELF。显式构建仍沿原backend加载库并检查ABI，报告单列；运行必须显式trial并保原guards/fallback，默认原函数。5CALL样例已验证native3、dtype回退1，strict两个资源义务仍未解，非真实科学整入口加速。分析记录现按slot ID排序，落盘重读后可直接重算比较。材料：`artifacts/baseline/semantic-successor-mainline.json`。

真实科学区域的可控域入口：`scar plan-controlled-region --kind STATEFUL_SPECIALIZATION --admission SPECIALIZATION/result.json --out FRESH`自动cold-admit原F4/已编译candidate，建立既有SemanticGraph及source→native stages→private slots/硬件引用，默认未知时间保持UNRESOLVED。`--entry-profile PROFILE`只接明确提供的条件分析成本，不能把旧预测变为合格；`--check FRESH/preparation.json`独立重放。另支持`--kind PIPELINE_FUSION --admission REGION_BUILD/result.json`。原全入口义务另存，不把CPU0私有阶段绑定扩大到guards/commit，不把payload预算当RSS/可用物理RAM。`calibrate-guard-regime`只比较同一已有canonical probe的两guard配置；本轮2warm+7对测量的入口差中位7085ns、完整canonical调用差中位6274ns，不归因旧误差、不导出component/profile资格。下一层先使canonical编译/内联/shape上下文匹配真实lowering，再冻结新的成本模型；不重跑旧模型碰阈值。验收见`artifacts/baseline/controlled-scientific-region.json`。

离线分析的重复重放已在单次冷事务内合并：硬件表/拓扑和entry cost共享已验结果，逐行仍走原严格decoder，原`record_codec.py`未改。命中和成功退出均核对传递文件内容；返回副本、32项上限、独立check新开冷事务。可选Python audit只观察纯metadata工厂的文件读取，不是目标采集器或native I/O跟踪器；非文件live条件由调用方key绑定，支持域外不复用。冻结实现的四新进程ABBA中，各模式2样本、warmup0，分析median42.3199→26.3717s（1.60475×），7份模型正文完全相同；无目标执行、native加载或新编译。默认未知成本仍拒绝选择，旧DECLARED预测仍未合格。该收益仅为SCAR离线分析器，见`artifacts/baseline/replay-mainline.json`。

当前实现进展：完整入口canonical回归已代替手填参数做第一次真实校验，TRAIN与canonical/target holdout分开，rank/SVD/误差可查；系数为DECLARED有效模型，不复用旧工具的scope中位数差。构造式时序已接原source producer及C++：关键路径优先、局部实现/route选择、整数端点传播、有限容量冲突加调度序边；原语义边不变，独立验证，不证明全局最优。compact fused已接原C1 guard与普通入口，独立scope UNKNOWN，不用macro模型授权。

后续逐层验收：①扩到真实科学caller的需求/完整成本证据，严格区分逻辑访问与实际跨层流量；②将已有结构delta产生的区域接到共同硬件/时间投影，避免只规划固定CALL骨架；③扩展前补不同shape/位置/并发和循环周期模型的适用域证据，再构造CPU/链路/存储日历；④同份证书接已有C++/compact，并在真实程序的完整同范围边界校验。测得min/max不升级数学界，未知成本不补零，不承诺由硬件名牌算出任意程序唯一实际最优。

真实stateful边界复用同一流程：`scar observe-region-inputs --plan REGION --out OBS -- ORIGINAL_COMMAND`只记录声明input.path的元数据；`scar specialize-stateful --program PROGRAM --inputs OBS/inputs.json --out FRESH --memory-budget-bytes B --compile`按观测shape重放原F4并生成显式单worker验证候选；`scar check-stateful-specialization --specialization FRESH/result.json`独立冷读，不加载native。scalar每次读新值，metadata不是不变性证明。`stateful_entry_cost`与`region_implementation`接原完整入口scope到共同IR；物理资源义务独立于cost，logical分析不授权lowering。真实driver验证见科学主链checkpoint，当前仍默认NO_OP，未生成完整程序C++或自动部署。

成本主链继续复用 `time-schedule → refine-time-envelope --scope-cleanup → bind-time-resources → cost-activities`。`scar model-stateful-activity-cost --activities COST_RECEIPT --out FRESH`按final source ownership聚合条件点模型，缺数值/串行关系/来源即拒绝，保留原表及转用残差；`scar validate-stateful-entry-model --model MODEL/result.json --holdout HOLDOUT/result.json --policy FROZEN_POLICY --out FRESH.json`从独立完整方法样本评估，失败不返回qualified_profile。raw DECLARED profile可用于假设分析，自动成本决策只能消费经相应政策验证的profile；物理资源、源码合法性与部署仍各自核对。禁止用holdout反调原模型、阈值或把component样本范围相加当物理界。

`scar check-cost-context --scope-observations OBS --holdout HOLDOUT --model MODEL --out REPORT`机械核成本转用字段，区分记录事实与固定源码配置；已证 canonical `trial=False`、真实harness `trial=True`，不把缺失runtime metadata补成已观测分支，也不把残差归因给此差异。下一次采样用`scar stateful-holdout prepare ...`预绑定future v2 design/epoch/policy/timer/源码/完整argv，`check --design D --sha256 SHA`只读核验，`run`才实际执行。通用验证支持无文件输出，合成记录不能获成本资格，future policy不能授权旧采样；本轮只验metadata，未执行新holdout。下一层先补匹配上下文的canonical证据及可控CPU/私有payload资源合同，再在真实科学入口冻结模型做独立验证。

研究依据： [DaCe/SDFG](https://htor.inf.ethz.ch/publications/img/dace-sc19.pdf) 的控制/数据/搬运分层，[Halide 2016](https://graphics.cs.cmu.edu/projects/halidesched/mullapudi16_halidesched.pdf) 的边界分析与解析分组，[MLIR bufferization](https://mlir.llvm.org/docs/Bufferization/) 的先tensor变换后存储规划，[Hong–Kung](https://www.eecs.harvard.edu/~htk/publication/1981-stoc-hong-kung.pdf) 与 [Roofline](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/Archive/EECS-2008-134.pdf) 的数据移动/性能界，以及 [HEFT](https://disco.ethz.ch/courses/fs14/seminar/paper/Jochen/4.pdf) 的构造式调度。Halide贪心/HEFT启发式并非全局最优证明；借鉴机制不等于直接替换SCAR前端。研究记录 `artifacts/baseline/constructive-planning-research.json`；实测诊断 `artifacts/baseline/pipeline-architecture-diagnosis.json`。

F5可增量接入经过完整前端的新源码，不重建旧图：先用
`scar.analysis.source_file_seed.seed_source_file(path, namespace_root, out, module=None)`建原路径单源seed，
再运行`python -B -m scar.analysis.source_dependency_frontend --model <seed> --explicit-seed --out <fresh-frontend>`，
用`source_interface_build --facts <frontend/facts> --out <fresh-interfaces> --jobs 1`投影摘要。
`source_context_build`新增`--supplement <frontend/facts> <fresh-interfaces>`（可重复），原base和exec父收据不变；
目录追加所有新端口，读回器重建同序目录并严格核对。module参数只是搜索候选，不是实际import选择。

新定位的缺失源码可自动成批进入前端：
`python -B -m scar.analysis.source_file_seed --admission <plan> --out <fresh-seed> --max-files 16`。
当前只接计划中实际ready且完整hash的新search候选；旧inventory候选与预算前沿仍保留原引用。
批量seed仍未授F0–F4；接上述`source_dependency_frontend --explicit-seed --jobs 16`才按原顺序处理，
超过4源采用调用者给定进程数，不会因material-index为空错误退回单进程。

`source_context_build --observe-port UNIT LOCAL`可重复指定少量原端口，发布独立`solver-observations.json`。
只读当前分析器token、pending和已有订阅，每类最多4样本；它不执行目标，也不把pending为0当全图收敛。

数据材料入口：`python -B -m scar.analysis.source_data_materials --file INPUT.yaml --out FRESH_STORE`。
可加`--loader-source /path/to/omegaconf/_utils.py`，只识别已固定AST版本的resolver规则，不导入OmegaConf。
`DataMaterialStore(path).read()/raw()`返回typed图/原bytes。材料不是已组合cfg，未知tag保对应字段的obligation。

`python -B -m scar.analysis.source_configuration_requests --targets CONTEXT_GRAPH --out FRESH_REQUESTS`
从已准入版本化库调用及入口输入生成条件请求，`read_configuration_requests(path)`核父收据并typed读回。
factory参数、decorator/task关联、初始argv与文件候选分别记录；不直接给cfg注入常量。

`python -B -m scar.analysis.source_configuration_composition --requests REQUESTS --out FRESH_GRAPH --hydra-source /path/to/site-packages/hydra --loader-source /path/to/omegaconf/_utils.py`
沿条件文件请求自动发现defaults依赖，保存共享材料、有序版本及provider/选择缺口。
`read_configuration_composition(path)`核父收据并返回(report, typed graph)；本层不输出最终cfg。

`python -B -m scar.analysis.source_configuration_overrides --composition GRAPH --out FRESH_OVERRIDES --hydra-source /path/to/site-packages/hydra`
复用固定ANTLR语法，按已固定库源码中的argparse声明筛选初始argv，生成带原输入端口的表达式和有序写入图。
`read_configuration_overrides(path)`严格读回；库初始化、visitor、自定义函数均不执行，组选择和当前argv仍需条件。
此静态输入工具的可选依赖为`configuration-source`；现有lewm环境已具备，未重新安装。

`python -B -m scar.analysis.source_configuration_values --overrides OVERRIDES --out FRESH_VALUES --omegaconf-source /path/to/site-packages/omegaconf`
生成有序状态表达式和已知路径的字段/引用图；可重复加`--path-json '["world","num_envs"]'`限定查询。
`read_configuration_values(path)`返回(report, typed graph)，root ID均指本图expression ordinal；原版本另见roots表。
字段结果始终保父条件；OPEN、missing、absent、null、循环及成员封闭性分别读取，不将其直接注入当前任务形参。

`python -B -m scar.analysis.source_configuration_state_binding --values VALUES --facts SOURCE_FACTS --out FRESH_BINDINGS`
定位已建图库API的原cfg实参及调用前后world，保存声明状态接点；Python返回值不承载该内容版本。
`read_configuration_state_bindings(path)`严格核源码版本、scope内alias及原端口；仍需证明输入与调用状态对应。

## 当前形式前端入口

```bash
python -B -m scar.cli source-bytes --file /path/to/program.py --out /path/to/fresh-material
python -B -m scar.cli source-bytes --model /path/to/model --out /path/to/fresh-f0 --jobs 4
python -B -m scar.cli source-bytes --model /path/to/model --out /path/to/fresh-check --verify
python -B -m scar.cli source-tokens --model /path/to/model --out /path/to/fresh-f1 --jobs 4
python -B -m scar.cli source-tokens --model /path/to/model --out /path/to/fresh-lexical-check --verify --jobs 4
python -B -m scar.cli source-syntax --model /path/to/model --out /path/to/fresh-f2 --jobs 8
python -B -m scar.cli source-syntax --model /path/to/model --out /path/to/fresh-syntax-check --verify --jobs 8
```

F0只保原始bytes与编码结果。解码失败仍存原bytes；文件不可用/预算停止准确报错，不给后层授权。
现有模型只追加每源raw/token分片；旧源码和IR保持。F1也可用`--file PATH`独立采集（先执行F0）。
`--max-tokens`是每源上限；verify按原工件预算重放，原预算超过本次上限则拒绝，不会静默忽略限制。
F1只转录词法，不证明语法有效；原生错误保prefix，预算停止不发布partial模型。
F2默认exec，也支持eval/single/func_type；--file先经过F0/F1，--max-nodes是每源AST上限。
源码始终只解析为AST；语法成功不证明作用域/控制合法。大literal超固定预算准确报frontier，不标UNKNOWN。
F0至F3及F4-A、全域模板接口已在声明材料域验收；F4-C正常路径边表与F5可能调用接线继续复用原ID。
当前profile固定本机CPython及tokenizer源码/静态解释器二进制指纹；内置tokenizer的shared Python暂不接纳。

F3-A：

```bash
PYTHONHASHSEED=0 python -B -m scar.cli source-symbols --model /path/to/model --out /path/to/materials --collect-only --jobs 8
PYTHONHASHSEED=0 python -B -m scar.cli source-symbols --model /path/to/model --materials /path/to/materials --out /path/to/check --verify --jobs 8
```

固定seed只用于分析进程；实际hash上下文写入profile，worker不一致时拒绝。每源scope/symbol上限可配置，
F1/F2及原材料的replay预算也受本次上限约束。默认capture会向可写模型只追加source_symbols；
collect-only是独立材料store，fresh核验绑定必需源码收据。PUBLIC_COMPILER_VIEW/UNJOINED不授AST绑定或scope完整性。
/home已恢复可写，同一main的只读期间提交已合并，普通入口可直接使用。
已验F3七层已按原收据登记到当前模型；`/var/tmp/scar-f2-recovery-7w7nd_0z/`仍保历史材料，canonical查询无需传外部store。

F3-B（只分类字段角色，不执行symtable或目标）：

```bash
python -B -m scar.cli source-roles --model /path/to/model --out /path/to/roles --collect-only --jobs 8
python -B -m scar.cli source-roles --model /path/to/model --materials /path/to/roles --out /path/to/check --verify --jobs 8
```

每store只保存一份`source_roles/catalog.json.gz`，每源为稀疏原槽ports和484条计数；
`--max-ports/--max-slots`限制本层，`--max-nodes/--max-tokens`同时约束上游重放。
默认capture向可写模型只追加source_roles；collect-only不改模型，verify重新计算角色与计数。
纯API以`read_source_syntax_material`取得验证后的不可变F2材料，用`SyntaxRoleView(material)`复用索引查询。

C1源码上下文（不求词法存储或建立真实frame）：

```bash
python -B -m scar.cli source-contexts --model /path/to/model --roles-materials /path/to/roles --out /path/to/contexts --collect-only --jobs 8
python -B -m scar.cli source-contexts --model /path/to/model --roles-materials /path/to/roles --materials /path/to/contexts --out /path/to/check --verify --jobs 8
```

外部B输入只允许只读采集/核验；B已注册model时可省`--roles-materials`并使用默认增量发布。
C1只存scope/region/private策略与稀疏原槽边界，catalog共享一份；同源F2/B各只fresh一次。
`SourceContextView(syntax, roles, contexts).resolve(slot)`或`.port_context(port)`复用父槽索引；不复制端口或逐槽context。
`--max-scopes/--max-regions/--max-boundaries`限制本层，`--max-slots`及其余输入预算同时约束重放。

C2a名字投影和整scope声明：

```bash
python -B -m scar.cli source-declarations --model /path/to/model --roles-materials /path/to/roles --contexts-materials /path/to/contexts --out /path/to/declarations --collect-only --jobs 8
python -B -m scar.cli source-declarations --model /path/to/model --roles-materials /path/to/roles --contexts-materials /path/to/contexts --materials /path/to/declarations --out /path/to/check --verify --jobs 8
```

外部B/C1输入只读、全收据固定；全部canonical时才允许默认append。每原B port一disposition，名字只intern一次。
`SourceDeclarationView(...).port/scope/contribution/name`查询原引用账本；`--max-names/--max-contributions/--max-conflicts`约束本层。
投影阶段固定`SYMBOL_VISIT_PROJECTION`；源码操作意图、visit flags、visibility和C1抑制分别保存，不授运行值或物理地址。

C2b条件化namespace接口与捕获（仍不授C3编译器投影）：

```bash
python -B -m scar.cli source-bindings --model /path/to/model --roles-materials /path/to/roles --contexts-materials /path/to/contexts --declarations-materials /path/to/declarations --out /path/to/bindings --collect-only --jobs 8
python -B -m scar.cli source-bindings --model /path/to/model --roles-materials /path/to/roles --contexts-materials /path/to/contexts --declarations-materials /path/to/declarations --materials /path/to/bindings --out /path/to/check --verify --jobs 8
```

三个上游store固定原收据；全部canonical时才允许默认append。每store一catalog、每源一类型化绑定分片。
`SourceBindingView(roles, contexts, declarations, material).port/scope/binding/interface/plan/capture/relay`查询原引用，`.verify()`核引用与逆边。
八项本层预算超限不发布部分表；源码冲突保端口与原因。namespace为源码模板，未观察运行字典、物理cell或地址。

C3-T编译器拓扑（需要固定分析进程hash上下文；不运行目标）：

```bash
PYTHONHASHSEED=0 python -B -m scar.cli compiler-topology --model /path/to/model --roles-materials /path/to/roles --contexts-materials /path/to/contexts --symbols-materials /path/to/symbols --out /path/to/topology --collect-only --jobs 8
PYTHONHASHSEED=0 python -B -m scar.cli compiler-topology --model /path/to/model --roles-materials /path/to/roles --contexts-materials /path/to/contexts --symbols-materials /path/to/symbols --materials /path/to/topology --out /path/to/check --verify --jobs 8
```

`CompilerTopologyView(roles, contexts, native, material)`查询`.scope/.region/.block`，传原SourceContextView后可查询`.port`。
模板保原construction parent/children；PUBLIC引用原F3-A行，INLINED/FUTURE_DETACHED保模板和规则，不伪造公开行。
`--max-blocks/--max-rewrites/--max-visit-steps`约束本层；只验拓扑与scope元数据，符号来源/参数仍未投影。

C3-S符号来源与参数投影：

```bash
PYTHONHASHSEED=0 python -B -m scar.cli compiler-symbols --model /path/to/model --roles-materials /path/to/roles --contexts-materials /path/to/contexts --declarations-materials /path/to/declarations --symbols-materials /path/to/symbols --topology-materials /path/to/topology --out /path/to/compiler-symbols --collect-only --jobs 16
PYTHONHASHSEED=0 python -B -m scar.cli compiler-symbols --model /path/to/model --roles-materials /path/to/roles --contexts-materials /path/to/contexts --declarations-materials /path/to/declarations --symbols-materials /path/to/symbols --topology-materials /path/to/topology --materials /path/to/compiler-symbols --out /path/to/check --verify --jobs 16
```

`CompilerSymbolView(declarations, topology, native, material)`查询slot/fact/source/port/symbol/parameter，`.why(scope,ordinal)`返回有界来源引用。
source slot、compiler slot与native行分域；scope code不是低位flag；varnames只是有序参数，不冒充runtime locals布局。
五项独立本层预算与原D.max_contributions分开；超限不发布部分来源表，外部五store均固定原收据。

F3-D只读组合验收（不复制程序图，不运行目标）：

```bash
PYTHONHASHSEED=0 python -B -m scar.cli frontend-check --model /path/to/model \
  --roles-materials /path/to/roles --contexts-materials /path/to/contexts \
  --declarations-materials /path/to/declarations --bindings-materials /path/to/bindings \
  --symbols-materials /path/to/symbols --topology-materials /path/to/topology \
  --compiler-symbols-materials /path/to/compiler-symbols --out /path/to/fresh-check --jobs 16
```

全部材料已注册到model时可省略对应sidecar参数。只写验收summary，固定输入/实现收据并重验原Q/S及父层。
`read_frontend_bindings(F0,F1,F2,B,C1,D,Q,A,T,S)`返回原表组合view；
`.port/.implicit/.endpoints/.parameter/.why`分别查询原引用，不合并源码namespace与compiler分类。
endpoint中的KEEP路径显式标记association；query预算耗尽返回incomplete与最多一条frontier及remaining，不当成无依赖。

已接受材料的登记（原字节复制，不重新授予语义结论）：

```bash
PYTHONHASHSEED=0 python -B -m scar.cli frontend-register --model /path/to/model \
  --acceptance /path/to/accepted-f3d/result.json --acceptance-sha256 ORIGINAL_ACCEPTED_SHA \
  --verifications /path/to/seven-verification-reports.json --out /path/to/fresh-registration --jobs 8
```

verification映射的键是七种artifact名，值为各原fresh报告的绝对路径。SHA须来自先前已接受的F3-D记录；现场重签不是验收。
登记核source/父层收据、原报告、实际DTO、catalog及当前profile；字节、实现或输入域变化时拒绝。`jobs`控制复制并发。
保持原ID和证明范围，报告标`HISTORICAL_FRESH_ACCEPTANCE_REUSE`；正常语义reader仍执行原strict检查。
每层独立持久化；中断可留下已登记前缀，重新使用同一输入和新的输出目录可复用完全相同的已登记层。

F4-A共享结构目录（不执行目标、不构造程序实例图）：

```bash
python -B -m scar.cli program-types --out /path/to/type-materials
python -B -m scar.cli program-types --materials /path/to/type-materials --out /path/to/fresh-type-check
```

`program_type_catalog/read_program_type_catalog`保原记录类型、字段语法和实现收据；
`validate_program_shape(graph,max_visits=...)`核exact名义类型与构造约束，返回cross-record/opcode语义未核标记；完整图仍须后层核验。
类型图递归用REF表示，值对象循环拒绝；原图和输入目录不改，不存每次迭代数值。
当前已恢复的共享目录位于`artifacts/current/type-grammars/`；独立fresh记录在`queries/type-grammar-restoration/verification/`。

F4-B接口查询：`ProtocolSignatureIndex(graph, structured, interface, metadata, language=language).match(original_operation_id)`，
入口在`scar.ir.program.protocol_signature_context`。规则来自typed protocol，world/ABI等从原接口绑定；缺原参数返回带字段的
`MISSING_SIGNATURE_CONTEXT`。显式补参仅供诊断，不能覆盖自动绑定或授予源码/变换许可；当前还不是规范图发布入口。

源码产生规则的结构入口在`scar.analysis.source_productions`：`production_type_catalog/read_production_type_catalog`核共享类型目录，
`validate_production_plan/production_plan_document/read_production_plan`核有界记录、原引用类型、区域归属和有序端口。
`ProductionSourceIndex(F0, SourceContextView, FrontendBindingView).match(plan)`复用已严格读取的父层；输入SHA只计算一次，不重parse或复制F2。
源码role port与namespace access分开；属性/keyword标签不能被接成变量读写。输出明确保`source_rules_replayed=False`，当前不能发布canonical新图。

局部表达式规则（F4-B，尚未闭合全域）：

```bash
PYTHONHASHSEED=0 python -B -m scar.cli source-expressions --model /path/to/model --source /original/source.py --out /path/to/materials --jobs 2
PYTHONHASHSEED=0 python -B -m scar.cli source-expressions --model /path/to/model --source /original/source.py --materials /path/to/materials --out /path/to/fresh-check --jobs 2
```

`--source`可重复；省略时处理声明源码全集。每源流式写一个`expressions.jsonl.gz`，共享目录只存一份，fresh逐记录重新产生并比较。
规则目录绑定recipe类型目录摘要；输入语法或记录构造契约变化时拒绝沿用原材料。
材料v2逐原表达式分区为局部recipe、父属表达式、已知不单独求值的源码、规则前沿，计数必须精确覆盖原分母；SUPPRESSED_SOURCE保C1原因/原槽，不冒称缺失规则或删除源码。
`SourceExpressionRules(ProductionSourceIndex, world=…, exception_abi=…).build/verify`不parse目标；按原字段定义顺序，保持state流与短路对象身份。
显式rule frontier保原occurrence/构造/用途；child接口、效果和state来源仍是义务。不能用这个入口宣称完整CFG/SSA或执行图已生成。
调用/显示使用已有PythonProtocol签名，item/pair数量由原字段分组导出；父属描述符另记`PARENT_SOURCE_DESCRIPTOR`，不计作缺失的独立表达式规则。
容器、格式化/模板规则分别在`source_display_rules.py`、`source_string_rules.py`；代码SHA和精确字面值输入类型均进入共享规则目录。
推导式在`source_comprehension_rules.py`；同步/异步子句共用循环模板，异步advance复用已取得的await iterator，按StopAsyncIteration耗尽。原C1求值归属决定eager/惰性异步传播。
`source_suspension_rules.py`按原C1执行owner产生Yield规则，区分普通／异步yield和bare None；normal／raise恢复入口显式接收新world，不能保存旧STATE。
`source_delegation_rules.py`产生Await/YieldFrom共享循环；recipe v3显式保存delegate端口，4种协议强制核签名。采集仍不展开driver/native，准入、close规范化及frame/context保活另验。
`source_callable_rules.py`产生Lambda及GeneratorExp的创建/延迟body；参数、初始默认值和活跃转交cell沿`source_deferred_interfaces.py`接原F3，不在创建时读取cell内容。v5分别表示异步generator的AWAIT推进及ASYNC_YIELD。
语句入口：`python -B -m scar.cli source-statements --model /path/to/model --source /original/source.py --out /path/to/materials --jobs 3`；保相同`--source`选择，加`--materials /path/to/materials`并换独立输出即可fresh核验。当前28类：Expr/Pass/Assign/Delete/Return/Global/Nonlocal/If/While/For/AsyncFor/Break/Continue/AugAssign/AnnAssign/Assert/Raise/Import/ImportFrom/FunctionDef/AsyncFunctionDef/ClassDef/TypeAlias/With/AsyncWith/Try/TryStar/Match。两入口复用collector但保独立domain、catalog和分片，不互认收据。
v6语句suite保normal/return/raise/break/continue端口，循环跳转由原F2 occurrence定位并由对应模板消费。`source_statement_interfaces.py`只给保守child接口，不授child实现、路径可行或清理证明；未来组合须检查实际child出口是该接口的子集。升级须重采集并fresh核验，不改旧材料的version冒充迁移。
v7原子语句在`source_statement_atoms.py`；compile option、inplace operator及annotation publication都有强制接口检查。target和声明地址中的slice/tuple按原父槽分类，避免多生成一次普通写入或容器构造。DEFERRED annotation另计`ANNOTATION_REQUEST` child义务；编译选项未解析和请求body未降低分别保false，不冒称已完成。
v8导入规则在`source_statement_imports.py`；保原alias/module/level引用和已知名字投影，正常路径按原序请求、取成员、绑定。`alias`作为父属描述符单独记账；不能将from-list、成员查找及绑定名字混为同一字符串，也不以一次静态请求证明module body已执行。
v9函数创建在`source_statement_functions.py`；defaults与退出路径复用`source_callable_rules.py`。docstring为父属metadata；函数注解另计`FUNCTION_ANNOTATION_REQUEST`义务，调用准入、请求body和native frame仍未证明。
v10的`source_type_parameter_interfaces.py`接原F3 scope/声明/resolution/capture，`source_type_parameter_rules.py`生成工厂激活和三类type parameter的顺序动作。lazy evaluator另计bound/default request义务；其body与native factory布局仍未降低。v11的`source_definition_interfaces.py`和`source_statement_definitions.py`接类及类型别名，复用泛型工厂与调用参数组装；class callback保独立namespace入口，类型别名另计`TYPE_ALIAS_VALUE_REQUEST`。class namespace初始化/收尾是待展开的typed边界，不能当作纯操作。
v12的`source_statement_contexts.py`接同步/异步with，复用完成路由和await推进；`withitem`为父属描述符。`restore_handled_completions`在所有完成上恢复先前异常状态；不能用未绑定的普通attribute lookup代替type-special方法查找，也不将正常exit结果用于抑制return/break/continue。当前只授局部规则与端口一致，native派发、frame/lasti及引用清理仍未证明。
v13的`source_statement_exceptions.py`接普通try；ExceptHandler为父属描述符，`suite_at`复用原字段列表调度handler body。MatchExceptionType保验证及匹配异常，DeleteExceptionBinding只接受原清理声明；finalbody为一份SHARED模板，通过显式完成路由保留/覆盖pending结果。所有调用须在同一词法activation内，不能借共享访问另一函数的参数或world。
v14的`source_statement_exception_groups.py`接TryStar，clause在同一dispatcher按序构造，不递归复制后续clause。分开body异常收集与分派失败，复用原异常名清理和共享finally。PrepareReraise输入改为异常引用/world，输出仍依明确异常ABI；旧材料须重新生成并fresh，不直接改version。
v15的`source_condition_rules.py`接条件上下文，`SourceExpressionRules.build(occurrence, ProductionPurpose.CONDITION)`返回BOOL/world。材料v3自动遍历CONDITION child请求，在原occurrence分区之后追加`LOCAL_CONTEXT_SOURCE_RULE`记录；`condition_recipes`单列，不冒充新增源码。fallback用同occurrence的VALUE模板，键必须含purpose；`DELEGATED_VALUE_RULE`保留被委托字段。更新后须重新生成并fresh，旧v2材料明确拒绝。
v16的`source_pattern_interfaces.py`从原F2/F3导出捕获、分支一致性和native操作端口；`source_statement_patterns.py`接Match，八类模式由父语句拥有，不单独当表达式求值。匹配失败不输出假捕获，成功后写入再判断guard；失败/后续case用共享模板。局部规则不授native内部实现、引用清理或全图证明。
v17的`source_root_interfaces.py`及`source_root_rules.py`接三类执行根：`python -B -m scar.cli source-roots --model /path/to/model --source /original/source.py --out /path/to/materials --jobs 3`。同源选择加`--materials`并换独立输出即可fresh；根、表达式和语句各保独立domain。FunctionType为非执行语法，不编造入口。入口仅在外部namespace准入之后建模，scope注解函数保留待展开请求，不能据此宣布完整依赖图。
v18的`source_evaluation_interfaces.py`导出原请求环境并做受控编译器字符串化查询，`source_evaluation_rules.py`生成函数注解及类型求值器的共享body；创建操作必须引用真实body ID。材料v4增加`evaluation_request_templates`和`stringized_annotation_values`，原入口/命令不变，须重采集/fresh，旧v3不互认。内核不parse或执行目标；仅字符串化查询恢复原F2子树并静态编译wrapper。模块/类汇总请求仍待展开，全部native/准入证明不由模板存在替代。
v19的`source_class_namespace_interfaces.py`从原scope/resolution/role推导隐式访问及编译metadata，`source_class_namespace_rules.py`展开有序prelude/finalization。旧InitializeClassNamespace/FinalizeClassNamespace类型已删除；按现有三入口重采/fresh即可，不改旧recipe版本。logical name/cell、未知native捕获传递及class注解汇总仍分别保留，不据普通namespace动作已展开宣称全层完成。
v20的`source_capture_interfaces.py`复用原F3活跃hop，并从原C3 FREE证据补充源解析可定位的共享捕获父链；`CaptureReference`明确区分旧整数ID与新复合键。`owned_cells`只是原binding的逻辑cell要求。三入口和材料外壳仍相同，旧实现pin不能直接fresh；须重生成v20 recipe，不改旧header。模块/类注解汇总体、准入和物理frame仍未闭合。
v21的`source_scope_annotation_interfaces.py`为模块/类请求导出原scope、共享捕获和三种目标优化策略的登记表；不parse或执行目标。BuildScopeAnnotations/BuildClassAnnotations现在必须带annotations及body ID；ReadAnnotationRegistration保环境/新world和异常出口。材料v4新增两种request计数，不再把它们记作SCOPE_ANNOTATION_REQUEST整块义务。沿原入口重生成即可，原F2/F3材料不变；源码模板与原编译单元是否保留该代码分开记录。
v22的`source_annotation_state_interfaces.py`导出namespace/cell来源、原投影key和编译参数域；`source_annotation_state_rules.py`展开写入。PublishAnnotation保留为描述符，已退出instruction union；无条件class字段只留metadata。原三入口不变，重生成/fresh读取新recipe；不重签旧版本。绑定编译上下文与完整状态/引用证明仍待完成。
v23的`source_compilation_contexts.py`生成原代码引用和登记生产位置；`source.compilation_contexts.bind(parameter, code_reference, insert_offset)`核原参数及代码版本后返回AnnotationRegistration，不执行或改写目标。常量编码区分tuple/frozenset/slice并保IEEE bits；SourceInput继续授权原F2/F3，Native位置只授本查询范围，不授全CFG/SSA。原材料入口重生成即可。
当前recipe v24还包含`compilation_flow`端口/入口边/编译事实使用表，由`source_compilation_flow.py`机械生成并在source matcher重验。源码recipe必须有表；无源码的shape-only样本可省略。材料外壳仍为v4，旧实现pin不互认，按原三入口重采集/fresh；报告单列端口、边及fact-use数量，native上下文传播证明仍为False。
当前recipe v25另由`source_core_interfaces.py`检查核心签名：profile声明抽象world/异常ABI，变长端口沿原Call/ClassDef、解包、Slice及tuple字段确定。报告单列core签名数量；原三入口和v4材料外壳不变，重采集/fresh后使用新pin。省略profile只适用于无源码的结构样本。
三类规则的build/verify及采集入口均支持`max_visits`/`--max-visits`（默认1,000,000），控制单recipe结构校验预算；该值写入材料limits，fresh必须相同。大根模板可显式提高预算，例如20,000,000；原steps/regions及字节预算继续生效。ProductionError及其子类跨spawn保留code/path/detail/备注，worker错误附原source路径，不再用BrokenProcessPool遮住预算前沿。模板接线和正文展开也传递相同校验选项。

只读模板接线入口为`python -B -m scar.analysis.source_template_links --model MODEL --out FRESH_OUT --materials expressions EXPR_COLLECTION EXPR_VERIFICATION --materials statements STMT_COLLECTION STMT_VERIFICATION --materials roots ROOT_COLLECTION ROOT_VERIFICATION`；可加`--source PATH`精确选择输入，未选时要求全模型三域材料齐全。仅接受已独立fresh的相同生成器/输入版本。每源分别存templates/links两张gzip JSONL表及小header，正文保留在父材料；`read_template_body(store, unit, SourceTemplateKey(occurrence, purpose))`核收据后展开。该入口不修改采集生成器或canonical图。
接线可用`--jobs N`（1至32个spawn进程）按源文件并行；每个worker缓存自己的只读父索引，独立写本源分片，不共享可变图或FD表。主进程只核父收据/版本并汇总单位收据，输出的记录顺序及分片字节不随worker完成顺序变化。
接线时另核局部world的来源顺序：步骤及内嵌调用只读当前world，正常state结果替换后续world；只读协议无需制造新state，循环world必须走携带端口。`local_state_sequence_checked`只表示这项依赖约束，不替代native效果或引用清理证明；采集生成器和recipe版本不变。
`read_template_graph(store, unit)`读回强类型模板及连接，核原输入、state/异常profile、分片哈希/计数及模板引用闭合；父目录必须匹配当前类型/协议目录及编译器header/stack guideline。`verify_template_store(store, jobs=16)`按源并行读回全部分片，只汇总收据，不集中持有整图，也不重新授予源码规则或native效果证明。`read_template_body`在此基础上按原材料展开正文。
星号/keyword的失败顺序参考固定版本[CPython codegen](https://github.com/python/cpython/blob/v3.14.7/Python/codegen.c)；本机stack guideline从header读入并固定SHA。

源码身份桥（按unit分片，保持输入模型和旧ID）：

```bash
python -B -m scar.cli source-anchors --model /path/to/model --out /path/to/materials --jobs 8
python -B -m scar.cli source-anchors --model /path/to/model --materials /path/to/materials --out /path/to/fresh-check --jobs 8
```

两入口均重验F2上游；身份生成本身只读原occurrence字段，不再解析或恢复AST。无旧SG的unit明确保留source-only处置。
`SourceAtomAnchorView.atom/occurrence/field`按原ID/字段查询；None只表示非旧Atom域构造，低层view不自授fresh源码证明。

## 已有建图入口（前端形式验收尚未完成）

```bash
python -B -m scar.cli static /path/to/program --out /path/to/model --phase bodies --jobs 4
python -B -m scar.cli static /path/to/program --out /path/to/model --expand --jobs 4
python -B -m scar.cli static /path/to/program --out /path/to/model --link-calls --max-call-bodies 80
```

`--phase sources/graph` 为材料目录/源码图，`--phase bodies` 继续构建规范主体；
`--expand` 增量补主体。显式预算仅限本次工作，不缩小完整性分母。
依赖材料可用 `--inventory-dependencies --dependency-root PATH` 定位，
再用 `--expand-dependencies` 送回同一前端；这些命令均不执行目标。

## 动态采集及离线连接

```bash
python -B -m scar.cli trace --calls-only \
  --capture-plan /path/to/plan.json --capture-plan-sha256 PLAN_SHA256 \
  --capture-transport snapshot --project-root /path/to/program-root \
  --out /path/to/fresh-trace -- python /path/to/program.py
python -B -m scar.cli observed-calls --model /path/to/model \
  --trace /path/to/fresh-trace --out /path/to/fresh-bindings --expand-missing
python -B -m scar.cli observed-graph --model /path/to/model --expand-missing \
  --trace /path/to/fresh-trace --out /path/to/fresh-observed-graph
```

采集计划由 `scar.trace.lean_capture.LeanCapturePlan` 对原源码/代码身份生成；具体接口以源码为准。
采集不额外执行用户 repr、descriptor 或 hook 来探测对象，不保存每次 Tensor 的完整数值。
优先复用已保存 trace；`observed-calls` 只离线连接，缺主体按完整代码身份请求原前端补建。
`--expand-missing` 先核完整Code/profile及已有plan文件指纹；无历史指纹时可补当前源码候选，
再按匹配Code作用域补body并fresh接线，不将父模块/环境认作历史事实；
不执行源码、不猜import名，旧分片保留但覆盖仍限旧basis。文件/字节预算用`--max-source-files/bytes`。
已有graph=None诊断按当前静态规则判是否可重试；成功补建保留旧收据，已有规范图不覆盖。
尚缺static adapter时明确报告形状/入口需求，不能把artifact文件当已建图。
采集仍含 frame profile hooks，不能宣称零干扰；性能测量与建图采集分开。
需要实际 callable 类型/owner/ABI 时加 `trace --native-callables`：仅在新 trace 内隔离编译小 C probe，
不安装环境依赖；当前限 CPython 3.14.7/GIL/Linux x86-64，不执行用户 getter 或读取 Tensor 数值。

## 共享图查询

```python
from scar.analysis.program_dependency_session import ProgramDependencySession

with ProgramDependencySession('/path/to/model') as session:
    ns, = session.select_bodies(file='program.py', entry='function_name', kind='FUNCTION')
    body = session.load(ns)                 # 原共享函数图及 index
    loops = session.loop_constructs(ns)     # 原成员 ID、嵌套、初始/回边端口
    if loops:
        operations = loops[0].expand(session.graph())
        boundary = session.loop_boundary(ns, loops[0])
```

不存在的 structured/主体信息保留准确缺口，不当成空子图。`ProgramDependencyBudget`
控制时间、输入、缓存和关系读取；边界查询完成仍须检查前沿和字段覆盖。
CLI 的 `static-query --program-graph` 使用原 typed ID，不按名称制造依赖。
动态接口用 `session.attach_observations(trace)` 接入，再由 `session.graph().observed_incoming/outgoing(port)`
查询；作用域与端口由返回的provider提供。原trace重验后才授边，DTO读回不是证明。
`graph.observed_layouts(scope)` 返回条件ABI布局；v2显式保存布局，v1读回为NOT_RECORDED而非空布局。
`graph.observed_native_callables(scope)` 查询附着原 CALL/Use 的实际类型、owner、self、flags 和地址事实；
未对应静态 CALL 的记录保留 frontier。导出为v3，v1/v2的这些字段读回为None=NOT_RECORDED。
类型名称只是描述标签，地址只适用该进程事件；它们不单独证明源码实现、kernel、copy 或效果。
有native采集时，`observed-graph` 自动生成共享协议分片并导出v4；`--native-protocols DIR` 重放已有分片。
API以 `provider.build_native_protocols(fresh_dir)` 生成、`attach_native_protocols(dir)` 核验接入；
`provider.native_protocols` 返回可展开的原canonical模板。协议输入/声明出口没有实际Invocation，witness仅为原CALL。
旧v1/v2/v3读回的协议字段为None；异常引用包未适配时仍保已知world，不把整个出口丢掉。

## 维护

当前任务及验收结果只更新 [STATUS](STATUS.md)，结构变化更新 [MODEL](MODEL.md) 或
[ARCHITECTURE](ARCHITECTURE.md)；不新增专题、版本或 gate 文档。
先定位缺失的识别事实、端口、连接或推理规则，再修改负责层；只运行相关检查。
生成图仅发布到 `artifacts/current/static/`，查询只引用原 ID/收据，不另建权威程序。

正常路径图与跨文件可能调用（静态，不执行目标）：

```bash
python -B -m scar.analysis.normal_graph --templates TEMPLATE_STORE --out FRESH_GRAPH --jobs 16
python -B -m scar.analysis.normal_links --model MODEL --templates TEMPLATE_STORE --out FRESH_FACTS --jobs 24
python -B -m scar.analysis.normal_targets --facts FRESH_FACTS --entry /absolute/path/to/entry.py --out FRESH_TARGETS
```

原操作/区域正文共享，边表与连接事实按unit分片；可能调用另存稀疏表。
`CallLink`保留实参、形参、默认值、环境及状态依赖，源码目标仍带准入条件。
当前固定参数调用传播不覆盖全部动态分派，不能由工作队列为空宣布全程序建图完成。

跨文件图查询（只读，不重新建图）：

```bash
python -B -m scar.analysis.normal_query --targets TARGET_STORE --source /path/to/program.py
```

`NormalCallGraph.calls(source, occurrence=...)`返回原有类型的可能调用连接；
`expand(link)`沿收据展开原共享body，使用`link.body.region`定位区域。
展开不代表执行、目标唯一或具备优化许可。展开参数保留实际包端口，未确定内容时不捏造形参别名。

容器/运算内容约束使用同一提取/求解入口，当前事实schema为v4、接线结果为v4；旧收据不改签。
`NormalCallGraph.contents(source)`读取以该源为消费者的可能内容边；CLI加`--content`汇总。
边只描述引用可能来自哪个内容位置，不证明某次调用的内容版本、无副作用或复用合法性。

`NormalCallGraph.namespace_lookups(source)`读取名字空间查找及开放绑定；CLI加`--namespaces`查看。
记录保留原接收者/结果端口和源码模块；导入路径只是标准导入条件下的候选，不能当已加载对象。
原生接口提供方可沿这些端口继续接线，源码候选数0不表示该成员不存在。

生成算子接口与原始调用接线（静态工具进程，不执行目标）：

```bash
python -B -m scar.analysis.python_bindings --targets TARGET_STORE --out FRESH_BINDINGS --jobs 4
python -B -m scar.analysis.python_binding_query --graph FRESH_BINDINGS --symbol torch.randn --source /path/to/program.py
python -B -m scar.analysis.native_resource_declarations --bindings FRESH_BINDINGS --out FRESH_RESOURCE_DECLARATIONS
```

`PythonBindingGraph.definitions/native_calls/normalization/expand_native/wrapper`沿共享定义查询；
`NormalCallGraph.call_sites`独立于目标推导，保留完整原调用ABI。转换依赖不是值别名。
绑定图schema v3另提供`return_types/declared_members/declared_protocols/declared_operators/operator_frontiers(source)`，CLI按源汇总。
`NormalCallGraph.language_operations(source)`独立读取全部原运算接口，包括尚无协议目标的操作；循环不unroll。
运算协议从AST selector识别正向、反向、原地与比较分支；类型、NotImplemented与优先级保条件，不用输入到结果的alias边代替运算。
返回契约来自隔离torchgen及安装版wrap规则；Tensor接口从固定源码/pyi抽取，不import torch。
Tensor方法、元组投影和复制后内容只作条件类型传播；原调用ABI/world不变，未知索引保留整数协议准入条件。
生成目录按export及源unit分片；算子接口与软件图共享原端口，不复制函数/循环正文。
当前后端目标固定于`pyproject.toml [tool.scar.codegen]`：C++20、operator ABI v1、plan schema v1。
第一阶段内只固定接口/编译基线，不实现优化或调度runtime。

目标硬件图及软件资源关系（不运行目标）：

```bash
python -B -m scar.analysis.target_graph --out FRESH_TARGET --path /path/to/program --path /path/to/data
python -B -m scar.analysis.software_resources --bindings BINDINGS --declarations RESOURCE_DECLARATIONS --target FRESH_TARGET --out FRESH_RELATIONS
```

`TargetHardwareGraph.resources/connections/paths`读取目标描述；`SoftwareResourceGraph.sources/candidates/artifact_locations`
读取原软件关系。目标ID由snapshot限定，inventory编号不直接当框架device ordinal；位置/设备候选需满足各自条件。
容量是平台报告的total，预算和实际使用独立；PCIe max不是实测带宽，Optional/List类型路径不展开为实际资源实例。
软件资源关系schema v2将显式调用和语言协议的规范化依赖分开标记；原运算端口不伪装成SourceCallSite。

全域正常路径图可用`python -B -m scar.analysis.normal_graph_collection --templates TEMPLATES --graph STORE_A --graph STORE_B --out FRESH_DIRECTORY`建立小索引。
`NormalGraphCollection.role/source_edges`按原unit查询；大边表保留在原store，类型声明有独立role，不进入runtime候选。
读取会话固定codec/P/G版本并预编译不可变标签上下文；逐边解码及压缩/解压SHA、大小、计数、父引用校验保留。

事实schema v6保留完整原ABI，并用原code_scope分组；`NormalCallGraph.attribute_sites/properties/attribute_storage`分别查询原操作与条件接线。
`entry_activations/module_demands/body_demands/call_demands`查询激活与未决接口；队列收敛不代表这些需求已解决。
`python -B -m scar.analysis.source_state_connection --boundaries BOUNDARIES --targets TARGETS --out FRESH_JOIN`
把声明状态边界接到原CallLink的实参/形参、数据返回和world；结果保原表行号与scope/alias依据。
`read_source_state_connections`重新推导接点；0连接是未决前沿，不把候选内容当作当前任务状态。
按需字段/方法图：`python -B -m scar.analysis.source_field_graph --relations RELATIONS --catalog CURRENT_CONTEXT --unit UNIT --call CALL_ORDINAL --out FRESH_GRAPH`。
不传`--call`时自动识别该unit中field GET→method GET→call的结构。按unit/call分片，引用原source/call表，不运行目标或全域求解。
`python -B -m scar.analysis.source_field_graph --read GRAPH`检查原证据、调用栈与方法ABI；字段writer域、当前版本及dispatch仍保条件。
`python -B -m scar.analysis.source_declared_graph --parent FIELD_GRAPH --out FRESH_CONTEXTS`展开共享声明正文；`--read CONTEXTS`核原路径和descriptor/ABI。
`python -B -m scar.analysis.source_declared_state --boundaries BOUNDARIES --contexts CONTEXTS --out FRESH_JOIN`连接内容/world状态接口；`--read JOIN`重新推导。
这两层只保条件关系，不授入口准入、运行dispatch或当前状态版本。
`python -B -m scar.analysis.source_world_flow --join DECLARED_STATE_JOIN --out FRESH_WORLD_GRAPH`从原源码模板投影调用后STATE/CONTROL关系；`--read GRAPH`重核原计划。
循环保回边，protocol world变化单独表示。当前材料适配只覆盖普通源码单元，compiled变体显式留边界；不运行目标或全局solver。
`python -B -m scar.analysis.source_addressed_world --world WORLD_GRAPH --catalog CURRENT_CONTEXT --out FRESH_ADDRESSES`关联地址路径与world；`--read ADDRESSES`重核。
字段/index访问保根与selectors，call保输入来源及开放效果；不以符号路径相等证明动态对象或内容版本相等。
`python -B -m scar.analysis.source_world_returns --world WORLD_GRAPH --out FRESH_RETURNS --max-nodes 10 --max-edges 12`连接跨函数返回；`--read RETURNS`核验。
完整认证world父链一次后按原report SHA及表收据复用祖先；同源码单元的历史调用者一次扫描，预算前沿显式保留。
`python -B -m scar.analysis.source_reference_consumers --returns RETURN_GRAPH --out FRESH_CONSUMERS`连接原实参/形参和状态接口；`--read CONSUMERS`重核。
完整认证返回图一次后复用祖先；所有选中调用点共用一次历史关系表扫描。未知callee效果、未解析pack及构造协议保留前沿。
`python -B -m scar.analysis.source_value_flow --consumers CONSUMERS --out FRESH_VALUES`为共享正文投影原操作、数据/状态/控制边及局部位置；`--read VALUES`重核。
body与entry分表共享，不按调用或循环复制。输入来源查询是may-dependency，不授alias/内容等价；报告分开父链认证、正文投影和查询耗时。
`python -B -m scar.analysis.source_value_interface_graph --values VALUES --out FRESH_INTERFACES`连接派生调用输入的原source接口；`--read INTERFACES`重核。
同一进程可共享`ValueInterfaceParents`给build/read，父图完整认证一次，再以原report/表收据保有效性；不改变目标选择或内容证明强度。
`python -B -m scar.analysis.source_body_expansion --interfaces INTERFACES --out FRESH_BODIES`展开原source/compiled函数正文；`--read BODIES`重核。
`source_graph_material(catalog, unit)`统一普通/compiled材料读取。正文references表引用原父表或新增body表，entry/reach的body序号指references表；不复制旧正文。
`python -B -m scar.analysis.source_native_binding_materials --site SITE_PACKAGES --out FRESH_DECLARATIONS [--catalog RAW_MATERIALS]`独立收集生成绑定声明；`--read DECLARATIONS`重核工具、源码和原定义。
本目录不依赖旧source solver，不选择导出/overload，不连接native或资源；接口继续使用既有PythonBinding类型。原`source_binding_materials.py`仍属于C2b前端。
`python -B -m scar.analysis.source_call_context_graph --bodies BODY_EXPANSION --out FRESH_CONTEXTS`保存选中incoming接口的callee引用查询及独立构造阶段；`--read CONTEXTS`重核。
查询栈索引interfaces表，roots引用原body entry；外层caller未选定即保边界。构造sites/steps/entries分表，保原Endpoint与connection，不把init数据接到构造结果。
`source_constructor_contexts.constructor_context_inputs(ConstructorInterfaceGraph)`从原单参数route投影兼容class输入，供后续声明方法展开；不选择MRO或运行时精确class。
`python -B -m scar.analysis.source_constructor_members --calls CALL_CONTEXT_GRAPH --out FRESH_MEMBERS`连接构造class输入、声明成员接口及共享正文；`--read MEMBERS`重核。
`python -B -m scar.analysis.source_class_base_graph --catalog CONTEXT_CATALOG --inputs MEMBERS_OR_CLASS_INPUTS --out FRESH_BASES`建立有序base与namespace来源图；`--read BASES`重核。
也可重复`--class-value UNIT LOCAL_PORT`选择任意已保留类定义。根是静态查询选择，不授程序入口、当前类型或继承关系证明；本层不扫描历史calls/constructions表。
`python -B -m scar.analysis.source_namespace_callees --calls CALL_CONTEXT_GRAPH --out FRESH_NAMESPACES`将namespace来源接到原callee查询/state；`--read GRAPH`重推核验。
`python -B -m scar.analysis.source_inherited_member_graph --calls CALL_CONTEXT_GRAPH --out FRESH_INHERITED`接继承成员、原ABI与共享正文；`--read GRAPH`重核路径及正文。
两者build/read可复用同一个`ConstructorMemberParents`，完整父链只认证一次；不合并caller，不证明实际MRO/dispatch。
`python -B -m scar.analysis.source_connected_bodies --namespaces NAMESPACE_GRAPH --members INHERITED_GRAPH --out FRESH_BODIES`连接两类接口、共享正文及下游callee查询；`--read GRAPH`重推。
`ConnectedBodyParents(..., common=certified_constructor_parents)`复用原父链；`interface_scopes`记录派生返回的完整caller许可域。读取时不能把所有接口都经普通`admit`授予通用域。
`python -B -m scar.analysis.source_declared_calls --catalog CONTEXT_CATALOG --source-root SOURCE_TREE --out FRESH_CALLS --jobs 4`按unit并行生成全部原callsite的namespace子图；可重复`--source-root`。
`DeclaredCallStore.load_unit(unit)`核收据/ABI/owners/请求；`read_unit(unit)`再重推该unit。源码声明不授scope激活或caller选择，compiled变体保在既有正文入口。
`python -B -m scar.analysis.source_named_bindings --calls FRESH_CALLS --bindings GENERATED_DECLARATIONS --out FRESH_BINDINGS --jobs 4`按完整符号保所有overload及原ABI；`--read GRAPH --unit UNIT`单unit重推匹配。未匹配不能解释为没有native实现。
`python -B -m scar.analysis.source_binding_native_graph --named FRESH_BINDINGS --out FRESH_NATIVE --jobs 4`复用隔离parser和canonical lowerer，将精确selector、原实参及wrapper声明接至native端口；`--read GRAPH --unit UNIT`核全部native材料/图及该unit归一化。
输入/返回包装、world依赖独立保留；不假设Python返回与native slots一一对应，也不声明wrapper前后state版本相等。
上述三个模块也可通过`scar.cli call-declarations / native-bindings / native-interfaces`使用。
`python -B -m scar.cli resource-graph --native FRESH_NATIVE --target TARGET_HARDWARE --out FRESH_RESOURCES`复用库存接资源类型与源码依赖；`--read GRAPH --unit UNIT`重推全局类型/目标及选中unit路由，不重新探测硬件。
`python -B -m scar.cli name-flow --declared FRESH_CALLS --unit UNIT --out FRESH_NAMES --jobs 4`按单元计算正常控制路径的词法读写来源；省略`--unit`选择全部声明单元，可用`--frontier REASON`机械选择原查询前沿。
`--read FRESH_NAMES --unit UNIT`重推该unit的全部选中查询/到达定义。schema v2中anchors、函数bodies各存一份，tracks按ID引用；同函数的值流投影也只计算一次。原声明与异常/resume分支不被正常路径过滤覆盖。
词法槽版本不等于Tensor内容版本；未知回调/捕获来源保前沿，不能将过滤结果作为cache或删除操作的许可。
`python -B -m scar.analysis.source_fact_materials --catalog CONTEXT_CATALOG --graph GRAPH_A --graph GRAPH_B --out FRESH_MATERIALS`核所列图的全部源事实引用；`--read INDEX`重核。
`read_fact_materials(INDEX)`返回report/resolver；`resolver.resolve(reference)`核目录路由、分片bytes/decoded收据和locator，返回原locator与实际物化artifact。
旧collection相对URI不能直接当文件路径；compiled的records/0按原JSONL记录解释。共享父body所在图须显式列入`--graph`，索引不暗示未列祖先已验证。
`python -B -m scar.analysis.source_admission --targets TARGET_STORE --out FRESH_PLAN`支持普通及context图，按真实source provider关联需求与原依赖材料，不修改模型。
对旧inventory未索引的实际module需求，用原候选roots定点查找；可重复`--search-root`指定候选root。
新搜索表与原inventory的索引域分开，未固定内容版本不列ready；搜索根及文件存在均不证明运行时import选择。
`python -B -m scar.analysis.source_dependency_frontend --model ORIGINAL_MODEL --material-index INDEX --out FRESH_ROOT --jobs 4`将选中的原指纹源码经过既有F0--F5，写独立model、验收与facts；不扫目录、不改原模型、不运行目标。
`python -B -m scar.analysis.normal_facts --parent BASE_FACTS --parent EXTRA_FACTS --out FRESH_CATALOG`引用原分片组成多父目录，正文复制0 bytes；按源展开使用所属父模型和模板，旧收据不重签。
binding图schema v4继承入口范围，按scope增量订阅native协议，并发布本轮增强的source关系；不得退回全unit eager分析后宣称入口闭合。
静态函数接线在`source_function_connections.py`；原`source_invocations.py`继续用于动态适配，两个入口相互独立。

有限值字段：`python -B -m scar.analysis.source_scalar_graph --targets TARGET_STORE --out FRESH_FIELDS`；
可重复传`--source PATH`只采该源码slice，仍保父图跨层来源。`ScalarGraphFields(path).rows(table)`按类型读取
fields/groups/operations/derivations；先核收据，逐行解码，不展开content因子。固定点只针对有限字段域，不是must常量证明。

源码文本候选：`python -B -m scar.analysis.source_text_values --facts FACTS --entry-contract ENTRY --out FRESH_TEXT`。
编译输入/代码值：`python -B -m scar.analysis.source_compiled_values --facts FACTS --executions BUILTIN_OVERLAY --texts TEXT_OVERLAY --out FRESH_CODE`。
这两层不运行目标；保默认编码及路径版本条件，不把code digest或source-only控制版本当F4 body许可。
