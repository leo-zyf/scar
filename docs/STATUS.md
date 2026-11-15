# 当前状态

SCAR 已有三阶段贯通的 CPU 原型：原源码图和硬件 → 依赖/资源计划 → C++20 → 普通入口验证及成本门控执行。**三个阶段的通用覆盖仍未完成**，不能把受支持区域的闭环当成任意 Python 程序全部编译完成。

## 推进层次与当前验收点

先贯通主链，再按下面各层扩大覆盖；每层只消费上游已准入材料，记录输入、产物、明确缺口和实际验收范围。某一受限程序跑通不代表整层完备。

| 阶段 | 有序层次 |
|---|---|
| 建图 | G0 源码/编译上下文与版本 → G1 token/语法/作用域 → G2 局部操作、值、状态及控制端口 → G3 跨函数/模块的目标、参数、返回、world/alias连接 → G4 动态与硬件证据、覆盖/边界 |
| 优化 | O0 原图准入与依赖事实 → O1 结构化简及合法性 → O2 实现、内存和资源成本域 → O3 联合计划选择 → O4 计划重放与NO_OP |
| 交付 | C0 已选计划准入 → C1 C++20控制/算子ABI降低 → C2 紧凑包、guard与回退 → C3 正确性、干净A/B及部署门控 |

三阶段主链已在受支持区域贯通。当前顺序是：① G2/G3 按 region 展开并同步登记 state/content 请求；② O2 补独立融合成本，再交 O3 选择；③ C1 将普通 caller 图接入已有 PipelinePlan。每层保留原 ID 和上游证据，验收通过再扩大范围。

## 第一阶段：建图

- F0–F4 保存原源码、绑定、控制、值和 world 端口；函数和循环使用共享正文，不逐次展开。原始输入到优化计划的引用在降低前重新核验。
- 新增 `observe-call-bindings --all-sources`：从已准入源码机械生成模块/函数 CALL 选择器，一次短运行采集，达到各调用点预算即停止记录。没有全局 profile，不保存 Tensor 数值；当前通道限定 CPython 3.14.7、64 个代码体/512 个调用点。
- 实际 Gray–Scott 普通入口及库共 2 个源码单元：25 层前端完成，4235 个原操作、51425 个规范端点、55018 条规范关系，局部图/状态分片约 8.82MB。前端 144.989s；短动态运行 1.480s，103 个选择点中观察 68 个实际目标。
- 新增跨文件普通 `from … import … as …` 来源解析：复用原 F4 namespace/import/export 事实和全局有限固定点，按 `(unit, template)` 共享函数子图；source过滤只选输出根，仍保留依赖单元。4文件样例一次前端52.299s，统一建图7.340s、域规划11.803s；歧义导出、循环依赖和输入写入分别拒绝，未执行源码或扫描文件系统猜目标。
- Python CodeID 回接共享正文使用 `IMPLEMENTATION` 关系，不冒充动态调用完成或状态版本证明。新回接复用F4 body scope→C3 compiler constant path→完整descriptor，候选同时覆盖保留快照与replay filename；缓存按完整descriptor隔离，不凭同一个fingerprint借用冲突字段。
- 现已补上上述 metadata 到条件函数 ABI 的桥接：复用原签名绑定及匹配 frame，参数、world、正常返回可往返，多个调用点不混线。默认参数读取当前逻辑槽，pack 不冒充引用别名，deferred 无普通返回；原属性 GET/alias 可提供标准实例方法的条件 receiver。每种绑定分支有独立原证据选择，仍不证明实际 receiver、调用完成、纯性或数组内容版本。
- 实际验收：四文件短采集0.672s/26记录，6接口/40跨层关系；复用Gray–Scott已有37记录，10接口/64跨层关系（含4个条件receiver和1个当前default槽）。全部关系可在统一主图查到，缺失0。报告共享引用表分别约110KB/180KB；没有重跑前端、展开迭代或新增变换。Gray旧静态全目录inventory/接口因规则不匹配被明确拒绝，当前不宣称14单元物化覆盖已恢复。
- 已实现分级材料准入：全部header/parent/接口gzip字节仍校验，外部UnitFacts及辅助接口首次查询时调用原reader完整核验；失败不发布成功缓存。v4目录明确绑定reader规则，旧v3由显式producer SHA在新目录重发合同，原12份leaf/interface收据不变。
- 实际目录验收保留14单元、495591端口；冷时12份外部facts/interfaces均未加载，一次原接口查询0.301s通过类型、family和witness核验，其余11份保持冷。实际Gray仍37记录→10接口→64主图关系，缺失0；完整查询单次诊断102.325s→55.661s（独立打开101.577s→54.817s，64次邻接合计0.412s）。这是SCAR分析延迟，不是目标程序加速基准。当前约105.6MB模板图仍eager；目录/byte准入不冒充全部语义已验证。证据：`artifacts/baseline/demand-interface-admission.json`。
- 当前硬件图保留全部 204 个资源、477 条连接，包括 CPU/SMT/ISA、NUMA/DRAM/cache、PCIe、存储及网络库存。硬件存在、可访问、后端支持和实际使用分别表达。
- 实际修正此前13个“目录外”误报：同一37条Gray记录中的23个Python目标全部匹配保留正文，产生27个条件ABI alternative（含descriptor分支）、19个共享正文；匹配阶段7个外部单元没有正规化template。按原CALL查询准入，先检查所属scope，再展开所选接口。另选最小外部leaf的一次真实跨文件查询，6条参数/env/world/正常返回关系均在主图可见，缺失0；其他73个原query明确未选择。材料重开与完整身份验证仍使这次检查耗时119.534s，不能与旧10接口范围的55.661s混作性能比较；不宣称27个ABI正文已全部展开。证据：`artifacts/baseline/retained-code-bridge.json`。
- region 发射层已实现：真实循环模板单区生成24条边，5区合并与原109条边顺序一致，保留13个跨区端口及循环回边，未展开迭代。此验收尚未将主 provider/state/content 切换为增量模式；下一层接通该生命周期。

## 第二阶段：化简及硬件规划

- 原 F4 绑定写入推导 carry，CALL 参数映射与 Expr 依赖闭包推导不变量。支持单纯函数及多层纯函数组合；composite 使用独立 proof、融合证书和全部函数全局路径，不把融合图当作真实 Python callable。
- 规划比较循环内重算、整数组物化、逐元素时间融合；支持有序可变状态区域、buffer 计划及明确 CPU 绑定。逻辑工作减少不等于墙钟加速，局部 scalar 不等于已证明寄存器分配。
- 多核 C++ 按确定行分区运行，保留每个元素原运算顺序、worker affinity 与浮点环境。显式线程预览不宣称线程数最优。
- 独立固定原语校准：一次编译、3.273s，两个固定计算家族 × 1/2/4 个 CPU。前向模型使用串行组件之和加最慢 worker chunk；不把串行时间除以线程数，不轮流测试目标实现。
- 完整自动选择已推进到每 NUMA 的 1–28 线程及 SMT 代表域，共 **106 个 mask**：复用先前 14 个，只补采 92 个，全部成本有独立固定原语覆盖。原函数有限边界观察另取得同上下文原执行成本（中位数 604.747ms），不记录 Tensor 数值、不插桩函数内部，也不冒充干净完整入口基线。
- `compile-build --loop-cpu-model` 机械比较完整 106 个成本和原执行，选出 CPU 0–13/28–41 的 **28 workers**，声明模型预测区域 13.740ms。只降低这一种目标实现；未试跑其他目标变体。域不包含任意 CPU 子集、跨 socket 团队或 NUMA 绑页，因此不是物理全局最优。
- 复合区域也已进入完整自动CPU成本链：复用原106配置的数学校准，单独补充数字化函数路由/源码字节读取的规范边界模型（额外中位数59.729μs），不把原direct边界直接当作完整复合边界。source-unit namespace分组只是规范模型，不证明实际globals身份或别名。OMP/GOMP/KMP控制环境参与成本适用性，改变时旧事实不能复用。
- 固定原语到目标图只按匿名有序值依赖、常量和资源上下文匹配，不按函数/程序名匹配。早先缺成本的 cap4 结果和单独合成完整模型记录仍保留；不得拿合成成本作硬件性能证据。批量投影每份原观测只解析/索引一次，再写独立事实分片。
- 已修正stateful校准准入：`target_workload_executed`及`target_variants_ranked`必须显式False，缺失或True均保留拒绝证据与UNKNOWN成本。真实Gray的106mask主链接口已存在；旧融合阶段模型holdout误差22.8316%超冻结20%门槛，且旧校准硬件context不匹配，不能复制temporal时间值或降门槛。
- 原始状态method成本采集已接入主规划链：仅在guard外计首个原调用，每个样本独立进程，不加载native变体、不记录数组内容。真实512×512、4步入口的单次method采样为6.970ms中位数（6.671–7.095ms，stdev0.218ms），预备1个进程排除、正式3个进程；预备进程不预热正式进程的cache。这是成本模型输入，不是干净完整入口性能。主规划读入后NO_OP成本已知，106个native成本仍未知，保留原执行。完整OMP/GOMP/KMP控制环境纳入成本上下文。证据：`artifacts/baseline/stateful-original-observation.json`。

## 第三阶段：C++20 交付及真实结果

- 原图 → 计划 → C++20 → 同一 ELF 导出 → 普通 Python 入口已贯通。运行包无需 compiler、IR、planner、codegen 或 g++；状态包加载仍需要原 NumPy ufunc 提供者；入口和明确的兼容/I/O 仍保留 Python，完整程序不是纯 C++。
- 跨文件组合自动选择：4个准入源码文件，3个库文件、4条真实函数全局路径，512×512/1000步，完整106配置自动选28线程。预热1组、正式3组：原完整进程中位数849.953ms（800.455–854.940ms，stdev30.121ms），选中实现261.466ms（250.753–282.104ms，stdev15.935ms），3.2507×；输出一致，native4/fallback0，正常成本门控另native1/fallback0。它是跨模块编译回归，不是新的LeWM或三程序套件加速。
- 跨函数组合实际普通入口：256×256/20 步，结果逐位相同，native 1/fallback 0。替换深层 binding/defaults/代码/函数 globals 四种情况均准确整循环回退一次；恢复后重新启用 native。
- 自动线程选择的科学微程序：512×512/1000 步、float64，完整进程预热 1 组、正式交替 3 组。原版中位数 **857.374ms**（854.495–872.810ms，stdev 9.849ms），选中实现 **287.887ms**（287.682–309.074ms，stdev 12.292ms），**2.9782×**。输出逐位一致，预热及正式 native 4/fallback 0；正常成本门控入口另 native 1/fallback 0。这是编译回归程序，不替代真实三程序套件。
- 真实 Gray–Scott 完整入口：512×512/1000 步、float64、C++ 单线程 CPU 0；当前运行端版本重新预热 1 组，交替正式 3 组。原版中位数 **11.136s**（11.097–11.189s，stdev 0.046s），优化 **3.939s**（3.916–3.943s，stdev 0.015s），**2.8271×**。包括进程启动、导入、应用、I/O 和报告开销；这不是纯 kernel 计时或未来性能保证。
- 区域/单步/20 步轨迹/完整积分、u/v、必要状态、数组身份、stdout 和文件逐层通过；完整进程测量 native 4000/fallback 0。随后正常成本门控入口 native 1000/fallback 0；改变 argv 后保留原执行。
- 上述区域配置是明确单线程验证预览，尚未证明最佳。完整域成本缺口未被这份测量抹去。baseline 的最终 dt=0 行为原样保留。
- 先前整数组物化 LICM 的 0.9625× 继续记录为 not profitable；不得与本次不同实现/计时边界混算。Sunfish/Cirq 仍仅有基线，无新增加速结论；当前无 GPU，不作 LeWM/GPU 实测加速声明。

证据：`artifacts/current/experiments/stateful-entry-v1/`、`composite-licm-v1/`、`parallel-temporal-delivery-v1/`、`temporal-resource-model-v1/`、`automatic-temporal-cpu-v1/`、`cross-unit-composite-v1/`。摘要：`artifacts/baseline/three-stage-system.json` 、`automatic-temporal-cpu.json` 和 `cross-unit-composite.json`。成本计划 7.50MB 在构建时保留；运行端只核准入合同中的原计划哈希引用，不读取该计划或源码图。

## 接下来仍需完成

1. 建图：普通条件ABI与保留外部Code身份已接通；接下来做region级正规化与state/content增量请求，再补别名内容版本、创建环境及原生schema。不能仅将state初始化改成部分template而漏掉后续effect请求。
2. 规划：把完整 CPU 成本链扩展到有序状态和更多算子家族，打通一般嵌套/自适应循环与跨区域调度；之后扩大 NUMA 绑页、多进程、MPI 和 CPU/GPU 联合域。当前自动选择不覆盖 Gray–Scott 的完整 CPU 成本域。
3. 交付：已有多叶PipelinePlan/RegionCall及跨文件composite C++20 host；缺口是普通caller-matched图到可重放闭合PipelinePlan的桥接，之后扩大dtype/控制及宿主，目前尚未提供任意程序独立C++交付。
