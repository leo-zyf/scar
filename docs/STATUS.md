# 当前状态

SCAR 已有三阶段贯通的 CPU 原型：原源码图和硬件 → 依赖/资源计划 → C++20 → 普通入口验证及成本门控执行。**三个阶段的通用覆盖仍未完成**，不能把受支持区域的闭环当成任意 Python 程序全部编译完成。

## 推进层次与当前验收点

先贯通主链，再按下面各层扩大覆盖；每层只消费上游已准入材料，记录输入、产物、明确缺口和实际验收范围。某一受限程序跑通不代表整层完备。

| 阶段 | 有序层次 |
|---|---|
| 建图 | G0 源码/编译上下文与版本 → G1 token/语法/作用域 → G2 局部操作、值、状态及控制端口 → G3 跨函数/模块的目标、参数、返回、world/alias连接 → G4 动态与硬件证据、覆盖/边界 |
| 优化 | O0 原图准入与依赖事实 → O1 结构化简及合法性 → O2 实现、内存和资源成本域 → O3 联合计划选择 → O4 计划重放与NO_OP |
| 交付 | C0 已选计划准入 → C1 C++20控制/算子ABI降低 → C2 紧凑包、guard与回退 → C3 正确性、干净A/B及部署门控 |

三阶段主链已在受支持区域贯通。本轮推进自动内容子图、条件引用定理、串行指令成本和完整入口收益检查。下一层分别补activation/内容版本证明、并行资源成本、最终交付准入成本；不能把MAY依赖、单配置成本或已编译产物当作这些证明。

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
- region按需展开已进入主图：原typed引用反向索引保留跨区数据使用和实际调用步骤，控制嵌套与词法父子分别表达。真实循环先只展开1区/24边；97个端口、194次双向查询与原109条边对照缺失0。state存在后再追加2个attribute操作，请求从1增至3，重复查询幂等；显式全量请求覆盖该库3692个操作、363个binding访问和182个content请求，部分邻接明确保留frontier。
- 原始边用`(template,region,local ordinal)`定位，存储schema v3、source representation v2。批量读回已消除逐行重复标签表校验；同批2000条真实记录，1次预热/3次交替测量，严格解码中位数616.469ms→272.954ms（2.2585×）。完整库图发布/读回本次49.488s，原压缩表7,516,613字节逐字节相同；旧81.183s是历史诊断，不当作受控A/B。
- 当前图合同与使用到的types/codec/writer/mapping pins都进入coverage准入，2进程新发布完整2-unit库存24.063s：4235操作、1329模板、1410region、51425端点、55018关系，分片9,235,411字节；worker峰值约86/191MiB，地址规则缺口0，未运行binding固定点，也不证明alias/RNG/外部效果闭合。
- 主build已实际接入上述覆盖、当前短动态记录和完整硬件快照：入口`tests/workloads/gray_scott_entry.py`与库字节均匹配F0；32×32/3步采集1.525s，103站点中观察68个，主build11.051s、12个条件函数接口、273资源记录（含全部204物理资源引用）。修复了`--cache-root`只查子目录而忽略本目录报告的接线问题。另一个成本采样入口在外部testcases中，不属于此catalog；此前误记为版本变化的范围诊断已纠正，测量不变。
- 内容lineage已自动接入公共图并按unit保存独立记录/边表：真实2文件220条GET/SET连接，包含GET的state→result及SET的payload→state；全为有条件MAY依赖，不创建逻辑版本、别名或相等证明。UPDATE/DELETE有定向测试，未在该真实case称已观察。父表ordinal仅用于引用，语义边不随查询顺序改变。当前完整库存26.568s、9,342,103字节，最终主build11.355s；新图公共GET/SET查询均通过。
- 修复按需持久化的真实缺陷：先查询后部操作再保存，旧cold reader按默认顺序重建会拒绝自己的requests表。两种存储格式的读回现在均按保存的request/content次序独立重放；ID声明/关系按集合严格比较，保留有行号引用的访问顺序。543操作/38条lineage实际完整读回通过，未减少字段、类型或收据核验。
- 新增自动有序内容图pass：从原catalog自行发现并重新推导区域，切片描述、局部内容epoch、读写与F4来源进入同一公共图，真实Gray产生92节点/317关系并完成typed表读回，不执行目标、不展开迭代。带caller context时，在activation接口未连接前明确返回frontier，避免单侧换context导致反向查询不一致。
- 内建dict/list槽定理已自动发现producer/primitive selector并沿原balanced控制路径匹配最近SET→GET；小静态case自动2个条件引用证明，插callback反例拒绝。能确定的constructor、literal、部分release事实机械解除，其余检查/外部假设单列；Gray实际2unit分析为0证明，保留具体适用性边界，不声称通用逻辑版本已证明。
- 新主图继续贯穿域规划与C++20验证包生成：仍发现1个有序状态区域、保留106个CPU配置和全部204个资源引用；规划35.006s，生成/编译验证包55.766s。未执行新native候选，成本不全继续NO_OP。证据：`artifacts/baseline/region-demand-main.json`。

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
- 新增显式成本证据重绑定：完整重放原/目标domain、原profile和已支持的模型依赖，只允许相同schema/sha/size物料的URI别名及3个同值pointer的domain_id变化，原estimate/derivation/evidence不变。真实旧6,970,369 ticks成本通过新域准入，规划结果1个KNOWN/106个UNRESOLVED、仍NO_OP；坏收据、修改scope和未知模型依赖拒绝，未重测目标。
- 本轮在预先声明的CPU0–3校准点补10个资源signature和2个guard/scalar边界signature，已知组件合计1.513ms，只剩2个FUSED阶段。新的匿名DAG/拓扑模型和`barrier+max(memory,ALU,path)`模型分别用全新冻结holdout验收，最大误差约28.34%/28.49%，均超过20%并拒绝投影；原始数据、汇编和拒绝原因保留。没有测目标变体、降门槛或扩充106mask后宣称成功。
- 上述旧模型失真的静态证据已具体化：纯算术探针SIMD内环有51处栈引用，mixed内环为0；本机已有LLVM-MCA17/Broadwell模型用于指令资源分析，未安装工具或改变环境。新增串行原语模型使用固定mixed/memory原语，123配置，全新holdout最大误差9.39%/8.10%，通过原20%工程门槛；它不是目标误差上界或物理时钟模型。
- CPU0成本首次18/18项完整：两FUSED模型值348,806ns/298,977ns，全调用3,471,101ns；原成本consumer准入通过。原backend仅静态编译和MCA分析，没有执行目标或排序变体。完整106配置域实际为2个KNOWN（含原执行）/105个UNRESOLVED，继续NO_OP；不能据一个配置宣称选优。
- 新instruction模型成本重绑定只接受专用严格重放：原source/plan/placement、GCC汇编、natural loops、MCA输入输出、holdout/公式和原fact指针逐项核验。真实同域模型证据重放通过，数值未重估、未重新计时；其他未知model依赖仍拒绝。

## 第三阶段：C++20 交付及真实结果

- 原图 → 计划 → C++20 → 同一 ELF 导出 → 普通 Python 入口已贯通。运行包无需 compiler、IR、planner、codegen 或 g++；状态包加载仍需要原 NumPy ufunc 提供者；入口和明确的兼容/I/O 仍保留 Python，完整程序不是纯 C++。
- C1原caller-matched接口桥接已接通：同一主build的原CALL/frame/参数/world/返回关系按完整F4步骤组成既有PipelinePlan；两个跨文件叶的query顺序实际为[5,4]。C++20及ELF区域3次逐位验证通过。
- C2新增紧凑运行包，合同约4.5KB，host/operator ELF逐字节相同；运行端无需compiler/IR/codegen/planner。实际原入口8×8/3步native3/fallback0、stdout一致；9×9变为native0/fallback3，stdout仍一致。代码/default/callee/输入/observer等10类变化以及真实native状态103均整caller回退一次、私有结果不发布。监测包含sys.monitoring全局和局部事件，不缓存输入内容。
- 默认NO_OP不加载ELF、不安装import hook；只有显式measure-candidate执行候选。新运行命令已走轻量dispatcher，实际禁止导入compiler/analysis/IR/codegen/planner仍成功。该dispatcher扩展改变runner身份，旧完整入口部署收据不直接复用；历史A/B保留，本轮未重做性能准入。当前只支持普通函数、float64二维数组和pool host，不是任意Python独立交付。
- 新caller完整入口检查已实现：8×8/2步、warm1+正式3对，原235.342ms、候选391.834ms，0.6006×、not profitable；stdout/stderr/exit一致，native8/fallback0。此测量属于正分支修复前的确切版本，原样保留，不能作当前版本A/B。审查发现最终receipt重放开销未覆盖，现已机械拒绝仅有正candidate gain的授权；当前没有最终正部署成本证据，默认NO_OP，严格env SHA没有工具键例外。
- 跨文件组合自动选择：4个准入源码文件，3个库文件、4条真实函数全局路径，512×512/1000步，完整106配置自动选28线程。预热1组、正式3组：原完整进程中位数849.953ms（800.455–854.940ms，stdev30.121ms），选中实现261.466ms（250.753–282.104ms，stdev15.935ms），3.2507×；输出一致，native4/fallback0，正常成本门控另native1/fallback0。它是跨模块编译回归，不是新的LeWM或三程序套件加速。
- 跨函数组合实际普通入口：256×256/20 步，结果逐位相同，native 1/fallback 0。替换深层 binding/defaults/代码/函数 globals 四种情况均准确整循环回退一次；恢复后重新启用 native。
- 自动线程选择的科学微程序：512×512/1000 步、float64，完整进程预热 1 组、正式交替 3 组。原版中位数 **857.374ms**（854.495–872.810ms，stdev 9.849ms），选中实现 **287.887ms**（287.682–309.074ms，stdev 12.292ms），**2.9782×**。输出逐位一致，预热及正式 native 4/fallback 0；正常成本门控入口另 native 1/fallback 0。这是编译回归程序，不替代真实三程序套件。
- 真实 Gray–Scott 完整入口：512×512/1000 步、float64、C++ 单线程 CPU 0；当前运行端版本重新预热 1 组，交替正式 3 组。原版中位数 **11.136s**（11.097–11.189s，stdev 0.046s），优化 **3.939s**（3.916–3.943s，stdev 0.015s），**2.8271×**。包括进程启动、导入、应用、I/O 和报告开销；这不是纯 kernel 计时或未来性能保证。
- 区域/单步/20 步轨迹/完整积分、u/v、必要状态、数组身份、stdout 和文件逐层通过；完整进程测量 native 4000/fallback 0。随后正常成本门控入口 native 1000/fallback 0；改变 argv 后保留原执行。
- 上述区域配置是明确单线程验证预览，尚未证明最佳。完整域成本缺口未被这份测量抹去。baseline 的最终 dt=0 行为原样保留。
- 先前整数组物化 LICM 的 0.9625× 继续记录为 not profitable；不得与本次不同实现/计时边界混算。Sunfish/Cirq 仍仅有基线，无新增加速结论；当前无 GPU，不作 LeWM/GPU 实测加速声明。

证据：`artifacts/current/experiments/stateful-entry-v1/`、`composite-licm-v1/`、`parallel-temporal-delivery-v1/`、`temporal-resource-model-v1/`、`automatic-temporal-cpu-v1/`、`cross-unit-composite-v1/`。摘要：`artifacts/baseline/three-stage-system.json` 、`automatic-temporal-cpu.json` 和 `cross-unit-composite.json`。成本计划 7.50MB 在构建时保留；运行端只核准入合同中的原计划哈希引用，不读取该计划或源码图。

## 接下来仍需完成

1. G3：接通有序region的activation与caller接口，再把局部内容epoch、槽定理及创建环境用于可证明的别名/版本；跨调用不混线，未知协议保持边界。
2. O2→O3：串行模型已通过；下一步建立并行worker/访存/同步的独立证据与适用域，补其余配置再选优，不能把CPU0成本除以线程数。之后推进跨区域、NUMA/多进程/MPI及CPU/GPU联合调度。
3. C3：覆盖最终交付路径的准入/收据重放成本，再做当前版本整入口A/B并授权；已有候选clock不能代替这一层。再扩大dtype/控制及宿主。

本轮统一证据：`artifacts/baseline/ordered-content-instruction-cost.json`；代码、拒绝原因和真实结果一起版本管理。
