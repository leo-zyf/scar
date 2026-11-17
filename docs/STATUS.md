# 当前状态

2026-11-17 GCC方法论已接首条源码区域pass：`macro-pipeline` 重放原F4/GlobalPort及硬件来源，投影5个原CALL到既有v2 OIR，明确原Python/NO_OP和已有FULL_FUSION。requires、条件Q、完整受保护调用成本、5项失效账本及独立重放已接通，不新增后端或全局求解。声明正模型可构建C++20/ABI1、受控安装原模块函数并恢复；独立与只读别名逐位一致，dtype/callee/CPU域变化回退。原执行无native guards，成本未知/负模型不构建或加载库。一次序列化tuple/list重放问题已修，当前材料独立冷读通过。

实测该8×8源码例子：warmup3、9次交替顺序、每样本1次完整受保护调用，原NumPy median13.584µs[min8.998,max15.444]、融合111.705µs[min109.358,max137.556]，not profitable。实际完整边界证据已使pass选KEEP且跳过native；声明正模型不当作加速证据。无worker/绑核，仍有guard/ABI等调用开销。仅小源码区域，非科学程序整入口或LeWM；无新加速结论。新全图重写、自动分析缓存失效重算、解析组件成本与资源日历构造均未完成，现有C1 compact亦未接新融合交付。验收：`artifacts/baseline/macro-fusion-pass.json`。

GCC13.3本机实证保留：-Q选项、pass顺序、GIMPLE/SSA、LICM及最终IR；闭合C例子发生内联、外提、重复/死计算消除和16字节vector，未知外部调用保留。O2/very-cheap与O3/dynamic成本差异及默认线程数1已核，不把GCC当Python语义证明。记录：`artifacts/baseline/gcc-o3-study.json`。

2026-11-17 解析/构造式规划调研已完成。主线为语义化简、工作/数据移动需求和容量界、解析组织、资源日历构造、证书与C++；少量组件标定用于有效吞吐/边界开销/重叠条件，不默认整版本黑盒调优。已有OIR、融合后端和求解器保留，求解器可作局部补充/校验；首条pass已实现，构造引擎尚未实现。具体逐层任务见DEVELOPMENT，文献与适用边界见 `artifacts/baseline/constructive-planning-research.json`。仍未完成三阶段。

2026-11-17 性能归因定位了主链架构缺口：现joint producer只生成固定CALL的CPU选项，source original/fusion/caller直串行不在该域，NO_OP仅为部署默认。相同进程、warmup3、9组×32的8×8干净对照：原NumPy7.687µs，五ABI单worker299.995µs，复用既有融合后端101.155µs；两种C++均不盈利，不自动启用。独立输入与只读别名逐位一致；另9次host插桩确认启动及task包装显著，ABI区间包含自身guard，不能称纯数学时间。源码正常路径有10份affinity scope（host与leaf双层），这些是源码API次数、非syscall trace。未改数学/后端。下一轮以composite region覆盖及实现选择为主链，再接每种实现的硬件/完整边界成本和交付；不继续仅调CPU排列。报告：`artifacts/baseline/pipeline-architecture-diagnosis.json`。

2026-11-17 条件worker生命周期已贯通 source→v2规划→C++20/ABI1：5原调用、10条件host活动、34端点约束；低启动成本声明模型选2 worker，高成本选1 worker，生成源码实际只启动选中CPU。完整task保ABI+wrapper，join等待可伸长，未选活动不存在；原v1 wire/digest不变，单配置实测宿主成本继续UNKNOWN。实际native冷重放通过，独立/只读别名输入与原函数逐位一致。8×8区域干净call-boundary计时（warmup3、9组×32）：原Python median9.495µs，C++392.328µs，not profitable，默认NO_OP；非科学程序整入口或模型准确性结论。验收：`artifacts/baseline/pipeline-lifecycle.json`。接下来按层补：第一阶段接完整host边界证据；第二阶段加入原Python/NO_OP整区域实现与同范围成本及可验证宿主预测；第三阶段延续证书到原入口/guard/fallback，并在闭合范围定期A/B，不将声明模型时间当实测加速比。

2026-11-17 原 pipeline host envelope 已接回值/存储/硬件/时间主图：32任务、55活动、157事件，保留分配、异步创建、原ABI、发布、join及释放；5物理槽/2560B各计一次。7次保留观测（warmup2）中32活动可重算插桩区间，23活动成本仍未知；5 ABI绑定原选CPU，39项物理执行需求未知。离线源码、硬件投影及成本重放通过且禁止CDLL，无新目标执行。观测≠未来service成本，等待≠busy CPU；没有整调用预测或新加速比。验收：`artifacts/baseline/pipeline-host-envelope.json`。下一层是按已选worker生成条件启动/退出活动，补可用于预测的宿主成本和并发条件，再接联合规划，不能将本轮区间直接喂作固定服务时间。

2026-11-17 统一实现规划主链已接通：原区域/软硬件来源→实现与位置选择→精确上下文成本查询→CP-SAT→独立资源/依赖/物化检查→C++20/ABI1。5-call源图的10个CPU service组件完成7样本/warmup2的C++内部标定（40.45s，非整目标版本搜索），共享2560B pool只计一次；孤立样本施加逻辑串行约束，并将该资源顺序与原依赖分别落实到C++。选中CPU为1/0/0/0/1，原函数与native在独立输入及只读别名输入下逐位一致。58,465ns只是service点模型完成时间，host启动/分配/派发/join、并发带宽和整程序性能尚未闭合。通用7项关键检查已覆盖并行、启动成本、预算、搬运、共享物化和UNKNOWN域；未知域不得报告已知子域的零gap。缓冲策略亦已进入同一接口，默认UNKNOWN保持原执行；声明模型选择仅允许显式验证，不冒充实测。验收：`artifacts/baseline/joint-implementation-planning.json`。当前下一层集中补host envelope成本和可校验的并发条件，再扩大联合规划域；不新增零散detector/backend。

私有容量复用本轮同时验收：4槽/8,454,304B，13次私有读取由当前步写入覆盖；每步前后清内容事实和动态输入强引用、仍完整snapshot。真实Gray外层3组原版/试验的状态/stdout/PNG一致，native4/5/3、容量创建1/1/2；重绑数组及时释放，fallback结束epoch。运行端隔离验证通过，默认无CDLL。它是有条件合法的容量变换，未测本方案整入口加速比；原provider同步借用Q继续显式保留。

2026-11-17 外层原控制链已接通：自动发现原共享循环，78操作/1811边/1946interned ports；7词法join中t/s两项多writer按类型不动点合并，保1回边与dt字段写，不展开。`session-outer` 将源图→activation/lease覆盖→compact入口接通；效应未闭合只允独立leaf guards后fresh C++计算，计算前后销毁，taint不清、无新增I/O quiescence假设。原Gray `integrate()` 的正常/频繁I/O/另一分支三个独立原版与试验进程，返回、状态、stdout与PNG逐字节一致，native分别4/4/0。另 stdout sink四次原地改输入仍与原版一致、native4；未遮罩print或替换原host函数。完整外层入口隔离运行禁用建图/优化包仍native4并保存真实PNG，default无CDLL。验收：`artifacts/baseline/session-outer.json`。次数/时长/host容量仍未知，不宣称驻留收益或全局最优。下一层优先证明仅复用私有容量、清输入引用和内容事实、每步仍full snapshot的策略，再接资源占用与同范围成本；不借机假定未知I/O纯。

2026-11-17 caller 自动接入主链已验：原 F4 的完整操作账本→引用路径/类型约束与直接lookup证明→compact scope→原导入入口自动安装/恢复；Gray `update()` 29项操作闭合为有条件profile，保未观测分支的原调用及失效barrier，不按业务名选择。6组独立原版/自动入口的逐步状态摘要、时间与stdout一致：正常/改输入native4，切分支native2，实例影子与自定义defaults元数据native0，unscoped kernel native4且逐次释放；dict override callback0。运行端禁用全部建图/优化导入仍native4，default无CDLL。动态session资源投影另接原15项body证据、4槽/8,454,304B，保未知次数/时长/outer CPU与额外metadata，不伪装固定N4/VRS。外围 `integrate()` 未闭合，每次最外caller退出释放，暂无跨caller驻留收益或新加速比。验收：`artifacts/baseline/session-caller.json`。下一层优先补原循环的词法carry/类型不动点、字段版本及未知I/O边界，再将外层scope寿命与成本接入规划；保持现C++算术正文和轻量运行端。

2026-11-17 跨调用工作区 session 主链已实现：原 F4 自动识别有序复制及整数轴降维，Gray 四次 halo 写均与两份内侧驻留 ROI 不相交；新 C++20/ABI1 create/step/invalidate/destroy 持有一次工作区，每步新 scalar、dirty/full 刷新和当前调用回退。受控接入原 `update()`，四步轨迹/时间逐位一致，native4/resident3；内侧改写显式失效、scalar 更新、对象替换及回退前释放均验证。独立冷读核对原图、生成源码/ABI 和 compact；运行隔离进程禁用 compiler/analysis/IR/planner/optimization/codegen 和外部图读取仍 native4/resident3，只读 compact 两个 JSON。局部 CALL 的6个调用与原 F4 端口接通，未保存数组值；CALL 无法覆盖直接写入，完整 caller/while/I/O 自动替换仍未证明，新 session 无性能结论。可视化已具离线展开/收起和资源视图，收尾后主力回到三阶段。下一层：以原 caller 的29个生产操作为覆盖分母，补接收者/descriptor、真值、加法及未知效应前失效，再接 lease 生命周期到时间/容量图，最后从该证书自动生成入口安装与 C++ 交付。验收：`artifacts/baseline/workspace-session.json`；不把受控接入冒充整程序优化。共享 guard 更新导致旧 compact 包需重新导出，原数学模板/基础 native 不变。

2026-11-16 COMMIT ROI驻留闭环已验并测得局部收益：自动从副本/槽尾部写集/carry证明首轮full snapshot、后续只刷新已提交ROI补集；Gray四步snapshot源码载荷16974400→4342288B，有限性/FP/provider/COMMIT全保留。C++20真实构建、两源冷读、原入口/partial2+2/default/runtime隔离、Gray原始与改scalar/halo逐位核对通过，同步provider诊断改写halo仍与原版一致。独立clean raw-N4 AB/BA，warm2/7pairs、CPU0 Xeon E5-2680v4/NumPy2.5.2/GCC13.3：原median9.754001ms[min9.737479,max9.872894]，驻留8.763597ms[min8.711060,max8.815263]，1.113013×，7/7对更快；完整variance/原mask+fenv恢复记录保留。仅固定四步区域，非整程序或LeWM。provider借用端口返回后安静是明确外部SYNC CPU Q（非已源码证明/非runtime可检），故default NO_OP，无自动部署。finite/新residency共用构建导出核心，旧finite分支重新构建/原入口回归过；模板/数学body不改。验收：`artifacts/baseline/workspace-shadow-residency.json`。下一层接真实caller与期间写效应/lease寿命，才能从显式四步组合推进原科学程序控制流；完整联合时间/CPU/layout规划仍未完成。

2026-11-16 共享循环硬件窗口主链已验：新增 `plan-workspace-windows`，抽出并复用原standalone的hazard/pair/anchor核心及原局部solver，保same-i端口、内容carry、FIRST/LAST和ROOT槽寿命，不展开迭代。原循环1窗，Gray2窗；Gray首窗stage0/1在原CPU0/固定pool的条件组件模型392468 ticks上打平，保原序；算术窗保FP/资源/成本前沿。原payload分别480B/12697904B，显式32MiB预算不变，未知anchor/residual不计0、总时长仍None。两源独立冷读通过，真实standalone旧HEAD与新共core全文输出一致，已有finite-reuse compact仍可metadata准入。新融合提示不冒充standalone lowering，NO_OP交付只引用原C++，不重复编译；没有新目标执行、变换或加速比。报告：`artifacts/baseline/workspace-closed-windows.json`。规则文件SHA更新使旧standalone窗口收据需重发。下一步优先审查COMMIT ROI副本一致性以削减跨轮snapshot搬运，再扩有来源的layout/CPU方案；完整联合时间规划和共享窗口变换交付仍未完成。

2026-11-16 真正的属性化简闭环已验：`prepare-workspace-guards → compile-workspace-guards → run-stateful-loop` 从原表达式/ROI/初始化/COMMIT/carry图机械证明跨轮FINITE_CONTENT，封闭4×5/N4输入扫描源码次数4→1（省60元素访问，非runtime计数）；数值仍变化，属性不跨调用缓存。真实新C++20/ABI1与原入口状态/stdout逐位一致，保首轮NaN/Inf拒绝、default无CDLL、N0/错N不读输入、partial2+2；独立冷读通过。原数学生成器/模板不改，仅新variant+显式compact profile；Gray自动因opaque provider拒，未冒用其约350us复合phase成本。报告：`artifacts/baseline/workspace-finite-property.json`。未测性能，收益UNKNOWN/default NO_OP，runtime升级后旧包需重新导出。下一层用既有window/implementation/resource solver补v4共享正文窗口入口，保原端口、未知anchor和借用slot寿命，再扩有来源的layout/CPU域；不再造求解器，不把局部模型当完整物理最优。

2026-11-16 v4执行桥已验：新增 `refine-workspace-execution → compile-workspace-refinement`，把共享正文、FIRST/LAST容量生命期、逐轮content carry、CPU域和有条件成本映射到source-bound证书，再用原backend重新构建C++20/ABI1。原循环13活动/2slot、Gray24/4均通过独立冷读；新源码/controller与证书逐项一致，原入口stdout/状态逐位相同，runtime无建图依赖、default无CDLL、部分提交2步后只续跑2步；Gray新区域C++四步与Python/原C++逐位相同，composition不导出caller替换包。离线画布另经真实浏览器展开18→37→收起18及资源切换验证。报告：`artifacts/baseline/workspace-execution-refinement.json`。这只接通已有workspace变换的三阶段交付，未补齐5/10个未知成本或全图数值规划，无新加速比。下一层先补封闭区域的稳定元数据/finite属性传递，opaque provider合同不足时保留检查，再推进v4联合资源时间规划。

2026-11-16 两端并行推进已验：workspace边界探针在两套源配置各采7组/warmup2，每组N4的30切点完整保留；仅编译/运行canonical检查、分配、绑核、恢复与释放片段，未运行目标数学DAG。6个完整UNKNOWN组接入后，原循环11subject中6有条件组件，Gray18中8有条件组件（原2copy未动），仍5/10未知且总时长None。另一条旧L4计算顺序证书已接C++20真实构建与compact入口：固定5调用声明成本例的输出一致、CPU0按C→A→E/CPU1按B→D启动；default无CDLL、runtime无compiler/大图依赖，输入变更与fresh输出检查通过。仅落实compute-order，尚未落实完整时间表，也不是新v4全图规划；声明模型不冒充实测。验收：`artifacts/baseline/workspace-boundary-and-pipeline-delivery.json`。下一步接v4计划执行证书，并针对已有成本证据分析可避免的循环检查；三阶段尚未全部完成，无新加速比。

2026-11-16 执行域与循环成本接线已验：`project-workspace-demands → bind-time-resources → cost-activities` 复用原v4数据/时间图，按当前C++源码的 bind→body→restore传播条件CPU事实。原循环3个阶段可推导CPU0，Gray前2个snapshot可推导CPU0；provider后即便后续操作封闭也不追认位置，COMMIT/分配/释放整体需求仍未知。真实原语只匹配Gray两项copy组件（18 subject中2有条件估计、16未知）；4×5输入未误用515×515事实。pool各作一个outer UNKNOWN group，仅计一次，逐轮残差不合并，总时长仍None。两条完整source/resource/cost独立冷读通过，无新目标运行、标定、C++生成或加速比。报告：`artifacts/baseline/workspace-demands-costs.json`。下一层补余下实施阶段的同范围成本及资源合同，再做联合时间规划与C++执行证书；三阶段仍未全部完成。

2026-11-16 共享循环已接统一时间/资源图：显式v4增加FIRST/LAST_NORMAL端点、`(i,local_epoch)`值/物化模板及COMMIT→下一轮输入的数据carry，v2/v3原字节不变。`project-workspace-time → bind-time-resources → view-graph`在原F4循环与Gray显式组合冷读通过，分别13活动/35事件/2私有slot/1 carry、24/64/4/2；正文只存一份，分配/逆序释放在ROOT，输入finite guard的数组读取及helper退出/progress均入图。新画布连原204资源/477连接，原C++入口仍完成4步。当前仅正常完成路径，失败/FFI时间未建模；13/24个服务时长和CPU执行域仍未知，不据此求数值最短调度。验收：`artifacts/baseline/workspace-indexed-time.json`。下一层补同源执行域与循环成本，再将联合计划转为C++执行证书；三阶段通用覆盖尚未完成，无新加速比。

2026-11-16 原入口交付已补通：`export-workspace-loop → run-stateful-loop` 共用轻量runtime，明确区分原分配与workspace配置；原F4/资源证明在导出端重放，原ELF不改。真实冷进程运行原入口，C++完成4步、stdout/状态逐位一致；默认无CDLL、零次不读输入、已提交2步后只续跑2步通过。阻断所有compiler/analysis/IR/planner/codegen导入和外部图读取仍成功；compact约62KB，composition拒绝冒充原循环。有限median选择抽为公共接口并接回原copy主链，真实成本重放仍选ORIGINAL；未新测性能或启用外提。验收：`artifacts/baseline/workspace-compact-runtime.json`。三阶段通用覆盖仍未完成，下一层集中接共享循环的最后使用/释放到统一时间资源IR，再补同源成本及联合规划，复用现有C++交付。

2026-11-16 workspace生命期主链已实现：当前F4源循环和Gray–Scott显式四步组合分别生成6节点/14边、10节点/26边的ROOT/共享LOOP_REGION资源图，正文各存1份；本轮私有read覆盖均证明，固定地址/容量不冒充不变内容。源码分配请求分别8→2、16→4，数组payload预算不变但生命期延长。C++20/ABI1实际运行逐位一致，默认回退、零次不读输入、改callee、错shape/count和已提交2步后只续跑2步验证通过。首次lowering因clone被内联丢失7个SIMD循环，4步9.85→12.19ms；保留单步编译边界后恢复7个SIMD循环，重测原9.849ms/workspace9.852ms（7对/warmup2，区间重叠），无可确认收益，not profitable to enable。负结果和编译诊断保留，无整程序加速比。报告：`artifacts/baseline/workspace-lifetime.json`。旧循环proof的producer pin过期已从原ProgramBuild重建；旧compact因guard抽公共core而指纹失效，未绕过校验。下一步把资源生命期方案并入统一成本选择，并补原入口交付；不强行开启该用例的外提。

2026-11-16 成本选择主链：新增 `calibrate-copy-window → choose-copy-window → compile-copy-choice`。原/融合窗口各用一个规范化计时cut，AB/BA交替、7对样本/warmup2，计时外重置与逐元素核验。真实原窗口median333215ns、融合543896ns，采样范围不重叠；规则据此保留ORIGINAL，融合在本组件条件下not profitable。两份成本表均22个条件组件+1源marker，融合缺证据字段已按真实FactRef解除，其余上下文残差/资源未知保留；不把组件范围当整程序保证。所选C++20/ABI1实际构建、独立冷读和原方法逐位验证/回退通过；三个合成策略分支仅测模型逻辑，未伪作实测。未重新标定旧组件、未运行完整目标或测新整程序加速比。报告：`artifacts/baseline/copy-window-cost-choice.json`。下一步继续有界联合规划，并优先推进共享循环内私有workspace生命期外提，而非强行启用无收益的循环融合。

2026-11-16 继续推进：`prepare-window-rewrite` 已生成可独立重推的真实替换图；两SNAPSHOT合为一个可展开COPY区域，31→30任务、76→74时间事件，原8值/12物化/6存储槽及204项硬件库存引用不变。就绪事件、success-fence、外部依赖和CPU占用通过完整映射重接；只删除已证明的内部源序边，不丢弃其他约束。成本23组中融合项重新UNKNOWN，保留21项原条件组件证据与控制marker，不拿旧复制成本和冒充融合成本。新C++20/ABI1独立schema消费这份图/成本/资源/证书；真实区域与原Python/原C++逐位一致，原方法native3、guard回退2、默认NO_OP回退1，版本/事件映射篡改拒绝。来源冷读通过，4项IR检查通过；本轮未测性能，无新加速比或全局最优结论。报告：`artifacts/baseline/g5-certified-copy-rewrite.json`。下一步将替换区域的成本证据接入统一选择，再扩展有界联合调度；当前证书只证明该局部变换及其C++实现，三阶段通用覆盖尚未完成。

2026-11-16：可视化已交付，回到三阶段主链。新增 `project-activity-time` 保留原31活动/24成本组、未知残差和保守存储生命期；`plan-closed-windows` 从原图机械抽出2个窗口，首窗的固定CPU条件模型已解，但与原序打平，第二窗保留依赖/资源缺口。独立的通用规则自动提出同迭代域、无读写冲突的raw-copy循环融合；`compile-window-trial` 已将真实Gray–Scott区域的两套循环融合为一套并编译为C++20/ABI1，原读写语句、内存池、CPU和事务边界不变。两输入场景各3步与原Python/原C++逐位一致，shape/alias/provider守卫和原方法wrapper回退另验；来源时间图、局部计划均独立冷读通过。仅为显式区域试验，融合成本UNMEASURED、默认NO_OP，无新加速比、整程序C++或全局最优结论。报告：`artifacts/baseline/g4-local-window-fusion.json`。下一层先补替换区域的成本接口和执行证书，再扩展联合CPU/内存/时间规划；不把局部条件模型当作完整v3求解。

当前按 G1→G6 落实时间模型：G1 同一 IR 的事件/活动/占用/约束 → G2 真实资源域映射 → G3 有来源的成本区间 → G4 放置/并发/内存/时间联合规划 → G5 C++ 执行证书 → G6 原入口正确性与 clean A/B。以下旧 L2–L4 结果只是已有有限声明模型的证据，不代表 G2–G6 已完成；未提交的旧 C++ solver-order 试作暂不扩展。

G1 已验：v3 时间字段及 `scar time-schedule` 接入现有 IR，v2 字段与哈希保持不变。CPU pipeline 为 16 tasks/41 events/5 owned slots；真实 Gray–Scott 为 20/54/4；状态循环保留一份 9-task 正文、27 events 及 distance=1 依赖。三份保存图均通过独立来源重放，Gray 原 F4 主链另经32.69s冷读确认；7项定向检查通过。耗时/服务需求未知项保留 UNKNOWN，旧求解器不得丢弃新约束；没有新目标执行、标定、加速比或部署授权。验收：`artifacts/baseline/g1-semantic-time.json`。

G2 资源接线已验，尚有显式未决字段：`bind-time-resources` 将每个占用 ID 接入原 snapshot；pipeline 的5个计算活动精确绑定原线程，其余11个需求单位仍未知，候选域保留56线程/28共享物理核。Gray/状态循环没有借用所选计算 CPU 冒充宿主活动的绑定。三源分别5/4/2份 owned slot 各计一次，HOST 候选按 CPU_ALLOCATE 合同推出，原预算保留；未证明页驻留、可用 RAM 或吞吐。三源独立冷读及9项定向检查通过，报告：`artifacts/baseline/g2-temporal-resources.json`。接下来补 G3 的活动资源需求和有来源的真实成本区间，再接联合求解；G4–G6 未完成，无本轮新加速比。

G3 集中按子层推进：a 原活动/组件范围与证据桥 → b 补实际 implementation envelope 活动 → c 成本/误差与资源需求闭合；不扩展 G4/G5。G3a 已验：`cost-activities` 将当前单 CPU0 的独立 canonical 样本连接到 Gray 原20活动，14个计费subject中10个有条件组件估计、4个未知；另5个实际边界term明确unmapped。7次采样/warmup2只花5.95s，没有运行目标或搜索变体；样本范围不是保证界，已知项之和不是总时延。loop异context保0已知，pipeline不冒用该成本；三源冷读、3项实际来源拒绝和4项定向检查通过。报告：`artifacts/baseline/g3a-activity-costs.json`。当前继续G3b，把缺少独立节点的guard、finite scan、team进入/恢复等建入同一时间图，不能仅填duration。

G3b 已贯通 standalone Gray 的“原图→实施阶段→时间→资源→成本”：原20活动加10个实施phase，原8值/12物化/6slots、全部旧HB与预算不变，74个事件；8个PRE phase纳入fence、2个POST仅正常交接。补上的2个输入扫描可接原样本，24个计费subject中12有条件估计、12未知，剩3个组合成本端点仍不匹配；不均摊、不重复计费。完整来源冷读与5项定向检查通过，原共享循环拒绝误套standalone边界，报告：`artifacts/baseline/g3b-implementation-envelope.json`。范围不包括compact/repeat实施，G3全层仍未完成；下一层G3c先明确剩余phase的资源/计时契约，再接缺失成本，暂不进入G4/G5。

G3c1 已验：`refine-time-envelope --bind-serial-demands` 静态重放 bind→闭合代码→restore；provider 的 affinity 保持合同缺失时立即失效，不用最终检查倒推。Gray 只证明前两个 SNAPSHOT 的CPU0单位需求，其后维持UNKNOWN，原任务/数据/时间边界不变；真实三链冷读与2项定向检查通过，报告：`artifacts/baseline/g3c1-serial-affinity.json`。继续顺序是 G3c2 排他边界时钟cut及明确残差 → G3c3 fused-DAG/tagged-scalar缺项 → G3c4 全范围成本和资源适用条件验收；完成前不开放G4/G5。

G3c2 已贯通：`calibrate-boundary-cuts` 用真实 shared guard→compiler adapter→StatefulArtifact.run 接原模板片段构成的 canonical no-op；同一整数 CLOCK_MONOTONIC 将每次调用分成7个连续排他区间，返回尾段合并两原phase，保留RAII/FFI/插桩及空DAG的适用残差。真实7样本/warmup2约3.00s、native9/fallback0、affinity恢复；`cost-activities --boundary-calibration` 独立重放原片段/编译来源和逐样本cut，仅替换UNKNOWN，不动12个既有primitive成本。Gray的30个活动仍各覆盖一次，23个计费组中19有条件估计、4未知（两个fused DAG、一个tagged scalar、一个成功检查）；全部仍非完整成本。2项定向检查和冷读通过；验收 `artifacts/baseline/g3c2-boundary-cuts.json`。当前继续G3c3/G3c4，尚不授权联合solver、部署或新加速比。

G3c3 已验：`calibrate-stage-dag` 从原 renderer 的只读入口快照构造有序/共享/typed-constant/物理槽别名完整子图键，源码身份只留引用；数学AST不改，捕获全文须逐字等于原renderer。原body直接进入C++20 canonical组件采样，明确承认拓扑特化及片段已编译，未执行完整目标；3组件各7样本/warmup2约3.79s，CPU/FP恢复且冷读成功。`cost-activities --dag-calibration`仅替换3个UNKNOWN，既有19组数值原样；30活动/23subject现22个条件估计，仅`success:precommit`未计时。2项结构检查通过，源码改引用使观测记录6.35MB降至115539B且结构键不变。见 `artifacts/baseline/g3c3-stage-dag.json`。下一G3c4按原源码范围核成功门槛/旧组合term收费归属、资源需求单位及模型残差；不将虚拟fence或原unmapped诊断粗暴当成额外物理耗时，不提前宣称全调用成本或最优部署。

G3c4 来源/账本子层已验，资源适用条件仍未闭合：`--scope-cleanup` 补上commit后、pool释放前的helper析构；源证书把无独立语句的成功门槛标为控制milestone，移除其虚构CPU占用。另修正SCAR原recipe建模缺陷：新recipe的pool释放为C++数组逆序3→2→1→0，旧收据仅历史模式重放，不授权现代多release时序；数值后端和目标未改，不算性能提升。新7-native-cut标定及完整recipe→time→hardware→cost冷读通过，31活动/24subject为23个条件估计+1源marker，18个原term均有收费归属，数值未知0；15份既有primitive/DAG数值证据复用，释放顺序差异仍保残差。源规则分离units/CPU位置：12个完整串行服务可证单位1，18个服务内部需求、28个CPU域和NUMA驻留仍未知；不拿caller参与1充当总资源。2项scope检查、2项旧G2回归通过，报告 `artifacts/baseline/g3c4-scope-accounting.json`。接下来做带残差的成本组时间投影与闭合窗口/opaque边界合同，再接局部规划；整scope资源可行性、物理成本闭合和全程序最优性仍未证明。

可视化支线：`view-graph` 已接第一阶段 ProgramBuild 原查询接口及版本化任务图，离线无限画布按分片展开/收缩，保原 frame、MAY/条件和证据，循环不逐次展开；布局单独保存。真实 Gray 导出8个程序片段（426节点/469边）及204资源/477连接的原库存，端点/深度/时间限制均明确，不冒充全程序覆盖。浏览器已验展开79→124→收起79、资源分组13→157、缩放/筛选/保存及零网络请求；未运行目标程序。可视化验收后回到时间/资源/规划主链，不继续扩展GUI产品。

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
