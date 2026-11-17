# SCAR 程序模型

SCAR 是宏观编译器：用有类型、可展开的图统一表示程序及硬件资源，支撑结构化简、hardware-aware规划与C++20代码生成。
建模、图化简/资源规划与C++交付分层推进；本文件描述建模层，图中出现重复或候选不构成变换许可。
范围、已知关系及缺口必须如实列出；缺口存在不算相应部分已完成。
当前没有做到所有 Python 行为的精确静态求解。

统一实现规划使用独立的 `PlanningProblem` 叠加层：`PlanningOperation` 引用可展开的原区域，`ImplementationOption` 记录后端入口、合法性来源、资源需求、私有工作区与输入/输出位置。`PlanningDataEdge` 保留逻辑值和 producer/consumer；位置不同时必须选择显式 transfer，不能从 `.to()` 调用或地址推出物理搬运。原始软硬件图仍是事实来源，求解器不重造源码语义。

`CostQuery` 的身份包含完整 scope/context，来源引用另保；`CostEstimate` 区分声明、经验、分析模型和 UNKNOWN。经验样本重算 median/min/max，样本范围不是物理保证；组件成本不提升为完整调用时长。缺项留在选择域，已知子域的下界和 gap 不得冒充全域最优。CP-SAT 在明确点模型中联合选择模式、放置、搬运和执行时间，完成时间相同时优先少变换。

存储约束按物理物化计费：一个 producer 值供多个消费者共享时计一次，跨域输入副本独立计费，借用的外部根不计入增量私有预算；全区域共享 pool 以 reservation 计一次。每个存活区间覆盖生产至全部相关消费者或搬运完成，工作区保守覆盖整个区域。该预算不是全进程 RSS、NUMA 页驻留或可用 RAM 的证明。

源 pipeline 的实现证书分开保存原依赖与资源顺序，交付端将二者接入 C++ lane；预测时间不作为启动时钟。孤立 service profile 默认施加容量 1 的逻辑互斥资源，不能据此声称已标定物理带宽。资源顺序必须真正落实，不能只落实 CPU 放置而丢弃其他约束。当前只交付受支持的正常完成 CPU 区域，完整 host envelope、共享带宽并发及全程序联合最优仍未闭合。

共享循环的时间/存储图使用显式 v4：ROOT 持有一次分配和末次释放，LOOP 保留一份 `(i, local_epoch)` 正文，跨层约束选择 FIRST 或 LAST_NORMAL。`TemporalContentCarry` 另将本轮 COMMIT 的物化连接到后续迭代的输入；它需要原来源和对应时序回边，不能由地址不变推得。当前仅建正常完成路径，失败恢复合同仍引用原提交游标；索引约束的结构合法不等于数值调度可行，更不授权执行。v2/v3 原有序列化不变。

执行域是沿已准入源控制路径传播的条件事实：每轮成功绑核建立 selected 域，未知 provider 使其失效，restore 只向后生效；调用者参与一个线程不等于整个调用只使用一个线程。循环成本另存 once/iteration 出现次数：外层 pool 不乘 N，逐轮残差保留各自的 i，组件采样范围不充当物理保证界。

数组区域另有定义期的 `ArrayValueTemplate`：内容模板是本区域输入版本的参数化函数，不是运行时 LogicalValueID 或跨调用缓存键。标准 ndarray C-order copy 可以与输入具有相同内容模板，但仍各有独立 ValueID 和 fresh 物化要求；slice 是 view，不能混为 copy。原 F4 CALL/receiver/参数/world 保留来源；编译入口把这些关系接入公共图，冷读重新执行源准入和结构推导。只有封闭类型/效果和稳定输入合同成立时才能转发内部 copy，最终独立输出仍须生成。

`CallerValueProjection` 按原 actual/formal 路由替换共享 leaf 的内容根，连接 `Value → CALL → fresh Result`；world 单独连状态接口，不参与内容相等判定。每个 CALL 保留原 Frame/FactRef，递归展开引用共享正文，不能把原 F4 Frame 直接套进另一个模板 namespace。资源计划另从数据边与 pool 复用推导 RAW/WAR/WAW；静态 wave 是它的执行细化，不能冒充源码必需顺序或最优时间表。

CPU 依赖就绪执行共享同一语义/存储 DAG：每个任务固定到显式 CPU lane，同 lane 容量为1；真正前驱完成才可领取，其他 lane 的无关任务无需先完成。领取次序是可执行资源策略，不新增数据依赖或永久 lane 顺序边；完成/消费者计数/slot 版本经同步发布，失败时等待所有已运行任务结束后才能释放私有存储和回退。

逻辑值 lifetime 由 producer 与全部 consumers 的偏序决定，单个 source-index `last_use` 只适用于指定串行顺序。RAW-only pool 仅在原图已有严格先后时共享槽位；不能因源码中后出现就复用。存储复用的 WAR/WAW 证据保留，但不得增加原 RAW 图的可达先后。更少容量与更多并行自由度是不同目标，最终应在有依据的时长和硬件容量下联合最小化 makespan。

封闭状态循环用一份共享 step 和内容坐标 `(数组输入根, iteration i, local_epoch e)` 表达递推：本轮最后成功提交的内容连接下一轮入口；只读数组和固定 scalar 是受 guard 约束的 live-in。该坐标不是运行时 LogicalVersion，不因对象引用未变就判为循环不变量。原源码循环证明、参数接线、资源计划和运行时准入分开，只有独立重放后才可用原 C++ repeat adapter 替换控制。

`FINITE_CONTENT` 等是附着于内容版本/ROI 的条件属性，不是内容身份。属性图用原表达式的操作数边、影子初始化、ROI 写入和已证明的 COMMIT→下一轮输入 carry 传递性质；未知 provider 会截断相关推理。有限输入经过封闭算术且原严格 FP/事务检查成功时，可证明下一版本仍有限，替代后续重复输入扫描；该结论既不复用数值，也不跨调用持有属性，不能单凭成功 COMMIT 推出。

COMMIT 的逐字节复制还可建立 `shadow[ROI] == external[ROI]` 的副本关系，即使数值已变化或尚未证明有限。关系仅覆盖被复制区域；在源写集、持有槽生命期、carry 和同步/无后写合同成立时，下一轮可保留该 ROI，只重新复制补集以重建完整 snapshot。补集不能直接视为相等。Provider 借用端口在返回后不再访问属于显式外部 Q，不从 NoPyAPI 或返回状态推出；新调用仍从完整 snapshot 开始。

显式 `WorkspaceSession` 将该关系扩展为跨单步调用的有主线程 lease：create 为 DIRTY，成功 COMMIT 才建立 ROI 副本对应；未知效应、重叠写、对象/descriptor 变化或失败使关系失效。每步 scalar 重新读取，固定地址不证明内容相同。只有调用范围内全部相关效应受约束时，才可把此原语接到真实 caller；原固定 N 次组合和该原语本身都不授予这项权限。

`ArrayCopyEffects` 另以原 F4 操作、formal 根和 epoch 解释索引复制：整数轴改变可见 rank，但物理 ROI 仍保留二维坐标；view 创建与后续读取分开，复制按源顺序读取当前内容并推进目标版本。写集与驻留 ROI 不相交只证明几何兼容，还需实际参数/根绑定和普通 ndarray 协议 guard。局部 CALL 证据仅将实际 callable 对回原源码端口，不表示所有写入或属性协议均已捕获；因此不能单独授权 session 延寿。

caller 的覆盖分母来自完整原生产操作，不是 CALL 计数。`GuardedSessionCaller` 沿原 DATA 边和唯一 formal 的词法写/读传递引用路径，另保原 world 和控制边；普通属性、标量协议和临时释放分别附带类型/保留条件，未知 source method 则由事前失效及持续到外层退出的 taint 覆盖。直接 lookup route 必须同时匹配实际 receiver、无实例影子和安装的 descriptor；同 CodeID 本身不够。当前只闭合有界普通 method profile，cleanup、loop、resume 等仍保前沿。安装层保留原 caller 控制与返回，最外入场丢弃旧副本事实、退出释放，尚不授权跨 activation 驻留。

动态 session 的生命周期与固定次数 VRS 分开投影：`SessionLifetime` 以 session instance 为参数，create/destroy 各一次，step/invalidate 为任意有限序列；invalidate 只丢内容对应，不释放容量。它引用原 kernel 的硬件证据和共享正文，不把旧 FIRST/LAST 或 N4 控制冒充新 session 事件。外层 CPU、服务时长和额外元数据容量仍独立未知，不据此声称完整资源可行性或数值最优调度。

`SessionScalarFlow` 另在原端口上计算有限 scalar kind 不动点。多 writer 的词法槽只合并类型，不合并对象、数值或逻辑版本；引用路径仅经有来源的 reference transport 传播。原循环回边、条件、字段 payload 与格式化操作均保留；外部调用的返回 kind 和入口类型仍是待检查条件。`SessionOuter` 将此源图接到嵌套 caller，host 效应未闭合时选择独立 leaf 交付，而非把整个计算区域拒绝。`SessionActivation` 区分一个外层 activation 与任意多个短 lease：每个 leaf 完整快照/检查/计算后销毁，原 host 调用处没有跨调用保留。未知效应不被声明纯或 quiescent，类型推导也不授予内容复用或全局调度权限。

## 五类核心实体

统一查询使用`ProgramView`与既有`ProgramDependencyGraph`。源码ID由原unit、编译/绑定profile、template和原Step/Region/完整Port限定，不随查询函数或展开顺序改变；节点投影保双向来源，不重新生成操作语义。
`get`返回公共节点元数据，`original`保原强类型记录，`expand`展开共享区域，`producers/consumers`查询依赖，`explain/explain_edge`保来源及原调用路径。Use、control等辅助端点有独立role，`kind=None`不是UNKNOWN，也不作为新计算值。
统一视图按单元发布：namespace/material各存一次，节点与边使用既有类型ID引用，原G.Edge以封存F4 template＋ordinal定位，不重复保存完整端点/来源对象。原指令正文不复制。
数据生成器、wire类型和材料协议精确核版本；facade producer SHA保来源，选中单元总由当前同版本映射规则重推验收。增加查询方法不要求重建未变的前端材料；改变映射/数据合同必须重新验收。

入口：`python -B -m scar.cli program-view --catalog CONTEXT_CATALOG --unit UNIT --out FRESH_VIEW`；可重复`--unit`。`--read VIEW --unit UNIT`对该完整单元按当前合同重推，而不是只检查文件哈希。

局部状态通过`ProgramView.state(unit)`接入同一查询。`BindingLocationTemplate`由源版本、原C2接口/绑定槽和key限定；实际frame、mapping或cell仍为实例参数。同一源槽不同激活不因此成为同一个物理位置；C2 binding索引也不是CodeScope ordinal。`BindingVersionTemplate`另保原函数、定义类型/站点及执行occurrence参数；绑定到同一引用不能证明对象或Tensor内容未变。读取保匹配recipe栈的全部may定义，`BINDING_READ`带条件和MAY精度；`STATE_LOCATION`统一为位置→版本。写入/删除、形参入场和开放来源分别表示；None payload不是Python None。

隐式类名字沿原C2分类或ordinary规则投影：prepared mapping、frame globals、frame builtins和捕获cell分开，有序lookup不是所有位置都已读取。显式环境/mapping参数通过条件ENVIRONMENT边绑定地址owner；没有原C2接口索引时保None，不借用另一scope的索引。`ReadBuildClass`是当前frame builtins中的语言查找，不等于导入builtins模块或已选择默认builder。escaped-root等原路由优先复用，无法确定不能自行改成源码globals_scope。

注解lookup按原`AnnotationStateAccess`分派，creator与deferred request scope分开：LOCALS_CHAIN不能借同名GLOBAL计划，class-cell保原implicit/binding/read_plan。`ReadAnnotationNamespace`可产生registry引用；`ReadAnnotationRegistration`产生bool，namespace read的payload为None，contains/index仍保原组合指令，不虚构registry GP。无显式owner时保实例参数，不拿world冒充dict/cell。namespace store使用原key/value端口；ensure只作条件UPDATE。默认求值器及type-parameter tuple关联也只是UPDATE输入，不执行其延迟正文。

`state.content()`将属性/索引协议接入同一namespace：地址是`(receiver引用发生实例, Field/Item/SliceBounds)`表达式，不是已选物理槽。属性lookup名复用已封存private-name投影，并以完整world对/receiver/action/payload定位原fact；索引DELETE直接来自原操作，不能依赖缺少DELETE的旧items表。原owned receiver Use、动态key及bounds均连到地址，不重造slice对象。`ContentStateProjection=project(world,address)`是原world上的符号状态变量，前后ID不同不证明写入或LogicalVersion不同；GET可能调用有副作用的descriptor，仍保前/后态。payload与GET结果保原端口，不建立结果等于slot内容的alias。实例、协议与实际内容版本保持明确开放前沿。

聚合更新用`WholeOwner`选择整个目标内容，不人为分配未知element key；UPDATE的payload是元素或源容器输入，不表示整个新内容等于它。DICT_MERGE保callee/目标/源三个原端口，comprehension用原ChoiceBinding区分LIST/SET/DICT。条件由原操作、owned Use、地址和真实completion端口限定，不用全局收集序号决定身份，便于独立并行构建。

保留的环境值与可变namespace地址分开：`ClassTypeParameterProjection`保原`ReadClassTypeParameters`、完整ClassBodyInterface/类型参数SlotRef、environment及其owned Use、结果和world对。它是当前callback环境→tuple结果的带activation条件依赖；不是环境与tuple的alias，也不是后续`__type_params__`字段读取，不造LocationID。`state.environment_values`提供类型化查询记录；持久化依赖边引用已共享的原操作/端口，完整interface不再复制到通用状态表，reader机械重建。实际创建上下文及版本尚待③选择。

源码函数接口保原callee来源、signature绑定、参数/env/world、pack/default及data/world返回，并索引所有未接调用和父前沿。`scar function-interfaces --calls DECLARED_CALLS --out FRESH --jobs N [--unit UNIT]`发布条件接口；`scar function-call-view --interfaces INTERFACES [--unit UNIT]`打开其统一视图。`SourceFunctionCallProvider(view,store)`默认不物化任何单元，查询按需求接入；原source bodies继续共享。进入使用包含目标body Region的完整Frame，返回须匹配pop；未知外层保持None/open suffix并显式报前沿，不能并入所有caller。FUNCTION_ENVIRONMENT和pack只是依赖，默认值创建上下文未选时留前沿，deferred creation没有普通正文/返回边。静态候选不授dispatch或内容版本证明。

函数对象的创建与使用分开：`SourceFunctionCreation`保原factory、creator/body scope、code/globals、default operand/原expression、cell handles、annotation和前后world；factory的多ref结果按原ABI保留。`scar function-creations --catalog CATALOG --out FRESH --jobs N`发布分片；`SourceFunctionCreationProvider`把初始输入角色接原owned Use/操作，`function-call-view --creations STORE`可合入视图。创建不执行body，输入字段也不是当前对象内容的alias。

构造/descriptor使用同一source端口与frame协议：`ConstructionInterfaceValue`表示原`NEW_RESULT/NEW_WORLD`，不等于fresh对象、storage或已知状态版本。`ProtocolCallableSelection`由原callee、阶段world和目标定义证据投影当前函数；`ProtocolDefaultSlot/State`保当前默认槽及读取，初始表达式仅为来源。`SourceProtocolCallProvider`把原SOURCE_NEW/SOURCE_INIT和method连接接到共享body；Frame.selection保原stage/edge事实，返回精确pop。INIT的数据返回丢弃，构造结果来自NEW_RESULT；OBJECT_INIT/RETURN_FOREIGN保NEW_WORLD→最终world的MAY依赖，不宣称版本相等。class receiver是type投影，pack仅是依赖；所有原连接guards保留。接口值的owner及CONTAINS让原call可展开到这些值。`bind_connection`复用capture校验，不伪造普通函数query/origin。

`ProtocolCallFrontier`保原DeclaredContext、可选Call FactRef、前沿事实及原因；它是现有节点的目标字段义务，不是UNKNOWN操作。`SourceProtocolFrontierProvider`按需挂到原call/callee Use/value/result/world端口，incoming/outgoing均可查询，按实际caller code_scope过滤；不消去未决类/MRO/descriptor条件。普通protocol视图自动注册，`--frontiers-only`可只挂接这些字段而不准备已知正文。source-protocol-view/2另引用按unit保存的frontier组件；原normal接口表与body引用保持。

普通模块和显式compiled exec上下文共享`FunctionInterfaceReader`及同一接线器，材料schema和来源分开。`scar compiled-function-interfaces --catalog CATALOG --unit COMPILED_UNIT --out FRESH`只对保留的compiled上下文使用既有namespace/binder规则，保完整候选/前沿；不把exec globals当可导入模块，也不加入import provider索引。`function-call-view --interfaces STORE`按schema打开两类材料；两个call可共享同一body而拥有不同frame。当前exec globals、捕获、默认值及namespace版本条件仍不因此证明。

返回函数增加`ReturnedCallableProvenance`：原函数创建、返回引用切片、完整creator接口序列和当前consumer各自保留。creator描述过去的引用来源，不能压到当前调用栈上；当前body仅压入当前call frame。`ReturnedCallableProvider`把选定创建点接到当前callee，保MAY及原前提，不读cell内容。所有source Frame.selection保原接口Fact；叠加叶中继承的接口仍引用父Fact。`function-interface-collection`只存不重叠叶的收据与unit路由，接口行和正文仍留在原分片。

`ObservedSourceCodeLink`保存实际CodeCapture/target/entry原锚点，经既有编译器的父LOAD_CONST、完整源码span及常量路径匹配原Function/Body，并核完整code指纹。统一图以辅助Provider记录及IMPLEMENTATION/MAY连接表示这种代码模板对应；不授函数创建实例、globals/cell相同或某静态调用已激活。普通源码只重放既有`flags=0,dont_inherit=True,optimize=0`合同；compiled exec需自己的原profile，不能由运行code的CO_NESTED等flags猜编译输入。
Lambda 还核原F3 `LAMBDA_BODY` 投影；同一行的多个表达式按完整跨度和真实常量槽分别对应。
GeneratorExp同样核原`GENERATOR_EXPRESSION`编译器投影；代码对应不代表创建已进入正文，也不建立生成器实例身份。

编译器生成的注解callback用独立`ObservedSourceRequestCodeLink`对应原deferred Region，而非伪造FunctionFact。完整code、父常量路径、MAKE_FUNCTION/安装协议、F2声明及C3请求scope共同定位原封存recipe；请求输入仍是environment/format/world，返回value/world。`SourceBoundaryCallback.body_kind`区分普通函数、求值请求及无源码接口；经native dispatcher的callback返回该dispatcher，不等于外层CALL结果。format、创建实例及globals/cells内容仍各自待绑定。

`CompiledCodeProfiles`沿原compile请求、文本与C1/F4收据恢复有效输入；相同root code只证明模板对应，不选择exec namespace分支。`ObservedSourceCallsite`再以完整caller code、scope、操作ABI及源码槽连接真实offset与原Call Fact；ApplyDecorator与工厂可有相同span，须用原定义的code常量位置区分。`ObservedSourceFunctionGroup`按原call/definition归组；已有等价接口保原Fact/frame，只有缺失目标增加条件接口。未知receiver以可能的0/1额外prefix记录，不用创建期default排除；未证布局只注释code证据，不新增错误参数路由。动态Invocation仍在原观察记录里，不能冒充静态Frame，循环body继续共享。
`LOAD_SUPER_ATTR`按原Attribute.value边对应内层源Call，后续方法CALL另算；以参数位区分0/2显式参数，不按`super`名称推断真实目标。`CALL_FUNCTION_EX`保留参数包协议；看见CALL仍不证明native入口、分配或普通形参绑定。
多行链式方法CALL也可指向末尾属性片段：用原F2 callee的Attribute终点、属性标记UTF8长度和Call终点推导候选跨度，再核完整caller code及原参数/world ABI。它仍对应整个原Call，不创建另一计算节点、不按方法名识别；同unit的语法父索引只建立一次。

采集计划是范围策略，不是程序事实。`scar refine-capture`只从已核源码的CodeID及原frame/path关系导出补采selector；新增正文可限定在完整父CodeID＋调用offset路径下，避免把共享库函数的所有调用全局展开。原selector和原图ID保持；同一CodeType的monitoring mask不能授予所有激活展开权。plan schema2持久化路径限制，schema1原计划仍可读取；未选择上下文保持开放，不能宣称未执行或无效果。collector同步子树与应用执行分开，即使它们调用同一stdlib代码。
正文entry与receiver身份分别判断：原caller的相邻CALL区间内，完整code、直接caller及site唯一匹配的非挂起帧可供应环境/world/返回接口。receiver不能weakref时不保留用户对象，也不猜地址身份；参数布局未证则保`NOT_MODELED`及原帧锚点。
这种普通正文的形参另保`CallInputEnvelope`：原callee、显式参数、packs及调用前world通过原CALL操作接到尚未精确路由的data formal。边是匹配frame下的MAY依赖，不是REFERENCE/alias或参数绑定证明；不造参数包、不引入调用结果作为输入。已有精确route优先，可挂起正文的创建不借此进入正文。

`SourceObservedBoundary`是原CALL拥有的辅助接口，不是新Operation或假callee body。按原call fact和target kind共享；callee、参数/packs、前后world及result引用原F4端口，观察事件另表保存。`TYPE_OBJECT`不等于标准构造阶段，未分类调用不重标native。可选native companion按同次caller/attempt/callable连接原carrier/ABI字段；无采集为`NOT_RECORDED`，不同于已采字段`UNKNOWN`。输入/状态/可能正常输出边保MAY条件，effects继续为UNKNOWN，不授native entry或正常完成证明。
边界内回调按原caller、CALL位置及相邻CALL/caller退出区间接线，按边界＋完整CodeID共享，帧观察另表。普通回调使用独立的匹配frame，参数来源保持条件投影，返回到dispatcher不等于外层CALL结果。可挂起代码只接共享body的IMPLEMENTATION关系，不重绑初始参数，也不把co_flags当作恢复位置或实例证据。缺源码仍保已知CodeID与入口锚点，body字段未决，不删除回调。

`SourceExecProvider`保原exec actions/edges、namespace选择及匹配body frame；ExecProtocolValue是辅助协议端点，不能伪装成源码GP。两分支有独立selection，body data进入discard，exec的None来自正常返回契约；审计、builtins初始化和释放保各自world/control边，BODY输出只由原正文返回供应。声明script入口不授运行激活。入口为`scar exec-view --catalog CATALOG [--out FRESH]`，`--read VIEW`严格重推全部typed表。

`SourceConfigurationProvider`把已版本化的输入表达式、字段投影、插值、读请求和声明root版本接入独立标记的`DECLARED_INPUT_ALGEBRA`子区，不能将这些操作计作运行事件。它复用canonical WholeOwner地址及post-call content-state公式，只以MAY条件把声明值接到receiver内容候选；Python返回值、整个world及未来task cfg均不相等。`scar configuration-view --catalog CATALOG --boundaries BOUNDARIES [--out FRESH]`发布小表，`--read VIEW`重推所有记录/边。原输入版本、源码API版本、实际receiver及后续mutation仍是独立条件；所有obligations保持可查询。

入口为`scar protocol-call-view --source CALL_CONTEXTS --catalog CATALOG [--members MEMBERS] [--unit UNIT] [--out FRESH]`；无unit且指定out时发布原接口域全部单元。输出按unit分construction values、callables、default slots/states、relations，引用原正文；未解namespace/MRO/实例、方法前沿与实际dispatch仍以原材料及条件保留。三个调用适配器先通过`prepare_source_ports`按unit/template收集去重，批量准备后才逐端口接线，避免交错读取反复解码整张模板表；准备不代表执行或准入。

`FunctionDefaultSlot`是源函数参数的逻辑默认槽；`FunctionDefaultState=project(pre_call_world,slot)`保当前绑定状态，不是tensor内容或LogicalVersion。它依赖当前callable和world，再以条件projection进入formal；初始GP仅是需creator/无mutation证明的provenance。`FunctionCaptureBinding`沿原C2 relay/derived transport把当前callee env连到原`BindingLocationTemplate`，位置ID公式与第二层相同；handle选择不读内容、不宣称外层frame同一cell。相关metadata保MAY、frame和未决实例前沿。

Native单元沿原codec namespace、entry/exit Value/Use和world投影。Tensor与Generator状态地址是`boundary slot＋Optional/List类型路径＋选中值实例`，不是已观察元素、StorageID或内容版本；列表不展开，Optional保存在条件。输入保原before/normal/raised world，返回仅在normal存在。原alias before/显式after字段保留，其阶段是annotation字段而非程序执行时间；标签按原declaration与boundary限定。分组是声明关系记录，查询时才产生条件MAY/PROJECTION边，禁止传递合并成对象或storage身份；wildcard、空after、无annotation与False may_write都不授无副作用/RNG或no-alias结论。

入口：`python -B -m scar.cli native-state-view --source BINDING_NATIVE_MATERIALS --out FRESH_VIEW`，`--read VIEW`独立重推所有native分片。`NativeStateSession(materials, view=existing_view)`复用同一查询和原core loader；基础core记录优先于扩展metadata，扩展状态不伪造canonical正文。新状态指向原图与声明，不复制原native图，也不声称已选择Python/native调用或dispatch。

`SourceNativeCallProvider`把原Python actual/pack/default接口接到同一native entry，将各normal output的原Use接回Python包装结果；world前后仅为状态依赖，不是版本相等。无actual的默认表达式仍保原声明，不造常量或引用alias。`CALL_TARGET`使用共享native Region；Frame的`selection`保原NamedBindingMatch FactRef，防止同call、同算子的不同来源路径混接，返回必须准确pop。所有转换/包装边均为MAY，parser、override、dispatch及未解receiver/conversion显式保留前沿。`scar native-call-view --source BINDING_NATIVE_MATERIALS [--unit UNIT]`打开视图，源码分片按需加载；`interface(unit,ordinal)`返回原输入、包装声明及frame，不复制native正文。旧schema2局部存储禁止context，故只对已知旧dependency版本允许逐行重推读回；含Frame的旧query wire不隐式兼容。

`SourceEffectRequest`保原操作和69种指令的闭合分类，动作标签可重叠，只表示待解释能力；不授pure、真实搬运、分配或RNG结论。已知绑定接口与尚缺的地址规则分开，`missing_rules`不伪装为运行时UNKNOWN。CLI为`python -B -m scar.cli program-state-view --catalog CONTEXT_CATALOG --unit UNIT --out FRESH_VIEW`，默认根据READ、原C2槽域/owner和原函数body作用域选择所有可跟踪局部槽，随后发布内容接口；捕获不冒充当前帧的局部槽。`--slot C2_LOCAL_SLOT`可缩小查询，`--interfaces-only`不求绑定固定点。schema `scar.source-state-view/1`中绑定与内容分表，均引用同一单元namespace与既有ID。`--read VIEW`独立重推全部已存记录及覆盖数，拒绝虚假的完成/内容证明标志；类型/求解器合同精确核版本，投影适配器SHA保来源，其变化必须通过全部表和计数的当前规则重推。缺少新环境合同的旧单元还须确认没有该投影；有则重新发布。跟踪的操作/控制边索引引用原正文投影，不复制指令。

| 实体 | 含义 | 对应程序部分 |
|---|---|---|
| Operation | 消费端口、产生结果并具有效果的操作 | 调用、计算、绑定、读写、分配、搬运、IO、等待 |
| Value | 原定义或区域参数产生的值 | 数据、对象引用、状态版本；局部绑定和协议令牌使用独立类型 |
| StateLocation | 可读写的位置或抽象状态域 | 对象字段、容器内容、Tensor 内容、module、RNG、文件状态 |
| Region | 有入口、出口和内部结构的子图 | 模块、类体、函数体及受支持的控制结构 |
| Resource | 执行或承载状态所需的资源 | CPU/GPU、内存、存储端点、文件、队列、stream/event |

这是实体分类，不是把每行源码划入一个互斥类别。一行代码可展开为多个操作与关系。
输入/输出是端口角色；常量/不变量是限定范围的性质；shape/dtype 是属性。
Block、Use、控制出口等是连接上述实体的强类型记录，不能合并成无类型字符串边。
实际定义见 [核心 IR](../scar/ir/program/model.py) 与 [资源 IR](../scar/ir/program/resources.py)。

K / Σ / A / R / Q / M 分别观察控制、状态、动作、资源、契约和证据，彼此正交。
Action 的 VAL / REP / MEM / XFER / STATE / CTRL / ORDER / IO / OPAQUE 也是可重叠标签，
并非五类实体或底层 `Behavior` 枚举的替代品。Q 保留假设；M 区分静态依据与运行证据。

## 硬件化简契约

输入是保留数据、控制、状态、效果及逃逸依赖的程序图、硬件拓扑/容量/代价模型与契约 Q；未知字段和未决条件继续显式保留。
Operation、Value、StateLocation、Region、Resource 及其原 ID 保持；任务图是引用原语义与来源的派生 IR，不替换原程序图。
决策覆盖 Q 允许的结构等价变换、物理副本及存储位置、搬运与计算先后、buffer 分配/释放生命期、设备选择及并行度。
目标是在峰值 memory、容量/可用预算、transfer 与同步约束下最小化 makespan；allocation 不等于 ready，逃逸输出不能提前释放。
算术次数和逻辑 bytes 是静态需求，不直接等于 duration、实际 DRAM 流量、cache miss 或有效带宽；缺少的成本逐字段保留 UNKNOWN。
当前固定 placement solver 只在给定整数 duration、资源需求与预算的离散模型内求解；尚未完成 joint placement，也不证明物理全局最优。
当前可执行后端准入有运行时 guard 的纯 CPU 数组区域及直线函数 pipeline；任务图不自动关闭开放效果、控制或跨调用内容版本条件。
实测用于成本校准及正确性/性能验收；优化主链由程序图、硬件模型与 Q 驱动，不以黑盒编译版本排名替代。

数组 pipeline 是派生 profile，不是第六类核心实体：`PipelinePlan`引用原入口区域，`RegionCall`引用共享 leaf plan，
`inputs/output`是本次入口作用域内的值边。全局 callee 选择保留 CodeID guard，不能把源码声明当成实际 callable。
内存规划再从这些值边派生 producer/consumer/ready/last-use；逻辑值与物理槽分开，同槽上的后续值不等于旧值。
`slot_id`是分配规划变量，不是 Python ObjectID、实际 StorageID 或数据版本；每次入口的分配有独立生命期。
固定 pool 的逻辑 release 只授权复用，物理槽持有到入口返回；最终输出独立新建并逃逸到调用方。
该 profile 的最优目标是固定顺序下的 workspace payload 容量，不是动态峰值、RSS 或整体时间最优。
完整 elementwise 融合进一步允许逻辑数组值没有独立物理材料化。原调用图及 guards 不变，
`原 leaf/source Expr → raw 融合 Expr → 既有优化器 Expr → 原生算子`逐层保引用和摘要；
raw 融合证书不冒充后续 CSE 后的图证明。输入替换依据 SSA producer 值边，不依据变量名称或物理槽。
有 slice、开放效果、逃逸或死 call 的区域不由该融合规则跨越，仍保原表示及其条件。

## 图、端口与层次

图用规范化记录和 ID 引用保存；源码对应、包含、数据、引用、状态、控制及资源关系分别表示。
F2语法输入按F0/F1收据、parser profile和局部occurrence ID定位；字段/列表槽是有序子引用，保留精确presence。
语法occurrence由后续语言规则映射到核心实体和端口；一个表达式可产生多项操作，parser单例不合并计算身份。
文件及版本是并行构建和存储的分片单位，函数/循环是可展开区域；文件边界不强行划分静态与动态。
文件内外已知关系均静态连接，动态只补实际选择；未执行路径继续保留条件，不因缺观测消失。
结果/参数通过有序 Use 连接消费者，`f(x, x)` 的两个实参端口不可去重。
端点以原 artifact/namespace、版本、类型 ID 和适用上下文定位；名字或行号不能单独证明身份。
静态调用上下文不是动态 Invocation；未知外层上下文也不能冒充已闭合的调用链。
源码位置与求值用途也分开：同一表达式的VALUE模板返回对象，CONDITION模板返回控制布尔量，不能把短路返回的对象再truth一次。当前source recipe以限定profile/输入下的occurrence＋purpose区分模板；条件请求按需展开，原源码分区与上下文变体分别计数。
条件保留主体、有序操作数和依据；“在条件成立时精确”不等于条件已经获证。

函数体共享一个原始模板，调用点通过实参、形参、环境、状态及返回/异常接口连接。
F5事实的code_scope来自原CompilationContextFlow；声明子图存在不等于函数体被激活。入口求解按模块、调用及显式区域接口准入事实，未准入模板仍保留。
普通外部符号不自动成为namespace；只有模块或provider声明允许继续限定名展开，其余属性仍是原receiver/result/world关系，不制造无限符号字符串。
词法包含关系与调用边分开。F4-B recipe的`SHARED`区域允许同一词法所有者下的多个子区域通过显式端口调用一份模板，但不得跨namespace或延迟执行边界；原区域参数/结果不能变成隐式跨区引用。普通try/finally据此共享一份finalbody，不按pending completion复制源码。这是局部recipe能力，尚未生成canonical共享图。
字典显示沿分配、顺序求值、更新及异常状态展开；空字典也不能当共享常量。旧有限文件图保持原覆盖。
展开查询返回原 Operation/Value/Use，而不是为每个调用复制整份源码图。
循环也可作为整体展开：当前 `LoopTemplateView` / `LoopConstructView` 是原图的投影视图，
保留成员、嵌套关系、初始输入、header 参数、回边、正常退出、break/continue 及来源缺口；
它们尚不是独立 canonical Region，也不把每次迭代静态展开成一份新图。
见 [循环模板](../scar/analysis/program_loop_views.py) 与 [循环成员](../scar/analysis/program_loop_regions.py)。

跨边界接口必须保留正常返回、异常、挂起及适用的状态输入/输出，不只连接返回 Tensor。
循环外引用可以指向可变对象；引用不变不能证明对象内容跨迭代不变。
初始化输入、循环携带值、迭代局部值和退出值须沿原端口区分。
F4-B source recipe另以原F2 `SourceLoopTarget`连接局部子图的break/continue端口；对应共享循环消费自己的跳转，耗尽与break分开连接else/后继。suite只沿正常完成接下一语句。语句may-completion上界、局部产生规则与最终CFG/SSA是三种不同的证明范围；iterator引用不变不授内容不变或引用释放时序证明。
编译条件与运行状态也分开：`ReadCompilationOption`保目标的符号编译事实，不读取分析进程flags冒充目标事实。注解的声明、字符串化发布与后续请求沿原C1求值阶段连接；deferred来源不等于声明时已求出值。增强赋值的地址/读/运算/写由同一父语句安排，不能当作两次独立地址求值。
注解状态访问显式区分locals→globals→builtins、globals→builtins及class protocol cell；来源相似不能证明它们引用同一对象。查找消费当前world，不能冻结初始化时的容器。源码编译元数据不计为运行操作；重复编译展开的登记编号是编译上下文参数，不是多次执行或运行时任意选择。有限取值域已知而上下文未绑定时，保留参数及缺失证明；单值可静态折叠。
编译上下文通过原代码版本、优化策略、常量引用路径和指令偏移标识；不是动态Invocation。代码指纹保指令、参数/捕获布局、异常/位置表及精确常量树，含编译生成的slice/frozenset。源码参数可以拥有多个已证生产位置；可由某一具体位置机械绑定。没有保留的生产位置不证明消费者无用，未证路径也不能抹掉其他已证边。
共享模板另带`CompilationContextFlow`：每个区域有逻辑编译环境端口，每条入口引用按原步骤ID及类型字段/tuple序号定位。边区分调用、子表达式、代码声明、类型工厂、循环条件/正文/退出、完成处理与恢复；break/continue目标及实参里的区域引用不算入口。代码声明不执行正文。
`SourceTemplateGraph`进一步把不同局部recipe的子表达式/子语句接口连接起来：节点键为原occurrence＋purpose，并由源输入及版本限定；边保存调用步骤和有序实参，完成端口按kind/target/ordinal连接。节点引用已封存的原recipe，展开时核对原字节指纹；不复制正文、不展开循环，也不据此证明native调用或完整效果。这仍是F4-B接口图，后续canonical CFG/SSA另验。
当前F4-C的NORMAL_SUCCESS视图以这些既有操作/区域为节点，新增DATA、STATE、CONTROL、DECLARE边。端点沿原模板、区域/步骤ID及端口序号定位；调用/返回用相同CallSite配对，避免共享body的返回连接串到别处。循环携带量回接区域参数，不变量只接初始化输入。应用报错暂在契约外；迭代耗尽仍按正常控制连接。
源码recipe同时声明抽象world类型及异常ABI。核心操作的参数、正常结果和异常出口必须匹配；Call/ClassDef的显式与隐式参数、扩展解包、slice缺省项及tuple构造数目沿原F2字段推导，不能从待检查的端口数倒推。普通tuple、模板字符串段tuple和插值tuple是三种明确布局。无源码的shape-only样本可省略profile，但源码入口拒绝缺失profile；这些签名不是效果、清理或native准入证明。
同一模板的不同调用位置通过各自的边限定环境，循环重复使用同一静态边。进入函数、类或延迟求值器时更换代码环境；内联scope保源码身份而共享外层code owner。端口的source_scope表示当前逻辑代码环境，原语法namespace仍保存在region.context中，两者不能互代。编译事实使用者显式引用端口；这些是参数化接口，尚不是caller到原native指令位置的完整实例化证明。
区域 cut 查询列出已有跨边界依赖，不自动证明可外提、复用、删除或整体替换。
见 [公共依赖查询](../scar/analysis/program_dependencies.py) 与 [边界分析](../scar/analysis/program_regions.py)。

## 身份、版本与不确定性

CodeID 指向代码版本与位置；InvocationID 指向一次真实调用，两者不能混同。
对象身份、storage/allocation 身份、内容版本和逻辑来源也互不等价。
同一对象可以被原地改写；不同对象可以复制相同内容；同一 storage 可以有多个 view。
地址复用需要 allocation/epoch 约束，view 需要 offset/shape/strides；地址相等不证明同一分配。
引用来源路径解释“引用如何传入”，不自动证明读取结果、实际 callable 或数据内容相同。

F5按需引用视图用`(原Endpoint, 配对调用栈)`作为查询状态，不复制程序Operation；
alias、完整LEXICAL binding、实参、初始default、data return及构造结果使用不同边类型。
反向跨返回口进入调用时压栈，形参只匹配该调用再出栈；捕获环境及递归截断保留未决项。
`NEW_RESULT`是构造协议结果，`__new__`可返回旧对象；只有原OBJECT_NEW/EXPLICIT_OBJECT_NEW提供分配位置候选。
同一候选位置的字段SET→GET只产生带scope、实例、descriptor及版本条件的may-content边，保原world；
它不变成alias，也不证明当前内容。由此推导的方法接口独立于入口准入及runtime dispatch。
声明正文上下文以原函数body和类型化class-input为键，调用边显式连接原实参/形参、数据返回和world。
同键共享正文，递归保回边；它不是Invocation。descriptor包装的原调用、builtin查找和前序miss条件均保留。
非NORMAL连接仅声明创建边，不因创建generator/coroutine就进入正文。声明状态接点另表引用这些边，不改历史CallLink。
调用后状态视图复用原F4 STATE/CONTROL端口、guard及CallSite，不以事实表行号推断执行顺序。
原子操作的world输入→输出另记OperationStateTransition，表示状态变化，绝不是数据copy或内容不变。
迭代器耗尽沿原advance的RAISE world及匹配StopIteration控制退出；普通应用异常不因此扩大。
后继状态cut是保调用标记/分支条件的may路径；可达不证明must执行、当前版本、匹配上下文或效果已消除。
状态访问的地址另记原根端口＋有类型字段/索引selector，并保访问前后world。动态key引用原值端口，不按repr猜值。
仅同CodeScope的完整LEXICAL binding可用于局部路径推导；捕获、模块查找、循环及预算截断显式保前沿。
call-result是独立不透明根，不因参数来源相似就把返回对象或deepcopy输出当成原输入。
跨函数返回图分别保存world return、data return及引用路径依据。先沿已选择的声明上下文连接，再保留历史may-call的独立调用点；
引用路径未接通时只停止该引用的传递，world仍可返回。world返回不证明内容保留。图节点共享原body，不为循环迭代复制。
调用消费者另记原callee/位置/关键字/参数包端口、来源路径和形参route。字段投影是对根对象的数据依赖，不能变成根对象alias。
形参接线保原world输入/输出与函数创建种类；生成器创建不进入正文，未解析target、pack或构造协议不等于没有消费者。

正文值流保原DATA/STATE/CONTROL/DECLARE边，并为每个非结构操作保存原instruction及类型化输入/输出接口。
输入→操作→输出表示可能的数据推导，不是alias或LogicalVersion等价；零输入操作同样保留。
局部读写以完整(CodeScopeRef, LEXICAL Namespace)位置节点连接，避免writer×reader展开；跨作用域捕获和非词法内容另保边界。
DATA包含bool、协议值等，不只对象引用。轻量查询只传播DATA与操作的数据输入/输出，不以world序列污染全部后继值；
原world、调用配对标签、分支及循环回边仍在图中。该flow-insensitive may查询不证明平衡执行路径、当前状态版本或实际影响。
派生调用输入可另接原CallLink的actual→formal、world与返回接口；根形参→派生actual的may依赖仍在父值流图，不能拼成根形参alias。
历史may-target与当前caller上下文匹配分开；接到compiled函数的声明接口也不等于已展开或运行该正文。
普通源码与compile-context使用同一材料接口，按原catalog格式选择reader；compiled计划仍引用其原domain、行号、digest及编译profile。
正文展开用原函数端口去重，已有正文只存跨工件引用，新增正文才写本地表；入口分别保原binding/形参/world条件。
展开静态函数正文不选择动态activation；后续来源查询仍保may-dependency限制，循环仍用原回边。
调用上下文引用查询另以原incoming CallLink为条件：形参只沿当前栈顶接口回到它的实参，data return先压入该接口再由其形参配对弹出。
外层caller未选定时保边界，不合并其他caller；非词法namespace、捕获、deferred及构造结果各保原协议边界。
同一正文共享多个incoming接口；查询栈是静态路径证据，不是Invocation或循环展开。来源匹配与开放上下文分开记录，原may候选不被删除。
若引用路径得到原函数定义，可用原signature binder生成带路径条件的函数接口；函数环境仍来自原callee，不以CodeID替代函数对象状态。

构造是独立协议区域。SOURCE_NEW保cls/原参数/world，数据返回NEW_RESULT、状态返回NEW_WORLD；其结果可能是已有或foreign对象。
SOURCE_INIT从NEW_RESULT/NEW_WORLD入正文，只把world返回原调用；完整构造的数据仍来自new结果，init数据被丢弃。
OBJECT_NEW/OBJECT_INIT、foreign返回、new/type-test/init未决都保原stage/class及证据，不把pending无输出阶段补成完成路径或分配证明。
构造class输入只从单一ARGUMENT引用投影：init的NEW_RESULT给请求类的兼容INSTANCE，new的原class参数给CLASS_OBJECT。
兼容INSTANCE可能是子类；精确类型、MRO与override仍未证明。pack本身不当作单一class引用，无关参数包仍留在原接口而不增加类型缺口。
多个原构造entry可共享同一声明正文/类型上下文，其实际对象、world与entry条件继续由原connection分别保存。
构造兼容输入可作为声明成员展开的根；旧正文引用共享，新正文按source callable去重并按源码单元批量读取模板。
instance上的classmethod只有在绑定class等于记录source class的附加条件下采用该class参数；未确定子类分支另保前沿。

namespace来源图以原GlobalPort及待选择的属性序列为查询状态。源码slot写入只产生带namespace/版本/作用域条件的引用来源，
不宣称当前slot、对象或内容相同。编译exec的globals字典与普通source module分别保身份条件；compiled单元不进模块导入候选索引。
导入按原Import.module解释返回模块候选，requested_module保完整请求；所有路径/名称映射候选均保留。
GET属性选择与IMPORT_FROM的registry fallback分别记录，属性receiver与结果不成为alias。
标准`import a.b.c`返回`a`时，父包上安装子模块的每个已匹配属性前缀另保来源边；仍需当前slot及标准导入条件。
namespace来源子图按原端口共享，使用点以原callee查询/state锚定，因此各caller栈不合并。函数接口保原调用ABI与环境版本条件。
进入被解析函数正文时，outer stack取原调用者查询的根；factory内部的来源stack仍是来源证据，不能代替调用者的实参frame。
派生接口的data return另有完整caller-stack许可域；读取同一静态result端口必须匹配该域，不能把某个caller得到的目标推广到其他caller。
历史may接口保原开放条件。所有正文共享，成员递归路径保边界/原回边，不通过迭代复制正文。
源码声明图按原callsite及CodeScope覆盖完整选定源树，namespace路径可在没有选中incoming的情况下单独分析；它不伪造Invocation或scope激活。
同一原callsite的caller查询另保调用栈，两种视图不能相互冒充。生成绑定按完整路径连接，保所有overload和原source ABI；转换依赖不当作alias。
分析状态（尚未选定、尚未证明）保在请求证据及状态字段，不能当作运行时guard与“选择该binding”合取。
生成绑定可独立链接到精确版本的canonical ATen接口。实参→parser转换→native端口是推导依赖；wrapper返回形状与native返回slots分别保留。
source world→native world、native normal world→source result_world也是依赖；wrapper前后effects开放，不能折叠为同一StateVersion。异常边界留在canonical接口。
一个native整数数组参数可依赖多个显式位置实参，聚合单独标记，不可只取第一个实参；fixed/optional列表的广播或vararg选择未证时保转换OPEN。
词法名字的事实不是执行次序：WRITE记录的value是RHS，真实写站点须锚定原AccessBinding及world；参数是函数入口的独立定义。
到达定义以原控制端口＋匹配recipe调用栈为状态，成功WRITE/DELETE做GEN/KILL，未知world效果弱加入开放来源；循环通过回边固定点求解，不展开迭代。
不同词法槽的传播结果引用共享函数控制边/操作和共享名字锚点；这些原图记录不按分析的名字数复制，分析器也复用原函数值流投影。
读取在READ入口观察，合并所有保留上下文的may定义。只有原C2本地slot所属环境匹配、锚点及控制求解覆盖足够时，正常路径查询才排除不可达的显式写；捕获和未知入口保开放。
原声明图不覆盖或删除。NORMAL_SUCCESS过滤另带路径条件，不能从该投影排除应用异常/resume路径或证明当前内容、纯性、dead work及优化合法性。
原生Tensor/Device/Generator类型投影通过原归一化依赖接入硬件图；Optional/List投影不是实际元素、分配或副本。native输出到Python包装结果也只是依赖，不把整个Python容器置于GPU。
host执行、native executor与目标内存池分别保候选；保留CPU/GPU/内存/存储拓扑不等于证明程序使用这些资源。source文件所在磁盘也不是运行I/O。

`TargetHardwareGraph/v2` 按平台发现全部 OS 可见硬件，独立于 workload：CPU package/core/thread 与 ISA、NUMA DRAM、共享 cache、PCIe、加速器、物理 SSD/NVMe、NIC/RDMA 及存储 backing。cache 是不可直接分配的硬件域；独立 device memory 必须有证据。控制/性能计数接口可聚合为有原 PCI ID 的边界，不建驱动内部。affinity/cgroup/设备可见性与 accelerator 的 discovery/access/backend 状态独立存表；未知属性保持未知，设备仍保留。所有关系有原 inventory FactRef；链路规格和相对 NUMA distance 不冒充实测带宽/延迟。硬件生产者 SHA 是历史来源，读者以版本化类型及原库存重推接纳 v1/v2，不重签旧快照。
类基项保原求值端口、顺序和CLASSES fact所属创建scope；packed ABI仍含body/name，不能直接当base列表。
namespace图的class-definition来源是附条件的源码声明产物，不是运行时class或MRO证明；当前版本、builtin及动态属性缺口仍逐字段保留。
继承成员记录原有序base槽及完整来源路径；每条descriptor选择可追至记录，不选择第一条基类或写入作为实际MRO/当前成员。

源事实材料与程序的数据Materialization分开。旧集合相对证据URI通过`FactMaterialRoute`沿原`unit_parent`目录选择定位到primitive分片，
核每级目录和叶stored/decoded收据、unit及原locator；不按同哈希任找文件。compiled事实保原variant unit与JSONL records/0。
图保原证据键，`FactMaterialResolver.resolve`提供可实际读取的物化引用；源材料路由不声明对象alias、内容等价或运行I/O。

[逻辑值 IR](../scar/ir/v2/values.py) 已定义 LogicalValue、ValueVersion、ProvenanceRecord、
Materialization 和等价声明，用于分开逻辑来源、推导关系、版本及物理表示。
这些类型的存在不代表任意 deepcopy、transform、CPU/GPU materialization 已自动统一溯源；
当前静态值/引用分析与运行观测也尚未全面接通到该逻辑值体系。
不能以 Tensor ID + `_version` 作为通用内容证明：NumPy 共享写入和 inference tensor 等需另行约束。

UNKNOWN 应落在确实未知的字段、效果或关系上，不能吞掉已知操作、端口及 shape/dtype 等属性。
源码重放对应与正文接口覆盖分别计数：完整CodeID可跨进程复用同一静态源码表示，不复用原Invocation、函数对象或namespace。外部准入按未表示的CodeID分批推进；相同完整源码/编译配置/root及原捕获证据子集仅保一个材料代表。已有源码但缺少annotation等正文接线时保接口缺口，不能反复降低源码或当作正文已接通。
已知、条件成立时已知、符号值、候选集、未知事实，与未分析/不支持/材料缺失/预算耗尽分开报告。
求解达到不动点只表示当前规则不再产生事实，不证明整个程序已覆盖或所有目标已解析。
未见消费者不证明无用；未见异常不证明不会抛错；没有记录写入不等于 NONE。

脚本入口契约引用原源码版本、root及namespace读取端口，并声明argv、cwd、环境和初始globals。
argv/environment记录可变位置的初始内容，不作为不可变tuple/dict常量。初值字段推导受整个入口契约约束。
编译上下文保filename/mode、请求flags、继承策略、source有效future位和优化级别；未知继承不是0。
同路径/同bytes不保证同一编译上下文，compile也不等于执行；parser-only flags与运行代码flags分开。
compile/exec边界按原namespace/reference来源识别，保留全部actual、world和caller scope；同名局部变量不因此成为builtin。
语言默认值、原F2直接literal和一般source value分别记录，不把单个may候选升级为唯一值。
compile结果到exec输入只增加条件引用关系；payload解释、有效编译版本和body绑定需要分别完成，不伪造root展开。
文本候选与标准str(path)可供给filename；编译使用显式继承flags及dont_inherit=True隔离分析器上下文。
CompiledSourceCandidate保存代码值digest与独立source-only控制版本；相等也不自动证明F4对应，不同则不能沿用旧body。
编译变体沿C1→声明/绑定→带flags原生符号→C3对应→F4生成，命名空间由源码和编译输入限定，调用见证另存，不按调用复制body。
F4复用原规则，只为其编译查询绑定局部服务；根future策略与有效flags一致。新图和原source-only图可共存，exec的namespace/world路由另行接线。
可选audit只观察编译文本与执行代码指纹，不提供完整编译请求。离线枚举兼容配方并按完整CodeID核对，局部F0–F4图可作为代码等价表示；多解保留，原请求与创建环境未决。
这些材料按实际捕获的根/嵌套CodeID接入共享目录，不进入模块导入候选。读回同时守住原audit、匹配报告与局部材料；函数正文对应不等于当前参数、namespace或effects相同。 局部CALL声明域与import候选域分开；生成代码的迭代/可挂起/mapping摘要引用真实局部材料收据，不能伪造模块manifest或恢复实例。
实际CALL位置还可对应语言协议操作，而非普通F.Call：生成器创建保留五个环境分量、capture、初始iterator及world，类构造保builder、body callback、name、bases/kwargs或原pack。完整caller CodeID、源码位置、LOAD_CONST/MAKE_FUNCTION、iterator或LOAD_BUILD_CLASS标记共同选择原step。协议接口与观察独立存储，以原操作端口和definition fact连接共享body；IMPLEMENTATION边不证明body激活、namespace身份或generator恢复位置。

有限标量字段沿原scope、参数/默认值、数据返回、模块/属性及content关系传播；world、环境及参数包不作标量alias。
`ScalarTransferGroup`保父表行、原端口及条件，以线性大小私有join传递；join不是程序节点。字段上限4，宽化保open，不从singleton may集证明常量。

content的READ_SELECTOR_V1标签表示读取选择条件；未知位置的写入连接全部已订阅selector。
None用于整体/未知选择或未指定selector的直接依赖，literal None键仍有独立PythonLiteral。
dict.get默认值保直接来源边，不广播为容器内容；编码因子只分解已有关系，不据标签证明版本或身份。

算子结果的声明类型由`ValueShape`及`ReturnTypeEvidence`连接原结果端口，元组字段和序列元素有显式投影。
这些记录是Value的条件属性，不能代替ObjectID、StorageID或LogicalVersion。序列转成list/tuple只传播元素类型摘要。
绑定/未绑定方法的receiver独立保存，原调用的实参、参数包和输入/输出world保持原样；手写接口没有生成映射时不虚构ATen selector。
语言运算保留原Operation及有序operand/world端口；`DeclaredOperatorCall`连接协议分支、声明及有条件的wrapper实现。
stub契约与注册实现是不同依据，不表示发生两次计算；C++注册目标与生成wrapper的实际符号对应，不能按Python名字猜算子。
隐式实例调用通过类上的`__call__`接线，查找中间结果不写回原对象端口；反射优先级、NotImplemented与原地fallback保留条件。
协议依据为固定[Python 3.14.7数据模型](https://github.com/python/cpython/blob/v3.14.7/Doc/reference/datamodel.rst)，不是运行观测。
属性接口单独保存GET/SET/DELETE/IMPORT_MEMBER、receiver、输入/输出world及读取结果或赋值RHS；不伪装成源码CALL。
property accessor复用`FunctionConnection`的参数/环境/状态接线。getter返回连接原属性结果，setter/deleter丢弃Python返回值。
字段、类namespace及模块namespace路由分别标型；字段fallback的descriptor/MRO条件未获证时仍是may关系，不授对象身份或版本等价。
实例字段以`allocation来源＋receiver class＋member`分区；同一源码分配点可能返回多类，不能只保分配端口。
`AttributeContentLink`显式保存SET RHS→GET result及原读写表坐标，原world仍从站点查询；相同抽象内容的不同写入也保留。
它是可能供给关系，不选择读取版本，不证明先后顺序，也不因DELETE或未见写入而清除其他路径的内容。
密集content边可按完全相同的目标邻域存为`ContentFactor`；因子是连接关系的存储表示，不是新的源码动作或数据等价证明。
源索引投影环不能逐位置展开：增量队列处理内容变化，下降闭环保留start/stride/path；UNKNOWN_POSITION是保守位置宽化，原投影边仍在。
类调用分为`__new__ → 返回类型判定 → 条件__init__ → 返回`。`ConstructionPort`以原调用结果和原class端口为锚点，
区分分配结果与分配后world；它是派生协议接口，不伪造源码`G.Port`。初始化消费分配后world，丢弃init的数据返回。
已声明自定义new时不预先生成默认实例；默认object分配要求源码继承链及namespace条件。MRO/metaclass仍是条件，不当成已观测目标。
变参formal的tuple/dict内容与每次绑定的有序实参分开；只有原构造/协议证明形状封闭，才展开调用参数包，不从may内容数量推断长度。
参数包不封闭时，已知函数体仍可在`CALL_SIGNATURE_ACCEPTANCE`条件下接入；`ArgumentProjection`分别表示位置元素、命名元素及剩余参数。
`SYMBOLIC_PACK`与`CLOSED_PACK`分开，默认值缺席条件、原包和未决需求保留；投影传播may内容，不建立包对象到参数的alias。
这只补充条件静态图，不证明调用实际发生、绑定完整、路径可满足或变换合法；生成器/异步创建仍不激活body。
同步生成器分离创建、取iterator与advance：创建保存引用路由，advance才接body，并使用当前推进的world。
yield导出接循环元素；sent输入与terminal return分开，正常return经StopIteration接口表示耗尽，不当作元素。
挂起/恢复旁表引用sealed F4原端口及摘要；稀疏F5没有的world端口仍用其原G.Port，不伪造源码身份。
F4局部图先按源码单元投影为`UnitSourceInterfaces`，再由F5连接；摘要是原图接口表，不另建一套程序节点。
迭代保全部完成端口，惰性body按原code owner选择挂起；映射保确切协议、world及端点映射资格。
非直线模板、无匹配协议、未映射端点与缺摘要分开；这些状态均不授无副作用、参数包闭合或值等价证明。

效果至少区分 reads、writes、allocates、frees、aliases、escapes、RNG、异常、外部及顺序影响。
未知效果保持开放；异常前已经发生的写入、IO 和清理不可回滚成“无效果”。
`.eval()` 不证明纯函数，推理也不默认无 autograd；模式、hooks、buffers 和可变参数属于相关状态。
见 [效果模型](../scar/ir/effects.py) 与 [状态地址](../scar/ir/program/state_addresses.py)。

代码作用域不等于Python可调用对象：模块/exec/eval根、class body与函数正文分别记录。实际CodeID可对应共享Region和独立Frame锚点；class body不进入函数参数binder，当前namespace与创建对象身份仍需单独依据。

## 静态结构与动态证据

静态前端依据保存源码、语言版本、编译来源和受支持的库契约建立结构、可能路径及条件关系。
动态层只补必须运行才知道的实际 callable、entry、实例、运行路径或资源事件；两者保留不同依据。
一次观测可以选择该次调用的目标，不能关闭其它路径、将动态目标集合当全集或授权优化。
动态补图以静态图的未决接口为采集目标，限定时间、调用次数及新事实量；函数/循环始终复用静态模板，重复迭代不复制正文。
同一调用位置只有出现新目标或所需状态事实时才增补证据。达到本次需求或预算即停；未观测分支及失效条件继续保留，不能声称全路径已观测。
静态/动态按事实分工，不按文件边界：已知跨文件边可静态连接，同文件多态也可能需要观测。
运行 code 与原实现图的连接须核完整 descriptor、compiler path/profile 和已记录的源码版本，不能靠 qualname/行号猜测。
没有历史全文件指纹时，当前源码仍可建图；匹配函数 Code 只证明该作用域的对应，不证明历史文件、父模块或运行环境。
缺失 body 可由真实 code 证据驱动按需静态展开；已有规范图、原 ID 和端口继续复用。
见 [调用观测](../scar/analysis/observed_call_bindings.py)、[函数接口](../scar/analysis/observed_function_interfaces.py)
与 [按需展开](../scar/analysis/observed_body_demands.py)。这些接口不是静态全程序闭合证明。

## 语言、native 与资源边界

Python 建模遵守求值顺序，区分定义阶段与调用阶段、绑定与对象内容、全局/闭包/类命名空间。
import 的 module 引用、namespace 和初始化状态分别表示；属性访问描述 lookup 请求，
不能直接当作 `__dict__` 读取或推断其结果身份。类定义最终产物取 decorator 链返回值，不保证是类。
导入的请求参数和名字投影可静态保存为已知字段；实际importer/module实例、registry命中及初始化另属状态/证据。IMPORT_FROM保导入回退语义；star import保动态namespace写范围，不能当作已知静态binder列表。
短路、比较、索引、赋值、调用及解包都可能调用用户代码、产生部分效果或抛错。
受支持的 try/finally/with 保留 pending completion 和清理顺序；generator/async 创建与恢复分开。
函数定义的创建、注解请求与调用激活分开，docstring属于编译元数据；decorator可返回任意对象，最终绑定沿实际decorator结果，不保证仍是原函数。初始defaults和cell引用保持来源，不承诺后续内容不变。
泛型工厂使用原type-parameter子scope；外层已求出的defaults通过显式端口进入。TypeVar等对象、惰性求值器、初始化绑定和参数tuple分别表示；bound/default属于后续请求，不能因其源码出现在声明中就提前求值。
复杂异常组、模式匹配、动态生成代码及部分异步/委托协议仍可能有明确前沿，不能宣称全语言闭合。
见 [源码降低](../scar/analysis/program_lowering.py)、[控制结构](../scar/ir/program/structured.py)
与 [恢复协议](../scar/ir/program/resumption.py)。

PyTorch/native 边界可以声明参数、结果、状态和可能效果；这不等于已证明 ABI、alias 或 kernel dispatch。
Python → operator → runtime API → kernel 的每条跨层连接需要对应契约或真实证据，当前尚未完整覆盖。
CPython调用协议已有共享声明子图：实际callable与原显式实参作输入，输入/正常/异常world分别保留。
协议输入和声明完成端口不生成Invocation；条件C ABI描述与真实实现进入、C参数、kernel执行分别表示。
不能将 native 节点无依据展开为物理调用，也不能因缺实现源码就删除已有输入/输出与状态接口。
见 [native 接口](../scar/ir/program/native.py)；目标止于可指导程序改写的 runtime/kernel 边界，不下钻指令。

资源要求与真实资源实例分开：静态 CPU/GPU、主机/设备内存、文件/存储端点、stream/event 约束，
不证明实际设备型号、DRAM/HBM 容量、SSD/NVMe 拓扑、allocation 或数据搬运已经发生。
同设备 `.to()` 可能不复制，`expand()` 可能是零步长 view；host 返回不等于 GPU 完成或结果已可见。
queue/enqueue、执行、完成和跨 stream 可见性不能合并；没有证据不能补造同步。
保存视频与保存 Tensor 都沿 buffer、表示转换、编码/序列化、写入及 finalize 等操作表达，
不需要人类用途分类；只在已支持的协议范围内声明相应读写、生命周期和异常效果。
OS FD/OFD 表、驱动内部结构和硬件指令不是当前要求重建的实体，原生效果边界仍须保留。

可选的 `--resource-scopes` 把实际 native CALL 与 OS 观察连接起来：固定二进制
BEGIN/C_RETURN/C_RAISE 记录引用原 Invocation/CALL；用实际线程、独占标记文件的
pwrite 偏移和线程内 syscall 顺序确定包围关系，不按时间相近配对。
每次执行仍是证据记录；源码侧按原操作、参数与完成端口归并为共享资源接口，
连接输入/输出 world 和实际调用线程，不复制正文或循环。
该关系表示“调用范围内发生”，可能包含回调与采集开销，不证明独占因果、内容 alias、
设备放置或 ready。标记文件未被程序改写是显式条件；无法归属的工作保留为未归属记录。

图的结构覆盖分母取原目录全部 unit 与各 unit 的 sealed template 集合，不取入口遍历到的节点。
每个 unit 保存完整的节点/边投影和绑定/内容状态表；跨层材料继续引用原 ID。
共享局部图已保存、实际执行入口已对应、字段/效果已知是三个分别记录的结论；其中任一个不代替另外两个。
绑定轨迹的存储可共享同函数、同控制预算的拓扑，并以轨迹内位集池引用定义集合。
池编号只表示存储引用，不表示程序值相等；全部控制位置、入/出集合、读取与前沿均可无损还原。

已实现的是上述强类型基础、受支持源码降低、共享层次查询及部分静动连接；尚未完成的是
全 Python/native 协议覆盖、统一逻辑溯源和所有跨层依赖的自动闭合。实际验收范围见 [STATUS](STATUS.md)。
架构组织见 [ARCHITECTURE](ARCHITECTURE.md)，后续只能在这些边界内扩展，不把设计目标当成现有能力。

exec的连接区域保留命名空间选择、条件builtins插入、closure检查、audit、共享根调用及释放阶段；
正常结果是None，根body返回值只进入discard。显式globals且locals缺省时二者共用原引用；
默认优化frame locals是新字典快照。此协议连接只覆盖正常成功路径。
命名空间接口分别保留ROOT_LOCALS、ROOT_GLOBALS、ENV_GLOBALS、BUILTINS、词法绑定及类映射，
不把exec字典登记成module。名字访问保原C2有序lookup、WRITE/DELETE目标及F4 world；
函数globals保实际创建参数与ReferenceEnvironment来源，builtins按frame入口/函数创建时选择。
内容边保原容器成员、key、赋值端口、引用路径及读写world；有条件的may-reaching边不是常量证明。

实际入口的编译context与source-only输入分别持有共享图。exec字典不进入import registry；
caller frame globals须由原frame/capture关系绑定，不能以源码module名代替其身份。
builtins在根frame入口或函数创建world选择；选中的map内容保普通读依赖，标准fallback单独保guard。
类关系查询只在单次查询中共享DAG子问题，依赖变化后重查；不缓存过期的False/UNKNOWN结论。

FrameGlobalsBinding连接原函数factory、选中callee、创建world与frame globals引用；根namespace
由原globals/locals formal输入描述，字典view不是新的allocation，也不是module对象身份。
StaticInputPort以输入contract版本和字段路径标识声明初值，独立于原源码端口；content因子可引用该
外部输入端口。初始字段版本不等于当前内容不变；普通名字访问和字典读写、浅拷贝保持同一内容依赖。

新增依赖源码从F0→F4进入独立provider；F5目录按原source、compiled context、新source的顺序追加。
已有qualified端口和数值offset不变，源码正文及局部接口仍向各自父收据读取。compiled context只由exec边激活，
不成为import候选；同module的不同源码保may候选，不以路径或名字合并身份。源码准入、body激活、实际调用选择和字段值求解是不同事实。

F5的纯引用transport可在已安装的有向copy约束中迭代闭包，包括已准入alias、参数及数据返回，不必逐条线等待普通事件队列。
它只传播已有抽象token，不计算目标操作或合并物理storage；原端口、边和普通事件保留。
闭包经正常add可提前触发原有module激活，工作量另计；当前不宣称全部顺序敏感transfer的固定点等价。

下一输入层分开记录三件事：数据材料节点（scalar/map/seq、原bytes/tag/span和子引用）、
版本化组合操作（defaults/package/override及lazy引用）、与原call actual/formal/world绑定的输入候选。
map/seq不是PythonLiteral；mandatory-missing、null、缺键和未知字段不混用。解析材料不冒充程序执行事件，
组合结果也不能越过deepcopy、删除和callback效果直接冒充当前cfg。文件内容版本、逻辑组合版本与物理对象身份分开。

数据材料的DataNode表示scalar/map/seq定义，DataReference表示一次源码出现并携带独立span；
children/root引用reference ordinal，reference.target引用node ordinal，因此alias共享/循环不展开。
材料层的representation_literal只解释YAML表示；`???`和`${path}`此时仍是字符串。
已安装OmegaConf loader的完整函数AST与工具bytes固定到profile，仅复现resolver和内建scalar解释，不调用自定义构造器。
每材料独立保存source.bin、node/ref/obligation三表和小header；普通读回核全部收据及图不变量。

配置请求由版本化库API的SourceCallLink识别，不按workload名称或参数变量名选择。
参数候选保原formal/actual、literal来源及有向alias路径；factory到decorator及task创建也保独立路径。
CLI值由声明输入端口及明确的option projection推导，初始argv不冒充当前argv。
文件候选优先用实际compile context filename；环境覆盖、搜索插件及输入变更未闭合时继续带条件。

配置组合以有序ConfigurationStep保存根版本链，MERGE_MATERIAL引用共享材料及package，
只排除该文件根defaults；不展开所有字段成巨型字典。ConfigurationDependency保原reference与子实例，
显式package_override和已解析package分开；同文件不同package共享材料，不共享实例。
组override是选择约束，不是在声明位置进行merge；缺失provider以OPEN_PROVIDER及obligation保留。
库隐式根、每文件同名schema、CLI和lazy引用尚未闭合时，局部payload顺序只在列出的条件下成立，
不得把组合版本、已解析cfg或运行时对象身份等同。

初始argv覆盖以独立表达式图及有序写入保存，原输入端口、完整token、字符区间、quote和类型均保留。
复用固定版本的ANTLR生成语法工具，不初始化Hydra、不调用visitor或用户函数；CALL/SWEEP保结构和未求值义务。
覆盖图的版本0显式引用组合图的选中根版本，后续写入各有input/output版本；这些ID只在对应工件内定位。
普通字段候选与组选择保不同条件，missing/插值在此层仍非已解析值；不把初始argv投影冒充当前argv或当前cfg。

输入值层把上述步骤编译为共享代数DAG：EMPTY/INPUT/OPEN_PROVIDER/EMBED/MERGE/ASSIGN/UPDATE/DELETE。
每个根版本只引用已有表达式；字段投影以(root expression, selector path)定位，保原材料/argv来源、依赖及条件。
映射/序列另有members_complete，不因存在几个已知字段便认定整个容器封闭。merge missing保旧值与scalar assign设missing分开。
插值复用固定生成语法；惰性引用使用该查询选中的根和安装后的父路径。动态key先重建路径再拆分，循环保依赖环。
中途地址读取有独立read demand：update可指旧snapshot；merge内部尚未降低的状态点保None，不冒充最终root。
未知provider、schema、当前argv与未支持的别名效果继续阻止无条件结论；本层是输入来源图，尚非任务当前cfg或物理对象身份。

SourceStateProjection把输入内容候选连接到原调用的(receiver reference, result_world)，
同时保存world_before与独立data_result；返回None的修改操作也能产生新内容版本。
接点保原call/attribute坐标及所属scope的alias证据。声明接点不证明调用激活、输入版本匹配或后续任务内容，
不得越过callback、deepcopy、删除及环境效果直接向形参注入字段。

统一source查询的 `effects(point)` / `explain(point)` 返回强类型SourceEffectSummary：原world输入/输出、六个集合字段及RNG/异常/外部/排序字段分别保Knowledge。集合成员引用同一unit的绑定/content访问表；已知成员与集合完整性分开。直接词法读取可明确槽读取及不写槽，不能推出对象/Tensor内容或物理storage不变。未附加状态地址时保具体missing rule；不为查询其他操作而展开无关正文。

F4ResourceApplicability区分NOT_APPLICABLE、COMPOSITE、DECLARATION_ONLY、HOST、HOST_INTERFACE_OPEN、EXTERNAL_INTERFACE_OPEN。组合节点的需求由进入的子区域组成，不声称子区域无资源。原生executor路由保原CALL、named match、normalization、官方schema和dispatch声明；CPU/GPU/内存/SSD/NVMe兼容边始终是未选候选，独立于实际dispatch、allocation、placement或transfer证据。

### 共享时间循环的受约束 lowering

`ArrayLoopPlan` 引用原入口、一个共享 `PipelinePlan`、原参数角色与 F4 来源。原 for 的控制参数是 iterator/world；数组由 lexical binding 读写传递，优化 IR 不伪造原图中不存在的显式数组 phi。抽象递推为 `carry(t+1, i) = body(carry(t, i))`。

只有单数组输入、逐元素算术/标量常量、无 slice/逃逸/开放效果且输出形状保持时，跨元素依赖为空，才可交换 `(t, i)` 执行次序为 `(i, t)`。时间方向仍依序执行每次运算；R0 融合证书、优化后 P 与实际循环合同身份分开保存。次数是观测/声明后逐调用精确守卫的信息，不能从一次运行推出未来相同。

`TEMPORAL_FUSION` 用一个 native 运行循环、一个元素局部 carry、一个新建最终数组，IR 大小为 O(body)。预算为增量数组 payload，排除输入、allocator 元数据、线程 scratch 与 RSS。原 leaf 结果数量和算法字节量不等于真实 malloc 次数、DRAM 流量或 cache 事件。全局 FP 状态 flags 不作为该 profile 的程序输出；后端保存恢复调用者环境，并对非法浮点状态及非有限 carry 回退。

`scar.loop-layout` 在该语义上单独描述空间 lane/chain 布局：有来源的 ISA、固定寄存器预算、每 Expr 活跃区间、first operand 最后使用时的 two-address 复用、行宽/尾部及规范原语成本代理。共享 SSA 值不因复用机器寄存器而合并身份。逐 lane 时间顺序不变，向量链在 DAG 节点之间交错；只选中布局被编译。实际 spill/栈调整来自独立 ELF 审计，不能从抽象寄存器预算反推；模型不预测全程序时间或线性多核收益。

`SourceObservedCallTarget` 是原 F4 CALL 上的独立观察记录，记录实际 callable CodeID 或精确 native carrier/type 及原始收据。图以 `CALL_TARGET/MAY` 和 run/process/thread/attempt 条件引用它，不把相同 CodeID 的多次观察合成一次 invocation，也不凭 native 名称生成实现 ID。caller invocation、callee entry/completion、参数/defaults、内容与 effects 仍各自有开放字段；显式装配进 `BuildSession` 不改变这些证据边界。

部署投影 `scar.loop-runtime-contract` 不取代上述图：构建端完整重放后，把必要的源码/环境/数组/CPU/ABI/selection 合同及 library 身份投影到小包，admission 收据绑定其原始字节。运行端依调用者信任的 admission pin 核对，再执行 live guards，不重新规划或读 proof 档案。函数 builtins 是函数创建时保留的环境，不能用之后的 `globals['__builtins__']` 代替。

`scar.deployment-cost-profile` 区分 COMPONENTS 与 COMPLETE_SCOPE，并完整绑定边界、artifact、CPU、内存与假设。组件和可复用次数用有来源的显式量正向计算；整边界观测不能跨 scope 或外推次数，也不能互减冒充加载成本。成本允许加载候选与执行准入是两种资格，原始 NO_OP 始终有效。

宏观pass复用既有v2 OIR，不建立另一套SSA/CFG。原ProgramOperationID经namespace与FactRef投影成独立OperationDefinitionID，显式表保一一对应；GlobalPort是源码依赖端口，不伪装运行时LogicalVersion。完整区域保输入/结果/world边界与未知效果，native资格仅在原Q和live guards下成立；原Python/NO_OP自身无native guard。`FULL_FUSION`的delta移除原CALL定义、引入组合操作；已闭合直线区域将其实际应用到既有SemanticGraph，根及公共slots逐字保留，删除私有slots并接回边界边，相邻world端口不因顺序相邻就合并。组合节点以source_expansion引用原图，以lowered_body_expansion引用经融合证书核对的实际ArrayRegion执行计划；原ProgramGraph和执行证据不改写。后继图按完整内容生成revision，use-def/源码顺序/静态last-use在新图上重算，分析同时绑定revision与region；静态last-use不证明storage释放或dead work。旧pool、成本、物理寿命、硬件时序及native身份必须重建，尚无通用缓存管理器。成本比较覆盖相同warm受保护调用；allowed CPU mask是成本适用域，继承caller执行不等于绑核或独占预留。

实现规划的`ResourceObligation(field, reason, source_refs)`表示尚未明确的物理资源资格，与`CostEstimate`分开：成本已知不能消除caller执行域或opaque memory缺口，资源未知也不抹掉已有时间事实。非空义务启用problem v3，旧v1/v2字节和digest保留；两种planner与独立verifier均拒绝选择未解模式。仅logical分析可显式忽略物理义务，但不授physical lowering。stateful入口尚无OIR selection时只投影既有源码入口引用，不虚构source delta。

可控资源合同另有明确scope：一次同步native调用的一条caller lane不是独占CPU核或跨调用mutex；私有workspace/fresh output及合同明确计入的borrowed payload按原预算范围各计一次。`plan-controlled-region`保留严格全入口problem，另建可控域problem；全入口CPU/RSS/NUMA等未知记录为`OUTSIDE_CONTROLLED_SCOPE_NOT_DISCHARGED`，并未证明或补零。可控域可行不代表整进程可行、时间准确或允许部署。真实stateful源码投影以evaluate与Expr共用operation，保区域内epoch、slice alias及必要最终状态；其原26动作到11个native stage的对应可一对多，原write的private-write/commit不重复计源操作。原source-order与native stage-order是两种关系，既有合法标量提升不能冒充逐边顺序不变；阶段CPU绑定亦不扩大成whole-entry绑定。

来源闭合且串行的activity成本可按最终subject各加一次，生成同完整方法边界的条件DECLARED模型。COMPONENT_ONLY、转用残差和原表的未物理闭合状态保留；数值未知不补零，复用FactRef的不同occurrence仍分别计费，collective allocation/release只计一次。模型构造可重放与性能预测验收是两层：冻结policy及model后，用独立同scope样本生成qualification；预测误差超阈值时不返回qualified_profile，保留原点模型供假设分析，独立的正确性/收益观测继续保留。两层均不授硬件或部署资格。

`PipelineDemand`分别组合原逐CALL eager、逐leaf native与最终fusion工作；原source occurrence经优化映射保型分类，多个原操作映到同一CSE target仍计多次，缺失target的分类保UNKNOWN，并与原array allocations/copies交叉核对。每个CALL按原value→formal路由实例化shape，循环外层次数不外推。逻辑算术、逻辑读写、累计分配与峰值/实际DRAM区分；静态需求不包含完整guard/FFI等耗时。`pipeline-forward-coefficient-profile`绑定同一context/demand及系数原证据，完整入口残项与已列feature服务的无重叠是显式模型假设；共享affine evaluator用有理数求和，派生秒仍标DECLARED_MODEL并交既有deployment gate。缺项不产生有效总成本，完整边界观测也不相减伪造组件。

canonical完整入口模型以同shape/arity/拓扑的生成程序采样，原Python入口与完整保护入口分别计时；仅TRAIN拟合，冻结后检验混合程序和目标。低维basis需满秩，其余feature归入有效项并以有来源的zero_basis注明非物理零；截距不是独立测得guard成本。拟合保存SVD/condition和holdout误差，失败不发布可用参数。构造式planner复用原PlanningProblem和时间/存储verifier，新增资源序仅是计划约束；差分图处理START/END等式、条件活动和WAIT_TAIL，容量/leases不重复计费，UNKNOWN域保留。compact fused v2仍operator ABI1，不冒充pooled host ABI；原Q保引用，运行只需compact metadata/ELF，成本scope独立于compiler-assisted warm入口。

自动入口另绑定 entry 源版本、逐字 argv、runner 实现及 launch policy。成本资源必须等于实际 native 合同规定的 CPU 与新增输出 payload，不以成本文件自报的较小需求替代。导入候选不是实际 loader 证据；普通目标模块完成加载后才安装源码/环境守卫。已安装边界的调用计数与所有可能别名的覆盖度分开；测得一次 native 不证明整个程序只有一次调用。

`scar.execution-recipe/v1` 是上层执行组织的交接件：semantic_boundary、control、typed task_graph、全部库存与已选 resources、后端 implementation/capability、原 artifacts 和 obligations。估价与选择不写回 recipe，按同一 recipe_id 独立引用；降低重放同份结构并核真实 compiler/ABI/layout/implementation。ID 是版本关联，不代替合法性或执行授权。现有 cpu-temporal-loop-v1 保一份共享正文与每坐标严格 time carry；分配只准备存储，完整计算完成才令最终值就绪。局部端口 label 不是原 ValueID/LogicalVersion/StorageID，原图端口和来源仍以引用保留。全调用成本不擅自拆成 allocation/compute/export 时长，缺项不补 0。

整体优化目标是依赖及资源约束下的完成时间：数据/状态/顺序前驱必须满足，live allocations 受容量约束，执行器与通道受并发约束。不同设备不自动意味独立；物理任务图可因分块、预取或异步搬运增加节点。循环保持结构化控制，不按毫秒或迭代完全展开。算法与计算精度不变，允许设备实现固有浮点差异是当前全局意向；现有严格 CPU profile 保其较强的顺序、fenv、finite 及逐位验证合同，没有擅自放宽或加入量化、近似或删减迭代。

时间直接附在既有 `VersionedResourceSchedule.temporal`，有显式时间时使用 v3；未附加时保留原 v2 字段与哈希。`TemporalEvent` 表示开始、提交返回、完成、数据就绪和 scope 边界；`TemporalActivity` 连接实际工作，`TemporalOccupancy` 表示执行需求或一份物理 slot 的占用区间，`TemporalConstraint` 表示 `after[i+d] - before[i]` 的上下界。父 scope 只是容器，不重复计算子活动成本；函数可递归包含 scope，循环保留一份参数化正文和正迭代距离，不逐次展开。

SEMANTIC / PLANNED / OBSERVED 分开，未知时长不是 0。分配完成不代表数据就绪，提交返回与设备完成没有固定先后；就绪事件绑定实际生产活动。`time-schedule` 从原 CPU 任务图机械生成这些关系，保留不同预算范围，外部存储不虚构分配/释放；导出存储在 scope 退出后继续存活。循环适配先重放原源码证明，并匹配完整 step 内容，正文中的值、slot 均为逐迭代模板，不授权跨迭代复用。当前资源可行性、循环周期和物理耗时均未被证明，旧 task-only solver 对显式 temporal 返回 UNRESOLVED，不能忽略新增约束继续选优。

`bind-time-resources` 以占用 ID 连接原硬件图，不再复制一张执行图：`ExecutionBinding` 区分精确线程与有来源的候选线程，`core_refs` 保留共享物理核关系，不从 SMT 数目推出吞吐或核独占；`StorageBinding` 每份 owned slot 一条，HOST 候选域不等于页驻留。`OriginalBudget` 原样保留预算和统计范围，库存总容量不变成可用预算。选定计算 CPU 集合不证明分配/guard 等宿主活动的线程绑定；源信息不足时域和需求仍为 UNKNOWN。所有引用带原 snapshot，未使用设备与连接仍由完整原硬件图保留。

`cost-activities` 用 `ActivityCostTable` 引用同一时间图和资源投影；每个原活动恰归一个计费 subject。原整池标定对应一组 allocation/release，不能均摊或按 slot 数重复乘；同一 canonical 样本可供不同真实工作 occurrence 估价，occurrence 不能重复计费。现 CPU 组件桥重放原 recipe/task 图、阶段、形状、完整硬件/后端 context 和原样本统计，将 canonical 样本标为条件组件模型，其 min/max 仅为经验范围。整调用观测不拆成 leaf 成本；没有独立活动端点的 guard、输入扫描、team setup/restore 等保留 unmapped，不能伪装成 duration UNKNOWN 的既有节点或塞入 fence。全部 subject 有证据仍不自动证明完整调用闭合，更不授权 solver、C++ 或物理最优性。

选定 pipeline 的 host envelope 复用同一 VRS：caller、worker 和原 ABI 通过事件连接；异步创建的提交返回与 worker 入场可交错，发布释放由 PRE/POST 观测包围，join 等待不是固定 CPU 服务。`ResourceProjection` 只绑定原证据授权的线程和存储候选，逻辑 worker/mutex token 不冒充 CPU。`OBSERVED_SCOPE` 每项引用一个活动的确切端点、原样本及统计证明，表示插桩片段观测；不得直接提升为未来 clean service、数学上下界或完整调用成本。pool 只计一次，符号成员不额外计费，未知外围仍保留。

实现规划 v2 增加条件 `PlanningActivity` 和 `PlanningTimingConstraint`，引用既有 operation；`when` 是所列实现选择的 OR，约束仅在条件及两端存在时生效。`WAIT_TAIL` 无整段资源需求，允许等待伸长；join 的完成满足 `end ≥ max(caller_arrival, worker_exit) + tail_forecast`。完整 task 成本为原 ABI 与不重叠包装成本之和，整段只占逻辑 lane，ABI 内已知物理需求单独保留，外围未知不补零。没有扩展时保持原 v1 wire/digest；预测源绑定全部宿主模板及代码，单配置样本不授跨选择域预测资格。

`refine-time-envelope` 将既有 standalone stateful CPU 执行器的实际边界细化到同一 VRS；`IMPLEMENTATION_PHASE` 保留固定源版本、actor 资格域和明确 effects，允许读原外部输入，不准写 Tensor 或冒充 payload 分配/释放。所有 PRECOMMIT phase 进入原成功 fence，POSTCOMMIT 只描述不可回退的正常状态交接；派生图不可执行。原数据、slot、旧 HB 不变，函数/循环仍使用原 scope。当前实施范围限定 compiler artifact 与共享 stateful guard，不覆盖 compact runtime 或 repeat adapter。新增有限性扫描可接已有 canonical 组件；非连续 guard/setup 或不匹配后缀的组合成本继续 unmapped，不能以 scope 父成本与 leaf 累加。

`--bind-serial-demands` 对同图执行前向 affinity 状态推理：原串行 bind 成功后为 BOUND，闭合的自生成代码保持；遇无保持合同的 opaque/provider 则 UNKNOWN，不能从末尾检查成功倒推历史。仅已证明区间内的活动取得原 HardwareThread 的单位需求，其余不变。该记录是有来源的静态资格，非 CPU 运行观测、独占资源预留或 Tensor 内容版本；任务、数据、时长、事件和依赖不变，重复推导幂等。

`StatefulRegion` 是从原 F4/UnitFacts 提取的受约束数组状态区域，保留 SourceRef、原 step/value/world 来源。
参数及接收者字段路径的 array/scalar kind 是运行时守卫的候选约束，不是静态已知值；未知调用、控制和效果仍阻止该 profile 准入。

slice 是 alias descriptor，在使用处读取数组根的当前 epoch；已 evaluate 的计算结果保定义时 payload，后续写入不能将它改成新状态读取。
call-local storage epoch 只记录本区域有序写入推进，不是全局 LogicalVersion，也不以相同 storage、epoch 或来源证明内容相等。

写入保原 world 顺序；普通接收者字段、同源 bare staticmethod 的 CodeID 和原 NumPy provider 各有独立守卫，power 不改写成乘法。
计算和写入先发生于私有 shadow，全部检查成功后才提交原写入区域；失败从未改动的外部状态调用原方法一次。
此事务边界要求同步 CPU 执行及无外部并发状态读者或写者，仍受原浮点环境、异常和返回 None 的合同约束。

规划保固定 stages、就绪依赖、payload lifetime、最后使用后 slot 复用及显式 CPU resource binding；预分配 workspace 容量保留到方法返回。
预算包含引用的外部状态数组与固定 workspace payload，排除其他程序数组、allocator 元数据、线程 scratch 和 RSS，不把 cache 容量当 malloc 预算。
Q 保原算术依赖与次序、dtype 和独立 provider；当前只在声明的 stage/存储模型内规划，不宣称整程序闭合或 makespan/物理最优。
一份共享方法区域可由原积分程序调用 5000 次，不复制或 unroll 这 5000 次正文；外围调用、边界更新与 I/O 的证明范围仍各自保留。

公共 `ResourceSchedule/v2` 将不可变 LogicalValue、带独立 ready 的 Materialization、物理 StorageSlot/Region 分开。
同 slot 不同版本覆盖前，所有旧版本重叠消费者须 happens-before；ALLOCATION 不产生值就绪。
完整 SUCCESS_FENCE 后才准外部 COMMIT。最后写回若由 full snapshot、全部 ROI 覆盖及 exclusive-state 合同证明完整结果等值，沿用 shadow LogicalValue 的 producer，仅新增外部物化 ready；早期 overlay 不冒充原中间状态。
这里表达选定实现的任务顺序，不能直接将全部调度边当成程序固有依赖。成本、别名覆盖和全程序闭合仍各自需要证据，v1 wire 不变。
