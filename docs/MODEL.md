# 图与执行 IR

公开产物使用 `schema_version=1`。图格式为 `scar.program-graph`，逻辑计划为 `scar.logical-plan`，执行 IR 为 `scar.execution-ir`。后端依次交接 `scar.scheduled-ir`、`scar.cpp20-target-ir`、`scar.cpp20-project` 和 `scar.cpp20-executable`；`scar.physical-plan` 保存调度成本与决策依据。图根文件按编译单元引用子图，同一源文件在不同宏、方言和参数下属于不同编译配置。

构建子图通过根文件的 `build.graph` 收据独立存储。`BUILD_OPERATION` 是捕获的工具调用，`BUILD_ARTIFACT` 是具有路径及可选采集摘要的源文件、对象或可执行产物；`BUILD_INPUT/BUILD_OUTPUT` 边连接消费者和生产者，序号保留链接输入顺序。`LIBRARY_REQUEST` 只说明命令请求，`ELF_LIBRARY_DEPENDENCY` 只说明采集时 ELF 的直接依赖，均不等于运行时加载或已执行计算。工具和库事实也进入资源表。构建调用与应用运行操作属于不同图域，捕获到编译器执行不能替代应用程序执行证据。

文件依赖通过独立的 `artifact_flow` 收据连接子图，格式为 `scar.artifact-flow`。`FILE_LOCATOR` 表示路径定位表达式，不能视作 inode 身份；`FILE_VERSION` 是某次捕获的字节快照；`FILE_VERSION_REQUIREMENT` 表示工具需要读取、但历史字节尚未观测的版本。`BYTE_VALUE` 用 `bytes:sha256:<digest>` 表示与路径、文件对象无关的字节内容。`SOURCE_OPERATION` 引用原过程图与端口，`SOURCE_CAPTURE` 表示分析器读取，`BUILD_OPERATION`、`EXECUTION_IMAGE` 接到对应子图。`READS_CAPTURED_VERSION/HAS_CONTENT/REQUIRES_VERSION/BUILT_IMAGE_SOURCE` 分别连接这些实体。捕获时相同路径不证明历史版本相同，`CAPTURED_AT_SAME_LOCATOR` 明确禁止升级为版本相等。

F2 的 `artifact_effect` 保留接口、操作数字段路径、默认值和未闭合义务；路径在 F3 图中沿端口边下降，而不是重新遍历 AST。流句柄的静态站点代表每次成功调用的实例模板，别名与条件候选单独保存，不把循环中的多次打开当作同一动态对象。M2 的 `FILE_VERSION_TEMPLATE` 进一步连接已正规化的写入／关闭操作与已证明的字节值；值相同可以跨文件副本连接，但文件内容还受普通字节文件、独占写入和并发效果条件约束。对特殊设备成功写入不等于产生同样的可读文件内容。

未闭合协议的 `closed_fragments` 和 `prefix_fragments` 是分析产物，按内容收据放入 `analysis/`。前缀保留控制端口及尚未完成语句的求值边界，不能把分支体提升为无条件事件。M4 不把这些片段封为可执行函数。常量文件分析也分别报告值证明、物理存储义务和未实施变换，不以内容哈希匹配替代消费者版本证明。

M2 沿已闭合调用的形参、局部赋值和返回值传播标量事实；每个静态调用端口只有一个可放宽的输入元组，递归或循环不会产生无限上下文。事实须符合目的类型的精确表示范围，静态／线程局部初始化不代表每次调用的当前值。条件调用摘要带形参匹配及调用可达条件，不能当成实际调用记录。

C 字节缓冲使用 `BUFFER_LOCATOR/BUFFER_VERSION/BYTE_EXPRESSION` 表达区域、写入版本和格式化／复制依赖。分支产生合流，循环保留头部与回边。`COMMAND_REQUEST` 引用所读缓冲版本，分别保存已知字节、格式前缀和未闭合字段；已知 `mkdir ` 前缀不能证明目录参数或 shell 效果。F2 的接口合同来自编译器捕获的真实库声明，M2 不凭函数拼写重建绑定。命令请求不是执行授权，也不自动成为可删除操作。

程序镜像是一个独立链接与全局状态的命名空间。显式多镜像输入用 `images` 描述各镜像的 `id`、`entry`、交付文件名 `output` 和编译配置，`root_image` 指定交付入口。F0 先应用各镜像自己的 ABI，再去重编译配置；图中 `images[].units` 是共享单元表的索引。同一编译配置可供多个镜像使用，但不同镜像中的同名 `main`、全局变量和记录不会合并。该归属由输入构建清单声明，目前不从子进程命令字符串猜测。

多镜像执行包使用 `kind: "image_bundle"`，每个子镜像都有独立的完整包及其相对收据。中端、调度层和目标降级层分别用 `ir`、`scheduled`、`target` 引用下一层可独立消费的镜像产物；顶层仅在全部必需镜像闭合后为 `READY`。一个镜像自己的函数收据始终相对于该镜像包，不能相对于总包解释。既有单镜像格式仍表示同一种执行模型中的一个链接目标。

打印层将多镜像交付为 `project/images/<id>/{src,include,CMakeLists.txt}`，顶层 CMake 组织独立目标，构建层输出 `project/bin/<output>` 和按镜像 ID 索引的 `executables` 收据。编译镜像集合不表示进程间依赖已经闭合：启动的实际目标、每次启动的进程实例、文件版本、IPC 及跨进程资源竞争仍须分别建模。当前物理计划在各镜像内独立选择，不宣称全工作流最优调度。

每个过程由 `signature`、`nodes`、`entry`、`states`、`accesses`、`state_flow` 和调用索引组成。`nodes` 是类型化操作记录表，`{ "port": 4 }` 引用本过程的第 4 号操作结果或语句。表达式依赖通过这些端口组成图；语句的有序区域表示控制关系，分支带独立区域。一个 `for` 操作拥有初始化、条件、更新和循环体端口，循环次数保持参数化。`state_flow` 用 `LOOP_CARRY` 和合流记录描述可变状态经过循环和分支后的版本关系。含未结构化跳转的过程保留原控制操作，并明确标记状态分析的保守边界。

F4 的版本是保守 may-def epoch，不是数据内容身份或动态迭代的 LogicalVersion。`may_write` 推进未知新 epoch，不能证明写入必然发生。未闭合的上下文、异常区域保留局部 `OPAQUE` 摘要及可能出口，不清空整个函数证据；迭代器显式区分一次 setup、符号化 advance 和成功后目标绑定。`coverage` 列出未展开区域、开放协议和未结构化控制。共享、闭包、静态或逃逸状态的读取同时依赖自身 epoch 与共享 memory epoch；F4 新引入的分析状态通过 `introduced_states` 交接，封存前加入状态表，避免悬空域引用。真正的 goto 尚只提供函数级摘要。源图证据进入 M1 后仍须经来源映射或 canonical 再分析，不能直接作为 M2 新建操作的完整优化证明。

`source_defined` 表示源码中有过程体；`defined` 表示当前操作体已经满足数值/本机执行资格。图的 `definitions` 统计前者，`qualified_definitions` 统计后者。动态 Python 过程可以完整保留其结构，同时尚不具备本机执行资格，两项不能混为同一个覆盖率。

中端绑定产物的 `activation_graph` 用 `roots` 和 `{owner,target,site,role}` 边保存激活依据。`site` 是源过程图 ID 与端口的组合，空站点表示不能依靠局部控制证明删除的硬边。M2 记录未取站点的守卫、规则和判定值，再比较裁边前后的根可达性，不能按函数名或一次运行未见调用而省略函数体。`and/or` 和比较链保留分支区域，不展开为无条件求值；逻辑源类型独立于最终存储类型，例如 `numpy_bool` 和 `bool` 不可仅因都存成 C++ 布尔值而互换。

库调用可在 M2 展开为复合区域，生成语句的 `origin_region` 指向源调用图站点；该字段是来源信息，不参与数值等价比较。`array_iota` 的合同是新建连续一维 binary64 坐标 `0…count-1`，计数为 i64 且在精确索引域内。区域内仍显式保留算术、控制和存储写入，不能将 `linspace` 等公开接口仅凭名称映射成一个语义不同的同名目标函数。计算、绘图、文件及进程等领域共享相同数据／状态／资源图，但各自具有能力、效果和成本合同。

例如两个语言前端都可以产生以下操作结构，在图存储时各表达式再被编码为端口。

```json
{"op":"call","callee":"bssn2adm_","args":[{"op":"unary","operator":"addr","value":{"op":"var","name":"n"}}]}
```

符号拼写和参数 ABI 来自选定编译环境。C/C++ 的链接名和 Fortran 的调用规则在前端写成统一签名。中端通过符号连接调用，并检查参数数目、传值/引用、类型和结果。库边界通过 `provider`、函数签名和版本化 ABI 显式记录。

对象、视图和存储具有不同作用。当前静态图中的可变 `states` 表示局部、参数、共享内存和全局状态域，`accesses` 表示读、写、取地址或未知效果。数组视图保留基址和索引表达式。`load_bytes` 和 `store_bytes` 表达同一字节存储上的类型化访问。它们支持旧 Fortran 的 COMMON/EQUIVALENCE，在目标端使用 memcpy 避免不合法的 C++ 别名访问。该静态状态版本分析仍不能作为任意动态 tensor 内容相等的证明。

主机接口在执行 IR 中表示为 `host_call`，包含版本化的 `interface`、已求值实参、结果类型、`effects`、`reads`、`writes` 和 `may_raise`。例如 `fs.copy2` 读取文件系统和当前目录，写入文件系统；`process.chdir` 写入当前目录；`time.wall` 读取主机时钟。`path.join` 仅操作文本，不读取当前目录。`{"kind":"handle","resource":"text_stream"}` 表示共享流身份，句柄复制不复制文件或流状态。`stream.open` 显式携带模式、UTF-8 和 `universal` 换行合同，不能把源语言的 `newline=''` 与默认换行混为一谈。当前仅闭合 POSIX 的受支持接口、默认缓冲和严格 UTF-8 文件模式，不包含任意流协议或进程启动。`console.input` 另携带已核验的标准流错误策略；支持严格 UTF-8 及输入输出一致的 surrogateescape。

源图的 `python_import` 分别保留根库 `source_provider`、实际请求模块的 `import_resolution` 和可选的 `member_resolution`。后者显式带有 `package_attribute_missing` 条件；子模块文件存在不等于包属性必须绑定为该模块。来源记录包含模块名、发行包名／版本、实际 origin、文件或解释器摘要及证明范围，查找或归属未闭合时保留具体状态。来源事实与执行合同分离，符合 [Python 导入的查找、初始化及名称绑定过程](https://docs.python.org/3/reference/import.html)；发行包名与导入名也分别存储，不能按同名推断归属（[官方元数据接口](https://docs.python.org/3/library/importlib.metadata.html)）。

异常值使用 `{"kind":"handle","resource":"error"}`，`error_make` 保存类别和完整显示文本，`throw` 与 `rethrow` 表示异常出口。`try` 有受保护体、按序匹配的处理器、正常落下才执行的 `else` 及必经的 `finally`；处理器的 `binding` 引用真实词法槽。`resource_scope` 只封装已证明的进入／退出协议，多个资源依次嵌套。异常发生前任意前缀都可能已执行，不能把受保护体内的赋值误认为处理器入口的确定绑定。`noreturn` 表示无正常返回，仍可能由外层捕获。退出异常分别保存进程退出码、捕获时的显示文本和未捕获时的可选 stderr 文本，避免将几种观察结果混为一谈。

`{"kind":"object_ref","record":id}` 是共享实例身份，区别于按值传递的 `record`。受支持布局显式声明 `fixed_fields/shared/unbound`，每个字段有类型及缺失读取的错误数据。`object_allocate` 创建身份；`object_get` 每次读取当前字段并检查绑定；`object_set` 按 `value → base` 次序写入同一身份。三者分别带 `MEM/STATE` 或 `STATE` 及 `may_raise`，不能按纯计算缓存。静态类定义和候选方法关系不等于动态调用；未激活候选只在对象查找与逃逸协议闭合后才能从执行包排除。

`text_format` 和 `text_print` 携带标量格式；二进制浮点用 `shortest_binary64`，明确最短回读数、负零、非有限值及定点／科学计数边界。打印操作先按源顺序捕获值与选项，再产生输出，不能依赖 C++ 函数实参的求值顺序。

来源类型不等于目标表示。M2 可在编译期合流同表示标量的可能协议，并在消费方证明共有行为；共有合同直接产生普通 `select`、比较或格式化 IR。可能协议不成为运行时 union/tag，选择也不隐式提升整数或降低精度。只在行为差异被实际观察时精化合同，未证明的差异仍是明确缺口。

`format_value` 的输入是已求值的类型化值、显式 `none/str/repr/ascii` 转换和已解析规格：填充、对齐、符号、负零规范化、进制前缀、补零、宽度、分组、精度及表示类型。规格同时保留空规格和显式字段的区别，例如布尔的空规格与显式数字规格。文本表示引用 `text_profiles` 中的源 Unicode 版本及不可打印区间，表的 ID 为内容摘要，多个作用域和操作共享同表。求值次序属于源版本合同；后端不再解析源格式串、推测 Python 版本或使用本机字符分类替代该表。

物理 `f64` 不等于源操作域。M2 区分 Python float 和 NumPy binary64 标量，并通过调用签名及结果身份传递；二者可以使用同一种物理表示，但幂、除法和告警合同不同。`scalar_power` 规定实数域及异常；可能产生复数而未证明其结果域时不能直接降级。`array_binary` 明确操作、尾轴广播、NumPy 默认数值策略及告警位置；普通结果分配独立存储，`array_update` 写入原区域并保留所有活跃视图，重叠读写先快照。原生库执行前后隔离并恢复浮点环境，按来源版本封存幂分派，不能仅凭两个 double 猜测原操作。

`array_format` 携带固定默认输出选项，`text_concat` 使用有序扁平文本成员，避免为已捕获的序列构造深度线性的表达式。当前数组元素为 binary64，默认告警与展示合同可用；自定义 NumPy 错误处理／warning hooks、其他 dtype、复杂值和数组构造式 repr 仍有明确边界。

图形接口使用 `graphics_context/graphics_plot/graphics_grid/graphics_save`。它们持有共享上下文句柄，显式读写当前图形、坐标轴、曲线和画布状态，并保留初始化、默认配置和导入选择条件。`graphics_plot.data_capture=snapshot_at_call` 区分曲线数据版本与后续被改写的输入数组；返回 `graphics_lines` 集合句柄，丢弃返回值不能删除坐标轴状态写入。`plot/grid` 可以创建 Figure 与 Axes，单独保存只要求 Figure，成功后仍有重绘效果。所有操作带状态／IO／内存效果及可能异常，不能当成纯数值运算。

`operator_requirements` 声明逻辑接口所需的实现，`operator_family` 将实际操作连到该要求。语义合同与目标实现资格分别成立；未找到等价原生实现必须保留拒绝，不允许用 Python 回调、空函数或省略输出替代。

MPI 仍使用普通 `call/indirect_call`，由已绑定 provider 补充 `mpi_interface(s)` 和 `mpi_contract`。合同区分 communicator、request、阻塞／非阻塞／持久通信，数据区域遵从 datatype typemap 与 count，不简化成连续 `count*sizeof`。释放 request 不代表通信完成，非阻塞 Barrier 也只在其完成时提供同步。`operator_requirements.mpi` 使用 C API、版本 1 和实际接口集合；SDK 能力、源／部署头证据及链接库身份在后端分别核验。未证明 rank 分配和实际 MPI 线程支持前，B1 只限制 SCAR 自行新增的 rank 内计算线程为一，不把这个安全上限解释为最优值。

变量的拼写不代表状态身份。全局引用带 `scope: "global"`，F3 分配 `global/<链接名>`；形式参数、局部声明和闭包单元具有各自的域。引用声明保留绑定自身的域，用 `alias_to` 指向被引用状态。引用初始化写绑定，之后通过引用的赋值写目标。隐式引用参数、构造和虚调用也记录借用，无法闭合的指针访问进入共享内存域；未知调用推进相应版本。

操作声明的外部读写进入 `external/<状态名>` 并登记为 `external_state`，例如 cwd 的版本不能只出现在访问表。封存前机械校验操作端口、状态域以及按域区分的 epoch 定义／引用；这种引用闭合不等于全部效果或优化证明已闭合。

M1 保留 `state_evidence`，包含图身份、状态表、访问和版本关系。M2 生成每个函数的 `state_contract`，列出帧内标量、初始化状态、可见逃逸及证据来源。M3/M4 必须使用这份合同；同名、相同类型或多次访问本身均不能证明值不变。重复声明、引用借用、静态/volatile 状态等不进入私有标量集合。M3 新增变量时同步更新合同；公共循环域的 `value_definitions` 保留定义表达式及被支配的使用位置，使 M4 能恢复多维因子，而不必重新分析来源或丢失数据关系。

对象记录包含字段、目标布局、基类和特殊/虚方法协议。`object_construct`、`object_new`、`object_dispatch`、`object_destroy` 明确表达构造、分配、虚调用和清理。构造函数初始化表达式也进入同一操作表。可展开函数与对象方法通过符号连接，实际类实例化要求保留相应具体虚方法及析构定义。

`process_launch` 的 `result_handle.site` 是静态分配站点，每次成功动态调用产生新实例。`process_wait/poll/communicate` 的 `handle` 引用原目标的端口投影，`sites` 是可能来源，不代表这些进程都已发生。`process_stream` 保留通道、方向及句柄关系；绑定方法取创建时的接收者快照，不能在调用处重读后来被改绑的变量。全部操作保留调用目标和守卫，尚未绑定到子程序镜像的部分仍为开放边界。

资源使用独立类型化表，含 CPU、物理核、逻辑 CPU、NUMA 内存、缓存容量、存储和可见加速设备。拓扑边表示资源归属及可达性。时间成本记录在后端物理计划中，包含测量、估计、线程范围与选择理由。硬件能力的观测和软件操作的合法性分别保留来源。

`CPU_POOL` 表达同一进程可使用的 CPU 集合及共享时间配额，通过 `CPU_POOL_MEMBER` 指向具体逻辑 CPU。`capacity_upper_bound_cores` 保留按 quota/period 推导的持续核时间上界，不是瞬时并发上限或计算吞吐；原始 quota 和 period 同时保留。资源 `capacity` 的单位明确为保守 worker 预算。cgroup 路径只作为限额证据，不展开为待优化的操作节点。未知约束保留 `issues`，工具并发回退为一；无争用和最佳并行度均不能由核数直接推出。

中端解码闭包中的图端口，形成语言无关的执行 IR。IR 拥有函数、类型、记录、全局状态、库接口以及结构化操作。每个函数用 SHA-256 收据封存在 `functions/`。计划审计包含改写及未解决问题；状态为 `INCOMPLETE` 时执行 IR 不能交给后端构建。后端独立读取执行 IR 包及其成员。

源协议可先进入绑定产品，供 M2 完成资格判定。`execution_kind: "python_protocol"` 表示结构已知但尚未形成执行合同，不能出现在 `READY` 执行包中。模块版本和名称绑定不等于不可变常量；可变标量保持为全局状态，函数每次读取当前值。`text` 是拥有长度的 UTF-8 文本，`unit` 是空值。`text_print` 带有显式值、分隔符、结尾、刷新、关键字求值顺序和标量拼写合同；副作用实参先按源顺序捕获，再发生输出。因此内嵌 NUL、空值和布尔拼写不依赖目标 C++ 的默认格式。

M4 描述合法遍历和守卫，B1 再选择物理实现。例如多维切片可以提供 `first_axis_fastest` 坐标遍历，附带稳定整数基数、形状相等守卫和线性回退。B1 对每种遍历分别计数、测量相关原语并估价；整数商余和浮点除法不能使用同一成本。

B2a 将选定执行方式降低为 `Type/Expr/Stmt/Decl/Attribute/Unit` 组成的目标树。例如整数相加产生 `Expr(kind="binary", operator="+", args=(...))`，不是一段含加号的原始程序字符串。B2b 只按目标树语法打印，随后进行 LLVM 格式化。未知目标节点或字段直接拒绝。固定运行库和测量探针是独立的 C++ 文件，不包含原应用代码。

包内收据含相对路径、字节数和 SHA-256，根文件原子发布。读取时检查路径范围和摘要。源码文件路径用于输入定位和追溯，后续阶段消费的函数操作已完全包含在产物中。已有源片段格式的旧产物需要重新建图。

图的表示能力、字段的可知程度、闭包的可编译程度分别报告。当前 Python 动态对象与部分库接口仍存在未闭合操作；完整图结构不等于所有字段已被证明，也不等于任意程序已可生成 C++20。不会把缺失的效果当作无副作用。

`ndarray` 类型分别声明元素类型和维数，轴长度作为值参与计算。`array_create/literal/copy/unary` 明确产生新存储，`array_view` 保留原存储所有权，普通句柄赋值保持共享。`array_load/store` 携带整数索引；写入合同指定尾轴广播，重叠源先形成快照。序列捕获不能延迟读取后来被重绑的变量，整数操作必须有溢出界证明，浮点除数必须证明非零。当前 CPU 实现支持 binary64 和整数前缀索引，其余类型及索引协议保留明确缺口。

`array_size` 读取当前区域的元素数；`array_reshape` 以显式 C-order 形状重解释同一连续区域，保留 storage 与 offset；`array_broadcast_copy` 按尾轴广播到新的独立区域，直接写入新分配，不先清零。静态字面量事实使用 allocation token 加 `(offset, shape)` 描述连续区域，reshape 和前缀 view 共享失效 token；不同形状不能再借用原索引前缀推定值。

`process_spawn` 引用已声明的 `image_id`，参数是有序文本值，标准流路由及环境继承均显式表达。结果是共享 `process` 句柄，一次成功启动产生一个新实例；调用位置不等于进程身份。`process.stdout` 投影共享流，`process.wait` 观察并缓存终态，带 ORDER 效果。图中声明镜像与部署产物的关系由 B1 对外层镜像包验证；原始脚本、旧可执行文件和未经绑定的 shell 命令均不作为后端补缺手段。
