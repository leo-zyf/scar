# SCAR

State · Control · Action · Resource：面向Python程序的宏观编译器。

目标流程：**统一程序/资源图 → 结构与硬件感知化简 → C++20宿主执行代码**。
算子实现通过版本化接口接入官方高性能库或外部算子生成agent，支持单算子、通信及融合区域。
当前并行开发建图、图化简与规划、C++20代码交付；静态优先，软件边界到算子接口。
已具备共享控制/数据/状态图、跨函数与容器传播及生成式算子接口接线；仍需补完整资源映射和外部接口。
完整程序建模尚未完成；已贯通受约束的 CPU 数组区域编译，没有LeWM加速比结论。
主测试组为 Sunfish、Gray–Scott、Cirq。真实 Gray–Scott 512×512 / 5000 步：进程内完整积分 **3.71×**；相同 C++ 库的独立进程自动入口 **5.22×**（3 对中位数），正式成本门另作一次完整验收。不同测量范围分开报告。旧单 stencil 版本 0.895× 仍拒绝，另两项只有基线，详见 [当前状态](docs/STATUS.md)。
LeWM 是外部 testcase；通用规则不按业务名称选择处理方式。
后端基线为C++20、算子ABI v1、执行计划schema v1（资源绑定用v2），配置见`pyproject.toml`。

## 使用

有序数组状态区域：从原 F4 自动发现字段读、求值、切片写入，再规划 barrier/临时存储并生成 C++20；当前需要显式形状、CPU 和数组 payload 预算。源程序不修改，产物尚未测量时不自动启用。

```bash
scar compile-stateful --source /path/to/program.py --catalog /path/to/catalog --shape 515 515 --target /path/to/target-hardware --cpu-threads 4 --memory-budget-bytes 33554432 --out /path/to/compiled-state
```

已保留的原 F4 区域可用 `--region REGION.json` 代替 `--source/--catalog`，避免重复目录加载。运行边界为 `scar.compiler.stateful_runtime.install_stateful_regions`；成本选择与合法性分开，未测产物仅可显式 trial。原 NumPy provider 精确绑定版本/二进制，无跨版本 ABI 保证。

共享执行图与原入口交付：

```bash
scar recipe plan-stateful --region REGION.json --plan STAGE_PLAN.json --out /path/to/recipe
scar recipe estimate --recipe /path/to/recipe/recipe.json --out /path/to/estimate.json
scar recipe lower --recipe /path/to/recipe/recipe.json --out /path/to/lowered
scar export-stateful-runtime --artifact /path/to/lowered/native --selection SELECTION.json --out /path/to/bundle
scar run --bundle /path/to/bundle --admission-sha256 PRODUCER_PIN --exclusive-inputs --memory-budget-bytes 33554432 --reuse-count 5000 --deployment-plan COST_PLAN.json --report /path/to/run.json -- /path/to/entry.py [arguments]
```

缺少成本计划时保留原执行；`--measure-candidate` 显式测候选，不表示部署盈利。公共资源图 v2 保状态版本、独立 storage slot、就绪与提交屏障；缺任务成本继续 UNKNOWN。轻量包仍需要原源码、Python/NumPy 和系统库，尚非独立 C++ 主程序。

三阶段区域闭环（CPU、精确 float64 二维 ndarray；其他输入回到原执行）：

```bash
scar compile-region /path/to/program.py --catalog /path/to/catalog --cpu-threads 4 --out /path/to/compiled
scar benchmark-region /path/to/compiled/region-0/native --out /path/to/compiled/benchmark.json
```

有catalog时直接降低原F4操作和端口；F0编译仅恢复CodeID，不重新用AST选择表达式语义。
硬件规划按模型推导；动态只补静态不能确定的输入字段，实测用于原语成本校准及最终验收：

```bash
scar compile-region /path/to/program.py --catalog /path/to/catalog --target /path/to/hardware --cpu-threads 4 --memory-budget-bytes 8388608 --out /path/to/preview
scar observe-region-inputs --plan /path/to/preview/region-0/plan.json --out /path/to/inputs --max-seconds 60 --stop-when-collected -- python /path/to/entry.py
scar calibrate-cpu --plan /path/to/preview/region-0/plan.json --inputs /path/to/inputs/inputs.json --out /path/to/calibration
scar plan-resources /path/to/preview/region-0/resource-plan.json --plan /path/to/preview/region-0/plan.json --inputs /path/to/inputs/inputs.json --profile /path/to/calibration/profile.json --out /path/to/planning.json
scar lower-resources --plan /path/to/preview/region-0/plan.json --resources /path/to/preview/region-0/resource-plan.json --planning /path/to/planning.json --out /path/to/native
scar benchmark-region /path/to/native --inputs /path/to/inputs/inputs.json --out /path/to/benchmark.json
```

这些命令的输出路径须尚不存在。预览计划须声明增量内存预算，不能将硬件总内存当可用预算。
输入采集要求 CPython 的 `sys.monitoring`，只对完整 CodeID 匹配的普通函数启用局部进入事件；
最多记录两次 shape/dtype/strides 等元数据，不记录数组内容。`--stop-when-collected` 会中断本次启动的子程序，
所得结果只覆盖执行前缀；省略该参数则让原程序继续。采集不加载原生库、不应用变换，也不是性能测量。
原语校准可用 `cpu-cost` 可选依赖（SciPy），不运行或排名目标程序的多个实现版本。
成本 profile 限定硬件快照、CPU 集合、编译器及完整参数、runtime 指纹、形状与布局；
当前是有效仿射模型，不能推导物理最优、实际 DRAM 流量或 cache miss。

固定线程参数当前只生成未排名的可行分配预览；未指定时，硬件分配缺代价模型则报告`PLANNING_REQUIRED`，不编译多版本比时。
`plan-resources --model COSTS.json`接显式任务成本；资源容量、CPU需求和切片输出大小从原计划推导，未知成本或预算不补零。
求解器当前只覆盖固定放置、非抢占任务和增量内存预算；模型最优不等于物理最优，联合放置/搬运/融合仍未完成。
资源任务图区分分配、值就绪和导出；C++已消费同步CPU三任务的 timed 计划，逐调用守卫 shape/layout/字节预算/CPU 绑定。
它按依赖执行，不等待预测 tick；未实现的 copy/wait 或 NUMA 页放置继续拒绝。

未指定线程数的硬件预览会输出未绑定的 `plan.json`，可自动构造 CPU 实现域并正向选择：

```bash
scar choose-cpu --plan /path/to/unbound/plan.json --target /path/to/hardware --inputs /path/to/inputs/inputs.json --cpu-pool 0 1 2 3 4 5 6 7 --max-threads 8 --calibrate-missing --out /path/to/cpu-choice
scar lower-cpu-choices --choices /path/to/cpu-choice/choice/choice.json --out /path/to/chosen-native
```

可重复传 `--profile FILE` 复用已有成本。缺失且未要求校准的实现保留为前沿，不能删除后宣称完整域最优。
校准只运行规范原语；模型选择完成后仅编译所选目标实现，最终是否采用仍由正确性和 A/B 验收决定。
当前域是给定 CPU 池中每 executor 的前缀 CPU 集合及 1/2 的幂/上限线程数，不是所有物理实现；NUMA 页未绑定。

原 F4 中只调用已准入数组函数的直线子图，可自动组成多区域 pipeline：

```bash
scar compile-pipeline --source /path/to/program.py --catalog /path/to/catalog --out /path/to/discovery
scar observe-region-inputs --plan /path/to/discovery/pipeline-0/pipeline.json --out /path/to/inputs -- python /path/to/entry.py
scar compile-pipeline --pipeline /path/to/discovery/pipeline-0/pipeline.json --inputs /path/to/inputs/inputs.json --target /path/to/hardware --cpu-threads 8 --memory-budget-bytes 12582912 --out /path/to/pipeline-build
scar benchmark-pipeline /path/to/pipeline-build/pipeline-0/native --inputs /path/to/inputs/inputs.json --out /path/to/pipeline-benchmark.json
```

默认 `--strategy auto` 在满足契约时将完整 elementwise 子图融合为单 kernel，以减少材料化；不满足时保原调用与 pool。
`--strategy pool` 显式保留原调用划分。预算先在选定表示上判定，原 pool 不可行不代表合法改图后也不可行。
原调用和 callee guards 始终保留，融合证书与后续算子图优化映射分开验证，不重关联浮点运算。
C++ host/pool 路径按已验证的 last-use 复用槽位，融合路径不分配中间 workspace，最终输出都独立新建；
输入与实际 global callee 由运行边界守卫。`scar.compiler.pipeline_runtime.install_pipeline` 可在测量通过后安装，
传入完整 benchmark 报告作为 `selection`；`trial=True` 仅用于受控验收。
线程数目前是显式未排名配置，表示选择的目标是增量 payload，整体时间尚未建模；不声称最优调度、NUMA 放置或实际传输。

共享循环主链已接入：原 F4 的 `range(count)`、局部 carry 读写与 world 回边组成一个可展开循环节点，正文只存一份。当前 native profile 限单个形状保持的数组 carry、纯 elementwise pipeline。

```bash
scar compile-loop --source /path/to/program.py --catalog /path/to/catalog --out /path/to/loop-preview
scar observe-region-inputs --plan /path/to/loop-preview/loop-0/loop.json --out /path/to/loop-inputs --stop-when-collected -- python /path/to/entry.py
scar compile-loop --loop /path/to/loop-preview/loop-0/loop.json --inputs /path/to/loop-inputs/inputs.json --target /path/to/hardware --cpu-threads 8 --memory-budget-bytes 12582912 --out /path/to/loop-build
```

可对已绑定的 body plan 校准独立 SSE2 原语，再由硬件模型选择向量链布局。校准不执行目标程序或其变体：

```bash
scar compile-loop --loop /path/to/loop-preview/loop-0/loop.json --inputs /path/to/loop-inputs/inputs.json --target /path/to/hardware --cpu-threads 8 --memory-budget-bytes 12582912 --plan-only --out /path/to/bound-plan
scar calibrate-loop-primitives --plan /path/to/bound-plan/loop-0/execution-plan.json --out /path/to/primitive-profile
scar compile-loop --loop /path/to/loop-preview/loop-0/loop.json --inputs /path/to/loop-inputs/inputs.json --target /path/to/hardware --cpu-threads 8 --memory-budget-bytes 12582912 --simd-profile /path/to/primitive-profile/profile.json --out /path/to/simd-loop-build
```

布局按原 DAG 的 last-use、SSE2 two-address 复用、寄存器预算和有来源的依赖/吞吐代理选链数；只编译选中布局。v2 artifact 附 layout 和实际 ELF 的 ISA/栈审计，loop ABI 仍为 1，旧无 layout 的 v1 可读。代理成本不是纯指令延迟或整程序耗时预测，抽象向量存储预算不是实际线程栈上限。

同一份上层 execution recipe 现在可贯通语义边界、任务/资源图、估价和 C++ 降低：

```bash
scar recipe plan-loop --loop LOOP.json --strategy STRATEGY.json --plan BOUND_PLAN.json --layout LAYOUT.json --cpu-hierarchy CPU_HIERARCHY.json --out /path/to/recipe
scar recipe estimate --recipe /path/to/recipe/recipe.json --out /path/to/estimate.json
scar recipe lower --recipe /path/to/recipe/recipe.json --out /path/to/lowered
```

输入来自现有静态循环发现及 plan-only 流程；layout/cache sidecar 可省略。recipe 引用原组件，不复制循环正文或展开迭代，完整保留未选资源。语义边界不随 CPU 数改变；recipe ID 还绑定资源、后端实现、编译器和布局。估计与 lowering 收据引用同一 ID。缺真实 temporal 成本时输出 UNKNOWN，编译仅取得待测产物；不是自动部署资格，也不表示全程序时序已求解。降低产物在 `lowered/native/`，继续使用下述导出及运行入口。

部署前一次性准入原 proof，再导出小运行包；完整 IR、profile、硬件档案和模板不进入运行进程：

```bash
scar export-loop-runtime --artifact /path/to/simd-loop-build/loop-0/native --selection /path/to/measured-result.json --out /path/to/runtime-bundle
```

```bash
scar inspect-entry --entry /path/to/application.py --contract /path/to/runtime-bundle/runtime.json --out /path/to/entry-map.json
scar run --bundle /path/to/runtime-bundle --admission-sha256 TRUSTED_SHA256 --exclusive-inputs --memory-budget-bytes BYTES --deployment-plan /path/to/cost-plan.json --report /path/to/execution.json -- /path/to/application.py [arguments]
```

`inspect-entry` 静态解析有界本地导入候选；`run` 在普通目标模块自然导入完成后安装守卫，无需 testcase 手写导入/替换，也不改应用源码。不支持的 loader、入口自身定义的目标函数等保留原路径。`--measure-candidate` 可替代 `--deployment-plan` 测量已准入候选，但不代表成本已批准。无成本计划时默认 NO_OP。

pin 必须来自调用者信任的构建记录，loader 不从包内 report 自行取信；哈希不等于签名。运行包保 library/ABI、原应用源码与嵌套 globals、实际函数 builtins、shape/count/owner、CPU/ISA/fenv 等守卫。仍依赖 Python/NumPy、原应用源码和系统共享库，尚非独立 C++ 程序。手动嵌入仍可用轻量 `scar.runtime.native_loop.load` 和 `scar.runtime.loop_guard.install_loop`。

部署成本可先在元数据上门控：`scar plan-deployment --profile PROFILE --context CONTEXT --reuse-count N --cpu-ids 0 1 --memory-budget-bytes BYTES --out REPORT`。完整 scope 观测只在其原 scope/count 回放；组件模型必须分别有来源的加载、安装及调用成本。缺失值保持 NO_OP，不从 warm/fresh median 差值推导 setup。正向结果仅 `LOAD_NATIVE_CANDIDATE`，不能替代准入和动态守卫。

外部调用可按原 F4 code/offset 轻量补位，再自动挂到统一图：

```bash
scar observe-call-bindings --source /path/to/entry.py --entry main --catalog /path/to/catalog --out /path/to/call-targets -- python /path/to/entry.py
scar build --entry /path/to/entry.py --catalog /path/to/catalog --target /path/to/hardware --call-bindings /path/to/call-targets --out /path/to/unified
```

每个位点默认只留一次实际 callable 元数据，局部 CALL 事件不采集 operands 或数组数值。`BuildSession.view` 通过 `CALL_TARGET` 条件 MAY 边查询独立观察句柄；它不是 InvocationKey，不能证明 callee 进入、完成、effects 或未来分派。

循环仅观察一次入口，保存数组元数据、精确整数次数及真实嵌套 callable CodeID，不保存数组数值。规划先证明各元素互不依赖，再把时间递推放入元素局部的 C++20 循环；不 unroll，也不按次数复制 IR。只读初值、写独立最终数组，无中间数组 workspace。这是增量 payload 下界，不是时间或物理最优证明；CPU 数量仍显式指定。

原入口、正文及各自 globals 中的叶函数均需匹配；次数变化、零次/负次数、布局或浮点守卫失配时执行原循环一次。正常启用需要该 artifact 的正确性与测量收据。

未测量的编译产物默认不选用。Python API `install_regions(module, artifacts, exclusive_inputs=True, selection=report)`
仅在已测计划及输入 shape/layout 范围内使用原生代码；`trial=True` 只用于受控 A/B。
契约要求同步 CPU 执行及无外部并发写入；这些不是由 ndarray 类型推断的全局事实。
前端自动扫描函数，不按业务名称选区域；图保留源码版本、CodeID 与原 F2 occurrence 引用。
当前支持切片、常量和基础算术，拒绝未知状态/调用；C++20 拥有区域循环，Python保留其他程序行为。

正式前端按层构建局部图，不运行目标程序：

```bash
scar frontend /path/to/source-root --out /path/to/frontend --jobs 8
scar build --entry /path/to/entry.py --catalog /path/to/frontend/catalog --cache-root /path/to/frontend --out /path/to/build
```

`frontend` 顺序执行源码→词法→AST→作用域/绑定→共享区域模板→局部接口→调用参数/返回接口；三类独立模板可并行。
每层检查原报告再推进，失败保留具体阶段。`local_declarations_complete` 只表示所选源码的局部声明图完成。
已有采集可追加 `--trace TRACE --project-root ORIGINAL_ROOT` 离线接线；随后 `build` 使用同一目录的观察接口。
已有F0–F3模型可用 `frontend ROOT --source-model MODEL --source FILE --out FRESH` 从F4继续；原source ID和分片保持，当前源字节先核对，F4仍执行收集与独立验证。
同一分析索引同时发布共享的类型/原生调用边界和独立观察表；有native companion时保原ABI字段，无companion时明确未记录。
边界可展开到实际Python回调的共享代码/body；普通入口与可挂起body分开，恢复位置和参数来源没有证据时保留未决字段。
编译器插入的生成器创建、类构造CALL单独接回原协议操作及其参数/状态/结果，不套普通函数ABI；共享正文不代表已证明实际激活或generator恢复实例。
可选 `scar trace --resumptions --project-root ROOT --out TRACE -- python app.py` 记录真实首次进入、恢复、yield 和最终完成，以同一激活连接原 InvocationID，不保存帧或 payload。
`scar source-resumptions --interfaces OBSERVED_FUNCTION_INTERFACES --metadata TRACE/resumption-meta.PID.json --out FRESH` 将边界接回原正文端口；委托驱动的继续执行不等于已完成规范化恢复，接线表尚待纳入统一 build。
动态生成的代码可用 `scar trace --compile-events --project-root ROOT --out TRACE -- python app.py` 保留限额内编译文本与执行代码指纹。
带此采集的 `frontend --trace TRACE --project-root ROOT` 会自动匹配兼容编译配方、生成局部F0–F4图并接回调用；只编译，不执行保存的文本。
同文件名的不同代码用完整CodeID区分，编译选项多解保留；原始编译请求、创建环境与参数值不由代码相等推出。
有trace时还会自动寻找外部文件及官方frozen源码，以完整CodeID验证后并行降低局部图；不导入目标模块。
最终catalog路径在frontend报告中；用 `scar build --entry ENTRY --trace TRACE --cache-root FRONTEND --out BUILD` 自动选取该目录及接口。

统一入口优先组合与入口源码匹配的 F4 catalog、调用/状态接口和资源材料；没有材料时建立源码、函数/循环子图和跨函数连接：

```bash
scar build --entry /path/to/entry.py --trace /path/to/trace --out /path/to/fresh-build --jobs 4
scar build --entry /path/to/entry.py --trace /path/to/trace --body-scope observed --target /path/to/target --out /path/to/fresh-build --jobs 8
scar build-query /path/to/fresh-build --action expand
scar build-query /path/to/fresh-build --action select --file model.py
scar build-query /path/to/fresh-build --action inputs
scar build-query /path/to/fresh-build --action resources
scar build-query /path/to/fresh-build --action dependencies --file model.py --entry forward --node program_value:ORIGINAL_ID
```

已提取的状态区域可按硬件产生 CPU 域，再将可信成本选择交给同一 C++20 后端：

```bash
scar choose-stateful-cpu --region REGION.json --target HARDWARE --shape 515 515 --memory-budget-bytes 33554432 --out PLAN
scar choose-stateful-cpu --domain PLAN/domain/domain.json --cost-profiles PROFILES.json --no-op-profile ORIGINAL.json --out CHOICE
scar lower-stateful-choices --choices CHOICE/choice/choice.json --out NATIVE
```

省略 `--target` 时自动发现平台硬件。成本不足保留原执行，完整声明模型只降低选中的一个实现；编译不授权部署，也不证明物理全局最优。已有区域正文仅按 artifact 引用共享，不为每个 CPU mask 复制。

可直接从已准入的统一程序图发现区域并规划，复用同一会话的原 F4 正文和全部硬件：

```bash
scar plan-build --build PROGRAM_BUILD --shape 515 515 --memory-budget-bytes 33554432 --out DOMAINS
scar choose-stateful-cpu --domain DOMAINS/region-0/domain/domain.json --out CHOICE
```

`plan-build` 当前为受支持的状态数组区域生成规划域，其余函数保留具体拒绝原因。输入 shape 是显式约束；局部绑定的 frame-isolation 投影保留未解除的 Q，引用稳定不能证明数组内容不变。

循环外提已有原 F4 → 依赖闭包 → C++20 验证入口：

```bash
scar frontend tests/workloads --source tests/workloads/licm_program.py --jobs 4 --out FRONTEND
scar compile-invariant-loop --catalog FRONTEND/catalog --source tests/workloads/licm_program.py --memory-budget-bytes 33554432 --out LICM
```

当前支持普通函数的固定 `range` 循环：循环体将一个纯数组函数的结果赋回 carry，其他数组实参只读。carry 由原绑定写入推导，外提节点由输入依赖闭包推导；输出保留原调用、回边和参数映射证据。生成 C++ 沿用算子 ABI v1；验证包装器另检查原 callable/range 和输入条件。非正次数保原输入别名，失败不暴露私有中间结果。该入口只编译供验证，尚不自动部署，也不覆盖一般自适应循环。

`scar compile-build --domains DOMAINS --calibration RESOURCE_OBSERVATIONS --out PIPELINE` 可复用已有区域/硬件域，串联依赖查询、全部 CPU 配置的前向成本和已选方案降低。`--calibration` 可重复传入；缺失或被拒绝的成本仍阻止选择。

统一三阶段入口现在还支持原图自动发现的纯数组循环。shape/count 是输入约束，carry 与外提节点由图推导：

```bash
scar build --entry ENTRY.py --catalog FRONTEND/catalog --out PROGRAM
scar plan-build --build PROGRAM --shape 256 256 --memory-budget-bytes 33554432 --out DOMAINS
scar compile-build --domains DOMAINS --loop-trip-count 400 --out COMPILED
scar validate-build --pipeline COMPILED --out VALIDATION -- ENTRY.py --size 256 --steps 400
scar run-build --pipeline COMPILED --deployment VALIDATION/deployment.json --exclusive-inputs -- ENTRY.py --size 256 --steps 400
```

资源计划比较重算、物化和逐元素时间融合；当前后端支持最后一种单线程实现，保留全部硬件图。运行包复用同一份 ELF，正常执行不加载编译器/图。成本门严格绑定已测入口与实现，不匹配时回原程序。兼容入口和 I/O 仍由 Python 执行；这不是整个程序已转为纯 C++。

无 trace 时可省略 `--trace`。`--catalog` 选择 F4 局部声明或 context catalog，`--cache-root` 指定接口材料目录；
`--model` 显式选择旧 source model，`--source-root` 增加依赖源码根。
新源码构建的 `--body-scope observed` 根据完整运行 CodeID 选择正文，并自动准入 trace 涉及的外部源码文件；保留其余源码声明，不扫描整个依赖安装目录、不证明未观察路径为死代码。
该模式默认直接连接运行中的实际调用，跳过全局静态目标求解；`--call-binding static` 可显式恢复静态求解。
已有 source model 可用 `--model MODEL --call-binding observed` 接新 trace。`build` 默认自动发现当前平台并保存到 `OUT/hardware`；`--target` 可复用明确指定的独立硬件快照。硬件发现不按 workload、CPU affinity 或 CUDA 可见性删设备；库存、访问限制、后端支持和实际资源使用分别保存。

硬件图包含 CPU package/core/SMT/ISA、NUMA DRAM、共享缓存、PCIe 端点/互连、加速器、物理磁盘及文件系统 backing、网卡/RDMA。控制与监测接口在完整 PCI 台账中保留并聚合为边界；发现显示设备不推断可执行 CUDA 或独立显存。可用预算与带宽成本仍需独立证据。单独采集：`python -m scar.analysis.target_graph --out FRESH --path /path/to/input`。
`manifest.json` 引用原正文和关系分片；资源按原操作/区域延迟接线，观察依据独立保存。
入口源码树及显式依赖根的 YAML 自动进入现有配置材料解析器；不执行 tag、插值或 Hydra 组合。
`gaps.json` 保留局部缺口、跨层前沿、未接运行资源及查询截断。
`workflow_complete` 只表示构建/存储/读回流程通过，`first_stage_complete` 当前始终为 false。
候选 GPU、文件和 SSD/NVMe 关系带条件；它们不是实际 dispatch、存储身份或物理复制证据。

Python 中用 `BuildSession(build_path)` 共享一次查询会话的 catalog/代码/关系缓存。
`session.view` 提供原 `get/expand/producers/consumers/explain`。
`source_region_ref` 可先连接原声明的函数正文；`get/expand` 再读取和核验正文，循环与函数仍共享原 ID。
F4 构建的观察选择保留原 trace 与调用接口的 run scope，`attach_trace()` 返回已接入材料摘要；
静态调用 frame 不是动态 invocation。旧 model 构建继续通过 `observed_scopes/observed_ports` 查询独立观察调用域。
F4 自动按需建立实际调用 scope：原共享端口加 run/InvocationKey，同一 caller 内重复 CALL 再加 attempt 锚点。
参数、数据返回、环境和 world 状态关系引用原图，不复制正文；结构共享不授予内容版本相等。
实际调用 provider 的 `scoped(scope, original)`、`body(scope)` 和 `shared(point)` 支持 scope 内正文查询与原点审计。
内部条件和循环回边保留，不因一次进入证据推导实际分支或迭代内容；caller attempt 始终单独表示。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lewm
cd ~/AAA/scar
python -B -m scar.cli static /path/to/program --out /path/to/model --phase bodies --jobs 4
```

静态入口不导入或运行目标。按需 CALL 采集和离线连接用法见开发说明。
已有函数/循环展开、跨分片依赖查询、有限属性推理；存在节点不代表依赖或效果已闭合。

已有 F4 图与 facts6 后，先按单元并行投影接口，再跨单元接线：

```bash
python -B -m scar.analysis.source_interface_build --facts /path/to/facts --out /path/to/interfaces --jobs 48
python -B -m scar.analysis.normal_targets --facts /path/to/facts --interfaces /path/to/interfaces --entry /path/to/entry.py --out /path/to/targets
```

摘要沿用原函数/循环/状态端口；不导入目标，也不按调用或迭代展开正文。

已有局部声明或 context catalog 后，可独立分析整个源树的调用声明，无需等待入口求解队列收敛：

```bash
python -B -m scar.cli call-declarations --catalog CONTEXT_CATALOG --source-root /path/to/program --out FRESH_CALLS --jobs 4
python -B -m scar.cli native-bindings --calls FRESH_CALLS --bindings GENERATED_BINDING_DECLARATIONS --out FRESH_BINDINGS --jobs 4
python -B -m scar.cli native-interfaces --named FRESH_BINDINGS --out FRESH_NATIVE --jobs 4
python -B -m scar.cli resource-graph --native FRESH_NATIVE --target TARGET_HARDWARE --out FRESH_RESOURCES
python -B -m scar.cli name-flow --declared FRESH_CALLS --unit SOURCE_UNIT --out FRESH_NAME_FLOW --jobs 4
```

每个源码单元独立存储。原调用、scope、参数和循环不复制；完整符号匹配保留所有候选overload。
声明图不代表scope已执行；参数类型、当前namespace、parser接受及实际dispatch仍需各自依据。
`name-flow`在共享控制流上计算词法槽位的may到达定义；新正常路径视图与原声明分开，未知效果保留。
资源图共享已有硬件库存，连接原生值的类型与源码依赖，不把设备候选当成实际placement。

## 文档

- [模型](docs/MODEL.md)：实体、身份、边、可展开结构与正确性约束。
- [流程](docs/ARCHITECTURE.md)：形式前端、建图流水线及动静分工。
- [开发用法](docs/DEVELOPMENT.md)：命令、查询接口和工作区约定。
- [当前状态](docs/STATUS.md)：真实结果、缺口与下一步。

当前CPU形式前端及建图证据在 `artifacts/current/experiments/cpu-*-current/formal/`，
精简验收摘要见 `artifacts/baseline/formal-frontend-results.json`；已有源码/查询材料仍共享原分片。
大产物不提交 Git，只保留一份正在开发的源码。

当前 CPU 主测试程序位于项目外、与 `le-wm` 同级：`sunfish`（一般程序）、`gray-scott`（PDE）、`cirq`（量子电路模拟）。
外部入口及命令、10 秒至 10 分钟的实测负载见 `artifacts/baseline/cpu-workloads.json`；测试程序的运行成功与完整建图验收分别报告。
该清单的 `current_run` 指向本轮结果；原始记录在 `artifacts/current/experiments/real-cpu-suite-v1/`。
