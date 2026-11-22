# SCAR 的分层和交接合同

`scar/cli.py` 负责命令分派。`scar/compiler/pipeline.py` 编排原生主链。各层只从上游规定的产物取数据。跨阶段的文件引用包含 `uri`、`schema`、`size_bytes`、`sha256`。原生运行把这些引用集中写入 `result.json` 的 `artifacts` 字段，原始图、证据和计划分别存放。

## 1. 前端

| 层 | 功能 | 输入 | 交接产物 |
| --- | --- | --- | --- |
| 源码接入 | 锁定源码字节和工作目录，为每个编译单元确定语言、方言和参数 | 入口、源码、编译配置 | 原生 `request.json`，Python 源码单元清单与 F0 字节记录 |
| 词法 | 识别 token，保留源码位置及无法识别的范围 | 锁定字节和方言 | Python F1 token、范围和覆盖记录。原生编译器内部消费 token |
| 语法 | 确定语法结构、嵌套范围和对应源码 | 源码、词法记录和解析配置 | Python F2 AST 和节点位置。原生编译器交出解析后的带位置记录 |
| 符号与类型 | 识别作用域、定义、引用、类型和调用位点 | 语法结构、编译参数和依赖声明 | Python F3 符号、作用域及绑定材料。原生编译单元交出 Clang 类型记录或 Fortran 编译语义记录 |
| 构图 | 将函数、基本块、循环回边和版本化值接入共享 IR | 每单元的类型、绑定和控制流事实 | `program.json` 中的 `ProgramGraph`，原生 `compiler-cfg.json`，以及 `graph-report.json` 中的源码锚点和缺口 |
| 运行证据 | 对照实际调用与资源活动，记录静态分支中仍待判定的字段 | 有界 trace、调用锚点和资源观测 | 可重放的执行证据和绑定记录。Python 建图可消费这组材料 |
| 硬件建模 | 发布平台容量、拓扑和访问关系 | 本机 CPU、NUMA、内存、存储和设备库存 | `hardware/result.json` 及其引用的资源表。实测带宽和时长作为独立成本证据 |

Python 正式层由 `analysis/source_frontend.py`、`compiler/frontend/stages.py` 驱动。原生语言适配器是 `compiler/frontend/clang.py` 和 `fortran.py`。原生阶段接口 `frontend/graph.py::build_graph(manifest, directory, jobs=...)` 返回 `GraphBuild`，其中含程序图、硬件和证据引用以及摘要。入口无法生成 IR 时，程序引用为空。图报告仍保留缺口。跨语言和间接调用的绑定覆盖以证据为准。

## 2. 中端

| 层 | 功能 | 输入 | 交接产物 |
| --- | --- | --- | --- |
| 图准入 | 核对图摘要、版本和 IR 结构，确定可分析的入口 | `GraphBuild.program` 和图证据 | 可读的 `ProgramGraph` 及其分析范围 |
| 结构化简 | 根据依赖、作用域和效果判断合法改写，关联原操作和值 | 程序图和源码锚点 | `simplified.json` 和 `rewrites.json`。当前原生规则支持 SSA 拷贝传播 |
| 资源与成本 | 把可实施选项、成本证据、核组、内存和时序约束表达成规划问题 | 图与硬件引用、成本证据 | `planning-problem.json` 中的 `PlanningProblem` |
| 选型和重放 | 以完工时间为目标选择合法的排程，并校验依赖和资源占用 | `PlanningProblem` | `plan.json` 中的执行计划以及验证结果 |

`compiler/middle/plan.py::simplify_copies` 逐字节核验前端图引用。`plan_entry` 接收入口、图引用、硬件引用和成本。独立的 `scar plan --problem` 重放规划，`--exact` 使用精确求解器。原生整入口的现有计价范围是原执行方案。结构改写的结果与可应用到源码的方案分别记录。Python 分派只在 CLI 处理。

## 3. 后端

| 层 | 功能 | 输入 | 交接产物 |
| --- | --- | --- | --- |
| 交付选型 | 确定采用哪一份源修订和哪个被允许的执行方案 | 锁定请求、图改写和排程审查结果 | 交付模式与采用记录。当前原生路线使用锁定的原始源码交付 |
| C++20 降级 | 将各编译单元映射为 C++20 源码，保留本地头文件和必要的原生库 | 已选源码修订及其编译参数 | `project/src/`、必要的语言转换记载和 CMake 输入 |
| 构建与链接 | 生成和构建独立工程，核对二进制及依赖 | 已生成源码、CMake 输入、库约定 | `project/CMakeLists.txt`、可执行文件、构建日志和 ELF 依赖 |
| 验证 | 对照原版与生成版的退出状态和输出，按声明合同比较 | 两个可执行入口、同一输入和 CPU 配置 | `validation/` 的原始输出，以及 `result.json` 的逐用例正确性和计时 |

`compiler/backend/cpp20.py::build_reference_project` 是现有原生交付接口。它接收锁定清单和新工程目录，返回可执行文件与 ELF 记录。Fortran 降级留在同层的 `fortran.py` 和 `_fortran_freeform.py`。构建共用 `codegen/native_project.py`，运行发布合同由 `runtime/` 负责。排程驱动的源码改写需要新增可重放的降低合同。

## 共享职责

`ir/` 定义图、数据、状态与计划类型。`ir/program/` 中按职责归并的模块包括名字流、调用入参、值内容、状态端口、源码观测和原生 ABI 合同。`artifacts.py` 只负责工件字节和原子发布。图查看仍由 `visualization/` 独立提供。构建证明记录了分析实现的摘要，旧的已封存产物按对应版本回读或重新构建。
