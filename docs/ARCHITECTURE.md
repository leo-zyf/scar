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
| 跨单元绑定 | 将调用点关联到目标函数区域，记录间接调用、外部边界和入口可达性 | 原生编译器调用表、Clang 声明、图锚点和可用的 ELF 符号 | `call-links.json` 中的操作 ID、区域 ID 和证据来源 |
| 运行证据 | 对照实际调用与资源活动，记录静态分支中仍待判定的字段 | 有界 trace、调用锚点和资源观测 | 可重放的执行证据和绑定记录。Python 建图可消费这组材料 |
| 硬件建模 | 发布平台容量、拓扑和访问关系 | 本机 CPU、NUMA、内存、存储和设备库存 | `hardware/result.json` 及其引用的资源表。实测带宽和时长作为独立成本证据 |

Python 三阶段入口位于 `compiler/frontend/python.py`、`compiler/middle/python.py` 和 `compiler/backend/python.py`。前端统一管理源码目录构建和分层 worker。原生语言适配器是 `compiler/frontend/clang.py` 和 `fortran.py`。`frontend/graph.py::build_graph` 接收锁定请求，交出程序图、跨单元连线、硬件和每个编译单元的证据。入口无法生成 IR 时，程序引用为空，流程在首个需要程序图的层给出阻塞原因。跨语言和间接调用的绑定覆盖以证据为准。

## 2. 中端

| 层 | 功能 | 输入 | 交接产物 |
| --- | --- | --- | --- |
| 图准入 | 核对图摘要、版本和 IR 结构，确定可分析的入口 | `GraphBuild.program` 和图证据 | 可读的 `ProgramGraph` 及其分析范围 |
| 结构化简 | 根据依赖、作用域和效果判断合法改写，关联原操作和值 | 程序图和源码锚点 | `simplified.json` 和 `rewrites.json`。当前原生规则支持 SSA 拷贝传播 |
| 资源与成本 | 把可实施选项、成本证据、核组、内存和时序约束表达成规划问题 | 图、改写清单、硬件引用和锁定用例 | `cost-observation.json` 与 `planning-problem.json` 中的 `PlanningProblem` |
| 选型和重放 | 以完工时间为目标选择合法的排程，并校验依赖和资源占用 | `PlanningProblem` | `plan.json` 中的执行计划以及验证结果 |

`compiler/middle/plan.py::simplify_copies` 核验前端图引用并验证改写后的 IR。`entry_problem` 固化图、改写清单、硬件和成本记录的引用。独立的 `scar plan --problem` 重放规划，`--exact` 使用精确求解器。原生整入口的现有计价范围是一次基准启动。SSA 改写目前仅作用于 IR。中端提交的可降级实施方式是 `preserve_entry`。

## 3. 后端

| 层 | 功能 | 输入 | 交接产物 |
| --- | --- | --- | --- |
| 交付选型 | 重读并校验引用，重组规划问题，独立核验所选排程及 CPU 可用性 | 锁定请求、图改写、成本和排程 | `project/delivery.json`，含已采用的排程、CPU、源请求和可执行文件摘要 |
| C++20 降级 | 按编译单元生成 C++20 源码及其编译参数，保留本地头文件和必要的原生库 | 已选源码修订及其按单元记录的编译参数 | `project/src/`、必要的语言转换记载和 CMake 输入 |
| 构建与链接 | 生成和构建独立工程，核对二进制及依赖 | 已生成源码、CMake 输入、库约定 | `project/CMakeLists.txt`、可执行文件、构建日志和 ELF 依赖 |
| 验证 | 对照原版与生成版的退出状态和输出，按声明合同比较 | 两个可执行入口、同一输入和 CPU 配置 | `validation/` 的原始输出，以及 `result.json` 的逐用例正确性和计时 |

`compiler/backend/cpp20.py::admit_entry` 重放交接合同后，`build_reference_project` 才降级、构建和检查 ELF。验收逐用例核对校准时锁定的基准程序和输入，在交付记录指定的 CPU 上比较结果。`result.json` 的 `layers` 记录各层的执行状态，失败时返回具体阻塞层。Fortran 降级由同层的 `fortran.py` 和 `_fortran_freeform.py` 负责。构建共用 `codegen/native_project.py`，运行发布合同由 `runtime/` 负责。混合目标的独立 C++20 构建、完整现代 Fortran 降级和排程驱动的源码改写仍待闭合。

## 共享职责

`ir/` 定义图、数据、状态与计划类型。调用出口、编译器符号、源码字节、执行观测等类型和紧邻的查询按职责归并。`artifacts.py` 负责工件字节、摘要和原子发布。图查看由 `visualization/` 独立提供。构建证明记录了分析实现的摘要，旧的已封存产物按对应版本回读或重新构建。历史源码保存在 Git。
