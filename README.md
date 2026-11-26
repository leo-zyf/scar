# SCAR

SCAR 将多语言程序和平台资源建立为统一图，证明逻辑改写，再生成独立的 C++20 工程。当前开发平台为 Linux CPU。

```bash
python -m pip install -e '.[cpu,test]'
scar build --manifest request.json --out work/graph --jobs 4
scar plan --graph work/graph/graph.json --out work/plan
scar compile --ir work/plan/exec-ir.json --out work/delivery
./work/delivery/project/bin/scar_entry
```

C/C++ 分析需要 libclang 18，Fortran 分析需要 gfortran。生成工程使用 clang-format、CMake 3.20 以上和 C++20 编译器。并行区域需要 OpenMP，MPI 目标需要对应的开发库和启动器。GNU 格式化、无格式 I/O 使用声明的 libgfortran.so.5 接口。安装配置中的 Python extras 只安装 Python 包。

MPI C 接口沿同一三阶段处理。前端记录编译器实际读取的外部头摘要，中端给已绑定的 MPI 调用补充通信和生命周期合同，后端用独立探针核对目标 SDK 的类型、函数 ABI、头及直接链接库。生成工程随附 `mpi-sdk.json`，CMake 再核对实际选择的 SDK。探针不调用 `MPI_Init`。源程序已有的通信调用、参数和顺序保留；当前验证覆盖本机 Open MPI 的双 rank 非阻塞收发与归约。自动 MPI 分区、最优 rank 选择和通信时序成本仍未实现，原生 Fortran MPI 接口也未统一到该 C 接口合同。

已有 Make 工程可以先用 Bear 3 捕获一次真实构建：`bear intercept --force-preload --output /absolute/events.json -- make -j4 target`。这会实际执行原工程的构建；应使用可重建的独立构建目录。将 `request.json` 写为 `{"source_root":"/absolute/project","entry":"main","targets":["target"],"build_trace":{"provider":"bear-3-intercept","path":"/absolute/events.json"}}`，随后依次运行上述三个阶段。前端从构建记录恢复编译配置、对象依赖、链接顺序和库需求，另外封存构建子图。增量构建未记录的对象及未闭合依赖会明确报告；采集前应确保所需目标的完整编译被记录。后两阶段无需 Bear、Make、原始构建记录或原工程。

构建记录也可与脚本控制器组成一个请求：保留 `build_trace`，另加 `root_image` 和含 `source` 的 `images`。构建产物到子镜像的来源连接随图封存，不能据此省略有日志或文件副作用的运行时构建。固定 rank 的单机 MPI 控制器可使用直接 argv 或无扩展的简单 shell 命令；后端核验实际启动器与子程序 MPI SDK，并要求原相对路径指向同一发布代的子程序。该通路不选择最优 rank 或自动划分计算。

需要将 Python 启动点与实际构建进程关联时，可使用可选的进程采集入口；它会实际运行给定脚本。CPython 3.12+ 仅监测 `Popen` 的启动与构造返回，不逐行／逐函数跟踪；源码快照与字节码验证在退出封存时完成。Bear 3 的包装 PID 通过独立进程系统调用记录核对，调试日志不进入应用管道：

```bash
strace --seccomp-bpf -f -qq -ttt -s 4096 \
  -e trace=clone,clone3,fork,vfork,execve,execveat -o process-tree.log -- \
  bear intercept --force-preload --output events.json -- \
  python -m scar.frontend.languages.python.process_capture \
    --source-root /absolute/project --out processes.json /absolute/project/driver.py
```

在原 `build_trace` 内增加 `"process_observation":{"path":"processes.json"}` 和 `"process_tree":{"provider":"strace-process-v1","path":"process-tree.log"}`。F0 锁定三份记录，F1 解码、F2 转交事实，F5 只用封存事实连接源码端口与实际进程；后续阶段不再调用采集工具。默认最多保存 64 条 Python 启动意图；无返回 PID、证据不足或采集被截断时保留未决，不按相似命令猜测。旧版 Python 可记录意图，但缺少局部返回监测时不证明进程身份。采集运行与最终性能测量分开。

若只需确定一个模块级 `from ... import ...` 在一次脚本运行中实际绑定了什么，可让 CPython 3.12+ 只监测选中初始化器。以下第二个命令在已有图包旁发布派生索引，复用所有编译单元，输出可直接传给 `scar plan --graph` 的路径。观测关联入口和模块源码摘要、解释器版本、字节码跨度及有序绑定端口。一次观测只描述这一次运行。单程序图省略 `--image`。

```bash
python -m scar.frontend.languages.python.import_observation \
  --source /absolute/package/__init__.py --line 60 --out bindings.json /absolute/driver.py
python -m scar.frontend.observation --graph work/graph/graph.json \
  --image driver --import-observation bindings.json
```

图包另通过 `artifact_flow` 引用文件与构建依赖图。它区分与路径无关的字节值、捕获的文件快照、动态文件版本模板和未观测的历史读取。中端可从已资格化的操作推导有序常量文本，遇到后续协议缺口仍保留先前分析片段；这些片段只供分析，不会作为未闭合程序的可执行函数。字节相同不自动授权删除生成、复制或构建操作。

已知进程协议可表示为目标未绑定的 `process_request`，继续分析管道、等待和错误控制。逻辑 IR 的 `READY` 仅表示协议已闭合；后端仍须核对执行目标和原生算子，不能据此执行任意命令或回退原 Python 程序。

CPU 数组算子固定使用 xtensor 0.27.1/xtl 0.8.0。执行 `python tools/provision_cpu_sdk.py` 可安装到用户目录；首次配置这些上游头文件需要 CMake 3.29 以上，可用 `CMAKE_COMMAND` 指定独立工具路径。已安装 SDK 的消费工程仍可使用系统 CMake 3.28。`SCAR_CPU_SDK` 可指定安装前缀。供应工具校验源归档及安装收据，不修改系统环境。NumPy 只作为受版本约束的源接口和验证基准，中端、后端及生成程序均不运行 NumPy。

一维 complex128 的受支持 FFT 合同另使用固定修订的 PocketFFT；`python tools/provision_fft_sdk.py` 安装用户级 SDK，`SCAR_FOURIER_SDK` 指定前缀。目标包封存实际头文件和许可证，后续生成工程不依赖安装时的 SDK 路径；这不代表完整 NumPy FFT 接口已受支持。

执行 IR 含任意精度整数运算时，CPU 后端会核验 GMP 6 的头文件、静态库与 LP64 接口，并把库及许可证封存到目标包。可使用系统的 `libgmp-dev` 或通过 `SCAR_INTEGER_SDK` 指定安装目录。已证明有界的机器整数运算继续使用原生类型。

三阶段分别在独立进程运行。`build` 读取源码、每个编译单元的语言和参数，输出 `graph.json` 及内容寻址的单元子图。`plan` 读取完整图包，输出审计报告 `plan.json` 和自包含 `exec-ir.json` 执行 IR 包。`compile` 读取执行 IR 包，选择目标调度，输出 `physical-plan.json`、CMake 工程、C++20 源码、头文件和可执行文件。搬动中间包后仍可独立消费。各阶段的代码只导入本阶段和公共格式层。

当前语言适配器位于 `frontend/languages/{c_family,fortran,python}`。每种语言都先捕获事实，再正规化为同一操作词汇，随后共用构图和状态分析。中端分别做绑定、规范化、等价化简和执行 IR 封存。后端经物理计划、类型化 C++ 目标树、独立打印器和构建层交付。

Python 的受支持文件、目录、文本流、结构化循环及异常／清理可经同一 IR 交付原生 C++20；默认编码和标准库来源在前端捕获。尚未闭合的自定义异常与迭代器、异步上下文及子进程执行合同会阻止执行包就绪，不会回退执行原 Python。

真实 CXSparse 的所选构建已通过 C++20 输出对照。AMSS 使用官方 `GW150914_5M` 输入和 CPU 构建配置。上游从原始 Python 入口运行至绘图结束，退出码 0，带有界导入采集约 1754 秒，产出 535 份 PDF。同一次 `scar compile` 会在 `physical/components/project/` 交付 ABE 和 TwoPunctureABE 两个 C++20 目标。此处假定输出目录为 `physical`。完整驱动仍返回退出码 2，它的首个未闭合操作记录在 `physical/result.json`。

在 Xeon E5-2680 v4 上，ABE 固定 16 个 MPI rank，每个 rank 单线程。相同输入、绑核和 CPU 环境下，上游与交付的生成版各预热一次，再交错计时三对。8 次运行均比对 156 个二进制场文件和 7 个文本输出，文本只规范化创建日期。

| ABE 单步 | 中位数 | 最小至最大 |
| --- | ---: | ---: |
| 上游原版 | 51.385 s | 51.250–51.544 s |
| 交付的 C++20 | 45.307 s | 45.263–45.695 s |

单步中位加速比 **1.134×**。独立的 13 步演化各运行一次，上游 617.17 s、生成版 562.77 s。此工况的 156 个场文件均在初始时刻写出。两张演化诊断表共 468 个数值有差异，最大绝对差 4.16e-12，最大对称相对差 3.37e-5，未设验收容差。TwoPuncture 在 16×16×8 网格预热一对、交错计时四对，上游中位数 5.375 s，生成版 5.356 s（1.004×）。

5M 图含 880 个编译单元、2,026,039 条操作和 243 项资源。M2 已沿主入口分析至文件末尾。第 436、441、445 行各保留一个循环区域，求值和取得迭代器只发生一次，目标写入和循环体属于取到元素的分支，正常耗尽后进入退出分支。三个循环体分别含 2、1、1 条语句。静态分析没有给它们指定迭代次数。产物见 `evidence/handoff/source-loop-flow.json`。

M2 在保留的 32 个调用操作中，将 7 条条件调用边连到 7 个函数体。给定图中指定的代码和活动帧，这 7 个函数都能沿结构化控制流到达返回。M4 封存的 513 个语句和控制区域含 28 个条件、13 个循环。B1 只读执行 IR 即可核对这些边。每条边的调用者、参数和返回目标在 `evidence/handoff/function-call-flow.json` 中。

用 5M 原始输出分别运行 6 个原 Python 绘图函数，生成的 13 份 PDF 除创建日期外与官方输出逐字节相同。已执行的分支和嵌套循环能按代码摘要和指令偏移对应到静态图。两颗黑洞的输入只覆盖部分分支。可从本机 `artifacts/current/verification.json` 的科学对照收据查看输入和输出。

在真实源码的导入节选中，第 37 行首次加载配置，两个绘图模块各自第 17 行都命中同一缓存对象。独立冷启动只导入绘图模块时，最先到达的站点进入配置代码一次。图为每个站点保存条件边，相同配置代码共用一份模块体。该节选跳过了求解部分。

M3 在 TwoPuncture 的 85 个外部数学调用点追溯操作数。热点中的 `Pion` 由入口参数 `n` 决定，`j` 随外循环变化，`k` 随内循环变化。大网格前三次 BiCGSTAB 迭代中两个调用点分别执行 9750 万和 5850 万次 `cos`。跨迭代复用还需证明浮点环境和库调用的可观察效果；本轮交付的两个二进制与之前实测版本的 SHA-256 均相同。对真实 C17 输入，仅共享已证明精确、私有的整数实参计算，保留原来的数学调用，7 组输入输出通过对照。证据见 `evidence/handoff/m3-math.json` 和 `evidence/solvers/public-run.json`。

真实角度的独立探针显示，2,190 种输入中有 2,189 种调用 `cos` 会置位 `FE_INEXACT`。打开该陷阱时，重新计算和读取缓存有不同的控制流。在固定舍入、关闭陷阱的探针里，反复使用系数表可获益；若每次读取并清除浮点标志，恢复标志的版本反而慢约 1.8%。原始计时、库绑定和位级对照见 `evidence/handoff/libm-domain.json`。

M1 对直接可达且唯一的原生定义在首次校验单元时生成函数绑定。正式 TwoPuncture 从 4 次解码减到 2 次，完整绑定结果逐字节相同。晚发现的目标仍校验相应单元。M4 把已封存的图成员交给计划报告，在同一 5M 图上少序列化 76,649,709 字节，两次生成的 292 个计划文件逐字节相同。

B1 用同一份可执行调用索引做 MPI、外部接口和链接资格检查。来源证据可以引用已删除的调用，这些引用只用于解释来源。真实 ABE 的规划耗时中位数从 8.441 秒降到 7.944 秒，两版各独立校准三次，未预热。固定校准数据后 ABE 的 187 个产物、TwoPuncture 的 58 个产物分别逐字节一致。

在 TwoPuncture 上检查了一个减少临时数组分配的实验。正式 50×50×26 网格中，从第一个 BiCGSTAB 残差到第四个残差，发布版和实验版各测两次，中位耗时分别为 12.760 和 14.244 秒。独立计数显示一个热函数中的 12,416,792 次数组分配全部消失。这个变换未启用。运行次数、区间、两份 ELF 和计数证据见 `evidence/handoff/two-puncture-storage.json`。

B2 将 C++ 目标记录的必填字段规则预计算一次。在真实 ABE 的 210,034 个目标节点上，解码、校验与打印的中位耗时由 2.555 秒降至 2.143 秒，两版各测四次。包含 LLVM 格式化的完整阶段各测一次，40 个文件及清单逐字节相同。两个求解器的 ELF 摘要均与前述应用基准一致。见 `evidence/handoff/cpp20-target-performance.json`。

另用交付的 TwoPuncture 可执行文件复测 16×16×8 网格，绑定逻辑 CPU 26，单线程，预热一对并交错计时四对。原版中位 5.398 秒（5.370–5.460），生成版 5.381 秒（5.307–5.401），比值 1.0032×。10 次运行的两个科学输出、stdout 和 stderr 对照一致，`Ansorg.psid` 归一化了首行日期。另一次测量的耗时分布在 5–11 秒，单独保留在 `solver-recheck.json`。见 `evidence/handoff/performance-payload/solver-clean.json`。

在 50×50×26 网格的前三次 BiCGSTAB 迭代上，另测了保留每次写回、按需重载目标元素的实现。该实现减少 273,206,400 次加载，增加 494,426,400 次有效性检查。同核单线程预热一对、交错十对，现行生成版与实验版的中位数分别为 11.996 s 和 12.278 s，配对中位比 0.977×。该变换已拒绝。见 `evidence/handoff/slot-reload-performance.json`。

GCC 编译反馈使用 20×20×12 训练网格。独立的 16×16×8 网格绑定单核，预热一对并交错计时十二对。生产版和实验版的中位耗时分别为 5.328 s、5.135 s，配对中位加速比 1.03863×。训练和额外编译花费 36.023 s，按每次配对中位节省的时间计算，需要约 182 次同规模求解回本。

官方 50×50×26 网格独立验证了同一反馈构建。单核、无预热，两对反向交错的完整求解分别为生产版 860.281、867.074 s，反馈版 851.709、858.304 s。配对中位加速比为 1.01014×，每次节省 8.671 s。计入 36.023 s 额外构建和训练成本，单次求解总耗时增加 27.352 s；若实测收益复现，第 5 次求解回本。两版的完整场和收敛输出均与上游一致，场文件比较时仅归一化创建时间。一轮同核有竞争的生产版计时已单列。两对样本仅说明这个网格和平台的实测结果。见 `evidence/handoff/pgo-official-50/index.json`。

提交 `b9b27f68` 的安装包执行 236 项 CPU 阶段测试，209 项通过。27 项失败中，19 项首先缺少原生模块初始化，5 项缺少加载结果与后续状态的衔接，3 项缺少导入效果等价性证明。测试宿主提供活动帧和缓存时，生成的 C++20 接口库可以使用已封存的常量、创建函数、写入名称并返回。本机 `artifacts/current/verification.json` 索引逐项诊断和实物。

5M 入口第 425 行展开为 6 个加载站点和 5 个模块体模板。后端生成了 5 个导入区域；模块区间可按实际帧、代码及下一条指令位置进入。调用方可在原控制流到达区间后进入。完整驱动仍需独立的模块加载服务，端到端加速比尚无实测。

779 个 Python 模块初始化器有 6,162 个函数定义，其中 3,766 个已关联到创建指令。1,113 个位置默认值元组来自定义帧的常量池，116 个需在定义时依次求值并构造元组。上版有 17 个区间从扩展参数后的操作码开始，本版撤回这类入口。两版解释器 3.12 和 3.14 已核对。真实 5M 的 C++20 区间在验证宿主中执行了 `warnings` 的 25 条指令（PC 64 至 114 前），13 种状态覆盖名称重绑定、对象寿命及查找或发布失败。驱动的首个未决操作是 `warnings.py:201` 定义类时读取基类 `Exception`。

中端按镜像转交独占的已绑定图，共享记录只在仍有其他消费者时复制。冻结同一 880 单元图，在单核上反序交错运行两对完整 `scar plan`，中位耗时从 114.058 s 降至 94.842 s，峰值内存中位数减少 518 MiB，296 个输出文件逐字节相同。

在同一项目的 352 处模块级单目标下标赋值中，F3 已为 332 处绑定了求值顺序与写入指令。索引目标端口连接接收者和键，未知别名仍由共享内存状态覆盖。配置初始化器的 M2 分析已到正常返回。B1 为其中八处完整写入生成独立区间。第 117 行从偏移 378 构造列表，388 经实际接收者写第 0 行，390 是缓存槽，正常续行在 392。生成的 C++20 已在受控宿主中与原配置比较了 11 种状态，包括写后抛错时数组已经改变、查名失败时尚未写入，以及入口不匹配时零源操作。

层间接口见 [架构](docs/ARCHITECTURE.md)，图和 IR 形式见 [模型](docs/MODEL.md)。`tests/stages` 通过禁止跨阶段导入及移走上游源码、图和报告，检查阶段独立性。GitHub Actions 从安装后的 wheel 执行同一检查。

Python 源码使用 Black 26.10.1 默认格式，C/C++ 使用 `.clang-format` 指定的 LLVM 风格。生成工程在编译前进行同样的格式化，默认编译启用 warning-as-error，CI 同时检查源码和生成产物的格式。
