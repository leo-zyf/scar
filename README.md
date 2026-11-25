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

第 425 行的加载器返回实际模块对象。M2 分别建立三个条件模块体，其中有 11 处函数创建。在保留的入口、模块及前缀中，27 个动态调用有 7 个能沿导入、属性读取和函数发布追溯到候选函数体。每条边以原调用选中的函数对象及活动代码为条件。7 个函数共还原 54 条连续语句，随后分别到达 3 个比较分支、3 个循环和 1 个数组索引赋值。第 429 行的函数体包含 `outdir → os.path.join → file0 → numpy.loadtxt → data` 的实际参数槽、属性读取和返回值写入。可逐条检查 `evidence/handoff/function-call-flow.json`。

独立对照取正式 5M 科学输出的 13 行数据，运行原 Python 入口第 430 行的调用切片一次。观察到的函数、代码、globals、builtins 和两个参数槽均与候选边匹配。生成 PDF 与正式上游产物仅创建日期不同，100 dpi 渲染像素相同。该对照覆盖一次原绘图调用，整驱动保持 `INCOMPLETE`。见 `evidence/handoff/plotting-frame-observation.json`。

F1 对 2,554 个普通函数捕获到独立代码身份，生成 9,467 条带指令位置的语句记录和 19,478 条名称读取记录。与上一版对照，11,753 条模块级证据在原字段上全部保留。函数体由 F3 图 ID 与子代码摘要绑定，调用点共用其内容寻址文件。见 `evidence/handoff/module-coverage.json` 和 `evidence/handoff/function-call-flow.json`。

M3 在 TwoPuncture 的 85 个外部数学调用点追溯操作数。热点中的 `Pion` 由入口参数 `n` 决定，`j` 随外循环变化，`k` 随内循环变化。大网格前三次 BiCGSTAB 迭代中两个调用点分别执行 9750 万和 5850 万次 `cos`。跨迭代复用还需证明浮点环境和库调用的可观察效果；本轮交付的两个二进制与之前实测版本的 SHA-256 均相同。对真实 C17 输入，仅共享已证明精确、私有的整数实参计算，保留原来的数学调用，7 组输入输出通过对照。证据见 `evidence/handoff/m3-math.json` 和 `evidence/solvers/public-run.json`。

真实角度的独立探针显示，2,190 种输入中有 2,189 种调用 `cos` 会置位 `FE_INEXACT`。打开该陷阱时，重新计算和读取缓存有不同的控制流。在固定舍入、关闭陷阱的探针里，反复使用系数表可获益；若每次读取并清除浮点标志，恢复标志的版本反而慢约 1.8%。原始计时、库绑定和位级对照见 `evidence/handoff/libm-domain.json`。

同一张已封存的 5M 图上，M1 合并对所选函数的重复扫描。固定一个逻辑 CPU，预热一对后交错运行三对，中位数从 41.546 s 降至 32.614 s（1.274×），绑定表与诊断指纹一致。此计时只覆盖编译器绑定层。

B1 用同一份可执行调用索引做 MPI、外部接口和链接资格检查。来源证据可以引用已删除的调用，这些引用只用于解释来源。真实 ABE 的规划耗时中位数从 8.441 秒降到 7.944 秒，两版各独立校准三次，未预热。固定校准数据后 ABE 的 187 个产物、TwoPuncture 的 58 个产物分别逐字节一致。

在 TwoPuncture 上检查了一个减少临时数组分配的实验。正式 50×50×26 网格中，从第一个 BiCGSTAB 残差到第四个残差，发布版和实验版各测两次，中位耗时分别为 12.760 和 14.244 秒。独立计数显示一个热函数中的 12,416,792 次数组分配全部消失。这个变换未启用。运行次数、区间、两份 ELF 和计数证据见 `evidence/handoff/two-puncture-storage.json`。

B2 将 C++ 目标记录的必填字段规则预计算一次。在真实 ABE 的 210,034 个目标节点上，解码、校验与打印的中位耗时由 2.555 秒降至 2.143 秒，两版各测四次。包含 LLVM 格式化的完整阶段各测一次，40 个文件及清单逐字节相同。两个求解器的 ELF 摘要均与前述应用基准一致。见 `evidence/handoff/cpp20-target-performance.json`。

另用交付的 TwoPuncture 可执行文件复测 16×16×8 网格，绑定逻辑 CPU 26，单线程，预热一对并交错计时四对。原版中位 5.398 秒（5.370–5.460），生成版 5.381 秒（5.307–5.401），比值 1.0032×。10 次运行的两个科学输出、stdout 和 stderr 对照一致，`Ansorg.psid` 归一化了首行日期。另一次测量的耗时分布在 5–11 秒，单独保留在 `solver-recheck.json`。见 `evidence/handoff/performance-payload/solver-clean.json`。

在 50×50×26 网格的前三次 BiCGSTAB 迭代上，另测了保留每次写回、按需重载目标元素的实现。该实现减少 273,206,400 次加载，增加 494,426,400 次有效性检查。同核单线程预热一对、交错十对，现行生成版与实验版的中位数分别为 11.996 s 和 12.278 s，配对中位比 0.977×。该变换已拒绝。见 `evidence/handoff/slot-reload-performance.json`。

提交 `5b4c7d8d` 的安装包执行 223 项测试，196 项通过、27 项失败、0 项跳过。失败用例与上一版相同。操作数最少的用例只有 14 个操作：F3 记录了导入和后续查找，M2 的特化路径直接调用项目函数，B1 因缺乏等价的导入、异常与缓存命中续体拒绝生成可执行文件。相同源码在普通启动、审计拒绝和预置模块时分别得到两种不同输出及一种错误退出。层间证据见 `evidence/handoff/smallest-entry-gap.json` 和 `evidence/handoff/qualification-gates.json`。完整 5M 入口仍无 C++20 运行时间或加速比。

层间接口见 [架构](docs/ARCHITECTURE.md)，图和 IR 形式见 [模型](docs/MODEL.md)。`tests/stages` 通过禁止跨阶段导入及移走上游源码、图和报告，检查阶段独立性。GitHub Actions 从安装后的 wheel 执行同一检查。

Python 源码使用 Black 26.10.1 默认格式，C/C++ 使用 `.clang-format` 指定的 LLVM 风格。生成工程在编译前进行同样的格式化，默认编译启用 warning-as-error，CI 同时检查源码和生成产物的格式。
