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

图包另通过 `artifact_flow` 引用文件与构建依赖图。它区分与路径无关的字节值、捕获的文件快照、动态文件版本模板和未观测的历史读取。中端可从已资格化的操作推导有序常量文本，遇到后续协议缺口仍保留先前分析片段；这些片段只供分析，不会作为未闭合程序的可执行函数。字节相同不自动授权删除生成、复制或构建操作。

已知进程协议可表示为目标未绑定的 `process_request`，继续分析管道、等待和错误控制。逻辑 IR 的 `READY` 仅表示协议已闭合；后端仍须核对执行目标和原生算子，不能据此执行任意命令或回退原 Python 程序。

CPU 数组算子固定使用 xtensor 0.27.1/xtl 0.8.0。执行 `python tools/provision_cpu_sdk.py` 可安装到用户目录；首次配置这些上游头文件需要 CMake 3.29 以上，可用 `CMAKE_COMMAND` 指定独立工具路径。已安装 SDK 的消费工程仍可使用系统 CMake 3.28。`SCAR_CPU_SDK` 可指定安装前缀。供应工具校验源归档及安装收据，不修改系统环境。NumPy 只作为受版本约束的源接口和验证基准，中端、后端及生成程序均不运行 NumPy。

一维 complex128 的受支持 FFT 合同另使用固定修订的 PocketFFT；`python tools/provision_fft_sdk.py` 安装用户级 SDK，`SCAR_FOURIER_SDK` 指定前缀。目标包封存实际头文件和许可证，后续生成工程不依赖安装时的 SDK 路径；这不代表完整 NumPy FFT 接口已受支持。

执行 IR 含任意精度整数运算时，CPU 后端会核验 GMP 6 的头文件、静态库与 LP64 接口，并把库及许可证封存到目标包。可使用系统的 `libgmp-dev` 或通过 `SCAR_INTEGER_SDK` 指定安装目录。已证明有界的机器整数运算继续使用原生类型。

三阶段分别在独立进程运行。`build` 读取源码、每个编译单元的语言和参数，输出 `graph.json` 及内容寻址的单元子图。`plan` 读取完整图包，输出审计报告 `plan.json` 和自包含 `exec-ir.json` 执行 IR 包。`compile` 读取执行 IR 包，选择目标调度，输出 `physical-plan.json`、CMake 工程、C++20 源码、头文件和可执行文件。搬动中间包后仍可独立消费。各阶段的代码只导入本阶段和公共格式层。

当前语言适配器位于 `frontend/languages/{c_family,fortran,python}`。每种语言都先捕获事实，再正规化为同一操作词汇，随后共用构图和状态分析。中端分别做绑定、规范化、等价化简和执行 IR 封存。后端经物理计划、类型化 C++ 目标树、独立打印器和构建层交付。

Python 的受支持文件、目录、文本流、结构化循环及异常／清理可经同一 IR 交付原生 C++20；默认编码和标准库来源在前端捕获。尚未闭合的自定义异常与迭代器、异步上下文及子进程执行合同会阻止执行包就绪，不会回退执行原 Python。

真实 CXSparse 的所选构建已通过 C++20 输出对照。AMSS 的 ABE 单步演化在 Xeon E5-2680 v4 上使用 16 个固定 MPI rank，每个 rank 单线程。每个程序预热一次，交错测量三次。第一轮中，上游原版、上一版 C++20、消除有界循环取余后的 C++20 中位时间分别为 51.85、74.30、61.01 秒。第二轮中端还保留了常量乘积的轴顺序，使后端可以生成等价的多维遍历。此轮原版为 51.39 秒，生成版为 53.23 秒，相对原版加速比 0.965×。两轮共 20 次运行，每次 156 个二进制场逐字节一致，7 个文本输出规范化创建日期后也一致。逐次计时、代码版本和产物摘要见 `artifacts/current/native-performance/abe/cpu-schedule.json`。

对完整 Python 驱动建图后，中端现已能规范化网格坐标生成、数组索引和整数网格参数的文本写出。进程启动后的 Python 模块绑定及其异常路径仍有未闭合的效果合同，完整约 30 分钟工况尚未验收。当前三阶段状态见 `artifacts/current/verification.json`。

层间接口见 [架构](docs/ARCHITECTURE.md)，图和 IR 形式见 [模型](docs/MODEL.md)。`tests/stages` 通过禁止跨阶段导入及移走上游源码、图和报告，检查阶段独立性。GitHub Actions 从安装后的 wheel 执行同一检查。

Python 源码使用 Black 26.10.1 默认格式，C/C++ 使用 `.clang-format` 指定的 LLVM 风格。生成工程在编译前进行同样的格式化，默认编译启用 warning-as-error，CI 同时检查源码和生成产物的格式。
