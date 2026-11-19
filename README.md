# SCAR

**State · Control · Action · Resource**：面向 Python 程序的宏观编译器。

```text
静态源码 + 必要动态绑定 + 硬件库存
→ 有类型、可展开的程序/资源图
→ 结构化简、成本区间及资源计划
→ C++20 / 算子 ABI1
→ 原入口、guard 和完整区域回退
```

当前优先完成三阶段核心开发，定期在同一完整边界验证。受支持的 CPU float64 二维数组、复合循环和事务区域已贯通；通用覆盖仍未完成，完整入口及兼容/I/O 仍保留 Python。

C++20 交付生成现成算子的调用及资源调度。稀疏求解使用 SuperLU 与 OpenBLAS。对 PyTorch 可用 `scar link-aten-provider --provider-revision REV --schema SCHEMA --out LINKED` 从完整声明生成并校验 LibTorch 调用。输出的 `operators.hpp` 提供类型化入口，`result.json` 记录编译和链接参数。能否替换原区域还取决于语义、资源与收益证据。详见[架构](docs/ARCHITECTURE.md)。

用 `scar trace --execution-evidence --torch-call-scopes --os-capture-root RAW --out RAW/trace -- python ENTRY.py` 采集，随后运行 `scar import-capture RAW --out MODEL`。从 `MODEL/native/native-call-scopes.json.gz` 选择原始调用序号，再用 `scar link-aten-capture --capture-model MODEL/capture-model.json --capture-model-sha256 SHA256 --call-attempt N --out LINKED` 直接生成外部调用。多次指定 `--call-attempt` 可批量生成。

`scar frontend --trace TRACE --project-root ROOT --adapter-directory ADAPTER ...` 可指定外部源码重放使用的 CPython 符号适配器。`scar observed-formals BUILD` 读取已准入的 F4 绑定状态和原始帧入口，输出调用范围内的形参端口、静态绑定版本及实际引用。调用方实参路由和对象内容仍由独立证据确定。

`scar plan-aten-capture --build PROGRAM --capture RAW --call-attempt N1 --call-attempt N2 --out PLAN` 将同一次调用方激活的原始 F4 端口与本次采集的 ATen 端口接入共同规划器。它保留状态顺序、本次看到的引用和未决成本，当前支持保留原执行。重复源码位置需要由共享循环接口接入。

规划器可复用 F4 已准入的原生绑定图，也可通过 `--binding-graph DIR` 指定。它按原始 CALL 和完整算子声明冷重放参数路由。V2 采集提供已映射 ELF 的包装器入口证据。`wrapper-contracts.json` 保留弃用签名、可能的原生写入、返回别名及仍由包装器处理的实参。

`scar plan-aten-boundary --observations RAW --observations-sha256 SHA256 --target HARDWARE --case ID --out PLAN` 将独立 C++20 调用的配对测量接入共同规划器。它按输入、库、线程设置和目标 CPU 保存成本，核对增益方向，再为整次调用保留串行或双核方案。成本作用域为该原生调用，Python 替换资格单独保留。

`scar compile-aten-boundary --plan PLAN --selection-sha256 SHA256 --adapter ADAPTER.hpp --adapter-sha256 SHA256 --out NATIVE` 冷重放选型与资源计划，再用明确提供的 C++20 适配器交付选定模式。适配器实现 `aten_boundary_abi_v1.h` 的入口，可直接使用交付中保留的原测量源码。执行 `NATIVE/launch.sh` 使用同一台主机的已核实 LibTorch、指定 CPU 和环境。

`scar compile-aten-graph --recipe RECIPE.json --recipe-sha256 SHA256 --linkage LINKED/result.json --linkage-sha256 SHA256 --out NATIVE` 从声明的张量数据流直接生成 C++20。`scar.aten-native-recipe` v1 声明固定形状输入、精确算子签名、按顺序绑定的值和整数标量，以及输出 ID。`NATIVE/launch.sh` 按声明顺序接受原生字节序的 float32 输入文件和新输出路径，成功时报告输出形状。C++ 产物直接调用现有 LibTorch，运行依赖记录在 `artifact.json`。当前支持有序的独立 CPU 图，原 Python 区域替换单独准入。样例与实测见 [CPU 图验证](artifacts/baseline/aten-native-graph.json)。

`scar project-aten-graph --build PROGRAM --capture RAW --call-attempt N1 --call-attempt N2 --linkage LINKED/result.json --linkage-sha256 SHA256 --out PROJECTED` 可从原始 F4 和同次采集自动生成上述配方。采集时追加 `--python-call-targets --native-tensor-inputs`，`LINKED` 使用同一采集和调用范围执行 `link-aten-capture` 生成。投影用精确 TorchOp 输入元数据确定 dtype、设备和布局，用 F4 路径核对内部依赖与原返回分量。原生回放由调用方提供实际输入值，原 Python 决策仍单独保留。见 [采集到 C++ 验证](artifacts/baseline/aten-capture-graph.json)。

`scar discover-aten-regions --build PROGRAM --capture RAW --contract WRAPPERS.json --contract-sha256 SHA256 --out DISCOVERY` 按实际调用分组，用原 F4 核验完整函数边界。加 `--capture-model MODEL.json --capture-model-sha256 SHA256` 可直接链接并投影唯一闭合区域，省去手填 CALL 编号。包装器效果由显式合同声明。发现报告可供后续成本规划和原入口验证使用。

`scar compile-aten-entry --help` 默认将已闭合的发现区域交付为独立 C++20 程序，计算调用 LibTorch。校准 Tensor 按原形参传 `--input NAME=FILE`。带容量预算的同 NUMA 计划可联合选择算子执行核和核内顺序，并与原序就绪计划配对验证。未采用并行计划时，容量合格的串行执行仍走 C++20。需要对照原 Python 程序时，`--entry-function` 单独运行完整入口差分实验。

`scar compile-aten-recurrence` 从已验原生 RHS 与固定网格步骤生成整个求解入口。`scar compile-exported-aten` 接收 `torch.export` 文件，为指定 `--input-shape NAME=DIM,DIM` 生成独立 CPU 推理程序。两条路径都写出 `plan/program.json`、`plan/problem.json`、`plan/execution.json` 和可构建的 CMake 工程。`scar validate-native-project` 冷读程序图和资源档案，核对文件与原生动态库，再按可选参考文件做逐字节验收。当前 AI 范围是受支持的纯函数 CPU FP32 算子和单个静态形状特化。

`compile-aten-recurrence` 可通过 `--discovery REPORT --discovery-sha256 SHA256` 从原入口推导求解方法、步长和状态输入端口，交付范围为自治固定网格求解的终态，运行入口使用原生文件参数。默认用四步回放标定算子，并按存储容量、估计成本和原生入口配对测量选择并行或串行。`compile-exported-aten` 可用重复的 `--calibration-input NAME=FILE` 为当前形状标定 CPU 串行版本。冷读验收通过 `plan/schedule.json` 重放执行顺序、绑核和释放点，并核对由同一计划生成的 C++ 源码。计时范围和验收结果见 [CPU 计划重放](artifacts/baseline/cpu-schedule-replay.json)。

`compile-aten-recurrence` 在既有读写依赖和 RHS 调用顺序下求共享循环体的张量存储峰值。峰值更低的顺序进入 C++20 交付，原图和生命周期证明可离线重放。真实 RK4 在 41 MB 预算下采用三核，38 MB 时采用单核，两种入口的 16 步终态均逐位一致。[CPU 生命周期排程验收](artifacts/baseline/cpu-recurrence-lifetime-order.json)。

使用 CPU-only LibTorch 时，可给 `compile-aten-recurrence` 或 `compile-exported-aten` 加 `--require-cpu-only`。构建会验证 Linux 加载器解析出的完整 ELF 依赖，遇到 CUDA Runtime、CUPTI 或其他未准入库即停止交付。`validate-native-project` 会重放同一依赖合同。产物当前绑定构建时的 LibTorch 安装。[纯 CPU 验收记录](artifacts/baseline/cpu-only-libtorch-delivery.json)。

`scar export-native-cpu --project NATIVE --manifest-sha256 SHA256 --out RUNTIME` 打包已验收的 CPU-only 产物、LibTorch 依赖和权重。`scar export-native-cpu --check RUNTIME --manifest-sha256 SHA256` 校验迁移包，可加 `--reference FILE` 和 `-- ARGS` 验收输出。运行直接执行 `RUNTIME/launch.sh`。当前验收覆盖本机 Linux 下的目录迁移，并沿用原计划的 CPU 绑定。[纯 CPU 三阶段验收](artifacts/baseline/native-cpu-end-to-end.json)。

重复传入 `--shape-profile PROFILE:NAME=DIM,DIM` 可把多个形状交付为同一个 C++20 推理程序，使用 `launch.sh PROFILE INPUT... OUTPUT...` 选择。各形状的图、物理问题和证书存放在 `plan/profiles/PROFILE/`，权重共用。各形状可用 `--calibration-input PROFILE:NAME=FILE` 标定。`--simplify-normal-completion` 允许按精确算子合同删除无可见消费者的计算，合同覆盖成功完成的返回及可见效果。生成时保留原图和修改收据。见 [形状与设备交付验证](artifacts/baseline/native-shape-and-copy-replay.json)。

`--plan-cpu-ready` 使用各形状的校准输入试验同 NUMA 物理核组。它核对独立算子、Tensor 容量和线程服务成本，并与同一调用核上的串行执行配对。只有原生入口也达到收益门槛，才把多核排程编入该形状。选型证据覆盖正常完成，Tensor 容量统计排除提供方内部工作区。CUDA 实验可用 `PYTHONPATH=. python scripts/verify_cuda_cublas_schedule.py --out NEW_DIR` 重放拷贝、cuBLAS 与流事件的离线交付。

`scar compile-exported-aten-cuda` 从 `torch.export` 构建连续行主序 FP32 `aten.mm` 图的 CUDA 候选。传入 `--model`、`--model-sha256`、`--cuda-model`、`--cuda-model-sha256` 和 `--out`。声明的 CUDA 假设模型只给离线规划提供容量与时间参数，实机时间保持 `UNKNOWN`，默认仍 `NO_OP`。产物用 `launch.sh --describe` 查看计划，用 `--probe` 检查设备。只有显式 `--trial` 才启动 GPU。`validate-native-project` 可离线重放源图、搬运、容量和 C++ 降低。[模型示例与验收记录](artifacts/baseline/export-cuda-source-delivery.json)。

同一命令接收 `scar.aten-export-hybrid-offline-profile` 时，前端会从源图识别相连的 GPU 矩阵乘法区域，将受支持的二维 `relu` 和 `add.Tensor` 留在 CPU，并在同一物理计划中记录跨设备依赖。每个逻辑值到同一目标内存的搬运只安排一次。CPU 算子通过 LibTorch 写入预分配的固定页内存，GPU 侧调用 cuBLAS。当前候选仍采用声明成本。[混合模型示例与验收记录](artifacts/baseline/typed-hybrid-export-delivery.json)。

`scar assess-native-device` 用 `--cpu-project`、`--cuda-project` 及对应的 `--cpu-sha256`、`--cuda-sha256` 核对同一源图的两个产物。传入 `--input NAME=FILE`、按顺序重复的 `--reference FILE` 和 `--out`。CUDA 到达准入后才开始交替运行新进程，比较全部返回字节和包含文件 I/O 的耗时。`--check ASSESSMENT --check-sha256 SHA` 重放证据而无需重新计时。无设备时保留 CPU，完整通过的收益证据会进入采用审查。[配对验收记录](artifacts/baseline/native-device-assessment.json)。

混合包可以用 `--with-activity-calibration` 加入专用采集入口。`scar calibrate-native-device` 接收同一组包、输入和参考参数，用 CUDA 流事件与主机单调时钟分别测量绑定后的执行步骤。分配、I/O 和同步单列。命令输出可重放的诊断成本图，采样缺失或低于计时分辨率的步骤保留 `UNKNOWN`。整入口收益仍使用普通试跑模式单独验收。[活动成本验收](artifacts/baseline/native-device-activity-calibration.json)。

`scar plan-native-device-placement` 可从已验证的 CPU 包比较合格矩阵乘法的 CPU/CUDA 放置。传入 `--cpu-project`、`--cpu-sha256`、`--device-model`、`--device-model-sha256`、`--out`。声明模型使用 `scar.aten-export-placement-offline-profile`。规划按共享搬运、依赖与容量排除不合格方案，搜索或成本未决时保留 CPU。`compile-exported-aten-cuda` 接收同一模型时默认构建被提名的 CUDA 方案，仍须独立实测才能进入采用审查。[放置模型示例与验收](artifacts/baseline/native-device-placement-selection.json)。

加上 `--empirical --input NAME=FILE` 和按输出端口顺序重复的 `--reference FILE`，规划只用与当前执行方案匹配的服务成本。可重复传入 `--device-evidence CUDA_PROJECT CALIBRATION_JSON SHA256`。CPU 普通内存、固定页内存上的 CPU 算子、GPU 算子和双向搬运分别匹配。缺少任何一项时保留 CPU 并列出缺项。服务时间排程与完整入口收益分别验收，`--check` 可离线重放放置决策。[实测成本验收记录](artifacts/baseline/native-empirical-placement.json)。

在同一源模型、形状和容量配置下，给 `compile-exported-aten-cuda` 加 `--placement-ordinal N` 可构建 `analysis.evaluated` 中指定的 CUDA 放置。该产物自带活动标定入口，其 C++20 排程及物化计划可独立重放。`calibrate-native-device` 的结果按放置方案及实际执行身份回填，部署默认保留 CPU。[指定放置的验收记录](artifacts/baseline/native-placement-treatment.json)。

实测放置沿数据边、队列顺序和同步 CPU 调用的提交依赖，分别传播每个服务的样本最小值、中位数和最大值。只有 CUDA 方案的上界严格低于其余全部容量合格方案的下界时才提名。候选区间无法严格区分时保留 CPU。样本范围用于服务模型，完整入口仍须另做验证。[区间验收记录](artifacts/baseline/native-placement-service-interval.json)。

加上 `--compare-ready --tensor-budget-bytes BYTES --entry-function main` 后，SCAR 沿完整张量数据图查找可同时就绪的操作，在同 NUMA 物理核上校准全部合格算子的服务时间。每轮用同一 CPU 掩码启动原程序及通过区域验证的候选，平衡每对方案的先后顺序。`delivery.json` 给出完整入口选择，`entry-validation/validation.json` 保留各候选结果。可用 `scar validate-aten-entry-candidates --check REPORT --check-sha256 SHA256` 重放选择和输出校验。

追加 `--allow-cross-numa-ready` 可比较跨节点核组。规划先提名核分配，再按该分配重放提供方算子并采样张量页。跨节点候选须提供完整观测。成本范围仍有重叠时优先选择同节点，最终执行经过区域和完整入口验证。

对用 `compile-aten-graph --ready-dag --allow-cross-numa` 构建的图，可在 `plan-aten-ready` 加 `--try-input-replica`。SCAR 为有重复远端读取的只读输入提名一个副本，将复制时间和页对齐后的容量纳入计划。候选需要采样位置、位级等价和配对收益证据。入口试验可使用选中的直接读取或副本计划。

为独立 CPU 图追加 `--concurrency-contract CONTRACT.json --concurrency-sha256 SHA256 --target HARDWARE --cpu CALLER --cpu WORKER` 可生成双核试验入口。`scar plan-aten-graph --bundle NATIVE/artifact.json --bundle-sha256 SHA256 --input X.f32 --input Y.f32 --out PLAN` 用单算子服务时间和交接开销求出互不重叠的相邻窗口组合。独立进程只对选出的并行方案与串行基线配对验证。`PLAN/plan.json` 逐段验算核心占用，`PLAN/launch.sh` 固定输入和已选窗口。性能报告的范围是预热后的调用。见 [多窗口调度验证](artifacts/baseline/aten-native-multi-window.json)。

`plan-aten-graph --tensor-budget-bytes BYTES` 给图持有的 CPU Tensor 后备存储设置预算。生成代码在最后一次读取后释放中间 Tensor 句柄，并行窗口等待两侧完成后再释放。规划器用原生校准记录的存储字节数估算各段峰值，筛掉超预算窗口，再验证核与存储预算。预算范围覆盖输入与算子结果，算子内部工作区、分配器保留与实际 NUMA 页位置仍独立记录为未建模。见 [Tensor 预算验证](artifacts/baseline/aten-tensor-budget.json)。

在满足三核准入且同 NUMA 的区域，可给 `compile-aten-graph` 加 `--ready-dag`，再用 `scar plan-aten-ready --bundle NATIVE/artifact.json --bundle-sha256 SHA256 --input X.f32 ... --tensor-budget-bytes BYTES --out PLAN`，按真实张量依赖安排现成 LibTorch 算子。新产物在中间张量的最后一次读取后释放句柄，规划使用涵盖各核可能完成顺序的张量存储上界。通过配对验证后，可用 `PLAN/launch.sh` 运行固定输入，或把 `PLAN/selection.json` 交给原函数试验。

`scar build-aten-python-trial --help` 提供显式原 Python 函数接入。它核对函数 CodeID、直线算子依赖、声明的包装器效果及已验证的原生计划，生成借用 PyTorch Tensor 句柄的 C++20 扩展。`scar validate-aten-python-trial --trial TRIAL --trial-sha256 SHA256 --out VERIFIED` 在原模块临时安装入口，测量带守卫的完整函数调用并验证原路径回退。试验按记录的 CPU、Tensor ABI 和输入内容执行，退出时恢复原绑定。见 [原函数试验](artifacts/baseline/aten-python-trial.json)。

`scar validate-aten-python-entry --trial TRIAL --trial-sha256 SHA256 --entry-function main --out VERIFIED -- ARGS '{output_dir}/result'` 分别在新进程运行原入口与守卫入口。准备工作线程的时机延至首次合格调用，应用可先配置 Torch。试验逐对核对返回值、梯度模式、标准输出及输出目录内容，并按进程总耗时决定是否保留原入口。`--allow-ambient-grad` 适用于保持调用方梯度模式、输入无需梯度的试验。见 [完整入口验证](artifacts/baseline/aten-python-entry.json)。

受信的本地 LibTorch 安装可在同次开机、同一挂载命名空间复用构建时的库文件内容校验。运行时仍核对打开的文件和已加载映射的身份。文件身份改变、旧交付缺少凭据或文件系统不支持时，执行 SHA256 内容校验。完整入口报告记录两条路径的字节数。见 [库文件身份验证](artifacts/baseline/aten-provider-identity.json)。

`build-aten-python-trial --geometry-contract CONTRACT --geometry-contract-sha256 SHA256` 可显式生成允许值变化的试验。构建时在已安装的 ATen Meta 上传播固定形状与步长，并与 CPU 存储观测核对。每个算子须声明与数据值无关的存储和效果，并承诺回退重跑不会重复可见副作用。原生桥接核对输入和中间结果的几何，运行时无需保留参考输入。用 `validate-aten-python-trial --input X.f32 --input Y.f32 ...` 可测未参与校准的同形状输入。见 [变化输入验证](artifacts/baseline/aten-variable-input.json)。

`run-aten-python-entry --mode OBSERVE_ORIGINAL` 汇总原入口实际执行的目标调用次数、耗时和 GradMode。`forecast-aten-python-entry` 用这份观测、独立输入的热调用校准及一次性准备成本，先估算进程内入口收益。独立的新进程 A/B 核对最终输出和实际收益。见 [torchdiffeq RK4 验证](artifacts/baseline/aten-ode-entry.json)。

主线采用编译器式顺序：原图与效果分析 → 合法结构改写 → 需求/成本推导 → 资源时序构造 → 证书 → C++20 编排。下面的 `macro-pipeline` 是早期数组融合实验，含自有数值循环。它将已准入的原 F4 调用区域投影到既有 OIR，由同范围成本决定是否按试验合同采用整区域融合。成本缺失或不盈利时保留原 Python 路径。

```bash
scar macro-pipeline --inputs INPUTS.json --out UNKNOWN
scar macro-pipeline --inputs INPUTS.json --cost-profile PROFILE.json --compile --out SELECTED
scar macro-pipeline --check SELECTED/preparation.json
```

`INPUTS.json` 沿用已有 source/value/memory/hardware 输入；`PROFILE.json` 沿用 `scar.deployment-cost-profile`，绑定完整 warm 受保护调用、shape、CPU 可用域及实现版本。原 CPU affinity 继承而非绑核，不创建 worker；显式试验可用 `scar.compiler.macro_pipeline.install(module, SELECTED/result.json, trial=True, exclusive_inputs=True)` 受控接入并恢复原函数。它不是整程序或 compact 自动交付。验收见 `artifacts/baseline/macro-fusion-pass.json`。

完整入口的正向需求/成本桥接也已接通：

```bash
scar model-pipeline-cost --inputs INPUTS.json --out DEMAND
scar model-pipeline-cost --inputs INPUTS.json --coefficient-profile COEFFICIENTS.json --out MODEL
scar model-pipeline-cost --check MODEL/result.json
scar macro-pipeline --inputs INPUTS.json --cost-profile MODEL/profile.json --compile --out SELECTED
```

第一条机械生成原逐CALL eager、逐leaf native和最终fused需求，以及全部成本未知的系数模板；不运行目标或加载库。原算术按source occurrence恢复，避免将CSE后的工作量误算成原工作。后续复用有来源的有效仿射模型与既有成本gate：完整入口开销及非零需求系数缺失时KEEP；逻辑字节不当作DRAM流量，声明参数不当作实测时间。验收见 `artifacts/baseline/pipeline-forward-model.json`。

固定拓扑的 canonical 完整入口标定已可执行：

```bash
scar calibrate-pipeline-corpus --inputs INPUTS.json --out CALIBRATION --build-jobs 4
scar calibrate-pipeline-corpus --observations CALIBRATION/observations.json --out REFIT
```

先并行编译、再串行计时；只用TRAIN拟合，混合canonical和真实源区域作冻结holdout。不相减不同scope的中位数，不将拟合截距称作独立guard/allocator时间。秩不满或holdout失败保持UNKNOWN；系数始终为有效 `DECLARED_MODEL`，不能外推到其他shape/CPU/guard版本。采样与离线拟合分离，模型发布在 `CALIBRATION/model/profile.json`；支持的原输入目前为等形、非空CPU二维数组。

已选融合也可交付为不依赖建图器的compact运行包：`scar export-fused-runtime --build SELECTED/result.json --out COMPACT`。它复制原单operator ELF，沿用C++20/ABI1，继承CPU、不创建host/worker；使用已有 `run-caller-pipeline --bundle COMPACT --admission-sha256 PIN --exclusive-inputs --measure-candidate -- ENTRY.py` 显式验证。compact有独立成本scope且保持UNKNOWN，macro收益不授权其自动执行；runtime更新后旧compact包需要按原producer重导。

以下规划接口默认使用关键路径与容量日历正向构造，不依赖求解器；`--planner solver`保留CP-SAT作显式局部补充。构造计划独立检查端点约束、搬运和存储峰值；贪心死路是UNRESOLVED，不证明不可行或全局最优。规划端可选安装 `scar[planning]`（OR-Tools 9.15.6755）；本机使用隔离的 `.venv-planner/bin/python`，交付运行端不依赖求解器。

```bash
scar joint-plan --problem PROBLEM.json --out PLAN.json
scar joint-pipeline --inputs INPUTS.json --cpu-ids 0,1 --out UNKNOWN
scar calibrate-pipeline-services --preparation UNKNOWN/preparation.json --out COSTS
scar joint-pipeline --inputs INPUTS.json --cpu-ids 0,1 --cost-profile COSTS/profile.json --compile --out SELECTED
scar joint-capacity --preparation CAPACITY/result.json --steps 4 --epochs 1 --out CAPACITY_PLAN
scar profile-pipeline-host --native SELECTED/native --out HOST_OBSERVATIONS
scar model-pipeline-host --observations HOST_OBSERVATIONS/observations.json --out HOST_MODEL
```

`joint-plan` 同时选择实现、数据位置转换与资源受限时间区间；独立检查依赖、搬运和存储峰值。源生产端负责完整源码/合法性重放。`joint-pipeline` 将选中的 CPU 绑定和资源顺序真正编入 C++20/ABI1；孤立 service 样本缺少并发证据时采用逻辑互斥并落实到运行顺序。模型时间仅覆盖 ABI1 service，不包括外围分配、线程启动、派发和 join，不是整程序时延或加速比。`joint-capacity` 保留未知成本，次数参数不授予原循环替换权；完整声明模型仅供显式试验，默认仍 NO_OP。

`profile-pipeline-host` 对已准入 native 的实际线程启动、分配、计算、发布、join 和释放做有界观测；不改原算术或派发策略。`model-pipeline-host` 离线重放源码及样本，输出原值/存储关联的时间图、硬件投影和逐活动成本表；`--check HOST_MODEL/result.json` 独立重放。观测成本保留插桩影响，不作为未来 clean service 预测；join 等待不重复计入 CPU 工作。未知 Python 外围、物理 CPU 需求及驻留位置继续保留，尚不授权完整时延规划。

`joint-pipeline --host-lifecycle` 进一步建立按所选 CPU 条件存在的启动、入场、退出、join 和清理活动，并将 ABI 与完整调用包装成本分开查询。`cost-contexts.json` 导出宿主预测模板；通过 `--host-cost-profile FORECAST.json --compile` 可显式验证完整已知点模型及其 C++ 线程集合。当前支持声明或解析预测，单配置实测样本保留 UNKNOWN；跨 worker 子集和 lane 顺序的固定系数仍是模型假设。原 Python 整调用成本、自动盈利选择和宿主物理 CPU 需求尚未闭合，默认 NO_OP。
LeWM、Sunfish、Gray–Scott、Cirq 都是外部 testcase，分析规则不按业务名称选择优化。

## 使用现有环境

```bash
conda activate lewm
python -m scar --help
```

使用已有环境，不升级 PyTorch/CUDA。安装入口需要时可用 `python -m pip install --no-deps -e .`。以下命令的输出目录应为新目录。

## 三阶段主链

先用已有形式前端逐层处理源码，再建立统一图。动态目标不确定时，可按原 CALL 位置作有限观察；不逐次展开循环或保存 Tensor 数值。

```bash
scar frontend /path/to/program --jobs 8 --out FRONTEND
scar observe-call-bindings --all-sources --catalog FRONTEND/catalog --out CALLS -- python ENTRY.py [arguments]
scar build --entry ENTRY.py --catalog FRONTEND/catalog --call-bindings CALLS --out PROGRAM
scar plan-build --build PROGRAM --shape ROWS COLS --memory-budget-bytes BYTES --out DOMAINS
scar compile-build --domains DOMAINS --stateful-interval-plans REGION_TO_INTERVAL_PLAN.json --out COMPILED
```

`observe-call-bindings` 是必要动态绑定的可选步骤；纯静态 build 可省略 `--call-bindings`。shape/预算必须来自声明或观测，不凭模型名称猜测。
`--stateful-interval-plans` 按精确 region ID 选择已有区间计划；也可使用 `--calibration` 等已准入模型输入。未决计划保留 NO_OP，`--plan-only` 只保存计划。

`plan-build` 自动保存正文、caller 接口、局部内容 epoch、生命期和共享循环 carry。`compile-build` 独立重放后消费，并校验同源硬件域的 URI 别名；未知成本不冒充零时长。标量与区间计划共用原 C++ lowering。

查看第一阶段的图：`scar view-graph --build PROGRAM --out GRAPH_VIEW`，随后打开 `GRAPH_VIEW/index.html`。支持拖动、缩放、双击展开/收起、关系筛选、原记录查看和本地布局保存；程序图与原硬件库存分别浏览，离线使用，不修改 IR。默认有导出范围限制，可用 `--depth`、`--max-parts`、`--max-nodes` 调整，未导出的边界会保留标记。已有任务图也可用 `--schedule SCHEDULE.json` 查看。

本机现有入口：`artifacts/current/experiments/graph-viewer-v1/program/index.html`；共享循环及时间/资源视图：`artifacts/current/experiments/workspace-time-v1/view/index.html`。画布按需加载子图，循环只保留一份正文；完整覆盖范围见各自收据。

已有来源核验的 standalone 串行区域成本可继续进入局部时间规划与 C++ 试验：

```bash
scar project-activity-time --costs COSTS/result.json --out TIME
scar plan-closed-windows --activity-time TIME/result.json --out WINDOWS
scar prepare-window-rewrite --windows WINDOWS/result.json --out REWRITE
scar compile-window-trial --rewrite REWRITE/result.json --out TRIAL
```

这条路径保留未知残差、原存储生命期及外部边界；自动发现同形状、独立 raw-copy 循环，生成真正的替换任务图，重接其就绪/时间/硬件关系，再由同一证书生成 C++20/ABI1。仅在唯一可试验窗口时自动定位，否则在准备替换图时给 `--window`。融合减少循环控制，保留原数据复制；新成本重新标为 UNKNOWN，默认 NO_OP，不构成全图选优或部署。各命令均支持 `--check`；旧 `compile-window-trial --windows` 保留为直接方案试验接口。

替换区域还可接源忠实的规范化组件成本与有限模型选择：

```bash
scar calibrate-copy-window --rewrite REWRITE/result.json --out COPY_COST --repetitions 7 --warmup 2
scar choose-copy-window --calibration COPY_COST/observations.json --out CHOICE
scar compile-copy-choice --choice CHOICE/result.json --out CHOICE_TRIAL
```

原完整窗口与融合窗口交替采样，各自一个计时区间；不累加旧单项成本，不运行完整目标来搜索变体。选择器按这两个固定 CPU 组件的 median 选择，平局保留原窗口；C++ 构建端消费选择并绑定对应子产物。经验范围和未证明的上下文差异保留，默认仍 NO_OP，仅显式试验，不据此宣称整程序收益或全局最优。这三个命令也支持 `--check`。

原源循环的私有 workspace 生命期外提可使用：

```bash
scar prepare-workspace --loop-build SOURCE_LOOP/result.json --out WORKSPACE
scar compile-workspace --preparation WORKSPACE/result.json --out WORKSPACE_NATIVE
scar export-workspace-loop --native WORKSPACE_NATIVE --out WORKSPACE_RUNTIME
scar run-stateful-loop --bundle WORKSPACE_RUNTIME --admission-sha256 EXPORT_SHA --report RUN.json --validate-only --exclusive-inputs -- ENTRY.py
scar project-workspace-time --native WORKSPACE_NATIVE --out WORKSPACE_TIME
scar project-workspace-demands --time-schedule WORKSPACE_TIME/result.json --out WORKSPACE_DEMANDS
scar bind-time-resources --time-schedule WORKSPACE_DEMANDS/result.json --out WORKSPACE_RESOURCES
scar cost-activities --resources WORKSPACE_RESOURCES/result.json --out WORKSPACE_COSTS
scar view-graph --time-schedule WORKSPACE_TIME/result.json --out WORKSPACE_VIEW
```

准备端重放原 F4 循环/单步证明，检查本轮所有私有读取均由本轮写入覆盖；图保留一份 LOOP_REGION 正文，外层持有固定容量，每轮内容仍有独立版本。C++ 保留单步编译边界及每轮 guard、计算、FP/CPU 还原和提交，只移动私有分配/释放。导出命令返回 `expected_admission_sha256`，作为上面 `EXPORT_SHA`；运行端只读取 compact 合同、动态库和原程序源码，不加载分析器或大图。省略 `--validate-only` 时仍执行原程序，显式试验失败只续跑未提交后缀。`prepare-workspace --recipe RECIPE --trip-count N` 仅生成固定输入的显式调用组合，不授权替换未经证明的原调用循环，也不能导出为原入口替换包。减少分配请求不保证更快；当前均未自动启用。

`project-workspace-demands` 当前只细化单线程 C++ 源路径：成功 bind 后才推导选定 CPU，provider 使后续位置知识失效，restore 后的 COMMIT 不沿用选定 CPU。成本命令默认保留 UNKNOWN；可追加 `--domain DOMAIN --candidate ID --calibration PRIMITIVE_JSON`，严格匹配来源、CPU、形状与布局后接入条件组件。外层 pool 计一次、正文按迭代计费，未知残差与完整时长仍保留，不把组件和当成整程序时间。

边界组件可用 `scar calibrate-workspace-boundary --time-schedule WORKSPACE_DEMANDS/result.json --out BOUNDARY --repetitions 7 --warmup 2` 采集，再给成本命令追加 `--boundary-calibration BOUNDARY/observations.json`。微型探针保留实际检查、分配、绑核和恢复片段，跳过目标 DAG、COMMIT 和 provider；每份保留逐迭代原切点及未映射间隙，计时开销与实现差异仍为残差。

将以上 v4 图、资源与成本接入执行证书和重新构建：

```bash
scar refine-workspace-execution --costs WORKSPACE_COSTS/result.json --out REFINEMENT
scar compile-workspace-refinement --refinement REFINEMENT/result.json --out COMPILED
```

两端均可用 `--check RESULT.json` 独立重放。当前要求准备记录固定原 base native；证书逐项保留共享循环的数据 carry、外层容量生命期、原顺序和每步事务，构建端核对重新生成的 C++ 源码与 controller ID。原源码循环输出 `COMPILED/bundle` 及 admission SHA，可交给上述运行命令；固定输入组合仅输出 native。此链只落实已有 workspace 变换，不代表未知成本已解决或联合最短时间规划已完成，默认仍 NO_OP。

封闭算术循环还可自动证明跨迭代的有限性属性，并仅省去后续输入扫描：

```bash
scar prepare-workspace-guards --refinement REFINEMENT/result.json --out GUARDS
scar compile-workspace-guards --preparation GUARDS/result.json --out GUARD_NATIVE
```

`PROVEN_CONDITIONAL` 要求首次输入扫描、封闭算术、每步严格 FP 成功、完整影子与 ROI/COMMIT/carry 证明；有未闭合 provider 时输出 `REJECTED` 并保留原执行。新 compact profile 明确标识扫描复用，C++ 首轮仍扫描，后续仅在前一步成功后跳过；scalar、CPU/FENV、事务和失败后缀检查不变。属性不跨调用缓存，不表示数据数值相同，也不把原复合 preflight 成本直接当作新版本成本。两端支持 `--check`；新运行包仍使用 `run-stateful-loop` 显式验证，默认 NO_OP。runtime 版本更新后，旧 compact 包需从原 native 重新导出。

共享循环的局部硬件模型可用 `scar plan-workspace-windows --refinement REFINEMENT/result.json --out WINDOWS` 构建，`--check WINDOWS/result.json` 独立重放。它复用原窗口/冲突分析和求解器，只在同一符号迭代内规划，保留数据端口、carry、FIRST/LAST 与外层工作区寿命；未知边界不计为零，也不拼成完整物理时间表。当前域为已知 CPU 绑定和原固定 pool，平局保留原序；融合提示尚无共享循环 lowering，不能直接交给 standalone backend。无已选变换时交付端引用原 C++ 蓝图，不重复编译。共用规则文件更新后，旧 standalone 窗口收据需重新发布，基础 native 与独立 compact 交付不受这次分析重构影响。

跨轮 shadow 驻留使用 `scar workspace-shadow --refinement REFINEMENT/result.json --out RESIDENCY` 生成 ROI 副本证明，再用 `scar workspace-shadow --preparation RESIDENCY/result.json --out RESIDENT_CPP` 编译。首轮完整 snapshot，后续仅复制成功 COMMIT 区域的补集；所有有限性检查、算术、provider 与提交仍保留。原源码循环可导出 compact；固定输入组合仍不获得 caller 替换权限。同步 CPU Q 明确要求借用端口无返回后延迟写，此条件不是已静态证明的 provider 属性，默认仍 NO_OP。`--check RESULT.json` 重放准备或构建结果。finite-guard 与 residency 变体共用构建/导出核心，各自绑定计划和新 controller ID。

`scar benchmark-shadow-residency --original ORIGINAL_WORKSPACE --variant RESIDENT_CPP --inputs INPUTS.json --out AB` 在独立进程进行固定 N4、warmup2、7 对交替 A/B；INPUTS 保存一次 `.npy` 输入及精确 scalar。加载、重置和核验均在时钟外；计时范围为 raw ABI 调用及返回，包含四步 native 控制与计算，不是完整科学程序。真实 Gray–Scott 区域一次测得原 median 9.754001ms、驻留版本 8.763597ms（1.113×）；完整输入/配置、范围与方差见 `artifacts/current/experiments/workspace-shadow-v1/benchmark/`，不能据此宣称整程序收益或自动部署。

跨独立调用持有工作区的显式 session 原型：

```bash
scar workspace-session --residency RESIDENCY/result.json --program PROGRAM --out SESSION
scar workspace-session --preparation SESSION/result.json --out SESSION_CPP
python -m scar.compiler.inspect_workspace_session SESSION_CPP/result.json
```

准备端从原 F4 自动发现复制写集，保留整数索引降维和逐条读写顺序，证明哪些写入与驻留 ROI 不相交；C++20/ABI1 提供 create/step/invalidate/destroy。轻量运行端 `scar.runtime.native_workspace_session.load` 默认不加载动态库；显式 `trial=True` 后可在受控调用范围执行，每步读取新 scalar，绑定变化或未知写入使副本失效，原函数回退前释放 session。最初的受控 Gray–Scott `update()` 接入已验证轨迹逐位一致；原语本身不证明外围 caller 的全部效应，CALL 观察也不捕获直接数组写入。新 session 未测性能，不沿用固定四步的加速比。共享 guard 更新后，旧 compact 包需重新导出。

现在可从原图自动推导支持的 caller guard 和安装位置，并从普通导入入口接入，无需手工改类方法：

```bash
scar session-lifetime --session SESSION_CPP/result.json --resources WORKSPACE_RESOURCES/result.json --out LEASE
scar session-scope --session SESSION_CPP/result.json --lifetime LEASE/projection.json --out CALLERS
scar session-scope --export CALLERS/result.json --out SCOPED_RUNTIME
scar run-workspace-session --bundle SCOPED_RUNTIME --admission-sha256 EXPORT_SHA --validate-only --exclusive-inputs --report RUN.json -- ENTRY.py
```

caller 分解复用原 F4 操作和数据/控制边，唯一 formal 绑定只传递引用路径，不证明内容不变。直接 method lookup 必须由当前 descriptor、接收者和字段类型 guard 兑现；未知调用使当前 activation 持续失效，保留其原执行。已接通原 Gray–Scott `update()` 的29项生产操作；外围 `integrate()` 尚未闭合，因此每次最外 caller 入场失效、退出释放，不跨外部间隙保留副本。动态 lease 图保留原硬件和固定容量，次数/时长未知时不冒充固定循环时间表，也不授予全局调度或盈利结论。省略 `--validate-only` 保留原执行，不加载 native。

外层共享循环可以继续接入条件标量流及独立 kernel 交付：

```bash
scar session-outer --nested CALLERS/result.json --out OUTER
scar session-outer --export OUTER/result.json --out OUTER_RUNTIME
scar run-workspace-session --bundle OUTER_RUNTIME --admission-sha256 EXPORT_SHA --validate-only --exclusive-inputs --report RUN.json -- ENTRY.py
```

静态阶段用有限类型不动点表示循环变量，保留原控制回边、字段写和未知 host 调用，循环不展开。完整 host 效应尚未闭合时，新显式策略只执行已有独立源合同的 leaf：完整输入快照、原检查和计算后即释放，taint 不清除。打印、绘图、文件保存继续原执行，不遮罩 `print`，也不假设未知调用返回后无延迟访问。真实 Gray–Scott `integrate()` 已对比状态、stdout 和 PNG；这是完整原控制路径上的有界 leaf 交付，仍非整程序 C++、跨 host 驻留或全局最短调度。

已有完整有限模型证书可通过 `scar compile-pipeline-refinement --certificate REFINEMENT/certificate.json --out COMPILED` 构建 C++20 并导出 compact 包，再用 `run-caller-pipeline --bundle COMPILED/bundle --admission-sha256 EXPORT_SHA --validate-only --exclusive-inputs -- ENTRY.py` 验证。当前只落实计算调用的 CPU lane 顺序与原依赖，沿用原初始化、内存和 join 保护；模型分配时间、释放时间及 makespan 不作为原生运行保证，默认仍 NO_OP。

闭合数组 caller 可通过 `compile-caller-pipeline --pipeline-workers N --pipeline-dispatch ready` 使用原数据依赖和 pool 复用约束，按实际完成情况派发后继；显式 CPU 分配不代表最快方案。对应 `run-caller-pipeline --validate-only --exclusive-inputs` 只验证执行，不采集性能或授权部署。

追加 `--pipeline-memory-order dependencies` 时，只在原 RAW 已证明旧值全部消费者结束后复用，避免为省槽位新增等待；预算不足明确拒绝。默认 `source` 保持按源码串行生命周期规划。两种容量模型均不等同最短执行时间，联合 makespan 选择仍需成本和执行证书。

## 交付与非计时验证

已有 compact stateful C++ 产物和 DIRECT/COMPOSITE whole-loop 产物使用同一个最终交付核心。准备与运行保持同一原环境；payload/slot SHA 来自 prepare 输出。

```bash
scar prepare-stateful-delivery --pipeline READY_PIPELINE.json --out CAPSULE -- ENTRY.py [arguments]
scar run-stateful-delivery --capsule CAPSULE --slot CAPSULE/validation.slot --payload-sha256 PAYLOAD_SHA --slot-sha256 SLOT_SHA --report CAPSULE/run.json --exclusive-inputs --validate-only -- ENTRY.py [arguments]
```

whole-loop 对应 `prepare-loop-delivery` / `run-loop-delivery`。省略 `--validate-only` 保留原执行。当前交付槽为 UNMEASURED，不使用旧性能收据授权新路径，也不导出 RELEASE。
已有 native ABI、源/状态/形状/预算 guard 与失败后的整区域回退保留；运行端无需 compiler、IR、planner 或 g++。

原 F4 状态循环现可用 `compile-stateful-loop --help` 构建；省略 `--step` 时，从选定原 CALL 自动发现唯一状态区域。再以 `run-stateful-loop --help` 接入原入口，复用既有单步 ABI1，逐步提交；部分成功后只恢复未提交的迭代。当前仅支持固定参数、每轮一个普通 global step CALL 的封闭 profile，显式验证使用 `--validate-only --exclusive-inputs`，默认仍原执行。

## 模型与开发状态

图用共享正文、强类型记录、ID/FactRef 引用和分片表存储；循环和函数可以继续展开，静态模板、动态调用、逻辑内容与物理 storage 分开。
硬件库存保留完整平台，存在、可访问、后端支持和实际使用分别表达。当前单区域 CPU 方案选择尚非全程序联合调度。

- [模型](docs/MODEL.md)：实体、端口、状态、版本、循环和证据。
- [架构](docs/ARCHITECTURE.md)：分层组件与交接。
- [当前状态](docs/STATUS.md)：实际验收、范围与未完成项。
- [开发与查询](docs/DEVELOPMENT.md)：形式前端、接口和工作区约定；更多命令见 `scar --help`。

真实 testcase 在项目外；`tests/workloads` 仅保留小型编译回归。大产物留在 `artifacts/current`，精简验收报告在 `artifacts/baseline`。只维护当前工作区源码，版本历史由 Git 保存。
