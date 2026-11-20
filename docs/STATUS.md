# 当前状态

`compile-cpu-entry` 已接入按应答发送命令的原生递归控制器。`verify-cpu-entry` 用同一流程冷验编译收据、交替启动原入口和 C++20 工件，并重放输出合同和选择。有限元 96 轮、RK4 16 步及递归入口的留出命令流各预热 1 对、清洁测量 3 对，中位数分别为 **2.003 → 1.133 s、6.708 → 2.330 s、0.924 → 0.120 s**。有限元标准输出与原程序逐字节一致。RK4 的完整终态逐字节一致，两组标准输出分别满足各自格式。递归入口核对有序应答和四场搜索，按已有合同仅规范化 `time` 与 `nps`。三个候选均通过本次作用域的收益门槛。见 `artifacts/baseline/cpu-entry-cross-provider-acceptance.json`。

有限元的无轨迹调用图重放按实际使用的读取器校验版本，恢复了旧收据中的 353 份静态材料和 182 条绑定。Cirq 实跑中的三处 `vdot` 归约绑定到 NumPy 配套的 OpenBLAS ILP64，均逐位一致。独立 C++20 量子入口还需绑定复数切片乘加的别名、浮点行为及随机数初始化。

`calibrate-native-render` 现在可用已发布的两套 C++20 绘图 ABI 校准从预备输入到同步发布的完整渲染会话。CPU 0 上的 1 帧和 8 帧前缀各预热 1 对、测量 5 对。所有 108 个 PNG 与留存的原始输出逐字节一致。已有版 v1 与资产常驻版 v2 的会话中位数分别为 320.2 → 315.3 ms 和 2.432 → 2.389 s。两版每次都在 80 MiB 进程虚拟地址空间上限内完成，八帧的最大 VmPeak 分别为 46,760 和 47,368 KiB。该上限涵盖整个独立驱动。

`plan-native-plot-boundary --render-calibration` 将 CPU、挂载点、进程上限和两种作用域成本接入原保存操作，并为 1 帧、8 帧分别构造可行计划。两个收益都低于已有 5% 资格门槛。此前两轮匹配的整入口测试方向相反，均未通过该门槛，因此默认选择 v1，原同步发布顺序不变。完整调用方成本和算子私有工作区预算继续单独记录。见 `artifacts/baseline/cpu-native-render-service-calibration.json`。

`calibrate-native-png` 用正式原生库的同步 `save_png` 测量已准备 RGBA 到 PNG 发布的干净服务成本。CPU 0 上四帧各预热 1 次、测量 5 次，24 次输出均与原始 PNG 逐字节一致，各帧中位数为 237.2–238.0 ms。采集核对同次 CPU 拓扑、`/home` 的 ext4 挂载和原生库字节。`plan-native-plot-boundary --calibration` 将三个匹配样本的成本及 23,040,000 B 借用输入接回规划图。完整渲染成本和私有工作区预算继续保留 `UNKNOWN`，同步发布约束仍生效。见 `artifacts/baseline/cpu-native-png-service-calibration.json`。

`export-native-live-plot-phase --png-detail` 在两份独立 C++20 诊断包里记录了 Pillow 的逐行滤波、zlib 压缩、临时文件操作和最终发布。真实 CPU 0 八帧运行共做 3 对。两版的所有 PNG 和 stdout 与清洁入口逐字节一致。已有版的 PNG 区间中位数为 1.877 s，其中压缩约 1.229 s、滤波约 0.624 s，临时文件操作及发布合计的中位数约 1.9 ms。此处耗时用于定位。

`plan-native-plot-boundary` 冷重放原 F4 保存端口、控制器作用域、首次数值更新前的发布证据及目标 CPU 拓扑，将诊断接到共享规划问题的既有操作。保持同步发布约束，绘图工作区上界及同次文件系统资源仍待测定，完整成本为 `UNKNOWN`，选择为 `NO_OP`。原生交付保持已有算子和发布路径。见 `artifacts/baseline/cpu-native-png-boundary-acceptance.json`。

`export-native-live-plot-phase` 与 `observe-native-plot-entry` 已把绘图分段诊断接到两个真实 C++20 控制器上。八帧无视频，各做 3 对诊断，stdout 和 PNG 与原干净运行逐字节一致。按独占时间计，已有入口的 PNG 编码和发布约 1.949 s，占绘图 ABI 时间约 79%。常驻版减少约 17.9 ms 的资产读取和 21.6 ms 的坐标处理。分段数据只用于定位。

相同正式二进制和参数的未插桩复跑中，八帧中位数为 2.454 → 2.401 s，方向与上轮 2.457 → 2.504 s 反转。两轮均通过输出对照。现有证据无法归因上轮回退，默认继续采用已有入口。见 `artifacts/baseline/cpu-native-plot-phase-attribution.json`。

`project-controller-plot-residency` 从原循环、已准入启动镜像及打开记录建立绘图资源的生命周期。布局、字体和打包后的 X/Y 在一个控制器上下文中保留，Z 和时间继续逐帧读取。ABI v2 复用 FreeType 字体对象及 Z 缓冲区。完整入口已通过篡改字体试验，在首帧或日志发布前拒绝异常资产。这条路径限定为编译器包内不可变资产。

八帧诊断中，布局和字体各打开 9 → 1 次。CPU 0 的非插桩实测各预热 1 对、交替测 5 对，所有配对的标准输出、PNG 和约定的 MP4 均逐字节一致。已有 C++20 入口与常驻版的中位数在八帧无视频时为 2.457 → 2.504 s，在三帧加视频时为 1.465 → 1.449 s。八帧样例回退到已有方案，常驻版继续作为显式试验。见 `artifacts/baseline/cpu-native-plot-residency-acceptance.json`。

`project-controller-argv` 从原参数声明和 F4 调用接出 7 个选项、8 个输入。省略结束时间保留整数默认值，显式给值按浮点解析，内部日志频率来自被调用函数的原默认值。`export-native-stateful-entry --argv-plan` 将解析编入同一个 C++20 进程。已验收的普通入口可直接使用原参数，适用范围包括短选项附带值、长选项缩写、重复选项覆盖和未知选项丢弃。`--demo` 仍待降低。

搬迁后的原生包使用和原 Python 入口相同的参数。CPU 0 上各预热 1 对、交替测 3 对，两组全链路分别是 2.297 → 1.466 s、2.310 → 1.474 s，约 **1.57×**。所有配对的标准输出、三帧 PNG 和 MP4 逐字节一致。另一次相对目录、默认不编码视频的路径生成了两帧，与原程序逐字节一致。见 `artifacts/baseline/cpu-native-public-argv-acceptance.json`。

同一导出事务内可以复用已验证的更新图和更新计划。使用文件版本及缺失路径守卫，发布前重核完整依赖内容。CPU 5 上两次相同源更新校验耗时 74.03 → 39.50 s，实际回放 2 → 1 次，输出摘要一致。这是编译校验耗时。三帧诊断还记录到字体和布局文件各打开 4 次，成本和可外提范围待下一轮验证。

`project-controller-plot` 用原 F4 调用、字段和顺序收据绑定快照。数值场在当前控制器数组，坐标在启动资产，标题读当前时间。文件名的序号来自独立持久字段，写帧成功后加一。控制器复用原进程创建及等待合同，输出目录明确传入。该计划限定为已验证的单图分支。

`export-native-live-plot` 将标题、等值线、栅格化和编码接成进程内 C++20 提供者。带绘图计划的 `export-native-stateful-entry` 从原同步调用直接传递借用数组，发布 PNG、标准输出及 FFmpeg 视频。搬迁后验证两组输入，分别执行 3 步与 5 步。两组的三帧、视频、标准输出、最终数组及标量状态均逐字节匹配。稀疏保存覆盖了快照序号和仿真步号不同的情形。

CPU 0 各预热 1 对、交替测量 3 对。两组整入口中位数分别为 2.301 → 1.467 s 和 2.309 → 1.471 s，均约 **1.57×**。计时包括启动和 FFmpeg 完成，终态调试文件仅在独立诊断轮导出。FFmpeg stderr 单独保存，日志含路径及进度时序。当前入口使用带类型槽位，原始 argparse 参数映射及其它分支仍待接通，默认选择维持 `NO_OP`。见 `artifacts/baseline/cpu-native-live-application-acceptance.json`。

`capture-agg-text --with-contour-style` 从同一次原始运行把输入数组、等值定位、归一化、调色表、格式化标量及字体接到原 AGG 槽。源码位置按原 CodeID 和调用端口复验。中端独立核对两处颜色字段、路径表和标题绘制区间，再将这些区间从模板移除。模板只留当前适用范围的版面和绘制状态，运行时检查坐标数组摘要。

C++20 从初始数组、一次和两次原生更新后的状态计算了 50、48、45 个填充区间，标题实际读取对应的时间标量。三帧共享同一份布局模板，完整 AGG、RGBA 和 PNG 与原程序逐字节一致。第三帧单独作为组合验证。标题试验使用版本匹配的 FreeType 和不同小版本的 HarfBuzz，资格限于已测单字体 ASCII 路径。当前仍由分开的原生组件完成，整入口保持 `NO_OP`，完整成本未测。见 `artifacts/baseline/cpu-live-plot-layout-acceptance.json`。

`capture-agg-scene --contours` 接通了原始三路数组视图、ContourPy 的填充区间、逐带路径和 AGG 的实际调用槽。两帧的输入均为 257×257，填充区间数分别为 50 和 48。`analyze-agg-scene-layout` 重放有序绘制字段，记录两帧间的变化。坐标轴和静态文字的一致性只覆盖所测场景。

`plan-contour-scene` 验证原路径表的位区间和逐带次序。`export-native-contour-scene` 复用上游 ContourPy MPL2014 C++，两帧的顶点和路径代码与原提供者逐字节一致。`export-native-contour-agg` 从场景模板移除对应路径表，并在运行时由数值数组重新生成。空环境中，从构建期数组和一步 C++ 更新后的数组分别重算了两帧，最终 AGG 绘制记录、RGBA 和 PNG 均与原程序逐字节一致。

当前模板仍接收原流程确定的等值层、颜色、版面和字形。不同场景需要各自的相应输入。全应用默认 `NO_OP`，完整时序成本未知。见 `artifacts/baseline/cpu-contour-live-acceptance.json`。

`project-controller-startup-slice` 已闭合 9 个控制器输入的原始值来源。所选构造分支上，`F` 和 `kappa` 独立流入各自的标量字段。`plan-controller-startup` 将默认值和选定数值提供者接入构建期物化，隔离执行原构造器两次。原先观测的数组用于校验摘要。4 个独立数组根生成 2,146,592 字节的启动映像，其中 `u/v` 接到原生更新端口，`x/y` 保留为辅助数据。变化的参数继续由调用方提供。

`capture-agg-scene` 采集原 AGG 绘制操作及像素参考。`export-native-agg-scene` 复用原 AGG 的 C++ 栅格化实现。两个不同的实测场景用同一可执行文件重新生成 RGBA，随后编码为 PNG，两层输出均逐字节一致。场景的布局、等值线几何和字形位图仍由原流程提供。

组合入口使用 `--startup-plan` 与已有子进程计划。搬迁后用两组运行时速率分别完成一次更新，初始数组由构建期资产初始化。两组更新后的数组内容不同，且各自的数组、标量、PNG 和 MP4 与原程序逐字节一致。绘制记录由试验夹具传入。整程序发布和 A/B 仍待场景生成闭合，规划保留 `NO_OP`。见 `artifacts/baseline/cpu-controller-startup-agg-acceptance.json`。

`project-controller-initialization` 已把 Gray–Scott 的实际构造和初始化写入接到 9 个控制器输入。两次零步运行在原入口各取一次状态，所观测到的字段变化仅有 `F`、`kappa`，四个数组逐字节一致。依赖独立性与数值算子合同仍待验证，初始化保持原路径。

`plan-controller-child` 从原 F4 和子进程证据绑定 8 个字面量、2 个运行时目录参数与 FFmpeg 可执行文件。`export-native-stateful-entry --controller-plan PLAN --child-plan CHILD` 从原调用点同步启动并等待。C++ 使用原继承配置和 `check=False` 规则，区分退出码、信号和启动失败。

`export-native-rgba-png` 复用 Pillow 编码源码和同版本 zlib，由运行时 RGBA 重新压缩。原始帧及更新后的留出帧都逐字节一致。CPU 0 上，原程序和组合入口的一次数值更新、PNG、MP4 及最终数值状态通过对照。本次组合由验证夹具提供原渲染器的 RGBA。原生栅格化、启动数值生成及同边界 A/B 仍待完成。宽 CPU 掩码实测会改变视频字节，整应用默认仍为 `NO_OP`。见 `artifacts/baseline/cpu-controller-media-acceptance.json`。

Gray–Scott 的输出路径继续分层接入。前端 `project-controller-effects` 将快照和渲染的实际调用接到原 F4 函数体，保留字段读写、原实参及默认值。单步 CPU 运行生成 PNG 和 MP4。`project-controller-children` 按同一次调用的作用域标记连接进程创建和等待，重放了 FFmpeg 的原始参数及退出状态 0。

`observe-controller-console` 在真实入口记录两次内置 `print`，并与程序返回后的单次 53 字节 stdout 写入核对。`plan-controller-console` 允许在显式独占、空的常规文件 stdout 上试验原生实现，整入口默认 `NO_OP`。`export-native-stateful-entry --controller-plan PLAN --console-plan CONSOLE` 保留原结束发布点，运行时需要 `--exclusive-stdout`。

搬迁后的 C++20 入口在空环境完成 75 步。实际 stdout、最终数组及标量与原程序逐字节一致。其余效果继续由同步提供者接收，本次用记录探针验证快照边界。原生渲染、初始化和完整应用 A/B 仍待闭合。见 `artifacts/baseline/cpu-controller-output-acceptance.json`。

CPU 原循环入口使用三份可独立重放的记录。前端 `project-native-controller` 从 F4 恢复共享循环、条件、动态步长与调用实参，循环体只存一份。中端 `plan-native-controller` 合并完整更新和同一次采集的效果顺序与资源约束。Gray–Scott 的 75 次完整更新及首帧写入可回接原调用端口。首帧位于第一次更新之前，字节校验只能作为验收依据。完整入口成本保留为 `UNKNOWN`，规划默认 `NO_OP`。

`export-native-stateful-entry --controller-plan PLAN` 生成 C++20 项目。数组保持在原生进程内存，步长字段由共享循环实时更新。三个原外部调用点按源条件同步分发，入口要求显式提供与控制计划标识匹配的 C ABI 效果库。最终数组及标量状态在成功返回后发布。发布帧、日志及视频的实际提供者需另行验证。

用 Gray–Scott 的真实数组对照原 `integrate`，在 75 步、末步缩短的 6 步和零步三个输入上验证搬迁后的可执行文件。最终数组、时间、步长、步数，以及回调的顺序、实参与当时的数组均逐字节匹配。身份错误的探针在回调和输出发布之前被拒绝。验收以探针代替外部发布，尚无该完整入口的 A/B。见 `artifacts/baseline/cpu-native-controller-acceptance.json`。

原 F4 图已接通一个完整的 Gray–Scott `update`。受布尔分支守卫约束的原路径依次执行数值区域、两次边界更新及标量返回。共享规划图核对八次有序复制，逻辑读取与写入各 16,576 B。完整步耗时仍未知，规划保留 `NO_OP`。

`export-native-stateful-entry --update-plan PLAN` 可生成独立 C++20 更新入口。在同一组常驻数组上执行显式步数，并按原返回表达式携带标量。用真实初始化数据，搬迁后的产物在空环境通过 1 步和 75 步验证，最终数组含边界单元、返回标量和输入文件均逐字节核对。原 `integrate` 的终止条件、步长调整和文件发布仍在交付边界外。见 `artifacts/baseline/cpu-standalone-update-acceptance.json`。

75 步的预备数组入口在 CPU 0 上预热 2 对、交替测量 7 对。原混合路径使用受守卫的原生数值区域和原 Python 边界更新，完整进程中位数为 661.893 ms。独立 C++ 为 115.206 ms，约 5.75×。计时包括启动、加载和输入输出文件。所有样本的两个数组和标量文件逐字节匹配，收益仅供预备数组入口评审。

Gray–Scott 的两次数组更新已在同一次调用中定位到一次原始 `while` 回边。`trace --control-code CODE_ID` 只采集所选代码的实际分支和跳转。冷重放把回边接到原 F4 端口，并核对两次 `update` 之间的外层路径。该路径未执行直接字段写入或其他 `CALL`。数组字段的区间内绑定、协议与回调效果仍独立保留。

相同 CPU 核上各预热 2 对、测量 7 对。75 步时，预加载后的整段原驱动中位数为 523.445 → 455.624 ms（1.15×），其中受守卫区域从 135.531 → 58.643 ms。75 步进入发布评审。留出的 6 步输入中，整段驱动时间区间重叠，保留原执行。`plan-entry-effects --paired-benchmark` 重算各自的判定。全图时长与调度资源需求仍未闭合。

独立 C++20 入口已执行真实 Gray–Scott 的一个数值步。导出重放已选 NumPy 原生算子的代码、重定位和库符号版本。产物移到新目录后在空环境运行，两个含 halo 的数组逐字节一致。准入限定在经过核对的 NumPy 二进制、签名和 CPU 条件。见 `artifacts/baseline/cpu-exact-loop-native-region-acceptance.json`。

`trace --stateful-allocation-epochs REGION` 将 NumPy 原分配器的成功分配、重分配和释放接到同一次调用采集。Gray–Scott 两个 `_euler` 检查点之间，两块实际拥有者分配保持同一分配代。两次 `update` 及各自三次子调用都回接原 F4 端口并核对正常返回。分配代与字段绑定分别保存。

`plan-entry-effects --loop-content` 将该区间和同一采集中的首帧发布顺序接到共享规划问题。重放会拒绝把数组区域移到首帧发布之前，完整时间与资源需求未定时保留 `NO_OP`。独立 C++20 入口也通过无回调数组区域的回归验收。见 `artifacts/baseline/cpu-allocation-effect-native-entry-acceptance.json`。

公共前端已用真实 Gray–Scott 验证源码调用与系统接口的接线。快照采集重放了 570 个区间，其中 479 个是 Python 调用，全部接回原 F4 端口。按同线程的原始标记顺序，将打开、写入 106,172 字节和关闭 PNG 的记录接到 `_dump` 调用的 Matplotlib `Figure.savefig`。PNG 与保留基线逐字节一致。进程尾部的 53 字节 stdout 写入仍保留为未归属效果。

`resource-scopes` 以实际帧退出或原生完成记录闭合区间。`buffered-stream-publications` 可查询同一文本流中可能被显式刷新发布的早期写入。独立小程序观察到两次源写入对应的一次 10 字节内核写入，字节来源与隐式刷新仍为开放项。采集耗时仅用于核验采集流程。见 `artifacts/baseline/cpu-source-effect-scope-acceptance.json`。

第二阶段通过 `time-schedule --resource-scopes DIR` 将首帧发布顺序接入同一份区域 recipe。观察到首帧调用闭合后才进入 75 次数组更新中的第一次，图保留一个数组区域。20 个内部任务和容量约束可以重放，完整成本仍为 `UNKNOWN`。第三阶段用同一 recipe 重建了 C++20 区域。在原驱动的四步试验中，四次原生接管均通过，逐步数组状态、最终 halo、日志和 PNG 逐字节匹配。默认执行仍保留原路径。这份验收的下一步是补齐连续分配和调用方效果，再校准同边界资源成本，最后交付独立 C++ 入口。

公共前端已接通所选正常调用路径的数组区域与循环回边。Gray–Scott 的本轮采集记录了两个连续数组入口的拥有者和相对几何，两份数组在这两个边界的观测结果一致。分析从 F4 恢复三次子调用的正常顺序与字段路径，得到 32 个访问、24 个读取。固定几何和效果条件下，32 个读取分片来自上一轮写入，16 个来自本轮写入。原地协议请求与实际写入资格分开处理。图还保留区间内分配连续性、未选分支和外部效果的证据缺口。见 `artifacts/baseline/cpu-source-storage-loop-acceptance.json`。

纯 CPU 公共前端的实参连线验收已通过。用同一 F0–F4 流程生成三种字符串调用的原图。普通字符串与无插值 f-string 都能从实参反向追到相同的常量内容，有插值时保留原变量、格式化和拼接。新查询同时核对调用记录与图中的值到实参边，调用位置由原字节码与源码绑定提供。该验收覆盖局部调用端口，跨函数和外部效果另行验证。

纯 CPU 稀疏扫描的准备外提已闭合。源图与两份历史输入验证同一份 302,112 字节 CSR 数值数据，与运行时迭代数无关。计划把拓扑、网格、Basis 与 COO 准备移到构建期。C++20 将数值数据嵌入只读段，按实参生成 56 字节头，经密封的 `memfd` 交给原六进程求解器，再按迭代顺序发布摘要。

原 Python、原生实时准备和原生预计算在相同 7 核组上一轮预热、三轮轮换测量。预计算相对原 Python 的中位数加速为 **12.88×（3 轮）、9.12×（37 轮）、7.80×（173 轮）**，相对原 C++ 为 **23.87×、8.92×、3.21×**。每个计数的输出逐字节一致。搬迁后在空环境和关闭 stdin 的条件下，留出计数 5 匹配原摘要，越界计数 513 在启动求解器前拒绝。选型证据只覆盖三个已测计数，其他计数沿用实时准备的默认计划。契约覆盖固定几何、固定数值库及成功摘要，原 Python 的导入和失败路径效果仍待证明。见 `artifacts/baseline/cpu-parameterized-sparse-prepared-acceptance.json`。

运行时稀疏扫描已接通跨进程数据边、配对成本和 C++20 入口。37 轮诊断观察到四段准备管道、302,168 字节 CSR、37 个有序求解结果及一次 65 字节摘要发布。诊断只记录顺序和字节，耗时来自独立的清洁进程运行。

在同一 7 核组、相同输入、数值库和摘要契约下，各计数跑一对预热和三对交替测量。37 轮的两/六工作进程中位数为 **1.157 / 1.007 秒**，173 轮为 **2.030 / 1.304 秒**。3 轮的区间重叠，保留六工作进程。C++20 包对这三个计数及其余准入计数均执行六进程，包内只携带采用的实现。已在空环境复验已测计数 3、留出计数 5 和越界计数 513。其他宽度与未测计数的最佳宽度仍待证据。见 `artifacts/baseline/cpu-parameterized-sparse-width-acceptance.json`。

scikit-fem 的稀疏扫描已能用同一个 C++20 程序接收运行时迭代数。源图把 `argv` 连到循环、缩放系数、求解结果和有序摘要。96/128 轮两个原始归档的网格与数值载荷一致。新计数进入 CSR 协议头及每轮缩放，网格准备保持不变，旧求解结果前缀无法复用。重放验证了具备 4,225 行、20,865 个非零元的 CSR 数值载荷在新信封中逐字节一致。

同一 7 核组上，直接原 Python 脚本与原生应用分别做一对预热、三对清洁测量。留出计数 37 的中位数为 **1.037 → 1.003 秒，1.03×**，173 为 **3.203 → 1.297 秒，2.47×**。两组 65 字节的 SHA-256 输出均逐字节一致。搬迁后还用 3 和 37 核对完整应用。当前计划固定一个控制进程、六个工作进程和六个在途额度，计数范围为 1–512。见 `artifacts/baseline/cpu-parameterized-sparse-entry-acceptance.json`。

复跑使用 `project-source-sparse-count`、`plan-native-parameterized-sparse-entry`、`export-native-parameterized-sparse-solver` 与 `export-native-parameterized-sparse-entry`。计算沿用 LibTorch CPU、SuperLU 和 OpenBLAS，摘要沿用 OpenSSL。

独立 C++20 入口已能从运行时命令构造新局面。源图绑定八类命令、两种 FEN 朝向、后续走子和半回合更新。计划单独跟踪接收者和模块评分表。局面构造读取当前表，搜索可重新选择表，新游戏释放走子和分数状态并保留当前局面及模块表。构建沿用已经核验的搜索和 PV 后端。

留出输入包含四场不同深度的搜索，逐条核对 42 次探测、四组历史和半回合计数、根状态及最终有序表。直接原脚本与 ELF 在同一逻辑核一对预热、四对清洁测量，中位数 **0.922 → 0.123 秒，7.50×**。同一个裸 ELF 搬迁后通过旧输入和留出输入。当前范围是按应答顺序发送的 `go depth 1–4`、已实现的位置命令及默认配置，异步停止、限时和选项变更仍在范围外。见 `artifacts/baseline/cpu-parameterized-recursive-entry-acceptance.json`。

复跑使用 `project-scalar-recursive-parameterized-entry`、`plan-scalar-recursive-parameterized-entry`、`export-scalar-recursive-parameterized-entry` 和 `scripts/measure_scalar_recursive_parameterized.py`。下一步把独立交付推进到另一类纯 CPU 工作负载，复用当前的输入端口、状态所有权和运行验证机制。

纯 CPU 已生成独立 C++20 Sunfish 入口。建图把 7 条有应答顺序的输入命令接到原控制分支、启动输出、位置准备和实际搜索的输入端口。规划在命令匹配后才执行对应效果。11 行启动回复和两个初始局面经原源码与观察证据固定，搜索分数、PV、走法及状态由原生算法实时计算。

固定输入版本的复跑入口为 `project-scalar-recursive-entry`、`observe-scalar-recursive-entry`、`plan-scalar-recursive-entry` 和 `export-scalar-recursive-entry`。包内包含 CMake、源码、头文件和宿主 ELF。前端分析发生在构建与复验，独立入口只依赖宿主 C/C++ 库。

直接运行原 Python 脚本和生成 ELF，固定同一逻辑核，一对预热、四对清洁测量的中位数为 **1.957 → 0.262 秒，7.45×**。观察轮逐条核对 39 次探测，两局历史、根位置、节点数及有序表一致。发布的 ELF 单独搬迁并在空环境通过，未知命令返回 77，已发布的合法前缀保留。见 `artifacts/baseline/cpu-standalone-recursive-entry-acceptance.json`。

Sunfish 的原图已接通两次搜索之间的状态边。原 UCI 在第一次关闭后执行 6 次走子表查找和 6 次属性检查。第二次搜索继续使用同一份 714 条走子记录，在首个探测前换掉旧分数表。已经观察到的按键读取可直接走 C ABI。其他消费路径和外部别名仍是准入边界。

C++20 v2 在一次原生分配中执行两场搜索。第一次关闭保留有序走子表，第二次原初始化后原子重绑并清空原生分数表。最后一个关闭点回写仍可见的状态。四对无插桩运行的整进程中位数 **1.990 → 1.684 秒，1.18×**，基线为上一版每局回写。独立观察核对 39 次探测、两局最终字典摘要和 6 次跨搜索查找。发布快照累计从 6,563,979 B 降到 361,965 B。见 `artifacts/baseline/cpu-recursive-owner-continuity-acceptance.json`。

驻留桥接的复跑入口为 `project-scalar-recursive-owner-continuity`、`observe-scalar-recursive-owner-continuity`、`plan-scalar-recursive-owner`、`export-scalar-recursive-driver --rebind` 与 `scripts/measure_scalar_recursive_owner.py`。

纯 CPU 主链已扩到 Sunfish 的整个迭代搜索驱动。建图把原 UCI 调用、共享的循环体、递归探测、暂停/恢复和关闭接在一起。两次深度 4 搜索共观察 39 次探测，含两次关闭前的深度 5 探测。每局使用新的分数表，第二局继续使用走子表。

规划把状态的原生所有权覆盖到一次生成器的完整生命周期。C++20 共享库在同一状态空间执行多次探测，通过 C ABI 返回原驱动约定的记录。原 UCI 仍负责输入、输出和停止控制，生成器关闭时回写字典。独立 C 调用方验证了状态快照、暂停恢复和硬截止检查。该路径保留原 Python UCI。生产部署、任意局面和跨搜索原生驻留仍待验证。

相同两局在逻辑 CPU 7 上各跑一对观察预热、四对交替清洁测量。整进程中位数 **3.248 → 1.990 秒，1.63×**。原输出除 `time` 和 `nps` 外一致。观察轮逐条核对 39 次探测、节点数和两局最终有序字典摘要。详情见 `artifacts/baseline/cpu-recursive-driver-acceptance.json`。

复跑入口依次为 `observe-scalar-recursive-driver`、`project-scalar-recursive-driver`、`plan-scalar-recursive-driver` 与 `export-scalar-recursive-driver`。`run-scalar-recursive-driver-study --project-ui` 绑定已观察到的外层消费端口。`scripts/measure_scalar_recursive_driver.py` 采集独立进程 A/B，规划器用 `--plan` 和 `--benchmark` 发布验收，`--check` 独立重算。

Sunfish 的整个递归函数已生成独立 C++20 入口。原程序采样验证了三个递归调用点，以及同一字典实例上的查询、写入和淘汰。原生产物读取类型化输入，保留节点计数、按插入顺序更新的字典和时钟检查顺序。六个用例的返回值与完整共享状态逐字段匹配，其中两个取自原驱动真实根调用，还验证了已有状态、容量 32 的两个有序字典和停止时的状态前缀。

两组深度 6 递归体各预热一对，交替测四对，每臂每轮运行三次。中位数分别为 **220.40 → 30.19 ms** 和 **252.80 → 33.93 ms**，约 **7.30×** 和 **7.45×**。这组历史计时使用已准备的输入，边界外包含输入准备、状态输出与 UCI 驱动。见 `artifacts/baseline/cpu-recursive-body-native-acceptance.json`。

Sunfish 的排序、逐轮解包、走子到递归实参已连成一条可重放的数据链。排序和延迟走子共用一份 C++20 评分实现和冻结表。两个深度 2 局面中，3,315 次走子复用同轮评分，完整返回状态逐项匹配原走子。未匹配排序凭据的 19 次走子使用原路径。

深度 4 的三臂配对各预热一次、清洁测量四轮，中位数为原搜索 3.123 s、仅替换排序 2.885 s、排序加走子 4.344 s。计时字段以外的原输出一致。观察轮复用 25,093 次评分，走子守卫及 POD 交接仍有开销。成本阶段选择仅排序。前端已将下一层的三个递归调用点、五个共享状态成员和外部时钟调用建图。有一条递归边具备真实值交接证据，另外两条的动态目标与效果仍需核实。见 `artifacts/baseline/cpu-sorted-transition-and-recursion-boundary.json`。

Sunfish 的原排序调用已通过 C ABI 接到生成的 C++20 区域。守卫保留原工厂创建和外层迭代，在调用前确认两个生成器、原生数据端口和实际评分表。CPython 字典变更通知仅更新计数，相关绑定变化后重新验证。两个深度 4 局面各预热一次、清洁配对四轮，整入口中位数 **3.120 s → 2.885 s**，约 **1.08×**。观察轮接管 23,017 次排序，搜索输出除计时字段外逐字节一致。另一次深度 2 对照在同一进程修改评分阈值，2,995 次排序都与原表达式逐条比对通过。当前仍是隔离试验，完整递归搜索由原程序控制。

有限元排程按 worker 数、窗口和核放置绑定实测成本。窗口 2 时提名 2 worker，6 worker 被支配。其余 18 个配置的成本保持未知。相同预构建 CSR 的三对清洁求解测量中位数为 2/2 的 **631 ms** 和 6/2 的 **812 ms**，96 次求解的最终输出逐字节一致。默认调度保持原样。见 `artifacts/baseline/cpu-guarded-sort-and-observed-schedule.json`。

Sunfish 已将原排序表达式的生成、评分、过滤和排序合成独立 C++20 区域。沿读依赖将状态输入从 8 个标量端口缩到 5 个。原搜索两种表版本共采到 16 次区域调用、247 条记录。原生程序搬迁后验证全部记录的值与顺序。固定输入的配对测量各预热一次、测五轮，原表达式和 C++ 中位数分别为 59.54 µs 和 5.69 µs，约 10.46×。计时使用预备好的区域输入。

前端已证明评分记录的定义沿原始绑定和解包到达选定的走子点。两条空记录路径被原条件排除，生成器挂起及递归间的实时表版本仍待守卫。固定大小的 C ABI 在独立 C 调用中复验 32 个完整评分和 32 个归档评分传递样本。预备输入下，原生评分的单次调用由 1,345 ns 降到 642 ns。现行搜索路径保持启用。

有限元窗口为 2 时，6 worker 比 2 worker 多出的诊断循环时间为 253.0–258.3 ms，其中最重的有序求解链多用 251.8 ms。每个 worker 扣除一次求解后，CPU 服务均值范围仍分离。新的冷读收据保留求解与信用依赖，当前证据无法确定服务时间变化的原因。见 `artifacts/baseline/cpu-eager-sort-and-credit-cost.json`。

Sunfish 的评分与走子可导出为 C++20 共享库。隔离搜索对照把该库接到原始递归调用点。两个深度 4 局面在观察轮完成 25,289 次原生走子，保留最佳着法、搜索分数和节点输出。各预热一次、清洁测量三轮，原路径和跨 ABI 路径中位数分别为 3.023 s 和 4.782 s，保留原路径。插桩轮中参数打包与返回重建合计约 42 µs/次，FFI 连同库执行约 7.5 µs/次。原图验证了 32 次可运输的重复评分及四条同根同键的字典读写边，先前评分替换重算在 32 个完整状态上通过，实时值来源与版本守卫仍待补齐。

有限元的 6 worker、窗口 2 已由独立原生求解器处理相同 96 个请求。实际峰值在途请求为 2，全部按序完成，输出逐字节一致。插桩循环区间约 795–798 ms，已有 2/2 记录为 540–542 ms。两种配置的 owner 在途内存预算相同，6/2 的累计 worker 地址空间限额多 4 GiB，默认等宽策略继续使用。见 `artifacts/baseline/cpu-native-search-study-and-independent-window.json`。

`compile-scalar-transition-scope` 已把完整评分、走子和视角旋转合成独立 C++20。两次真实深度 2 搜索各保留 16 条链路；原图记录评分返回、旋转前后的六字段状态和递归消费。简化图把六处字符串切片和拼接改为单次构造。原生工件在搬迁和空环境下逐项匹配 32 条完整返回状态，受控负索引复现了原切片行为，溢出拒绝且输出为空。编译采用一次进程内图重放，同一证据的单次构建从 65.24 s 降到 35.70 s，生成源码、二进制和样本输出逐字节相同。完整搜索的状态写入和整场收益仍待验证。

有限元有序排程现可分别推导 worker 数及在途窗口。只提名当前可交付的 6/6，信用链条件下界 172.778 ms，同一输入的独立诊断范围为 189.018–191.951 ms。6 worker 配 2 个在途请求仍处于模型范围，在同一假设下与 2/2 耗时估计相同，累计 worker 地址空间限额多 4 GiB。验收记录见 `artifacts/baseline/cpu-state-transition-and-window-planning.json`。

`compile-scalar-score-scope` 已能把 Sunfish 的完整评分函数生成独立 C++20。一次原程序执行了两个深度 2 局面，同一个表对象出现两个内容版本。48 个完整评分和原分支记录通过冷读，五个条件的真分支有真实样本，第六个真分支通过独立的受控对照。简化图消除两处重复行查找，并在改写键后保留读取。原生工件按内容版本分派，12 个行引用共享 7 份只读数据。同一输入在两个版本分别得到 26 和 15，未知版本拒绝。完整搜索的状态更新和收益仍待验证。

有限元 96 轮新增四 worker 诊断，与已有两、六 worker 使用同一份求解输入。单次求解均值依次为 10.965、12.627、11.458 ms。两点线性外推与四 worker 后平坦的假设均扩大了六 worker 独立验证的误差，现行提名策略继续使用。验收记录见 `artifacts/baseline/cpu-score-versions-and-width-validation.json`。

Sunfish 已在深度 2 的一次原程序执行中接通八条链路，从表项差值到走子合计，再到递归形参。原图保留四处可能写表的位置。对采集到的一个 6×120 表内容版本，`compile-scalar-table-scope` 把两次行查找合并为一次，并保留两次索引读取。独立 C++20 工件在空环境和搬迁后均通过八个样本，越界输入拒绝。验收范围是这一评分表达式，整搜索交付与收益仍待验证。

有序 CPU 排程现分开推导发送端服务、结果可见时刻、接收端服务和按序等待。新模型接入 `compile-cpu-entry`，有限元 80 轮在原程序、两 worker、六 worker 三组各预热一次并测三轮，中位数为 1.734 s、1.445 s、1.107 s，stdout 逐字节一致，六 worker 约 1.57×。96 轮的独立诊断显示新模型仍低估六 worker 循环约 8–11 ms。下一步按内容版本扩大 Sunfish 可安全降低的范围，同时用独立证据收紧跨宽度求解成本。见 `artifacts/baseline/cpu-scalar-scope-and-transport.json`。

`compile-cpu-entry` 的有序 CPU 路径现从一个两 worker 原生诊断推导全部 2–6 worker 的循环耗时。按实际 CPU、NUMA 和内存约束提名六 worker，构建校准与提名两种宽度。六 worker 循环估计 179.9 ms，另一次诊断覆盖区间合计 189.0 ms。整入口清洁中位数为 Python 1.999 s、两 worker 1.536 s、六 worker 1.146 s，stdout 逐字节一致。原生诊断通过独立描述符记录，按序等待与载荷接收各有观测。

RK4 16 步完成 1 对预热、3 对清洁测量，Python 与 C++ 中位数为 6.735 s 和 2.351 s，终态文件逐位匹配，两套 stdout 各按自身格式验证。Sunfish 深度 2 的真实运行验证了 8 次返回对象到递归形参的直接传递，连接 5 条原始 F4 数据边。下一步补上 Sunfish 该区域的内容与效果沿边证明，并用原生诊断继续收紧跨宽度的成本估计。见 `artifacts/baseline/cpu-ordered-width-and-source-handoff.json`。

`scar compile-cpu-entry --provider A --provider B --entry SOURCE --out OUT --argv ...` 已用同一命令编译有限元和固定网格 RK4 两条 CPU 入口。SCAR 依据源码路径、字节和提供方合同选择唯一入口，`--check OUT/result.json` 可按原合同冷复验。

有限元 96 轮再次从一次原程序采集重建了 182 条动态调用绑定。它沿用经核验的静态图，单独离线验收的绑定与发布为 1.199 s。准备、求解和有序发布的 288 个区间已汇总到同一份源图。清洁测量的 Python 原入口和六 worker C++20 中位数为 1.981 s 和 1.125 s，stdout 逐字节一致，约 1.76×。

RK4 复用已验 RHS，重新构建递推图、规划 41 MB 内的三核排程、生成并搬迁 C++20 包。16 步终态与原程序逐位一致。两步原生比较中，串行和三核中位数分别为 0.914 s 和 0.585 s。Python 状态行和原生 ABI 元数据分开验收。下一步补采原生求解、进程间搬运与按序等待的成本，让成本图提名 worker 宽度，并继续扩大源码入口覆盖。见 `artifacts/baseline/cpu-entry-native-providers.json`。

`scar compile-triangle-profile --family FAMILY --entry ENTRY --argv 96 --out OUT` 已跑通已登记的 scikit-fem CPU 入口。前端在同一原程序进程中采集一次，接通 182 条调用观测和 10 个选定帧，入口图包含 324 条类型化依赖边。首次离线构图因并行编辑触发输入校验，随后通过 `--resume` 用同一份采集完成构图。F4 重放耗时 96.8 s，其中调用目标准入占 88.9 s。

排程层按实际 CPU 核、NUMA 和内存约束提名 2 和 6 worker，保留有序发布。C++20 工件与原入口在 CPU 49–55 各预热一次、交替测三轮，中位数分别为原入口 1.987 s、2 worker 1.539 s、6 worker 1.128 s，stdout 逐字节一致。SCAR 推荐 6 worker，约 1.76×。结果已冷读复验，工件搬迁、安装后在关闭 stdin 的最小环境中通过执行。下一步将已知调用目标的图投影与新运行事件分开复用，取得逐段成本证据，再以另一条 CPU 入口验证提供方接口。当前结论限于已登记的提供方、入口和参数，完整 Python 可见效果仍待闭合。见 `artifacts/baseline/cpu-triangle-profile-auto-entry.json`。

`scar compile-triangle-profile` 已把已准入的源码证据、CPU 排程、C++20 生成和整入口对照接到同一命令。`128` 轮有独立的原程序调用和输出证据，五组静态初始数据与 `512` 轮逐字段一致。两种工作线程配置共享一次上游编译和同一数值输入。CPU 49–55 上各预热一次并交替测三轮，Python 原入口、2 worker、6 worker 的中位数分别为 2.500 s、1.738 s、1.216 s，均逐字节匹配。选中 6 worker，约 2.06×。工件安装后在关闭 stdin 的空环境中实跑通过。当前命令仍以已验证的三角形提供方和源码证据为输入。见 `artifacts/baseline/cpu-triangle-profile-second-count.json`。

固定参数 `512` 的 scikit-fem 入口已能把从源码推导的初始网格和参考单元数据编入 C++20 工件。推导复验期间禁止导入数值库和目标包，生成的 472 字节输入与原运行记录逐字节一致。安装后关闭 stdin 仍能输出相同的 65 字节结果。入口效果验收涵盖这组参数和输出，日志及导入效果仍待证明。

紧凑 Basis 协议实际少传输 589,740 字节，并保持网格到摘要的结果一致。CPU 49–55 各预热一次后交替运行四轮，原入口、稠密工件、紧凑工件的中位数分别为 8.631 s、2.081 s、2.066 s。两种 C++ 布局的样本范围重叠，默认保留稠密。见 `artifacts/baseline/cpu-source-sealed-triangle-profile.json`。

固定 512 轮的 scikit-fem CPU 路径已从 472 字节初始网格和参考单元数据跑通到最终输出。C++20 原生执行六轮网格细化、边界提取、Basis、装配、求解和摘要发布。网格、Basis 和最终输出逐字节匹配，搬迁及安装后实跑通过。同一 CPU 组上的四对清洁运行中位数为原入口 8.643 s、工件 1.921 s，约 4.50×。

前端目录扩展到 14 个源码单元，普通入口投影覆盖 10 个条件可达单元和 324 条类型化边。55 个执行帧补充了构造和对象流的证据。成本图保留网格的两个消费者，并验证 Basis 的值可用 72 字节和零步幅视图表示，当前交付仍使用原传输路径。下一步验证源配置的自动生成、原入口效果和紧凑传输。见 `artifacts/baseline/cpu-triangle-seed-to-publication.json`。

scikit-fem 真实 CPU 入口图新增 `LinearForm` 源码单元，共 9 个单元、38 个条件调用接口和 312 条类型化边。另一次同源运行用 8 个局部入口和返回事件核实三条数据流、六个字段。成本收据标注了父子包含关系，同一主线程的互斥采集区间合计 32.526 ms。

C++20 的交付入口已上移到 Basis 数组和约束行。原生 ATen 组合按明确的归约次序逐位重产 73,728 项矩阵贡献和 24,576 项载荷贡献。随后原生执行 512 轮求解，并发布与原入口一致的 65 字节输出。共用同一份下游库后，拆分与统一入口的四对配对中位数为 1.470 s 和 1.474 s，样本范围重叠。网格和 Basis 的构建、原入口的可见效果，以及同边界完整原入口性能仍待闭合。见 `artifacts/baseline/cpu-basis-to-publication.json`。

纯 CPU scikit-fem 的真实 512 轮入口图已扩展到 8 个源码单元、33 个条件调用接口和 267 条类型化边。定向采集命中 112 个调用位点，记录耗时 9.143 s。阶段 II 复用同一份分片摘要并将 13 个原入口计时位点绑定到新图。三个上游生产者已定位，其独立耗时和完整的数据返回关系仍待验证。

从原入口捕获的 COO 矩阵、载荷及约束行可直接交给独立 C++20 包。包内复用上游稀疏例程，转换后的 302,168 字节 CSR 和最终 65 字节输出均与原程序一致。安装后和搬迁均已实跑。相同 COO 输入的四对清洁测量中，拆分启动和统一入口中位数分别为 984.0 ms 和 990.3 ms。原入口仍负责生成 COO 数据和约束行。见 `artifacts/baseline/cpu-return-producers-and-selected-publication.json`。

定向调用回放已接入正式建图。真实 scikit-fem 入口保留 97 条观测、28 个条件接口和 231 条类型化边，旧 F4 归档、稀疏入口投影及成本收据均可读回。17 个共享函数体的依赖汇总以 22 个分片保留，磁盘占用从 88.7 MB 降至 9.7 MB，冷重放结果一致。独立 C++20 已从规范 CSR 输出与原入口逐字节一致的 SHA-256 行。清洁同合同五对测量的中位数为拆分路径 981.350 ms、一体路径 992.148 ms，决策保留拆分。网格装配和原入口动态输出效果仍待闭合。见 `artifacts/baseline/cpu-focused-entry-and-digest-publication.json`。

scikit-fem 的模块入口已把四个实参接到循环形参，将循环返回的数据和状态边接到实际采集的 `print` 调用。三个上游源码单元已通过本地 F4。按原调用点和生产者 CodeID 发起的新采集命中 97 个位置，隔离的定向回放接通了 28 个条件调用接口。六 worker 的三次插桩运行核对了 1,536 次原生任务与有序发布，调用方原准备合计中位数为 685.8 ms。C++20 独立交付现从原始规范 CSR、右端项和约束行开始，计算对角槽和每轮标量，512 个结果逐位一致。四对清洁测量的分离版与一体化版中位数为 1.0246 s 和 1.0109 s，样本范围重叠。原入口还负责网格装配、边界选择和可见发布。见 `artifacts/baseline/cpu-entry-dependence-and-source-csr.json`。

同一份纯 CPU RK4 采集的 F4 整图装配从 439.7 s 降到 336.8 s。解析后的七张语义表在对齐读取规则版本摘要后完全一致，旧归档仍可冷读。scikit-fem 的已发布计划把六步原始归一化和调用方效果接入成本图，两段独立启动实测合计 697.8 ms，其余实际成本保留未知。独立 C++20 稀疏包从一份准备快照和 512 张票据重建请求，输入由 102,873,156 B 降至 311,384 B，512 份结果逐位一致。原入口仍负责网格装配与可见输出，守卫运行含 1 次原调用、511 次替换和 0 次回退。整图的对象内容与效果依赖，以及从网格装配到输出的完整 C++20 交付仍待闭合。见 `artifacts/baseline/cpu-indexed-assembly-and-compact-stage.json`。

RK4 绑定阶段已复用静态图分片。同一采集的 21 个单元重放得到 316 个固定点和 32,845 个绑定版本，所有表逐字节或逐记录相同。墙钟从两进程的 1,664 s 降为六进程的 227 s。整图已接入新绑定产物，749 次已采集函数体进入均匹配。现行 scikit-fem 原入口的六 worker 方案从串行原生的 6.934 s 降至 2.726 s。同一成本模型在 2 至 13 个 worker 中估计六 worker 耗时最低。独立 C++20 包直接由发布计划生成，迁移后 512 个结果与原入口逐位一致。见 `artifacts/baseline/cpu-inventory-and-ordered-closure.json`。

纯 CPU 新验收覆盖 RK4 和 scikit-fem 有限元扫描。RK4 的 749 次已采集函数体进入全部匹配源码图，接入 32,845 个绑定版本和 486 条资源记录。28 个形参入口已关联到 36 个实际 ATen 操作数。36 MB 图持有张量预算下，生命周期排程把单核存储上界从 37,748,744 B 降到 35,389,448 B，C++20 程序的 16 步终态逐位一致。scikit-fem 从现行 F4 重新准入串行 SuperLU，原入口中位耗时 8.691→6.940 s，正式入口的 512 次求解均走原生实现。历史有序排程的独立 C++20 包已复验 512 个结果逐位一致。整程序运行路径、对象内容和效果，以及现行源域的并行稀疏排程仍待闭合。见 `artifacts/baseline/cpu-observed-f4-and-sparse.json`。

已从一次新鲜的纯 CPU 原入口采集贯穿支持的 RK4 路径。静态前端通过 25 层验收，动态 JOIN 将 `dynamics` 的 4 次调用绑定到 8 个原生算子。41 MB 下选择三核，四步原生入口中位耗时 1.515→0.830 s。C++20 运行包移动后从包内加载 LibTorch，16 步终态逐位一致。整程序 F4 的路径、状态和效果关系仍有未闭合项。见 `artifacts/baseline/native-cpu-end-to-end.json`。

官方 CPU-only LibTorch 2.13.0 已完成固定网格 RK4 和导出推理交付。两份 C++20 产物通过完整 ELF 依赖准入和冷读重放，RK4 16 步终态与推理 5 个输出均逐位一致。严格模式拒绝带 CUDA Runtime 和 CUPTI 的旧构建。41 MB 预算下的 CPU-only RK4 四步新进程中位数为单核 1.515 s、三核 0.855 s。见 `artifacts/baseline/cpu-only-libtorch-delivery.json`。

纯 CPU 固定网格递推已接通张量生命周期排程。真实 RK4 的三核存储上界由 42,467,336 B 降到 40,108,040 B，在 41 MB 预算下获选。38 MB 下保留单核，两种 C++20 产物的 16 步终态都与原 torchdiffeq 逐位一致。41 MB 下的新进程四步配对中位数为单核 1.531 s、三核 0.789 s。见 `artifacts/baseline/cpu-recurrence-lifetime-order.json`。

实测放置按固定依赖和队列传播算子样本区间，异构路径补入同步 CPU 调用先于后续 GPU 提交的因果边。真实 CPU 归档的模型区间为 2.523–2.652 ms，点估计 2.551 ms，图持有的存储上界 200 B。三个 CUDA 方案仍缺实测，保留 CPU。见 `artifacts/baseline/native-placement-service-interval.json`。

同一份硬件配置现可生成任一容量合格的 CUDA 放置标定产物。真实导出图的 2 个备选放置通过 C++20 构建及独立重放。3 个容量合格的 CUDA 放置都已取得无设备收据，并在同一次排程中复验，当前保留全 CPU。见 `artifacts/baseline/native-placement-treatment.json`。

CPU/CUDA 放置已接入按方案匹配的实测成本。真实导出模型的全 CPU 产物提供 6 个原生算子服务样本集合，5 个输出与 PyTorch 逐位一致。8 种配置中 4 种被声明容量排除，3 种缺实测成本，其中 1 种已确认本机没有可用 CUDA 设备。离线重放保留全 CPU。见 `artifacts/baseline/native-empirical-placement.json`。

CPU/CUDA 放置已按合格算子从源图统一提名。真实导出的 3 个矩阵乘法生成 8 种配置，在声明的容量下 4 种被排除，模型提名 2 个 CPU 矩阵乘法和 1 个 CUDA 矩阵乘法。候选连同活动采集入口已生成 C++20 并通过重放。进一步收紧声明容量会保留全 CPU。无 GPU 实机样本时部署默认保持 CPU。见 `artifacts/baseline/native-device-placement-selection.json`。

混合原生产物已接通可选活动采集与规划器成本重放。在真实导出图上验证了 13 个活动索引、源算子、物化字节数和事件等待绑定。C++20 调用可分别记录 CUDA 事件和主机单调时钟，普通试跑不分配计时事件。本机无设备，实测项为零，13 项活动成本保持 `UNKNOWN`，诊断计划未产生完工时间。见 `artifacts/baseline/native-device-activity-calibration.json`。

纯 CPU 与混合包已接入同一配对验收。实测混合用例通过源图、权重、五个输出端口及二进制身份的准入。CUDA 探测返回 77，决定保留 CPU，双方均未进入计时。离线重放复现该决策。修改已锁定的二进制身份、重复返回的参考或采用结论均无法通过验收。见 `artifacts/baseline/native-device-assessment.json`。

混合 `torch.export` 已接通设备分区、共享搬运、C++20 交付与离线重放。真实用例识别出两个 GPU 区域，统一计划含三个 cuBLAS 调用、三个 CPU ATen 调用、五次上传和两次下载。五个输出的 CPU 原生参考均与 PyTorch 逐位一致。混合产物通过编译链接，设备探测仍无可用 GPU，尚未实测混合执行正确性与性能。见 `artifacts/baseline/typed-hybrid-export-delivery.json`。

真实 `torch.export` 的三个非方阵乘法已从类型化源图生成 CUDA C++20 候选。源图共享的四个外部值各上传一次，四个返回端口共用三次下载，内部中间值由设备上的下游算子直接读取。独立验收重放了源图、物理计划及 C++；同一输入的 CPU 原生产物四个输出均与 PyTorch 逐位一致。本机探测无可用 GPU，CUDA 实机结果、耗时及自动采用仍未验证。见 `artifacts/baseline/export-cuda-source-delivery.json`。

CPU 推理现可按形状选用不同排程。一个动态 `torch.export` 模型交付为单个 C++20 程序，大形状采用同 NUMA 三核，小形状保留单核。大形状原生入口中位耗时为 1.261 s → 0.819 s，两个形状的输出均逐位一致。CUDA 侧已将 H2D、cuBLAS FP32 SGEMM 和 D2H 放进同一份物理计划，编译链接通过，实机尚缺可用设备。见 `artifacts/baseline/cpu-ready-and-cublas.json`。

单个 C++20 推理包已可按名称分派批次 2、4、8，三个形状分别通过成本、存储、生成代码与双输出逐位验收。按正常完成合同自动删除两次无消费者的算子调用，原图保留。固定步长求解包可用内置源码重新推导求解边界，RK4 16 步结果仍逐位一致。单 CUDA 设备的 pinned host 与 device 搬运计划已生成并链接 C++，实机缺少可用设备。各验证范围见 `artifacts/baseline/native-shape-and-copy-replay.json`。

CPU 部署计划已接通算子标定、容量约束、C++20 生成及冷读验收。RK4 在 64 MiB 预算下采用三核，四步独立原生进程的中位耗时为 1.507 s → 0.788 s。40,000,000 B 预算下使用串行 C++。两种产物及自动推导求解参数的产物均通过 16 步原程序终态逐位验证。单形状 CPU 推理也通过标定计划及逐位验证。预算只计算图持有的 Tensor 存储。GPU/NPU 的实机排程和交付仍待验证。见 `artifacts/baseline/cpu-schedule-replay.json`。

固定网格 RK4 与 CPU `torch.export` 已共用程序图、物理问题、CMake 工程及冷读验收。RK4 的 16 步终态与原程序逐位一致。推理模型在批次 2、4、8 分别构建并通过同一验收，批次 4 还通过重新构建与安装路径复验。已验的 SuperLU/OpenBLAS 原生稀疏包由同一 CMake 生成器重建，512 个结果均一致。源 F4 收据仍需更新。GPU 经主机到 NPU 的两跳搬运已验证模型容量与时序，实机后端及通用 Python 入口仍未完成。见 `artifacts/baseline/layered-native.json`。

2026-11-19 `compile-aten-entry` 默认交付独立 C++20 区域。真实 RK4 的八个算子从发现记录生成原生程序，空环境运行和进程跟踪均确认无 Python 进程或链接依赖。缺少并行合同时另行生成串行原生程序，两条路径的输出均与原 RHS 逐字节一致。联合排程能改变核内次序，和现有就绪调度的八对实测为 24.09/24.12 ms，未过采用门槛，交付保留原序 C++ 就绪调度。此处已覆盖 `dynamics` 区域，完整 RK4 驱动仍待单独降低。见 `artifacts/baseline/aten-joint-native.json`。

2026-11-19 已接通跨调用 NUMA 只读副本的内容校验、失效、容量与热命中时间表。两组固定输入 RK4 区域各测八对，直接读取与保留副本分别为 44.25/45.21 ms 和 438.82/433.31 ms。结果逐字节一致，两个方案均低于采用门槛。已编译验证 CPython 桥接，未在完整入口启用保留副本。见 `artifacts/baseline/aten-ready-retained.json`。

2026-11-19 跨 NUMA 就绪调度可为复用的只读输入加入一次显式复制。图证书计入目标核执行、同一逻辑值的两种物理存储和副本容量。两个 RK4 用例均核对了本地页、位级输出与绑定。独立配对中，矩阵区域直接读取 38.70 ms、复制 40.42 ms，高复用加法区域分别为 432.22、451.70 ms。两例均保留直接读取。高复用用例的完整入口由原程序 5.750 s 降至直接就绪调度 4.835 s，三轮逐字节一致，并且零回退。见 `artifacts/baseline/aten-ready-input-replica.json`。

2026-11-19 跨 NUMA 就绪调度已接通页位置、原数据边和对应算子成本。8 步 RK4 的四组候选中，最佳跨节点候选的独立估值约 25.25 ms，按实际页位置和固定核分配重放后改为 30.92 ms。规划选同节点 14/15/16，完整入口 4.702→3.392 s。只提供跨节点核组时选 14/0/15，完整入口 4.704→3.824 s。两条路径均做了四轮配对，每轮 32 次原生调用、零回退，结果逐字节一致。见 `artifacts/baseline/aten-ready-numa.json`。

2026-11-19 就绪调度已改为按完整数据图准入并校准各核上的全部合格算子。16 步 RK4 用例的 8 个算子中，首批就绪位置为 0、2、4，原分组只能排相邻双核窗口。原程序、分组、就绪的整入口中位耗时分别为 7.002、5.638、4.188 s。就绪调度获选，完整进程 1.67×。四轮实测中两个候选每次各执行 64 次原生调用，零回退，所有结果逐字节一致。见 `artifacts/baseline/aten-fullgraph-ready.json`。

2026-11-19 完整入口选型已接入 `compile-aten-entry --compare-ready`。24 步大张量加法的同场中位耗时为原程序 11.761 s、分组 9.086 s、就绪 11.319 s。分组选中，完整进程 1.29×。4 轮实测中两个候选每轮各执行 96 次原生调用，零回退，所有输出逐字节一致。保留成本提名、区域验证、完整入口原始记录及可独立重放的决策。见 `artifacts/baseline/aten-entry-candidates.json`。

2026-11-19 就绪调度 ABI 2 已接入按最后一个实际消费者释放张量。RK4 矩阵区域的张量存储上界由 51.75 降至 45 MiB，能通过 48 MiB 预算，32 步完整入口中位耗时 18.970→14.653 s。大张量加法区域上界由 225 降至 200 MiB，原生区域比原分组快 1.16×。完整入口同场对比时原分组为 9.172 s，就绪调度为 10.901 s，因此该入口保留分组。两组验证输出均逐字节一致，零回退。见 `artifacts/baseline/aten-ready-lifetime.json`。

2026-11-19 已将原 Tensor 依赖接通三核就绪调度。真实 `torchdiffeq` RHS 恢复 8 个算子与 7 条内部数据边，C++20 后继在自己的输入就绪后即可调用 LibTorch。配对区域相比串行 1.45×，相比原分组调度 1.12×。32 步原入口进程中位耗时 18.927→14.746 s，3 对实测逐字节一致，每次 128 次原生调用且零回退。大张量逐元素加法路径实测慢于原分组，因此拒绝换用就绪调度。见 `artifacts/baseline/aten-ready-dag.json`。

2026-11-19 已从 Tensor 依赖追踪并测量汇合后的串行消费者。100 MiB 输入算例的点估计为 33.69 ms，同样本仅采用独立消费者耗时的估计为 56.38 ms。按顺序平衡后，原生复测 AB 为 54.88 ms、BA 为 44.99 ms。分段诊断显示顺序差主要位于并行组。规划保留两种校准的观测范围。24 步 ODE 完整进程约 1.28×，96 次原生调用、零回退、结果逐字节一致。最终重放的执行二进制与输入守卫一致。见 `artifacts/baseline/aten-team-post-join.json`。

2026-11-19 实测并发组耗时已接入时间表。64 步矩阵 ODE 选同节点三核，完整进程约 1.87×，256 次原生调用。100 MiB 输入的逐元素加法 ODE 进行 24 步，完整进程约 1.29×，96 次原生调用。两例零回退且结果逐字节一致。独立放置和宽度复测的顺序与当前估值一致。较大 Tensor 样例的整体估值仍高于复测约 11 ms。见 `artifacts/baseline/aten-team-group-timing.json`。

2026-11-19 三核组已支持跨 NUMA 的页证据和条件化成本。八个代表组中的初始跨节点首选 CPU 0/1/14 通过 64 步 ODE 验证，完整进程 1.75×。独立交替放置复测中，同节点组为 15.09 ms，跨节点组为 16.99 ms，顺序与初始点预测相反。已加入采样范围重叠时的同节点选择。新一轮自动选 CPU 14/15/16，完整进程 1.86×。两次各 256 次原生调用、零回退、结果逐字节一致。见 `artifacts/baseline/aten-team-cross-numa.json`。

2026-11-19 三核组已接入源码发现、成本规划和 C++20 守卫交付。64 步 `torchdiffeq` 的三个独立矩阵乘法在 CPU 14/15/16 上运行，算子区域相对串行实测 2.51×。完整进程中位数从 11.983 s 降到 6.486 s，256 次原生调用、零回退、结果逐字节一致。见 `artifacts/baseline/aten-three-cpu.json`。

2026-11-19 联合 NUMA 校准已支持有依赖的多个窗口。真实九算子图重放七个组合，并识别后续 worker 对前一窗口结果的读取。新采集的六算子 `torchdiffeq` 联合 RHS 在四个拓扑代表中选择 CPU 14/15 和两个窗口。48 步完整进程中位数由 11.734 s 降到 7.514 s，192 次原生调用、零回退，结果逐字节一致。见 `artifacts/baseline/aten-joint-numa.json`。

2026-11-18 NUMA 页位置已接入 ATen 并行窗口成本。双向跨节点校准均观测到 worker 结果的采样页转到另一节点，规划器沿图依赖替换对应 caller 消费成本。选择仍是 CPU 14/15。96 步 ODE 的完整进程中位数为 12.189→8.139 s，384 次原生调用、零回退、结果逐字节一致。见 `artifacts/baseline/aten-numa-locality.json`。

2026-11-18 较宽 CPU 域已接通 `compile-aten-entry`。真实 56 逻辑 CPU 目标按 NUMA、缓存和 ISA 提名四个代表对，利用现成 ATen 算子的两侧服务时间与交接成本选定 CPU 14/15。该对上原生区域实测 1.94×，96 步 ODE 的配对进程中位数从 12.337 s 降至 8.132 s。每个守卫进程执行 384 次原生调用、零回退、结果逐字节一致。见 `artifacts/baseline/aten-broad-cpu.json`。

2026-11-18 `compile-aten-entry` 把已验证的发现报告接到目标 CPU、范围内成本规划、C++20 交付及完整入口 A/B。96 步 ODE 自动恢复 CPU 4/5。原生区域实测 1.72×。完整进程中位数从 12.182 s 降到 8.079 s，约 1.51×。3 次实测的守卫进程各执行 384 次原生调用，零回退，输出逐字节一致。见 `artifacts/baseline/aten-entry-delivery.json`。

2026-11-18 ATen 区域发现从真实 ODE 采集中识别出 `dynamics` 的四次激活并生成闭合配方。C++20 产物运行两组状态输入，均与原函数逐字节一致。计算入口由 LibTorch 提供。含旧签名和 `out=` 写入的采集保留原执行。见 `artifacts/baseline/aten-region-discovery.json`。

2026-11-18 真实 torchdiffeq 0.2.5 RK4 CPU 入口通过了需求预测和完整进程验证。96 步矩阵 ODE 的原路径观察到 384 次 RHS，独立演化状态的热调用为 1.86×，A/B 前预测进程内为 1.58×。1 对预热、3 对实测的进程中位耗时从 12.164 s 降至 8.062 s，约 1.51×。每次守卫运行的 384 次 RHS 均进入 C++20 LibTorch，最终状态逐字节一致。求解器控制流沿用原实现。见 `artifacts/baseline/aten-ode-entry.json`。

2026-11-18 原 Python ATen 试验已支持固定几何下的输入值变化。构建时用已安装的 ATen Meta 核对四个操作的形状和存储，并要求显式的无可见副作用重跑合同。运行时核对输入及中间结果，参考输入读取为 0 B。稠密随机值与稀疏值模式两组独立输入均逐字节一致，热函数分别为 1.32× 和 1.34×。完整新进程在稠密组仍 `KEEP_ORIGINAL`。见 `artifacts/baseline/aten-variable-input.json`。

2026-11-18 本地 LibTorch 库身份复用已在同一 CPU 入口验证。守卫进程重用构建时 472,419,299 B 的内容校验，运行时重新散列 0 B。独立诊断中，完整校验为 1.375 s，身份复用为 3.438 ms，过期单库身份触发对应内容重验。相同负载的 7 对新进程实测中位数为 2.034→2.056 s，决策仍 `KEEP_ORIGINAL`，输出逐字节一致。见 `artifacts/baseline/aten-provider-identity.json`。

2026-11-18 已验证同一 PyTorch 用例的原 `main()` 新进程。守卫路径每进程实际执行一次双核，返回值、梯度模式、标准输出和结果文件一致。2 对预热、7 对实测的进程中位数为 2.036→3.441 s，决策 `KEEP_ORIGINAL`。独立诊断中核验约 472 MB 的已加载 LibTorch 文件字节耗时中位数为 1.373 s。见 `artifacts/baseline/aten-python-entry.json`。

2026-11-18 真实 PyTorch 采集的四算子函数已接回原模块做显式试验。核对同一 CodeID 后，C++20 扩展借用原 Tensor 并运行已验证的双核计划。7 对预热、31 对实测的完整函数调用中位耗时从 22.164 ms 降至 18.044 ms。输出逐字节一致，关键字、梯度和输入内容变化均走原函数，退出恢复绑定和调用线程亲和性。计时包含守卫与输入比较，范围为该热函数。见 `artifacts/baseline/aten-python-trial.json`。

2026-11-18 生成的 CPU 图已接入 Tensor 最后使用点与分段存储预算。新四算子 PyTorch 采集生成同一个 C++20 图，估算图持有的 Tensor 后备存储峰值为串行 9,437,184 B、双核 10,616,832 B。10,000,000 B 预算保留串行，11,000,000 B 预算选择双核，两者输出与原 Python 逐字节一致。宽预算下预热后的调用中位耗时从 22.246 ms 降至 16.771 ms，31 对均更快。预算未含算子内部工作区和分配器保留。见 `artifacts/baseline/aten-tensor-budget.json`。

2026-11-18 多窗口 CPU 时序已接通真实采集。新 PyTorch 用例恢复 9 个算子、18 个精确 Tensor 输入端口和 8 条依赖。独立校准后，规划器保留 2×2 窗口串行，选择两个 768×768 窗口并逐段验证核心占用。生成的 C++20 调用现有 LibTorch，三个返回值均与原 Python 逐字节一致。7 对预热、31 对实测的中位耗时为 49.599→24.890 ms。测量范围为预热后的独立原生图调用。见 `artifacts/baseline/aten-native-multi-window.json`。

2026-11-18 生成的 ATen CPU 图已接通双核时序和 C++20 执行。规划器根据已声明的算子效果、图依赖、两个不同物理核心和独立测得的服务时间计算方案，再验证整次调用。2×2 保留串行，768×768 的独立声明用例选双核，最终匹配试验中位耗时从 24.961 ms 降至 14.459 ms，31 对均更快。测试只计预热后的原生调用。见 `artifacts/baseline/aten-native-schedule.json`。

2026-11-18 已从同次 CPU 采集自动生成独立原生图。`project-aten-graph` 核对 6 个 Tensor 输入、2 条内部数据边、1 个原返回分量及外部算子链接来源。三次算子的 C++20 输出与原 Python 返回的第 0 项逐字节一致。不同采集的同摘要链接报告及 `add.out` 写入路径均被拒绝。原入口仍 `KEEP_ORIGINAL`。见 `artifacts/baseline/aten-capture-graph.json`。

2026-11-18 `compile-aten-graph` 已从声明的数据依赖直接生成 C++20 和原生输入输出入口。分叉汇合、双输出链、标量及空张量共四例与 PyTorch CPU 参考逐字节一致。运行只使用现有 LibTorch 算子和本机原生依赖。当前交付按声明顺序执行，原 Python 自动替换、硬件重排及性能提升仍待验证。见 `artifacts/baseline/aten-native-graph.json`。

2026-11-18 独立 ATen 的实测选型已接 C++20 交付。冷重放核对原样本、成本、硬件和资源计划，再以显式 ABI1 适配器编译固定模式。2×2 产物在 CPU4 串行，768×768 产物在 CPU4/5 执行，两者与原串行区域逐位一致。错误启动掩码和环境在调用前被拒绝。交付验证范围为已声明的独立原生调用。见 `artifacts/baseline/aten-boundary-delivery.json`。

2026-11-18 CPU4/5 上已做同边界 A/B。独立 LibTorch 调用各取 31 对，2×2 中位数为 5.560→25.582 µs，保留串行。768×768 为 24.869→12.978 ms，规划器选择双核，资源验证通过。完整 Python 函数边界另测 41 对，包含原 `empty` 续行，768×768 为 25.008→12.188 ms，逐位核验并通过回退对照。两组成本各自保留。见 `artifacts/baseline/aten-boundary-a-b.json`。

2026-11-18 ATen 规划已接入具名的包装器入口、生成解析声明和效果要求。新 CPU 采集核对 4 个 CALL，区分进入同一 `add.Tensor` 的现代和弃用签名，保留 `out=` 的写入端口与返回别名，以及 `requires_grad` 的包装器实参。两组规划均保留原执行。独立的 CPU4/5 采集和目标库存采用同一可用掩码，逐调用资源需求和完整成本仍待验证。见 `artifacts/baseline/aten-wrapper-contracts.json`。

2026-11-18 `plan-aten-capture` 已通过同次运行的原始 CALL 索引把 F4 和 ATen 接到共同规划 IR。真实 CPU 用例有 4 个 CALL、3 条保序约束和 2 条源码引用与 ETO 描述相符的路径，另核对两个工厂 CALL 各含 4 个直接原生算子。两组均保留原执行，效果、就绪、资源和成本待补。新采集另以进程映射及 ELF 字节核实 4 个包装器入口，仍需证明生成声明与二进制的构建关联。见 `artifacts/baseline/f4-aten-planning.json`。

2026-11-18 生产采集已接通 CPython 原生 CALL 到同次运行的 ETO/Kineto。单线程 CPU 用例核对 17 个调用范围及其中 16 条直接算子事件。同一循环位置的两次 `mm` 分别保有事件身份，一次异常保留 `C_RAISE`。按源码范围采集后，标记由 85 减到 17，两个采集均保留 16 条直接算子事件。`link-aten-capture` 从 6 个调用事件生成 3 个 LibTorch 入口，独立 C++20 数值逐位一致。F4 中端对 PyTorch 仍返回 `STATE_PROVIDER_CALL`，当前缺少外部算子的端口和效果准入。见 `artifacts/baseline/production-torch-call-scope.json`。

2026-11-18 C++20 后端已接入现有 LibTorch 调用接口的校验和生成。用 CPU0 单线程真实 PyTorch 同次采集识别出两个不同源码位置的 `mm`，并跟踪它们的结果对象进入 `add` 的两个端口。独立 C++20 验证通过生成接口调用 `mm`、`add.Tensor`、`empty.memory_format`，数值逐位一致，确认链接至原生 LibTorch。本地链接的资格与原 Python 替换许可分别保存。见 `artifacts/baseline/aten-provider-linkage.json`。

2026-11-18 LeWM CEM 的 487 条调用协议前沿均已接回 50 个共享源码接口。v5 沿同帧原字节码核实了 26 个接口上的 336 次正常续行，独立冷验核对 52 条条件结果和状态边。原采集缺少同次原生调用记录，返回内容、设备完成和副作用仍待证明。本轮未运行 LeWM。见 `artifacts/baseline/lewm-observed-normal-continuations.json`。

2026-11-18 LeWM CEM 的 v3 图以原调用、前驱返回和帧形参重放了一个隐式接收者布局。`solve → init_action_distrib` 有三条数据参数路由，附带帧槽隔离条件。缺少前驱证据时保留零或一前缀两种布局，现有 C++ 降级仍拒绝隐式接收者。单元构建并同进程复验耗时 64.83 s，独立冷读 63.99 s，旧 v2 冷读为 74.05 s。542 条观测前沿仍在。见 `artifacts/baseline/lewm-observed-positional-receiver.json`。

2026-11-18 用已保存的真实 GPU 采集与当前源码重建了 LeWM CEM 单元。静态图含 70 个调用，v2 回放确认 `solve` 实际进入 `init_action_distrib`，参数绑定仍待补齐。冷读复验一致，当前还有 542 条观测前沿。本机 CUDA 设备数为 0，本轮没有新推理或加速测量。见 `artifacts/baseline/lewm-observed-v2-review.json`。

2026-11-18 稀疏整入口的代价评审已接到发布入口。对照前固定了实现、资源和采样器。三组交替中，已发布并行、紧凑 C++、紧凑加补集消除的中位数分别为 2.820、2.868、2.890 s，三个路径输出一致。选择器保留已发布方案，独立启动复验完成 512 次求解。见 `artifacts/baseline/sparse-entry-review.json`。

2026-11-18 有限元入口的原 F4 调用边和源码用法证明固定返回槽无人读取。显式计划在首次调用后省去 511 次整数补集构造，保留原归一化、求解状态与发布。将编译移到离线后，准入中位数由 58.7 降至 31.1 ms。整入口三组交替的紧凑 C++、补集消除、既有 Python 并行中位数为 2.842、2.883、2.842 s，进程输出一致。该消除继续保留为试验。见 `artifacts/baseline/dead-result-item.json`。

2026-11-18 紧凑 C++20 工作进程接入原有限元入口。原 Python 继续准备输入并发布结果，首次求解使用原载体核对，后续 511 次经精确字节校验后返回工作进程结果。原始入口、既有 Python 并行、紧凑 C++ 的三组交替中位数为 8.540、2.825、2.852 s，输出及退出码一致。逐请求 `memcmp` 减少了比较成本，紧凑路径仍作为显式试验。见 `artifacts/baseline/native-detached-entry.json`。

2026-11-18 有序稀疏导出新增 `--detached`。从原有限元入口捕获一次固定输入，worker 用 OpenBLAS `DOMATCOPY` 和 SuperLU 还原 512 轮数值步骤，所有结果逐位一致。输入由 102,873,156 B 减到 311,384 B。相同的 8 个 worker、各 1 次预热和 3 次交替实测均约 0.916 s，显式紧凑模式未取得可辨识的耗时收益。计时范围为独立 C++ 输入到输出，原应用准备和效果验证另计。见 `artifacts/baseline/native-ordered-detached.json`。

2026-11-18 `export-native-ordered-sparse` 从配方生成可直接运行的 C++20 程序，调度和传输由 SCAR 生成，求解调用原生 SuperLU/OpenBLAS。移动产物后，用原有限元入口实际生成的 512 组矩阵值及 RHS 逐项对照，512 个结果按序逐位一致。错误配方、截断输入和非有限数均验证了拒绝及发布边界。范围为已归一化的 CSR 请求，流式边界尚无性能结论。见 `artifacts/baseline/native-ordered-stream.json`。

2026-11-18 `export-native-ordered-sparse` 把有序执行配方编译成独立 C++20 调度，重用打包的 SuperLU/OpenBLAS。8 个进程实测各自使用配方指定的物理 CPU 与地址空间限制。512 次请求按序发布；两份重复输入的结果摘要均匹配原始记录，注入失败时只发布之前的成功前缀。独立 C++ 批次的 3 对中位数为 6.447→0.894 秒。数据范围为两份已封存输入交替运行，原 Python 入口的发布资格仍单独校验。见 `artifacts/baseline/native-ordered-sparse-cpp.json`。

2026-11-18 稀疏区域的 C++20 交付可独立调用已有 SuperLU 与 OpenBLAS。`export-native-sparse` 从同一 F4 执行计划冷重放，保存原生 ELF 的可迁移依赖闭包及 ABI 1 绑定。移动后由独立 C++ 可执行程序验证两组数值的逐位结果及结构守卫，实际加载路径均在包内。原 Python 入口、成本发布仍各有自己的合同。见 `artifacts/baseline/independent-native-provider-export.json`。

2026-11-18 分层审查用独立输入补了源码定义身份与内置绑定验证、后继图冷回放、NumPy 提供方和当前浮点策略读法、运行中的解释器身份、实际 `spawn` 引导。两子进程的独立复现逐位一致，8 次原生和 16 次原路径调用均符合预期。此前的 Cirq 22 量子位、8 进程对照在 CPU16–24 下为 1,114.743→1,069.885 秒，试验组命中 42,064 次，应用报告结果一致。该单对 A/B 使用双方显式浮点策略和先前提交 `923ec891` 的交付链。源证据版本已更新，默认继续 `KEEP`。报告见 `artifacts/baseline/partition-spawn-and-scalar-mainline.json`。

2026-11-18 `run-partition-workers` 已验证能在正常 CPython `spawn` 子进程绑定同一份 C++20 产物，并通过真实父子 PID 和源码 CodeID 记录执行证据。Cirq 12 量子位的 2 个进程在默认 NumPy 浮点策略下回退 288 次，在双方使用显式试验策略时执行 264 次原生调用、回退 24 次，报告的数值结果一致。父进程校验一次 Python 可执行文件，匹配的子进程验证运行中文件身份。实测子绑定由约 125 ms 降至约 3 ms。Sunfish 的通用标量表达式已经从 F4 证明编译到原入口，固定深度 6 的整入口三对中位数为 3.897→4.239 秒，走法、节点数和分数一致。默认决策继续为 `KEEP`。见 `artifacts/baseline/partition-spawn-and-scalar-mainline.json`。

2026-11-18 三个 CPU 用例的真实入口继续推进。Cirq 中的整张量复制由通用源码规则识别，原图与条件后继图使用同一组边界端口。后继图仍保留对角乘、逐项乘加、循环携带的输出内容和原输出对象。`plan-partition-copy` 支持冷重放，`compile-partition-trial` 生成 C++20 CPython 扩展，经原导入绑定进入真实入口。17 量子位单核试验的两轮整入口为 3.233→3.333 秒，8 轮为 6.788→6.642 秒，逐轮完整状态的字节摘要一致。试验双方采用同一 NumPy 浮点策略。默认决策仍为 `KEEP`，8 进程的正式负载尚待验收。Sunfish 由当前 F4 识别到同一表达式内两次读取同一表行，4 个棋面的 78 次等值验证通过。逐次 Python 守卫的成本过高，保留原搜索。见 `artifacts/baseline/partition-copy-and-scalar-reads.json`。

2026-11-18 数值合同与结构化简已接通。旧规划在最近舍入时把 `1.0 + 2⁻⁵³` 折成字面值，向上舍入时旧 native 静默差 1 ULP。现只静态折叠有界的整数负号、加、减、乘。浮点常量算术在 C++20 的调用方浮点环境保护内执行。四种舍入模式逐位验证通过，空数组上的标量除零进入回退。五调用的源码管线已发布融合后继图及重新推导的 use-def、静态末次使用和逻辑需求。8×8 输入的逻辑分配从 7 次降到 1 次，逻辑读写量从 8,192 B 降到 1,536 B，保持 448 次算术的操作数顺序。独立冷读验证通过，篡改需求后重新计算哈希仍被拒绝。当前执行决策为 `KEEP`，成本资格仍待闭合。另已统一 holdout 采集与验证的收益公式，之前 46.8% 的预测误差仍超出阈值。见 `artifacts/baseline/strict-fp-and-structural-fusion.json`。

2026-11-18 完整入口成本已做新实测并拒绝：新增明确区分compiler adapter/compact、trial、owner与selection的成功路径协议；原ELF在canonical输入上完整调用2warm+7sample，均值919245ns。随后冻结原20%误差/5%收益政策及新模型，真实原入口2warm+7组AB/BA得单方法median(episode means)1967329→1727909.25ns（1.13856×）；预测误差46.8002%，不获资格。全部配对状态/文本/配置输出一致，36次native/0fallback。首调用与后续调用分布明显不同；scalar类型均float，不能归因整数tag或具体cache/NUMA分量，也不删除首调用。另暴露future验收器两种gain公式的浮点末位差异，该SCAR缺陷与模型拒绝独立。数据/模型/阈值原样保存；下一层修该验收缺陷，并按GCC方法优先补数值合同→实际图改写→重分析，不继续靠标定等待所有结构工作。报告：`artifacts/baseline/matched-entry-cost-mainline.json`。

2026-11-18 原native任务图已进入共同规划器：`plan-controlled-region` schema4保留整调用问题，并发布可展开的内部problem/projection/plan。真实科学区域的20任务、12物化、6槽全部保留；9阶段联合观测只作一个计费组，对应12个规划操作，原执行工作未减少。219份来源见证重接为18条外部前驱；原72时间约束保留，42条新增关系绑定资源/就绪/覆盖顺序。20任务保caller参与，9阶段有原CPU0参与合同，commit/alloc/release不冒用该绑定；3,203,376B固定payload只计一次。独立冷读通过；无group观测入口另验20操作、全成本UNKNOWN；删除依赖后重算hash及重复计费均拒。prepare70.83s/check70.10s、峰值约406/399MiB，均为离线分析；禁止目标执行、编译及native加载。12个成本frontier保UNRESOLVED，完整调用与实际可执行时序仍未闭合，无新加速比。材料：`artifacts/baseline/controlled-native-task-mainline.json`。

2026-11-17 跨函数effects已接现有COMPOSITE主链：共享callee摘要保逐CALL ABI/multiplicity，区分caller输入与已产中间值读取，provenance不重复计读，父子摘要不是可累加流量。原namespace/signature接口绑定3源unit、5共享函数、4 CALL/4 guards；独立schema保组合正文不等于新Python函数。Source window→callee effects/binding→movement→实际optimized C++ placement已贯通，C0/导出均冷重推；两种caller facade拒无证书micro，exact-copy既有标量发射已补。一次真实编译/导出后，5次native一致（含未改普通入口1次、stdout/stderr及结果bytes SHA一致），输入变化/输出污染/只读alias/零负trip/深层callee变更回退通过；全重算hash的假callee次数被source/native gate拒绝。direct原F4/optimized发射另以禁native metadata回归通过。3项定向metadata检查及1项既有轻guard检查通过，无新计时/加速比或后端；本例为编译器回归，不冒充真实CPU科学套件或LeWM。下一层转到真实科学区域同边界成本及硬件条件，任意callee/完整Python效果与三阶段通用覆盖仍未完成。材料：`artifacts/baseline/composite-effect-mainline.json`。

2026-11-17 源码移动证明已实际接入既有外提C++主链：原F4 loop窗口保匹配CALL/RETURN、零trip/回边及正常路径条件；严格样例28操作、15条原DATA传递边，沿5条边追到唯一只读来源，四项移动条件均CONDITIONAL。源/优化后DAG分别核身份，实际n2计算发射至循环前；source-motion不再是未消费旁证。C++20/ABI1真实4次native逐位一致，输入变化/输出污染/只读别名/零负trip/改callee回退通过；完整冷读及全重算ID的假读取摘要均被正确核验/拒绝。真实Gray–Scott自动得到2循环/9调用窗口，7个条件ONE_PER_ENTERED_BODY、2个存在绕过路径，均不展开迭代；此为静态控制事实，不是pure/reuse或实测执行次数。3项定向metadata测试通过；未测性能，无新后端/加速比。仅既有array-carry域闭环，跨一般callee的effects/alias组合、完整成本及通用硬件时序仍未完成。材料：`artifacts/baseline/motion-legality-mainline.json`。

2026-11-17 已收尾作用域内材料复用和源读写投影：真实科学入口prepare及独立check各开2个不同ProgramBuild session，entry推导、source witness、native准入各1次；原摘要展开内容SHA不变。logical FactRef经原resolver核physical/catalog链，未改原引用。54表达式/26动作有18访问、RAW7/WAR23/WAW1，80来源关联全接既有semantic/native阶段；原序与物理alias UNKNOWN保留，未授权移动。3项定向metadata测试及真实冷验通过，无目标执行/新编译。带计数的prepare68.20s/check68.15s不是A/B或目标性能。复核GCC13.3源码、实际O2/O3选项与21份旧收据后，下一层集中打通移动窗口→alias/no-clobber→must-execute/安全提前求值，再重建后继图成本和资源时序；不新增后端。完整三阶段仍未完成，完整调用成本UNKNOWN/默认NO_OP。材料：`artifacts/baseline/scoped-admission-mainline.json`。

2026-11-17 跨函数条件MAY摘要已接真实Gray–Scott入口：复用原F4与有界目标观察，8共享body/49CALL/9条件目标，formal/environment/world分开，CALL输出切边后经原routes替换，SCC有限集合固定点不展开递归；保control selector、receiver/key地址依赖及未知callee已知输入。来源SHA/CodeID+原body/point/definition绑定实际native，54表达式+26动作的80个来源投影全部接原F4和11native stages（投影非新增执行节点），两次stateful写保留。CLI及共同Problem接入schema3；真实独立冷验通过，无target/native/编译，whole-call成本仍UNKNOWN/NO_OP。依赖/unknown事实按ID共享，报告44.25→13.71MB，展开事实哈希完全一致；3项metadata验证通过。完整三阶段仍未完成，下一层补可用于移动证明的精确effect/alias摘要和同session重复冷准入复用。材料：`artifacts/baseline/source-summary-mainline.json`。

2026-11-17 共享循环移动分析已贯通现有三阶段主链：`array_motion`绑定原body/interface版本，统一依赖、structural invariant及可移动placement；source F4重发现适配接入`invariant_build`，CPU计划、匿名成本需求及串/并行C++均消费同一规则，优化后DAG独立重推并拒绝旧分析。3项metadata验证通过；按当前规则重发既有generic样例ProgramBuild后，原F4→计划→C++20/ABI1实际编译和冷load通过，4次native逐位一致，输入变化、输出污染、只读别名、零/负trip无native及callee变更回退均通过；自洽伪造source route被冷重放拒绝。旧材料不改，未测性能或升级部署资格。当前仍为严格array-carry域，通用MemorySSA移动及递归调用摘要尚缺，三阶段未全面完成。材料：`artifacts/baseline/unified-motion-mainline.json`。

2026-11-17 入口→native→阶段组上下文已接通：有界原driver捕获51目标，条件入口投影为2unit/10接口/8共享body/66原typed路由，源码SHA+CodeID自动见证实际native源。原303条主要为lazy视图/预算，非303个语义失败。新observer保全部原source字节及真实前后阶段，canonical两数组与原native逐位一致；2warm/7样本body组mean0.540264ms、预包装FFI mean0.760542ms（非A/B/整方法）。7阶段GCC模型字段匹配、2未知；独立冷读、两错误来源/映射拒绝及4项回归通过。来源和实测component已入共同Problem，但整方法cost仍UNKNOWN，默认NO_OP；全三阶段未完成。材料：`artifacts/baseline/entry-native-frame-context.json`。

2026-11-17 GCC反馈已接真实科学区域：实际C++的9个非COMMIT阶段关联原action/expr、GCC诊断及共同Problem；raw/投影用gzip保留，冷check不调用compiler/native。成本用途门禁已贯通：旧1,423,927ns在显式声明分析选1，在要求合格时间时保数值并加5项未解、选0；两条prepare+独立check均通过。完整函数与孤立标定的两个主计算stage都已向量化，但GCC相对准备/尾部成本不同；不能作为ns归因或加速比。三阶段仍未完成。材料：`artifacts/baseline/gcc-lowering-feedback.json`。

2026-11-17 离线分析已去除同一冷事务内的重复硬件解码/拓扑与entry cost推导。冻结代码四新进程ABBA，各模式2样本、warmup0，median42.3199→26.3717s（1.60475×），7份模型正文完全相同；命中/退出仍核文件，新事务独立冷验，原严格codec未改。真实stateful无成本/旧声明及fusion声明三条prepare+check均通过，成本未知和旧预测未合格状态不变。16项定向回归通过；未运行目标、新编译或加载native。这是分析器收益，非目标加速；三阶段仍未完成。材料：`artifacts/baseline/replay-mainline.json`。

2026-11-17 真实科学stateful区域已自动进入既有SemanticGraph：7输入、54表达式、26有序动作；24个evaluate与Expr共operation，无重复执行节点。实际C++的11 stages、私有slots与CPU阶段引用由原source_action_ids/expression_stages接通；两次write各对应private write+commit，原动作仍26。caller world exit原F4证据有歧义，保持字段frontier，未消除全程序gaps。`plan-controlled-region`按单activation可控资源合同保严格全入口与可控域两份problem；真实259的计入payload3203376B（borrowed1073296+private2130080）受33554432B预算，CPU0只覆盖私有执行阶段，caller/commit不指定core。无成本不补0；显式旧DECLARED profile仅作条件分析，不能升级时间/物理/部署资格。canonical guard两配置的唯一新采样已完成，入口配对差中位7085ns，非旧预测误差归因或新加速比。三阶段仍未完成。材料：`artifacts/baseline/controlled-scientific-region.json`。

2026-11-17 后继图→需求→硬件/时间Problem→既有C++验证候选已接`plan-semantic-region`。F4五调用样例实际after只有1个composite；原7次/3584B逻辑分配变为1次/512B，原eager、raw body与优化lowering分别记账。原完整hardware snapshot与两typed物理义务保留：strict UNRESOLVED；显式logical声明模型可构造1ms计划，该值不是实测。C++20/ABI1实际编译/构建时ABI admission有记录；独立冷检查禁止CDLL通过，默认安装保原函数且不加载，显式独占输入验证下3次native与1次dtype回退均逐位一致并恢复绑定。分析记录排序不稳定已修。成本兼容检查另机械证出canonical/真实harness的trial配置不同，记录级未知仍保留；future v2采样协议和通用无文件验证已接CLI，6反例拒绝，未执行新holdout/未改旧预测。三阶段仍未完成；下一层集中补真实科学后继区域、匹配上下文成本及可控资源资格。材料：`artifacts/baseline/semantic-successor-mainline.json`。

2026-11-17 GCC O3方法继续落实到真实图改写：复核本机13.3的O2/O3选项及既有GIMPLE/LICM/alias-vectorization材料；不以重复试跑寻找版本。`macro-pipeline`在原F4冷重放后，把选中OIR实际应用到已有SemanticGraph，5个CALL子节点变1个composite（含root共6→2定义，slots17→5），保公共边界、Q及UNKNOWN effects。新节点可追溯原图并展开实际融合body；重算use-def/源码顺序/静态last-use，拒绝旧图分析、错误region及错误body。KEEP逐字保图。完整prepare+独立check分别约16.65s/16.24s，均禁CDLL；REWRITE只用明确声明的结构验收模型，不是实测成本或部署资格。未新增后端/加速比，未重写全程序；下一层是从后继图重建需求、成本、资源时序和交付绑定。材料：`artifacts/baseline/semantic-region-revision.json`。

2026-11-17 完整方法成本出口已实现：`model-stateful-activity-cost`冷重放现有activity/resource/scope账本，只支持已证明串行的standalone方法，24个最终subject覆盖31活动各计一次；旧coarse条目不再累加，collective pool不按slot乘，唯一source milestone为0。保留COMPONENT_ONLY与全部转用残差，原whole_call_cost_closed=false不改；新条件DECLARED点预测1,423,927ns已接共同CostEstimate和实现IR，逻辑规划可用，物理资格仍独立未解。删掉V2 DAG证据会因3个UNKNOWN拒绝发布；禁CDLL冷重放及3项相关回归通过。

冻结模型后，真实原driver默认259规格做warmup2对/7对AB–BA，每对4次方法调用；计时内无hash/复制/trace，保留caller原0–55 CPU mask。原方法episode均值的median1.97499775ms[min1.8855935,max2.3318435,var0.0213078030ms²]，native1.05520175ms[min0.9159615,max1.62088875,var0.0575503260ms²]，方法范围1.87168×；28次measured native/0fallback，全部9对最终state/scalar/PNG/stdout一致。预测比实测高34.9436%，超过事前20%阈值；`validate-stateful-entry-model`独立由原样本重算，返回qualified_profile=null，未改模型/阈值、未拟合或重跑，默认NO_OP。本次legacy采样只绑定model，未携带政策引用及完整timer/harness预绑定；评估器亦将这两项列为未解，事后放宽到90%仍不会qualified。原采样/模型/政策均未改。局部收益已观测，不能当整程序或LeWM加速。下一层先固定统一采样design，再定位canonical→组合执行的上下文残差、用新独立样本校验，并补caller CPU契约与scoped内存资格；三阶段仍未完成。材料：`artifacts/baseline/activity-entry-cost.json`。

2026-11-17 真实科学入口继续贯通：只给 Gray–Scott 原 `run_simulation.py` 补做 F0–F4，再按引用合入原计算模块材料（fact正文复制0B，3个接口worker），当前规则下重发入口ProgramBuild。两次有界原边界元数据观测自动给出两输入259×259，不保tensor数值/地址、不固定5个scalar；`specialize-stateful`已按原F4精确重发现及硬件领域生成C++20/ABI1。真实driver `-T 2`显式trial的4次调用均native、0回退；逐次pre/post及完整entry后的数组（含halo）/scalar类型与字节、stdout/stderr和PNG逐字节一致，源码不变。新增`check-stateful-specialization`冷重放原F4/观测/recipe/6个native文件，禁CDLL通过，错shape和伪造selected拒绝；12项相关现有回归通过。未测目标A/B或视频、未部署。

改写区域和stateful完整入口已投影到共同实现IR；资源未解义务独立于时间成本，非空时用problem v3，旧v1/v2 wire/digest保留。真实259的NO_OP及两个CPU候选分别保留成本UNKNOWN、caller CPU与opaque memory/residency义务，strict规划诚实UNRESOLVED；无OIR selection时不声称有source delta。成本与计划均绑定实际编译domain，不沿用旧515模型或另一发布路径的context。当前ProgramBuild仍有303个来源/接线缺口，catalog全图物化未闭合，三阶段未完成。同规格canonical scope-boundary、源码stage-DAG V2（4/6/7）及原语标定已串行完成，均warm2/7样本；仅组件证据，完整入口成本未闭合。下一层接现有activity成本聚合出口及guard路径残项，再接资源资格与成本决策。材料：`artifacts/baseline/scientific-mainline.json`。

2026-11-17 三条主链继续落地：①`joint-plan/joint-pipeline`默认构造式规划，复用原typed problem、容量/搬运/共享存储寿命和独立verifier；v2条件启动、端点等式与WAIT_TAIL保留。真实source→构造→证书→C++已在禁OR-Tools进程中逐位验证；有限RAM/RAM→HBM仅为typed声明模型验证，不是实际GPU搬运。贪心可能死路，保UNKNOWN/domain gap，不声称最快。②融合C1 v2只带一个原ELF，无host、不pin；在禁止compiler/IR/analysis/planner/codegen导入且不读外部图JSON的进程中native3与原函数一致，实际普通导入入口的stdout也相同，default无CDLL。新scope成本仍UNKNOWN，旧compact运行实现pin变更需重导。

③真实canonical完整入口标定已取得数据：6 TRAIN、2混合holdout、1真实源区域holdout，warmup3/9次、4并行编译任务且计时前join。仅TRAIN拟合的两侧basis秩均4，冻结目标误差原Python10.16%/融合1.62%；目标实测median10.365/99.495µs，预测9.312/101.111µs，判断KEEP且不加载native。是小源码区域、固定shape与warm边界的有效模型，不是独立组件成本或科学程序加速。发布曾因共享dict引用导致digest变化，已改不可变快照/按落盘观测离线拟合；保留原样本，无重采样或放松校验。报告 `artifacts/baseline/constructive-corpus-compact.json`。

当前下一层：把结构改写后的区域接到共同的硬件/时序投影，在真实科学程序验证完整caller成本与计划；扩展模型适用域前先补对应证据。循环仍在上游共享模板表示，新构造器处理已闭合DAG区域；跨迭代时序、完整图重写/分析缓存及通用整程序C++仍未完成，不能把本轮局部闭环当作三阶段完成。

2026-11-17 正向需求/成本层已贯通：`model-pipeline-cost` 全源重放后机械组合原逐CALL eager与native/fused需求，复用现有单区域分析、shape权威及FactRef校验。原算术按source occurrence经保型映射恢复，CSE不会把原两次乘法误数一次。8×8例原/融合均448次逻辑算术；逻辑读写8192→1536B，累计payload分配3584→512B，不是DRAM或耗时。完整入口残项单独列出，由明确无重叠的有效模型覆盖；无系数保持UNKNOWN/KEEP，不默认为零。

两组显式DECLARED系数分别算出KEEP/REWRITE，正模型已交既有macro pass及C++20/ABI1；完整源码/需求/系数冷重放通过，独立输入、共享输入别名、同buffer改内容后三次native均与原函数逐位一致。所有系数为算法验收声明，尚未自动标定，不代表真实性能预测或新加速比。物理流量、cache、完整组件标定和资源日历构造仍未完成；C1 fused compact缺口保持。验收 `artifacts/baseline/pipeline-forward-model.json`。接下来先取得独立组件证据并校验预测适用域，再推进构造时序，不回到整版本黑盒排名。

2026-11-17 GCC方法论已接首条源码区域pass：`macro-pipeline` 重放原F4/GlobalPort及硬件来源，投影5个原CALL到既有v2 OIR，明确原Python/NO_OP和已有FULL_FUSION。requires、条件Q、完整受保护调用成本、5项失效账本及独立重放已接通，不新增后端或全局求解。声明正模型可构建C++20/ABI1、受控安装原模块函数并恢复；独立与只读别名逐位一致，dtype/callee/CPU域变化回退。原执行无native guards，成本未知/负模型不构建或加载库。一次序列化tuple/list重放问题已修，当前材料独立冷读通过。

实测该8×8源码例子：warmup3、9次交替顺序、每样本1次完整受保护调用，原NumPy median13.584µs[min8.998,max15.444]、融合111.705µs[min109.358,max137.556]，not profitable。实际完整边界证据已使pass选KEEP且跳过native；声明正模型不当作加速证据。无worker/绑核，仍有guard/ABI等调用开销。仅小源码区域，非科学程序整入口或LeWM；无新加速结论。新全图重写、自动分析缓存失效重算、真实组件标定与资源日历构造均未完成，现有C1 compact亦未接新融合交付。验收：`artifacts/baseline/macro-fusion-pass.json`。

GCC13.3本机实证保留：-Q选项、pass顺序、GIMPLE/SSA、LICM及最终IR；闭合C例子发生内联、外提、重复/死计算消除和16字节vector，未知外部调用保留。源码复核及补充编译确认：未知alias可生成带检查双路径，示例运行收益阈值3、已有restrict契约时为2；常量返回不消除两次volatile状态写。目标成本单位不是纳秒，未运行目标或测加速。SCAR已有首条pass，但全图修订/通用分析失效重算/规模阈值分派仍有缺口；逐层任务收在DEVELOPMENT。记录：`artifacts/baseline/gcc-o3-study.json`。

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
