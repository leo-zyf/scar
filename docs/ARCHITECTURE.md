# SCAR 编译器组织

`scar/cli.py` 定义公开命令。`scar/compiler/pipeline.py` 编排主流程并写出每阶段的产物。静态输入、运行证据和硬件库存进入前端。中端消费图与效果合同。后端消费已选择的图修订和计划。

## 1. 前端

| 层 | 输入 | 产物 | 代码 |
| --- | --- | --- | --- |
| 源码与编译参数 | 入口、锁定源码、逐编译单元的语言和方言 | 可复现的翻译单元输入 | `compiler/pipeline.py` |
| 类型与符号 | 原始编译单元 | 类型、源码范围、符号与编译器 SSA | `compiler/frontend/clang.py`、`compiler/frontend/fortran.py` |
| 图构建 | 编译器事实和可用的调用证据 | `ProgramGraph` 的区域、控制边、值和状态依赖 | `compiler/frontend/graph.py` |
| 平台与证据 | 目标主机、必要的运行记录 | 硬件图、资源位置和运行绑定 | `analysis/target_graph.py`、`trace/` |

Python 入口通过 `analysis/build.py` 驱动已有源码与执行证据层。`compiler/frontend/stages.py` 调度正式前端的隔离工作进程。原生入口记录未绑定的调用和无法证明的效果，供规划判断。共享节点、值、端口和记录格式由 `ir/` 定义。

## 2. 中端

`compiler/middle/plan.py` 校验图改写，生成或读取 `PlanningProblem`。它将可行方案交给 `optimization/` 的资源求解器。成本与资源容量有独立的证据。优化目标是缩短完工时间。已接通的原生整入口计价目前只有原执行方案，图改写与后端采用分别记录。

## 3. 后端

`compiler/backend/cpp20.py` 负责统一的源文件锁定和 C++20 降级入口。`compiler/backend/fortran.py` 与 `_fortran_freeform.py` 处理 Fortran 的源码降级。`codegen/native_project.py` 负责 CMake、构建与链接回读。生成版和原版的对照由主流程统一执行，运行时加载已发布合同的机制在 `runtime/`。

仍在使用的 Python 规则、硬件模型和提供方实现继续由原有目录提供。整理这些模块按上面的层和数据接口进行。每层只保留本层负责的实现，实验和工作负载放在编译器以外。`visualization/` 可独立查看图。
