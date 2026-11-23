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

三阶段分别在独立进程运行。`build` 读取源码、每个编译单元的语言和参数，输出 `graph.json` 及内容寻址的单元子图。`plan` 读取完整图包，输出审计报告 `plan.json` 和自包含 `exec-ir.json` 执行 IR 包。`compile` 读取执行 IR 包，选择目标调度，输出 `physical-plan.json`、CMake 工程、C++20 源码、头文件和可执行文件。搬动中间包后仍可独立消费。各阶段的代码只导入本阶段和公共格式层。

当前实现覆盖有界 Python 数值函数、C/C++ 类型化过程、GNU Fortran 过程、跨语言 ABI、结构化循环及部分对象生命周期和虚调用。Python 数组入口可以显式采集一次受限运行，记录形状、类型和布局，生成二进制数组 ABI。剩余能力缺口带源码位置记录；存在影响可执行闭包的缺口时计划为 `INCOMPLETE`。

真实 CXSparse `cs_demo2` 已通过原版与生成 C++20 输出比较。AMSS 的实际 C++ 调用方与 `bssn2adm` Fortran 过程已独立通过 288 项输出比较。最终验收为官方 Python 驱动、初值生成、完整 CPU ABE、数据和后处理的 5M 用例，原版约 29 分钟、16 MPI 进程。该完整生成版验收仍在开发中。

层间接口见 [架构](docs/ARCHITECTURE.md)，图和 IR 形式见 [模型](docs/MODEL.md)。`tests/stages` 通过禁止跨阶段导入及移走上游源码、图和报告，检查阶段独立性。GitHub Actions 从安装后的 wheel 执行同一检查。

Python 源码使用 Black 26.10.1 默认格式，C/C++ 使用 `.clang-format` 指定的 LLVM 风格。生成工程在编译前进行同样的格式化，默认编译启用 warning-as-error，CI 同时检查源码和生成产物的格式。
