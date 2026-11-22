# SCAR

SCAR 是基于程序图的宏观编译器。它把源码、运行绑定和硬件资源接到同一编译流程，再做合法的图改写和资源调度，最后交付 C++20 工程。

| 入口 | 用途 |
| --- | --- |
| `scar build` | 只建图。用 `--manifest` 读取原生编译单元，用 `--entry` 读取 Python 入口，用 `--formal` 生成 Python 源码目录，或用 `--static` 构建静态源码和函数体图。 |
| `scar plan` | 从 `--problem` 计算资源和时序，或用 `--build` 从 Python 图构造编译域。 |
| `scar compile` | 用 `--manifest` 编译原生输入，用 `--domains` 编译已建模的 Python 区域，用 `--plan` 重放原生交付计划。 |

`scar trace` 和 `scar import-capture` 提供前端证据。`scar build-query` 查询图。`scar check-program` 检查持久化的类型图。`scar run` 加载已生成的执行合同。`scar view-graph` 提供可选的图查看。

```bash
scar build --manifest entry.json --out graph
scar compile --manifest entry.json --out compiled
scar plan --problem compiled/planning-problem.json --out replay.json
scar compile --plan compiled/compiler-plan.json --out delivered
```

每个原生编译单元在清单中声明源码路径、语言和编译参数。原版与生成版按同一输入核对。原生程序覆盖、跨语言调用和调度成本的缺口在结果中单独记录。跨语言支持取决于入口的实际图与交付证据。

## 工作区

`scar/` 是可安装包，`tests/` 放验收和最小输入，`examples/` 放公开入口示例，`scripts/` 放开发和验收工具。大型外部程序放在仓库外。每次运行用 `--out` 指定独立输出目录。已发布的图、计划和证据会按引用复核，搬动前需检查其依赖方。

`artifacts/baseline/` 保留环境基线。旧实验报告和已退役样例可从 Git 的 `archive/scar-before-workspace-cleanup` 读取，例如 `git ls-tree -r --name-only archive/scar-before-workspace-cleanup artifacts/baseline testcases`。

[编译器组织](docs/ARCHITECTURE.md) · [IR 数据模型](docs/MODEL.md)
