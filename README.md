# SCAR

SCAR 是基于程序图的宏观编译器。它把源码、运行绑定和硬件资源接到同一编译流程，再做合法的图改写和资源调度，最后交付 C++20 工程。

CPU 编译可用 `python -m pip install .` 安装。采集 PyTorch 工作负载时使用已有的 PyTorch 环境，或用 `python -m pip install '.[torch]'` 安装采集所需的 PyTorch。

| 入口 | 用途 |
| --- | --- |
| `scar build` | 用 `--manifest` 构建原生程序图，用 `--python-build` 将已准入的 Python 源码构建投影成同一程序图。`--entry`、`--formal` 和 `--static` 提供 Python 前端证据。 |
| `scar plan` | 用 `--graph` 从类型化图生成封存计划，用 `--problem` 重放资源和时序。`--build` 继续支持已有 Python 编译域。 |
| `scar compile` | 用 `--plan` 重放原生入口或已支持 Python 区域的交付计划。`--manifest` 串接原生三阶段，`--domains` 继续支持已有 Python 编译域。 |

`scar trace` 和 `scar import-capture` 提供前端证据。`scar build-query` 查询图。`scar check-program` 检查持久化的类型图。`scar run` 加载已生成的执行合同。`scar view-graph` 提供可选的图查看。

```bash
output=$(mktemp -d /tmp/scar-run.XXXXXX)
scar build --manifest entry.json --out "$output/graph"
scar check-program "$output/graph/program.json"
scar plan --graph "$output/graph" --out "$output/plan"
scar compile --plan "$output/plan/compiler-plan.json" --out "$output/delivered"
```

每个原生编译单元在清单中声明源码路径、语言和编译参数。原版与生成版按同一输入核对。原生程序覆盖、跨语言调用和调度成本的缺口在结果中单独记录。跨语言支持取决于入口的实际图与交付证据。

对已准入的 Python F4 构建，可用 `scar build --python-build BUILD --out GRAPH` 发布编译图，再用 `scar plan --graph GRAPH --input-shapes SHAPES.json --trip-count 4 --out PLAN` 封存区域计划。`scar compile --plan PLAN/compiler-plan.json --out DELIVERY` 可独立生成 C++20 试验工程。当前接入直接不变量循环、组合不变量循环和单数组递推。计划明示试验范围及成本缺口。旧阶段计划需用签发版本重放。

## 工作区

`scar/` 是可安装包，`tests/` 放验收和最小输入，`examples/` 放公开入口示例，`scripts/` 放开发和验收工具。大型外部程序放在仓库外。每次运行用 `--out` 指定独立输出目录。已发布的图、计划和证据会按引用复核，搬动前需检查其依赖方。

`artifacts/baseline/` 保留环境基线。旧实验报告和已退役样例可从 Git 的 `archive/scar-before-workspace-cleanup` 读取，例如 `git ls-tree -r --name-only archive/scar-before-workspace-cleanup artifacts/baseline testcases`。

CPU CI 在独立进程中检查三个阶段的交接。可从项目根目录运行 `python scripts/run_gate.py gates/cpu.json --out /tmp/scar-gates.json`，报告按层记录每条判据、源码摘要与失败原因。CI 通过后构建并验收源码包和 wheel。 [CPU 验收清单](gates/cpu.json)

[编译器组织](docs/ARCHITECTURE.md) · [IR 数据模型](docs/MODEL.md)
