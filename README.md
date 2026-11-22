# SCAR

SCAR 是基于程序图的宏观编译器。它把源码、运行绑定和硬件资源接到同一编译流程，再做合法的图改写和资源调度，最后交付 C++20 工程。

| 入口 | 用途 |
| --- | --- |
| `scar build` | 只建图。用 `--manifest` 读取原生编译单元，用 `--entry` 读取 Python 入口，或用 `--formal` 生成 Python 源码目录。 |
| `scar plan` | 从 `--problem` 计算资源和时序，或用 `--build` 从 Python 图构造编译域。 |
| `scar compile` | 执行完整编译。用 `--manifest` 读取原生输入，用 `--domains` 读取已建模的 Python 编译域。 |

`scar trace` 和 `scar import-capture` 提供前端证据。`scar build-query` 查询图。`scar run` 加载已生成的执行合同。`scar view-graph` 提供可选的图查看。

```bash
scar build --manifest entry.json --out graph
scar compile --manifest entry.json --out compiled
scar plan --problem compiled/planning-problem.json --out replay.json
```

每个原生编译单元在清单中声明源码路径、语言和编译参数。原版与生成版按同一输入核对。原生程序覆盖、跨语言调用和调度成本的缺口在结果中单独记录。跨语言支持取决于入口的实际图与交付证据。

[编译器组织](docs/ARCHITECTURE.md) · [IR 数据模型](docs/MODEL.md)
