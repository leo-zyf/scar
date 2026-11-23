# SCAR

SCAR 将程序和执行平台建成统一的图，基于依赖、效果和资源约束规划执行，并交付独立的 C++20 工程。当前重写以 Linux CPU 为验收平台。

```text
scar/
├── cli.py         # 命令分派与三阶段串接
├── contract.py    # 图、资源、计划和工件收据的共同格式
├── frontend/     # 源码与编译事实、区域图、平台资源
├── middle/       # 等价图改写、逻辑依赖与封存计划
└── backend/      # CPU 探针、物理排程、C++20 生成和校验
```

```bash
python -m pip install -e '.[fortran]'
scar build --manifest request.json --out work/graph --jobs 4
scar plan --graph work/graph --out work/plan
scar compile --plan work/plan --out work/delivery
```

Python 源码可从 `scar build --entry program.py` 接入。闭合的逐元素数组循环用 `plan --input-shapes` 和 `--trip-count` 特化。生成的 C++20 入口接收固定布局的 float64 二进制输入。后端根据实测自动选核，`compile --cpu-threads N` 约束可用工作线程数。节点及工件含义见 [架构](docs/ARCHITECTURE.md) 和 [图模型](docs/MODEL.md)。

阶段入口可分别运行。中端从 `graph.json` 及其前端材料生成自包含计划包。后端只需 `plan.json`、同目录的 `plan.payload/` 和平台工具链。移动计划包后可独立编译。当前按已覆盖原生方言和 Python 数组 ABI 做正确性验收。成本缺失的计划保留基线。隔离验收入口在 `tests/stages`。旧版实现及旧验收记录可按 Git 历史回读。
