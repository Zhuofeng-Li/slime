# `create_rollout_manager` Code Walkthrough

从 `train.py` 调用 `create_rollout_manager(args, pg)` 开始，到所有 SGLang 引擎就绪、`RolloutManager` Ray Actor 返回为止。
所有文件路径相对于项目根目录 `/fsx-alignment/home/zhuofeng/slime`。

---

## 调用树总览

```
train.py:17  create_rollout_manager(args, pgs["rollout"])
│   [placement_group.py:183]
├─ RolloutManager.options(...).remote(args, pg)        ← 创建 Ray Actor
│   [rollout.py:353]  RolloutManager.__init__
│   ├─ configure_logger()
│   ├─ load_function(data_source_path)                 ← 加载数据源类
│   ├─ load_function(rollout_function_path)            ← 加载 rollout fn
│   ├─ load_function(eval_function_path)               ← 加载 eval fn
│   ├─ init_http_client(args)                          ← 初始化 HTTP 客户端
│   ├─ start_rollout_servers(args, pg)                 [rollout.py:962]
│   │   ├─ _resolve_sglang_config(args)                [rollout.py:1099]
│   │   ├─ (per model) _start_router(args, ...)        [rollout.py:892]
│   │   │   └─ multiprocessing.Process(run_router)     ← 子进程启动 sglang_router
│   │   └─ (per server_group) _make_group(...)
│   │       └─ ServerGroup.start_engines(port_cursors) [rollout.py:70]
│   │           ├─ ray.remote(SGLangEngine).options(...).remote(...)  ← Ray Actor
│   │           └─ engine.init.remote(dist_init_addr, port, ...)     [sglang_engine.py:117]
│   │               ├─ _compute_server_args(...)
│   │               └─ _init_normal(server_args_dict)  [sglang_engine.py:193]
│   │                   ├─ launch_server_process(ServerArgs(...))    ← 启动 HTTP 推理服务
│   │                   └─ POST /workers → router                    ← 注册到路由
│   ├─ init_tracking(args)
│   ├─ Lock.options(...).remote()                      ← rollout_engine_lock
│   └─ (if fault_tolerance) RolloutHealthMonitor(group, args).start()
│       [health_monitor.py:35]
│       └─ threading.Thread(_health_monitor_loop).start()           ← 守护线程
├─ rollout_manager.get_num_rollout_per_epoch.remote()  (optional)
├─ rollout_manager.check_weights.remote(...)           (if check_weight_update_equal)
└─ rollout_manager.offload.remote()                    (if offload_rollout)
```

---

## Phase 1: 工厂函数 — 创建 RolloutManager Ray Actor

### 1.1 `create_rollout_manager` 入口

**文件: `slime/ray/placement_group.py:183`**

```python
def create_rollout_manager(args, pg):
    rollout_manager = RolloutManager.options(
        num_cpus=1,
        num_gpus=0,              # ← Manager 本身不占 GPU，只协调
    ).remote(args, pg)           # ← 异步创建 Ray Actor，触发 __init__

    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch   # ← 动态算 rollout 总数

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))    # ← 权重快照
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())   # ← 初始化后立即 offload 释放 GPU

    return rollout_manager, num_rollout_per_epoch
```

`RolloutManager` 以 `num_gpus=0` 运行，只是一个逻辑协调器；真正占 GPU 的是它内部创建的 `SGLangEngine` actors。

---

## Phase 2: RolloutManager.__init__ — 加载函数和配置

**文件: `slime/ray/rollout.py:353`**

```python
def __init__(self, args, pg):
    configure_logger()
    self.pg = pg
    self.args = args

    data_source_cls = load_function(self.args.data_source_path)
    self.data_source = data_source_cls(args)                       # ← 实例化数据集

    self.generate_rollout = load_function(self.args.rollout_function_path)       # ← rollout 生成函数
    self.eval_generate_rollout = load_function(self.args.eval_function_path)     # ← eval rollout 函数

    if self.args.custom_reward_post_process_path is not None:
        self.custom_reward_post_process_func = load_function(...)  # ← 可选：自定义 reward 后处理

    if self.args.debug_train_only:
        self.servers: dict[str, RolloutServer] = {}                # ← 调试模式跳过引擎初始化
    else:
        init_http_client(args)
        self.servers = start_rollout_servers(args, pg)             # ← 核心：启动所有 SGLang 引擎

    init_tracking(args, primary=False)
    self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()  # ← 分布式锁 Actor
    self.rollout_id = -1
```

`load_function` 通过 Python import 动态加载用户定义的函数，使得 rollout 逻辑完全可插拔。

---

## Phase 3: start_rollout_servers — 解析配置、按 Model 启动服务

**文件: `slime/ray/rollout.py:962`**

```python
def start_rollout_servers(args, pg) -> dict[str, RolloutServer]:
    config = _resolve_sglang_config(args)          # ← 解析 SglangConfig（YAML/prefill/默认）

    servers: dict[str, RolloutServer] = {}
    gpu_offset = 0
    engine_offset = 0
    rollout_pg_offset = _compute_rollout_offset(args)   # ← rollout GPU 在 PG 中的起始偏移
    megatron_num_gpus = _compute_megatron_num_gpus(args)

    for model_idx, model_cfg in enumerate(config.models):   # ← 遍历每个模型（多模型时有多个）
        router_ip, router_port = _start_router(
            args, has_pd_disaggregation=model_cfg.has_pd_disaggregation,
            force_new=(model_idx > 0),              # ← 第二个模型起强制新建路由
        )
        server_groups = []
        for group_cfg in model_cfg.server_groups:
            group = _make_group(group_cfg, router_ip, router_port)
            handles, port_cursors = group.start_engines(port_cursors)
            server_groups.append(group)
        ray.get(all_init_handles)                   # ← 阻塞等待所有引擎就绪

        servers[model_cfg.name] = RolloutServer(
            server_groups=server_groups,
            router_ip=router_ip, router_port=router_port,
            update_weights=model_cfg.update_weights,
        )
    return servers
```

### 3.1 _resolve_sglang_config — 三种配置来源

**文件: `slime/ray/rollout.py:1099`**

```python
def _resolve_sglang_config(args) -> SglangConfig:
    if getattr(args, "sglang_config", None) is not None:
        return SglangConfig.from_yaml(args.sglang_config)     # ← 用户提供 YAML
    if args.prefill_num_servers is not None:
        return SglangConfig.from_prefill_num_servers(args)    # ← PD 分离模式
    return SglangConfig(models=[ModelConfig(                  # ← 默认：单 regular 组
        name="default",
        server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=args.rollout_num_gpus)],
    )])
```

### 3.2 _make_group — Engine 数量计算

**文件: `slime/ray/rollout.py:1001`**

```python
def _make_group(group_cfg, router_ip, router_port, overrides_extra=None):
    gpus_per_engine = group_cfg.num_gpus_per_engine          # ← 每个 engine 占多少 GPU（TP size）
    num_gpu_per_engine_local = min(gpus_per_engine, args.num_gpus_per_node)
    num_engines = group_cfg.num_gpus // num_gpu_per_engine_local  # ← 该 group 的 engine 总数

    group = ServerGroup(
        all_engines=[None] * num_engines,            # ← 预分配槽位，后续 start_engines 填充
        num_gpus_per_engine=gpus_per_engine,
        rank_offset=engine_offset,                   # ← 全局 engine 序号偏移
        gpu_offset=gpu_offset,                       # ← 全局 GPU 槽位偏移
        needs_offload=needs_offload,
        ...
    )
    engine_offset += num_engines
    gpu_offset += group_cfg.num_gpus
    return group
```

**Engine 数量公式：**
```
num_engines = group.num_gpus / min(num_gpus_per_engine, num_gpus_per_node)
```

例如：rollout 用 8 块 GPU，每个 engine 需要 TP=4，单节点 8 卡 → `min(4,8)=4` → `8/4 = 2 个 engine`。
多节点时（如 TP=8 跨 2 节点，每节点 4 卡）→ `min(8,4)=4` → 每节点各启动 1 个 actor，共同组成 1 个逻辑 engine。

---

## Phase 4: _start_router — 启动 sglang_router 子进程

**文件: `slime/ray/rollout.py:892`**

```python
def _start_router(args, *, has_pd_disaggregation=False, force_new=False):
    router_ip = _wrap_ipv6(get_host_info()[1])                    # ← 取本机 IP
    router_port = find_available_port(random.randint(3000, 4000)) # ← 随机找空闲端口

    router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
    router_args.host = router_ip
    router_args.port = router_port
    router_args.disable_health_check = True                       # ← slime 自己做健康检查

    if has_pd_disaggregation:
        router_args.pd_disaggregation = True
        router_args.disable_circuit_breaker = True                # ← PD 模式下禁用熔断

    process = multiprocessing.Process(target=run_router, args=(router_args,))
    process.daemon = True
    process.start()
    time.sleep(3)                                                 # ← 等 router 就绪
    assert process.is_alive()
    return router_ip, router_port
```

`sglang_router` 作为独立的守护子进程运行，所有 SGLang engine 都通过它进行请求路由和负载均衡。

---

## Phase 5: ServerGroup.start_engines — 创建 SGLangEngine Ray Actors

**文件: `slime/ray/rollout.py:70`**

```python
def start_engines(self, port_cursors):
    pg, reordered_bundle_indices, reordered_gpu_ids = self.pg

    RolloutRayActor = ray.remote(SGLangEngine)        # ← 包装为 Ray remote class

    rollout_engines = []
    for i in range(len(self.all_engines)):
        global_rank = self.rank_offset + i
        base_gpu_id = int(reordered_gpu_ids[self.gpu_offset + i * num_gpu_per_engine])

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=reordered_bundle_indices[...],  # ← 绑定到 PG bundle
        )
        rollout_engine = RolloutRayActor.options(
            num_cpus=0.2, num_gpus=0.2,               # ← 轻量占位，真正 GPU 由 sglang 进程用
            scheduling_strategy=scheduling_strategy,
            runtime_env={"env_vars": env_vars},        # ← 注入 SGLANG_* 环境变量
        ).remote(self.args, rank=global_rank, worker_type=self.worker_type,
                 base_gpu_id=base_gpu_id, ...)

        rollout_engines.append((global_rank, rollout_engine))
        self.all_engines[i] = rollout_engine

    # 分配端口后触发 init（不等待）
    addr_and_ports, port_cursors = _allocate_rollout_engine_addr_and_ports_normal(...)
    init_handles = [
        engine.init.remote(**(addr_and_ports[rank]),
                           router_ip=self.router_ip, router_port=self.router_port)
        for rank, engine in rollout_engines
    ]
    return init_handles, port_cursors              # ← 返回 ObjectRef 列表，由上层 ray.get() 等待
```

每个 `SGLangEngine` actor 拿到 `0.2` 个逻辑 GPU 作为占位符，实际 GPU 由 actor 内部 fork 的 SGLang HTTP 服务进程独占。

---

## Phase 6: SGLangEngine.init — 启动推理进程并注册到 Router

**文件: `slime/backends/sglang_utils/sglang_engine.py:117`**

```python
def init(self, dist_init_addr, port, nccl_port, router_ip=None, router_port=None, ...):
    self.router_ip = router_ip
    self.router_port = router_port

    server_args_dict, _ = _compute_server_args(
        self.args, self.rank, dist_init_addr, nccl_port, host, port,
        self.worker_type, base_gpu_id=self.base_gpu_id,
        sglang_overrides=self.sglang_overrides,      # ← 合并用户自定义覆盖参数
    )
    self.node_rank = server_args_dict["node_rank"]

    if self.args.rollout_external:
        self._init_external(server_args_dict, ...)   # ← 连接外部已有 SGLang 服务
    else:
        self._init_normal(server_args_dict)          # ← 正常模式：本地启动进程
```

### 6.1 _init_normal — 启动 HTTP 推理服务并注册

**文件: `slime/backends/sglang_utils/sglang_engine.py:193`**

```python
def _init_normal(self, server_args_dict):
    self.process = launch_server_process(ServerArgs(**server_args_dict))  # ← 启动 SGLang HTTP 服务

    if self.worker_type == "encoder":
        return                                       # ← encoder 不注册到 router

    if self.node_rank == 0 and self.router_ip and self.router_port:
        payload = {
            "url": f"http://{self.server_host}:{self.server_port}",
            "worker_type": self.worker_type,         # ← "regular" / "prefill" / "decode"
        }
        response = requests.post(
            f"http://{self.router_ip}:{self.router_port}/workers",
            json=payload,                            # ← 向 router 注册自己
        )
        response.raise_for_status()
```

`launch_server_process` 在 Ray actor 进程内 fork 出一个 SGLang 的 HTTP 服务，监听分配好的端口，最后通过 POST `/workers` 告诉 router 自己的地址。

---

## Phase 7: RolloutHealthMonitor — 故障容忍守护线程

**文件: `slime/utils/health_monitor.py:23`**

```python
def __init__(self, server_group, args):
    self._server_group = server_group
    self._check_interval = args.rollout_health_check_interval
    self._check_timeout = args.rollout_health_check_timeout

def start(self):
    if not self._server_group.all_engines:
        return False
    self._stop_event = threading.Event()
    self._pause_event = threading.Event()
    self._pause_event.set()                          # ← 初始化后处于暂停态，generate 时再 resume

    self._thread = threading.Thread(
        target=self._health_monitor_loop,
        name="RolloutHealthMonitor",
        daemon=True,                                 # ← 守护线程，随主进程退出
    )
    self._thread.start()
```

`RolloutManager.__init__` 为每个 `ServerGroup` 各创建一个 `RolloutHealthMonitor`，初始处于暂停状态，在每次 `generate()` 时通过 `health_monitoring_resume()` 激活，对应 offload 时再 pause。

---

## Phase 8: 初始化后收尾操作

回到 `create_rollout_manager`，`RolloutManager` actor 启动完成后还有三个可选的后续调用：

**文件: `slime/ray/placement_group.py:190`**

```python
# 1. 动态计算 num_rollout
if args.num_rollout is None:
    num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
    # → len(data_source) // rollout_batch_size

# 2. 权重校验快照（用于验证训练后权重确实更新）
if args.check_weight_update_equal:
    ray.get(rollout_manager.check_weights.remote(action="snapshot"))
    ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))

# 3. 与 Megatron 共用 GPU 时，立即 offload 释放显存给训练侧
if args.offload_rollout:
    ray.get(rollout_manager.offload.remote())
```

返回的 `(rollout_manager, num_rollout_per_epoch)` 供 `train.py` 驱动主训练循环。

---

## 完整调用栈

```
train.py:17  create_rollout_manager(args, pgs["rollout"])
│   [placement_group.py:183]
├─ RolloutManager.options(...).remote(args, pg)        ← 创建 Ray Actor
│   [rollout.py:353]  RolloutManager.__init__
│   ├─ configure_logger()
│   ├─ load_function(data_source_path)                 ← 加载数据源类
│   ├─ load_function(rollout_function_path)            ← 加载 rollout fn
│   ├─ load_function(eval_function_path)               ← 加载 eval fn
│   ├─ init_http_client(args)                          ← 初始化 HTTP 客户端
│   ├─ start_rollout_servers(args, pg)                 [rollout.py:962]
│   │   ├─ _resolve_sglang_config(args)                [rollout.py:1099]
│   │   ├─ (per model) _start_router(args, ...)        [rollout.py:892]
│   │   │   └─ multiprocessing.Process(run_router)     ← 子进程启动 sglang_router
│   │   └─ (per server_group) _make_group(...)         [rollout.py:1001]
│   │       │   num_engines = group.num_gpus / min(num_gpus_per_engine, num_gpus_per_node)
│   │       └─ ServerGroup.start_engines(port_cursors) [rollout.py:70]
│   │           ├─ ray.remote(SGLangEngine).options(...).remote(...)  ← Ray Actor × num_engines
│   │           └─ engine.init.remote(dist_init_addr, port, ...)     [sglang_engine.py:117]
│   │               ├─ _compute_server_args(...)
│   │               └─ _init_normal(server_args_dict)  [sglang_engine.py:193]
│   │                   ├─ launch_server_process(ServerArgs(...))    ← 启动 HTTP 推理服务
│   │                   └─ POST /workers → router                    ← 注册到路由
│   ├─ init_tracking(args)
│   ├─ Lock.options(...).remote()                      ← rollout_engine_lock
│   └─ (if fault_tolerance) RolloutHealthMonitor(group, args).start()
│       [health_monitor.py:35]
│       └─ threading.Thread(_health_monitor_loop).start()           ← 守护线程
├─ rollout_manager.get_num_rollout_per_epoch.remote()  (optional)
├─ rollout_manager.check_weights.remote(...)           (if check_weight_update_equal)
└─ rollout_manager.offload.remote()                    (if offload_rollout)
```

---

## 初始化后各部件一览

| 部件 | 类型 | 位置 | 作用 |
|------|------|------|------|
| `RolloutManager` | Ray Actor (0 GPU) | `rollout.py:350` | 整体协调器，驱动 rollout/eval/offload |
| `sglang_router` | 子进程 (daemon) | `rollout.py:892` | HTTP 负载均衡，每个 model 一个 |
| `SGLangEngine` | Ray Actor (0.2 GPU) | `sglang_engine.py:100` | 每引擎一个，管理 SGLang 推理进程 |
| SGLang HTTP 进程 | 子进程 | `sglang_engine.py:195` | 真正占 GPU，提供 `/generate` 接口 |
| `Lock` | Ray Actor (0 GPU) | `utils.py:39` | 权重更新分布式锁 |
| `RolloutHealthMonitor` | 守护线程 | `health_monitor.py:10` | 定期 `/health_generate`，检测引擎宕机 |
| `data_source` | 普通对象 | `rollout.py:360` | 数据集迭代器，存活在 Manager 进程 |
