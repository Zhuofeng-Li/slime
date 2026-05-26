# `/submit` Code Walkthrough

本文说明客户端执行下面这段代码后，Frontier-CS algorithmic judge 内部会经过哪些步骤，以及对应源码位置。

```python
r = requests.post(
    f"{url}/submit",
    files={"code": ("sol.cpp", code)},
    data={"pid": problem_id, "lang": "cpp"},
    timeout=30,
)
r.raise_for_status()
```

## 1. 客户端提交

相关源码：

- `FrontierSmith/slime_rm/frontiercs_rm.py:71`
- `FrontierSmith/verl/verl/utils/reward_score/frontiercs.py:103`
- `FrontierSmith/scripts/debug_frontiercs_eval.py:39`

客户端以 `multipart/form-data` 方式提交：

- `files["code"]`: 上传源码文件，文件名通常是 `sol.cpp`。
- `data["pid"]`: problem id。
- `data["lang"]`: 语言，这里是 `cpp`。

`r.raise_for_status()` 只表示 `/submit` HTTP 请求成功，不表示代码已经通过测试。服务端 `/submit` 成功后只返回一个 submission id，即 `sid`。真正的编译和运行发生在后台 worker 中。

客户端拿到 `sid` 后，通常会继续轮询：

- `GET /result/:sid`
- `FrontierSmith/slime_rm/frontiercs_rm.py:82`
- `FrontierSmith/verl/verl/utils/reward_score/frontiercs.py:114`

## 2. 服务启动和路由注册

相关源码：

- `FrontierSmith/Frontier-CS/algorithmic/server.js:49`
- `FrontierSmith/Frontier-CS/algorithmic/server.js:62`
- `FrontierSmith/Frontier-CS/algorithmic/server.js:66`
- `FrontierSmith/Frontier-CS/algorithmic/run_judge.sh:18`

`server.js` 做三件核心事情：

1. 创建 `SubmissionManager`。
2. 创建 `ProblemManager`。
3. 创建 `JudgeEngine` 并注册 API routes。

默认监听端口是 `8082`。`run_judge.sh` 会把容器端口 `8082` 映射到宿主机，并设置：

- `PORT=8082`
- `GJ_ADDR=http://127.0.0.1:5050`
- `JUDGE_WORKERS=8`
- `GJ_PARALLELISM=8`

## 3. `POST /submit` 路由入口

相关源码：

- `FrontierSmith/Frontier-CS/algorithmic/judge/src/router.js:12`
- `FrontierSmith/Frontier-CS/algorithmic/judge/src/upload.js:33`

入口是：

```js
router.post('/submit', upload.single('code'), async (req, res) => {
```

`upload.single('code')` 使用 multer 读取 multipart 中名为 `code` 的文件字段。multer 配置在 `upload.js`，使用 memory storage，文件大小限制是 5 MB。

路由处理逻辑：

1. 读取 `pid` 和 `lang`：
   - `router.js:14`
   - `router.js:15`

2. 读取源码：
   - multipart 文件：`req.file.buffer.toString('utf8')`，见 `router.js:20`
   - text field fallback：`req.body.code`，见 `router.js:25`
   - base64 fallback：`req.body.codeBase64`，见 `router.js:30`
   - text body fallback：`router.js:37`

3. 语言归一化：
   - `router.js:42`
   - 例如 `c++`、`cpp17`、`gnu++17` 都会归一到 `cpp`。

4. 校验参数：
   - `router.js:49`
   - 缺少 `pid`、`lang` 或 `code` 时返回 HTTP 400。

5. 提交到 judge engine：
   - `router.js:57`

6. 返回 `sid`：
   - `router.js:58`

成功响应形如：

```json
{"sid": 123}
```

## 4. `JudgeEngine.submit`: 入队并返回

相关源码：

- `FrontierSmith/Frontier-CS/algorithmic/judge/src/judge_engine.js:169`
- `FrontierSmith/Frontier-CS/algorithmic/judge/src/utils.js:142`

`submit(pid, lang, code)` 的职责是把任务放进内存队列，不在 HTTP 请求线程里完成判题。

主要步骤：

1. 分配 `sid`：
   - `judge_engine.js:170`
   - 底层由 `SubmissionManager.nextSubmissionId()` 维护递增 counter。

2. 记录初始状态：
   - `judge_engine.js:171`
   - `this.results.set(sid, { status: 'queued' })`

3. 创建 submission 目录：
   - `judge_engine.js:172`

4. 入队：
   - 正常情况下直接把 `{ sid, pid, lang, code }` 放入 `this.queue`，见 `judge_engine.js:183`
   - 如果队列非常大，则先把源码写成 `source.code`，队列里只放 `{ sid, pid, lang }`，见 `judge_engine.js:176`

5. 写 `meta.json`：
   - `judge_engine.js:186`

6. 返回 `sid`：
   - `judge_engine.js:191`

到这里 `/submit` 请求就结束了。此时状态通常仍然是 `queued`，后台 worker 还没有或正在处理。

## 5. 后台 worker 消费队列

相关源码：

- `FrontierSmith/Frontier-CS/algorithmic/judge/src/judge_engine.js:38`
- `FrontierSmith/Frontier-CS/algorithmic/judge/src/judge_engine.js:363`
- `FrontierSmith/Frontier-CS/algorithmic/judge/src/judge_engine.js:509`

`JudgeEngine` 构造函数中会调用 `startWorkers(config.workers || 4)`。每个 worker 运行一个无限循环：

1. 从 `this.queue` 取一个 job：
   - `judge_engine.js:511`

2. 如果没有 job，sleep 50 ms 后继续：
   - `judge_engine.js:512`

3. 获取源码并写入 `source.code`：
   - `judge_engine.js:521`

4. 加载题目配置：
   - `judge_engine.js:528`
   - `ProblemManager.loadProblem(pid)` 解析 `config.yaml`，见 `problem_manager.js:112`

5. 根据题目类型分发：
   - `interactive` -> `judgeInteractive`
   - `leetcode` -> 当前不支持
   - 默认 -> `judgeDefault`
   - `judge_engine.js:530`

## 6. 默认题目判题流程

相关源码：

- `FrontierSmith/Frontier-CS/algorithmic/judge/src/judge_engine.js:370`
- `FrontierSmith/Frontier-CS/algorithmic/judge/src/gojudge.js:75`
- `FrontierSmith/Frontier-CS/algorithmic/judge/src/gojudge.js:89`

默认题目走 `judgeDefault(problem, sid, pid, lang, code, subDir)`。

主要步骤：

1. 编译或准备用户程序：
   - `judge_engine.js:374`
   - `goJudge.prepareProgram({ lang, code, mainName })`

2. C++ 编译命令：

```js
['/usr/bin/g++', srcName, '-O2', '-pipe', '-std=gnu++17', '-o', outName]
```

对应源码：

- `gojudge.js:89`

编译是通过 go-judge 的 `/run` API 执行的：

- `gojudge.js:14`

3. 获取或编译 checker：
   - `judge_engine.js:378`
   - `getOrCompileChecker()` 在 `judge_engine.js:102`
   - checker 编译在 `gojudge.js:151`

4. 遍历所有 test cases：
   - `judge_engine.js:384`

5. 每个 case 调用 `judgeCase()`：
   - `judge_engine.js:226`

## 7. 单个 case 的执行流程

相关源码：

- `FrontierSmith/Frontier-CS/algorithmic/judge/src/judge_engine.js:226`
- `FrontierSmith/Frontier-CS/algorithmic/judge/src/judge_engine.js:239`
- `FrontierSmith/Frontier-CS/algorithmic/judge/src/judge_engine.js:268`

`judgeCase()` 做两段运行：

1. 跑用户程序。

   输入文件通过 go-judge `files[0]` 传入，stdout/stderr 分别被捕获：

   - stdin: `{ content: inf }`
   - stdout: `{ name: 'stdout', max: 128 * 1024 * 1024 }`
   - stderr: `{ name: 'stderr', max: 64 * 1024 * 1024 }`

   时间、内存限制来自题目 case 配置：

   - `cpuLimit: toNs(caseItem.time)`
   - `memoryLimit: toBytes(caseItem.memory)`

2. 如果用户程序不是 `Accepted`，直接返回该 case 的失败结果：

   - `judge_engine.js:255`

3. 如果用户程序运行成功，则运行 checker：

   - `judge_engine.js:268`

   checker 参数是：

```js
['chk', 'in.txt', 'out.txt', 'ans.txt']
```

checker 的输入文件包括：

- `in.txt`: 原始输入
- `out.txt`: 用户程序输出
- `ans.txt`: 标准答案

4. checker 退出码为 0 且状态为 `Accepted` 时，该 case 被认为通过：

   - `judge_engine.js:284`

## 8. 汇总分数并写结果

相关源码：

- `FrontierSmith/Frontier-CS/algorithmic/judge/src/judge_engine.js:387`
- `FrontierSmith/Frontier-CS/algorithmic/judge/src/judge_engine.js:410`
- `FrontierSmith/Frontier-CS/algorithmic/judge/src/judge_engine.js:419`

每个 case 的 checker 输出中会尝试解析：

- `Ratio: <number>`
- `RatioUnbounded: <number>`

如果没有 `Ratio`，则：

- case 通过时记 `1.0`
- case 失败时记 `0`

最终结果结构大致是：

```json
{
  "status": "done",
  "passed": true,
  "result": "Correct Answer",
  "score": 100,
  "scoreUnbounded": 100,
  "cases": []
}
```

最终结果会同时：

1. 写入内存 `this.results`：
   - `judge_engine.js:418`

2. 写入磁盘 `result.json`：
   - `judge_engine.js:419`

如果判题过程中抛异常，则结果会是：

```json
{
  "status": "error",
  "error": "..."
}
```

对应源码：

- `judge_engine.js:429`

## 9. 客户端轮询 `/result/:sid`

相关源码：

- `FrontierSmith/Frontier-CS/algorithmic/judge/src/router.js:65`
- `FrontierSmith/Frontier-CS/algorithmic/judge/src/judge_engine.js:195`

结果查询入口是：

```js
router.get('/result/:sid', async (req, res) => {
```

服务端会调用 `judgeEngine.getResult(sid)`：

1. 优先从内存 `this.results` 取。
2. 如果是 `done` 或 `error`，读取后会从内存删除。
3. 如果内存没有，则尝试从磁盘 `result.json` fallback。

客户端侧看到：

- `status == "queued"`: 继续等。
- `status == "done"`: 读取 `score`，转换成 reward。
- `status == "error"`: 记为 0 分。

## 10. 总体时序

```text
Python requests.post /submit
  -> Express router.post('/submit')
  -> multer 读取 code 文件
  -> 校验 pid/lang/code
  -> judgeEngine.submit()
      -> 分配 sid
      -> results[sid] = queued
      -> 写 meta.json
      -> queue.push(...)
  -> HTTP 200 {"sid": ...}

后台 worker
  -> queue.shift()
  -> 写 source.code
  -> loadProblem(pid)
  -> prepareProgram 编译 sol.cpp
  -> compile/cache checker
  -> 对每个 test case:
       run 用户程序
       run checker
       解析 Ratio 分数
  -> results[sid] = done/error
  -> 写 result.json

Python requests.get /result/{sid}
  -> 拿 done/error
  -> 计算 reward
```

