# 微博量化投研系统 — 设计文档

> 版本: v2.0 | 更新日期: 2026-05-28

---

## 一、系统总览

### 1.1 系统定位

本系统是一个基于微博财经博主观点的 A 股量化投研平台，核心能力：

1. **微博博主内容抓取** — 自动化采集指定财经博主的微博内容及图片
2. **智能选股系统** — 从博主观点中提取股票信号，结合行情数据生成选股建议

### 1.2 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Web Layer                        │
│              (app/web/routes.py + templates/index.html)      │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│  Crawler │  Market  │ Selector │ Analyzer │   Notifier      │
│  Module  │  Module  │  Module  │  Module  │   Module        │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│              SQLite Database (weibo_crawler.db)              │
├─────────────────────────────────────────────────────────────┤
│              APScheduler (定时任务调度)                       │
├─────────────────────────────────────────────────────────────┤
│   数据源: 微博移动端API / 新浪财经API / 腾讯财经API / DeepSeek LLM  │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 目录结构

```
app/
├── main.py                    # FastAPI 入口, CORS, 启动事件
├── database.py                # SQLAlchemy 引擎 + Session 工厂
├── models.py                  # ORM 模型定义
├── config.yaml                # 全局配置
├── scheduler.py               # APScheduler 定时任务
├── crawler/
│   ├── weibo.py               # 微博内容爬虫
│   └── weibo_image.py         # 图片爬虫
├── market/
│   ├── base_client.py         # 行情客户端基类
│   ├── market_data.py         # 统一行情接口
│   └── sina_tencent_client.py # 新浪/腾讯行情客户端
├── selector/
│   ├── stock_selector.py      # 多维选股引擎
│   ├── multi_factor_model.py  # 多因子打分模型
│   ├── predictor.py           # 趋势预测器
│   └── backtest.py            # 回测框架
├── analyzer/
│   ├── stock_analyzer.py      # 量化投研分析器
│   └── llm_extractor.py       # LLM 实体提取器
├── data/
│   ├── __init__.py
│   └── stock_sector_map.py    # 内置股票→板块映射 (627 只 A 股)
├── notifier/
│   ├── realtime_alert.py      # 实时价格预警
│   └── semantic_alert_detector.py  # 语义预警检测
├── reviewer/
│   └── daily_reviewer.py      # 每日复盘
└── web/
    ├── routes.py              # API 路由
    └── templates/
        └── index.html         # 前端单页应用
```

---

## 二、微博博主内容抓取

### 2.1 数据源

| 数据源 | 接口 | 用途 | 认证方式 |
|--------|------|------|----------|
| 微博移动端 API | `https://m.weibo.cn/api/container/getIndex` | 博主微博列表 | Cookie (SUB 字段) |
| 微博移动端 | `https://m.weibo.cn/detail/{id}` | 微博详情 | Cookie (SUB 字段) |
| 微博图片 CDN | `https://wx1.sinaimg.cn/` | 图片下载 | 无需认证 |

### 2.2 数据库模型

#### Blogger (博主表)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | Integer (PK) | 自增主键 |
| weibo_uid | String | 微博用户 UID |
| name | String | 博主昵称 |
| description | String | 博主简介 |
| avatar_url | String | 头像 URL |
| followers_count | Integer | 粉丝数 |
| following_count | Integer | 关注数 |
| post_count | Integer | 微博数 |
| is_active | Boolean | 是否启用抓取 |
| created_at | DateTime | 入库时间 |
| updated_at | DateTime | 更新时间 |

#### WeiboPost (微博内容表)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | Integer (PK) | 自增主键 |
| blogger_id | Integer (FK→blogger.id) | 所属博主 |
| weibo_id | String | 微博原始 ID |
| content | Text | 微博正文 |
| publish_time | DateTime | 发布时间 |
| reposts_count | Integer | 转发数 |
| comments_count | Integer | 评论数 |
| attitudes_count | Integer | 点赞数 |
| images | Text (JSON) | 图片 URL 列表 |
| is_original | Boolean | 是否原创 |
| source | String | 发布来源 |
| created_at | DateTime | 入库时间 |

#### WeiboImage (图片表)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | Integer (PK) | 自增主键 |
| post_id | Integer (FK→weibo_post.id) | 关联微博 |
| image_url | String | 原始图片 URL |
| local_path | String | 本地存储路径 |
| downloaded | Boolean | 是否已下载 |
| created_at | DateTime | 入库时间 |

### 2.3 爬虫核心逻辑

#### 2.3.1 博主内容抓取流程

```
用户添加博主(UID)
    │
    ▼
获取博主信息 → 存入 blogger 表
    │
    ▼
调用微博移动端API获取微博列表
    │  GET https://m.weibo.cn/api/container/getIndex
    │  参数: containerid=107603_{uid}, page={n}
    │  Header: Cookie: SUB={cookie}
    │
    ▼
解析返回JSON → 提取 mblog 字段
    │
    ▼
去重(weibo_id) → 存入 weibo_post 表
    │
    ▼
提取图片URL → 存入 weibo_image 表
    │
    ▼
翻页继续(直到无新微博或达到上限)
```

**关键参数 (config.yaml):**

```yaml
crawler:
  cookie: "SUB=xxx"              # 微博登录Cookie
  max_pages: 10                   # 每次抓取最大页数
  delay: 3                        # 请求间隔(秒)
  timeout: 30                     # 请求超时(秒)
```

#### 2.3.2 图片下载流程

```
遍历 weibo_image 表 (downloaded=False)
    │
    ▼
下载图片 → 保存到 static/images/{blogger_id}/
    │
    ▼
更新 local_path, downloaded=True
```

**图片URL转换规则:**
- 原始 URL: `https://wx1.sinaimg.cn/orj480/xxx.jpg`
- 大图 URL: 将 `orj480` 替换为 `large`

#### 2.3.3 定时调度

APScheduler 配置 (config.yaml):

```yaml
scheduler:
  enabled: true
  crawl_interval: 180             # 抓取间隔(分钟)
  image_download_interval: 60     # 图片下载间隔(分钟)
  market_data_interval: 5         # 行情更新间隔(分钟)
```

每个抓取周期:
1. 遍历所有 `is_active=True` 的博主
2. 依次抓取每个博主的新微博
3. 请求间延迟 `delay` 秒
4. 异常时跳过当前博主，继续下一个

### 2.4 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/bloggers` | 获取所有博主列表及微博数量 |
| POST | `/api/bloggers` | 添加博主 (body: {uid, name}) |
| DELETE | `/api/bloggers/{id}` | 删除博主 |
| POST | `/api/crawl/{blogger_id}` | 手动触发单个博主抓取 |
| POST | `/api/crawl-all` | 手动触发全部博主抓取 |
| GET | `/api/posts/{blogger_id}` | 获取指定博主微博列表 |
| GET | `/api/posts/{blogger_id}/latest` | 获取最新N条微博 |
| GET | `/api/images/{post_id}` | 获取指定微博的图片列表 |
| POST | `/api/images/download` | 触发图片下载 |

---

## 三、智能选股系统

### 3.1 整体流程

```
微博内容
    │
    ▼
┌─────────────────────────────────────┐
│  实体提取 (LLM优先 → 正则回退)       │
│  - 股票名称 + 代码                   │
│  - 行业/板块                         │
│  - 情感: bullish/bearish/neutral    │
│  - 交易信号: 买入/卖出/加仓/减仓/建仓/清仓 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  时序聚合                            │
│  - 指数时间衰减 (半衰期7天)           │
│  - 信号密度加成                      │
│  - 立场一致性加成                    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  持仓画像                            │
│  - 看好/可能持仓 (bullish)           │
│  - 看空/已清仓 (bearish)            │
│  - 仅讨论/提到 (neutral)            │
│  - 重仓板块                          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  多维选股 (结合行情数据)              │
│  - 博主观点因子                      │
│  - 行情技术因子                      │
│  - 多因子综合打分                    │
└─────────────────────────────────────┘
```

### 3.2 实体提取模块

文件: `app/analyzer/llm_extractor.py`

#### 3.2.1 LLM 提取 (主路径)

**数据源:** DeepSeek API (OpenAI 兼容接口)

**配置 (config.yaml):**

```yaml
deepseek:
  api_key: "sk-xxx"
  base_url: "https://api.deepseek.com"
  model: "deepseek-chat"
  timeout: 15                    # 单次调用超时(秒)
```

**批量提取策略:** 一次发送一个博主的最近 30 条微博，一次 LLM 调用提取所有实体。

**Prompt 设计:**

```
你是一个金融文本分析专家。请分析以下微博文本，提取：
1. 提到的股票（名称+代码如果能推断）
2. 提到的行业/板块
3. 对每只股票/板块的情感倾向：bullish(看好)/bearish(看空)/neutral(中性/仅讨论)
4. 是否有明确交易信号（买入/卖出/加仓/减仓/建仓/清仓）

以JSON格式返回，结构如下：
{
  "stocks": [
    {"name": "贵州茅台", "code": "600519", "sentiment": "bullish", "signal": "买入", "confidence": 0.8}
  ],
  "sectors": [
    {"name": "白酒", "sentiment": "bullish", "confidence": 0.7}
  ]
}
```

**数据结构:**

```python
@dataclass
class ExtractedStock:
    name: str                    # 股票名称
    code: str | None             # 股票代码
    sentiment: str               # bullish / bearish / neutral
    signal: str | None           # 买入/卖出/加仓/减仓/建仓/清仓
    confidence: float            # 0.0 ~ 1.0

@dataclass
class ExtractedSector:
    name: str                    # 板块名称
    sentiment: str               # bullish / bearish / neutral
    confidence: float            # 0.0 ~ 1.0

@dataclass
class ExtractionResult:
    stocks: list[ExtractedStock]
    sectors: list[ExtractedSector]
    method: str                  # "llm" / "regex"
```

**缓存机制:** 以微博文本批次的 MD5 hash 为 key，缓存 ExtractionResult，避免重复 API 调用。

**异常处理:** 超时/API 异常 → 返回空 ExtractionResult，不阻塞主流程，自动回退到正则提取。

#### 3.2.2 正则提取 (回退路径)

文件: `app/analyzer/stock_analyzer.py` → `_extract_stocks()`

**提取规则:**

| 模式 | 示例 | 说明 |
|------|------|------|
| `$股票名$` | `$贵州茅台$` | 标准标记格式 |
| `$股票名(代码)$` | `$中际旭创(300308)$` | 带代码标记 |
| `$$板块$$` | `$$半导体$$` | 板块标记 |

**行业关键词匹配 (INDUSTRY_KEYWORDS):**

内置 20+ 个行业的关键词表，匹配微博正文中出现的行业词汇:

| 行业 | 代表关键词 |
|------|-----------|
| 半导体 | 芯片, 半导体, 晶圆, 封装, 光刻 |
| 新能源 | 光伏, 风电, 新能源, 碳中和 |
| 白酒 | 茅台, 五粮液, 白酒, 酒企 |
| 军工 | 军工, 航天, 国防, 兵器 |
| AI算力 | 算力, AI, GPU, 大模型, 智算 |
| ... | ... |

### 3.3 时序聚合模块

文件: `app/analyzer/stock_analyzer.py` → `_temporal_aggregate()`

对每只股票的所有信号进行时序聚合，输出加权分数。

#### 3.3.1 指数时间衰减

**公式:**

$$W_{time} = W_{recent} \times 2^{-\frac{\Delta d}{T_{half}}}$$

其中:
- $W_{recent} = 1.0$ (最近观点权重)
- $T_{half} = 7$ (半衰期，天)
- $\Delta d$ = 发文时间距今天数
- 下限: $W_{old} = 0.15$

**衰减示例:**

| 距今天数 | 权重 | 含义 |
|---------|------|------|
| 0 (今天) | 1.000 | 最高权重 |
| 1 | 0.905 | |
| 3 | 0.741 | |
| 7 | 0.500 | 半衰期 |
| 14 | 0.250 | |
| 30 | 0.150 | 到达下限 |
| 60+ | 0.150 | 最低权重 |

#### 3.3.2 信号密度加成

检测同一股票在短时间窗口内的提及密度，窗口定义:

$$Windows = [1天, 3天, 7天, 14天, 30天]$$

**加成规则:**

从最短窗口开始检测，命中即停止:

| 窗口内提及次数 | 加成公式 | 示例 |
|--------------|---------|------|
| ≥3次 | $B_{density} = 1.3^{\min(n-2, 3)} \times (1 + 0.1 \times \frac{30}{W})$ | 3天内3次: 1.3×1.0=1.3; 3天内5次: 1.3³×2.0=2.197 |
| ≥2次 | $B_{density} = 1.3^{0.5} \times (1 + 0.05 \times \frac{30}{W})$ | 7天内2次: 1.14×1.21=1.38 |
| 1次 | $B_{density} = 1.0$ | 无加成 |

其中 $W$ 为窗口天数，$n$ 为窗口内提及次数。**窗口越短加成越大。**

#### 3.3.3 立场一致性加成

仅计算时间权重 > 0.1 的近期信号（约30天内）。

**计算逻辑:**

```
dominant_ratio = max(bullish_count, bearish_count) / total_recent_signals
```

| dominant_ratio | 加成 | 含义 |
|---------------|------|------|
| ≥ 0.7 | $B_{consistency} = 1.2 \times dominant\_ratio$ (0.84~1.2) | 立场一致，加成 |
| 0.5 ~ 0.7 | $B_{consistency} = 1.0$ | 中性 |
| < 0.5 | $B_{consistency} = 0.7$ | 立场矛盾，减权 |

#### 3.3.4 综合加权分数

**公式:**

$$Score_{final} = \left(\sum_{i=1}^{n} S_i \times W_{time,i}\right) \times B_{density} \times B_{consistency}$$

其中:
- $S_i$ = 第 i 条信号的原始分数
- $W_{time,i}$ = 第 i 条信号的时间衰减权重
- $B_{density}$ = 信号密度加成
- $B_{consistency}$ = 立场一致性加成

**辅助指标:**

$$TW\_Count = \sum_{i=1}^{n} W_{time,i}$$

$$Confidence = \frac{Score_{final}}{TW\_Count}$$

### 3.4 信号检测模块

文件: `app/analyzer/stock_analyzer.py` → `_analyze_signal_with_context()`

#### 3.4.1 信号类型与关键词

| 信号类型 | 关键词 | 权重 |
|---------|--------|------|
| **buy** (买入) | 买入, 加仓, 建仓, 抄底, 补仓, 买入看, 越跌越买 | 3.0 |
| **hold** (持有) | 持有, 持仓, 拿住, 不卖, 继续持有, 躺平, 坚守 | 2.5 |
| **sell** (卖出) | 卖出, 清仓, 减仓, 止盈, 止损, 割肉, 离场 | 2.5 |
| **no_position** (不持有) | 没有持仓, 没买, 不碰, 避开, 不建议, 没有仓位 | 2.0 |
| **watch** (关注) | 关注, 跟踪, 观察, 留意, 盯着, 注意 | 1.5 |

#### 3.4.2 否定词检测

检测信号是否被否定词翻转:

| 否定词 | 示例 |
|--------|------|
| 不/不要/别/不要 | "不要买入" → 买入信号被否定 |
| 没/没有 | "没有加仓计划" → 加仓信号被否定 |
| 不建议/不建议 | "不建议追高" → 买入信号被否定 |

**逻辑:** 否定词出现在信号关键词之前且距离 ≤ 3 个字 → 信号被翻转。

#### 3.4.3 情感判断 (正则路径)

| 信号类型 | 情感分类 | 基础分数 |
|---------|---------|---------|
| buy / hold (未被否定) | bullish | confidence × 3.0 |
| sell / no_position | bearish | confidence × 2.0 |
| 其他 (watch等) | neutral | 0.3 |

**情感判断 (LLM路径):** 直接使用 LLM 返回的 `sentiment` 字段 (bullish/bearish/neutral)。

### 3.5 持仓画像模块

文件: `app/analyzer/stock_analyzer.py` → `_analyze_likely_positions()`

#### 3.5.1 三分类体系

经过时序聚合后，根据主导情感 (dominant sentiment) 将股票分为三类:

| 分类 | 条件 | 最低分数阈值 | 说明 |
|------|------|------------|------|
| **likely_positions** (看好/可能持仓) | dominant_sentiment = bullish | 1.0 | 博主看好的个股 |
| **bearish_positions** (看空/已清仓) | dominant_sentiment = bearish | 0.5 | 博主看空的个股 |
| **mentioned_positions** (仅讨论/提到) | dominant_sentiment = neutral | 0.3 | 仅提及无明确立场 |

#### 3.5.2 冲突消解

如果一只股票同时出现在买入信号和卖出信号中:

```
如果 sentiment=bullish 且 股票在 sell_stocks 中 且 不在 buy_stocks 和 hold_stocks 中:
    排除该股票 (不放入看好列表)
```

#### 3.5.3 排序与截断

每类按 `score` 降序排列，取 Top 8。

#### 3.5.4 输出字段

```json
{
  "name": "中际旭创",
  "score": 5.62,
  "mention_count": 12,
  "latest_time": "2026-05-27T14:30:00",
  "sector": "光模块/AI算力",
  "sentiment": "bullish",
  "signal": "加仓",
  "confidence": 0.85,
  "density_bonus": 1.44,
  "consistency_bonus": 1.20,
  "reasons": ["加仓", "看好", "买入"],
  "contexts": ["[2026-05-27] 今天加仓中际旭创，AI算力确定性最高..."]
}
```

### 3.6 板块分析模块

文件: `app/analyzer/stock_analyzer.py` → `_analyze_positions()`

#### 3.6.1 板块识别 (三重来源)

| 来源 | 优先级 | 说明 |
|------|--------|------|
| LLM 提取的 sectors | 最高 | 直接识别板块名称 |
| 股票→板块映射 (STOCK_SECTOR_MAP) | 中 | 通过识别出的股票名称反查所属板块 |
| INDUSTRY_KEYWORDS 关键词匹配 | 低 | 正则匹配微博正文中的行业关键词 |

#### 3.6.2 STOCK_SECTOR_MAP

文件: `app/data/stock_sector_map.py`

内置 **627 只 A 股** → (板块, 代码) 映射，覆盖 28 个行业:

| 行业 | 代表股票 | 数量 |
|------|---------|------|
| 白酒 | 贵州茅台, 五粮液, 泸州老窖 | 15 |
| 半导体 | 中芯国际, 北方华创, 中微公司 | 35 |
| 光模块/AI算力 | 中际旭创, 新易盛, 天孚通信 | 18 |
| 新能源/光伏 | 隆基绿能, 通威股份, 阳光电源 | 25 |
| 锂电 | 宁德时代, 亿纬锂能, 欣旺达 | 22 |
| 医药 | 恒瑞医药, 药明康德, 迈瑞医疗 | 40 |
| 军工 | 中航沈飞, 航发动力, 中航光电 | 30 |
| 金融 | 招商银行, 中国平安, 中信证券 | 25 |
| ... | ... | ... |

**查询函数:**

```python
get_stock_sector("中际旭创")  → "光模块/AI算力"
get_stock_code("中际旭创")    → "300308"
get_sector_stocks("半导体")   → [("中芯国际", "688981"), ("北方华创", "002371"), ...]
```

#### 3.6.3 SECTOR_ROLLUP (板块归并)

将细分子板块归并为大类板块，共 280+ 条映射:

```python
SECTOR_ROLLUP = {
    "白酒": "白酒",
    "食品": "消费",
    "乳制品": "消费",
    "调味品": "消费",
    "光模块": "光模块/AI算力",
    "AI算力": "光模块/AI算力",
    "GPU": "光模块/AI算力",
    "半导体设备": "半导体",
    "半导体材料": "半导体",
    "芯片设计": "半导体",
    ...
}
```

#### 3.6.4 板块权重计算

**LLM 路径:**

| sentiment | 权重公式 |
|-----------|---------|
| bullish | $+confidence \times 3$ |
| bearish | $-confidence \times 1$ |
| neutral | $+confidence \times 0.5$ |

股票→板块映射权重:

| sentiment | 权重公式 |
|-----------|---------|
| bullish | $+confidence \times 2$ |
| bearish | $-confidence \times 0.5$ |
| neutral | $+confidence \times 0.5$ |

**正则路径:**

板块权重 = `∑(time_weight)` (每次关键词匹配累加时间衰减权重)

股票→板块映射: $+count \times 0.8$

#### 3.6.5 板块时序密度加成

与个股密度加成逻辑相同，对 `industry_signal_times` 中的时间戳列表检测短窗口密度:

```python
# 检测窗口内的信号密度
for window in [1, 3, 7, 14, 30]:
    recent_count = count_signals_within(times, window)
    if recent_count >= 3:
        industry_weight *= density_bonus  # 短窗口多次提及 → 加成
        break
```

#### 3.6.6 板块情感判断

| 条件 | 情感标签 |
|------|---------|
| LLM 可用且 bullish_w > bearish_w | bullish |
| LLM 可用且 bearish_w > bullish_w | bearish |
| 正则路径且 weight > 3 | bullish |
| 其他 | neutral |

### 3.7 行情数据模块

文件: `app/market/`

#### 3.7.1 数据源

| 数据源 | 接口 | 数据内容 |
|--------|------|---------|
| 新浪财经 | `https://hq.sinajs.cn/list={codes}` | 实时行情 (价格/涨跌幅/成交量) |
| 腾讯财经 | `https://qt.gtimg.cn/q={codes}` | 实时行情 (备用源) |

**数据字段:**

```python
@dataclass
class MarketQuote:
    code: str              # 股票代码
    name: str              # 股票名称
    price: float           # 当前价格
    change_percent: float  # 涨跌幅(%)
    volume: int            # 成交量
    turnover: float        # 成交额
    high: float            # 最高价
    low: float             # 最低价
    open: float            # 开盘价
    prev_close: float      # 昨收价
    timestamp: datetime    # 行情时间
```

**请求格式:**
- 新浪: `s_sh600519` (简版), `sh600519` (完整版)
- 腾讯: `sh600519`

**降级策略:** 新浪超时 → 切换腾讯源。

#### 3.7.2 历史数据

通过新浪财经接口获取日K数据:

```
https://quotes.sina.cn/cn/api/jsonp_v2.php/IO.XSRV2.CallbackList/xxx/CN_MarketDataService.getKLineData?symbol={code}&scale=240&ma=no&datalen=250
```

返回最近 250 个交易日的 OHLCV 数据。

### 3.8 多维选股引擎

文件: `app/selector/stock_selector.py`

#### 3.8.1 选股维度

| 维度 | 数据来源 | 权重 | 说明 |
|------|---------|------|------|
| 博主观点 | Analyzer 模块 | 40% | 多博主交叉验证 |
| 行情趋势 | 新浪/腾讯行情 | 25% | 涨跌幅/均线趋势 |
| 技术指标 | 历史K线计算 | 20% | MACD/RSI/布林带 |
| 基本面 | 市值/PE/成交量 | 15% | 流动性/估值筛选 |

#### 3.8.2 博主观点因子

**多博主交叉验证逻辑:**

$$Factor_{blogger} = \sum_{b \in Bloggers} Score_{b}(stock) \times Weight_{b}$$

其中 $Weight_b$ 为博主权重 (默认等权，可按粉丝数/历史准确率加权)。

**交叉验证加成:**

| 提及博主数 | 加成系数 |
|-----------|---------|
| 1 | 1.0 |
| 2 | 1.5 |
| ≥3 | 2.0 |

#### 3.8.3 多因子打分模型

文件: `app/selector/multi_factor_model.py`

**因子列表:**

| 因子 | 计算方式 | 方向 |
|------|---------|------|
| momentum_5d | 5日涨幅 | 正向 |
| momentum_20d | 20日涨幅 | 正向 |
| volume_ratio | 当日量/20日均量 | 正向 |
| ma_position | 价格相对20日均线位置 | 正向 |
| rsi_14 | 14日RSI | 中性(30-70最佳) |
| blogger_sentiment | 博主观点综合分 | 正向 |

**打分公式:**

$$Score_{multi} = \sum_{i} w_i \times normalize(F_i)$$

其中 $normalize$ 为 Min-Max 归一化到 [0, 1]，$w_i$ 为因子权重。

#### 3.8.4 趋势预测器

文件: `app/selector/predictor.py`

基于历史K线的简单趋势判断:

| 指标 | 判断规则 |
|------|---------|
| 均线多头 | MA5 > MA10 > MA20 → 看多 |
| 均线空头 | MA5 < MA10 < MA20 → 看空 |
| 量价配合 | 价涨量增 → 确认趋势 |
| 背离 | 价创新高量缩 → 警惕反转 |

#### 3.8.5 回测框架

文件: `app/selector/backtest.py`

| 参数 | 默认值 |
|------|--------|
| 初始资金 | 1,000,000 |
| 手续费率 | 0.1% |
| 滑点 | 0.2% |
| 最大持仓 | 10 只 |
| 调仓频率 | 周频 |

### 3.9 分析结果数据结构

```python
@dataclass
class AnalysisResult:
    # 统计
    total_posts: int = 0                    # 分析的微博总数
    analysis_time: str = ""                 # 分析时间

    # 个股持仓
    likely_positions: List[Dict] = field(default_factory=list)    # 看好/可能持仓
    bearish_positions: List[Dict] = field(default_factory=list)   # 看空/已清仓
    mentioned_positions: List[Dict] = field(default_factory=list) # 仅讨论/提到

    # 板块
    hot_industries: List[Dict] = field(default_factory=list)      # 重仓板块
    industry_distribution: Dict = field(default_factory=dict)     # 板块分布

    # 信号
    buy_signals: List[Dict] = field(default_factory=list)        # 买入信号
    sell_signals: List[Dict] = field(default_factory=list)       # 卖出信号
    watch_signals: List[Dict] = field(default_factory=list)      # 关注信号

    # 交易风格
    trading_style: Dict = field(default_factory=dict)            # 交易风格画像

    # 实用报告
    practical_report: Dict = field(default_factory=dict)         # 生成报告
```

### 3.10 实用报告生成

文件: `app/analyzer/stock_analyzer.py` → `_generate_practical_report()`

从分析结果中提炼可操作的建议:

| 报告项 | 生成逻辑 |
|--------|---------|
| 重点关注 | likely_positions 前3名，带板块和信号 |
| 风险提示 | bearish_positions 中的个股 |
| 板块倾向 | hot_industries 前3名 |
| 交易风格 | 根据 buy/sell/hold 比例判断 (积极/稳健/保守) |
| 操作建议 | 综合信号强度给出 (加仓/持有/减仓/观望) |

---

## 四、API 接口总览

### 4.1 博主管理

| 方法 | 路径 | 参数 | 返回 |
|------|------|------|------|
| GET | `/api/bloggers` | - | `{bloggers: [{id, name, weibo_uid, followers_count, post_count, is_active}]}` |
| POST | `/api/bloggers` | `{uid, name}` | 添加结果 |
| DELETE | `/api/bloggers/{id}` | - | 删除结果 |
| PATCH | `/api/bloggers/{id}/toggle` | - | 切换 is_active |

### 4.2 内容抓取

| 方法 | 路径 | 参数 | 返回 |
|------|------|------|------|
| POST | `/api/crawl/{blogger_id}` | - | `{success, new_count, posts}` |
| POST | `/api/crawl-all` | - | `{results: [{blogger_id, new_count}]}` |
| GET | `/api/posts/{blogger_id}` | `?page=1&per_page=20` | `{posts: [...], total, page}` |
| GET | `/api/posts/{blogger_id}/latest` | `?limit=10` | `{posts: [...]}` |

### 4.3 量化分析

| 方法 | 路径 | 参数 | 返回 |
|------|------|------|------|
| GET | `/analyze/{blogger_id}` | - | 完整 AnalysisResult JSON |
| GET | `/smart-pick` | `?strategy=blogger_cross` | 多博主交叉选股结果 |

### 4.4 行情数据

| 方法 | 路径 | 参数 | 返回 |
|------|------|------|------|
| GET | `/api/market/quote/{code}` | - | MarketQuote |
| GET | `/api/market/kline/{code}` | `?period=day&count=250` | K线数据 |

### 4.5 图片

| 方法 | 路径 | 参数 | 返回 |
|------|------|------|------|
| GET | `/api/images/{post_id}` | - | 图片列表 |
| POST | `/api/images/download` | - | 下载结果 |

---

## 五、定时任务

| 任务 | 间隔 | 说明 |
|------|------|------|
| 微博内容抓取 | 180 分钟 | 抓取所有活跃博主的新微博 |
| 图片下载 | 60 分钟 | 下载未下载的图片 |
| 行情数据更新 | 5 分钟 | 更新关注股票的实时行情 |

---

## 六、配置项总览

文件: `config.yaml`

```yaml
# 微博爬虫
crawler:
  cookie: "SUB=xxx"              # 微博Cookie (必填)
  max_pages: 10                   # 单次抓取最大页数
  delay: 3                        # 请求间隔(秒)
  timeout: 30                     # 请求超时(秒)

# DeepSeek LLM
deepseek:
  api_key: ""                     # API Key (空则回退正则)
  base_url: "https://api.deepseek.com"
  model: "deepseek-chat"
  timeout: 15

# 行情数据
market:
  source: "sina"                  # sina / tencent
  update_interval: 300            # 更新间隔(秒)

# 定时调度
scheduler:
  enabled: true
  crawl_interval: 180
  image_download_interval: 60
  market_data_interval: 5

# 数据库
database:
  path: "weibo_crawler.db"

# 服务器
server:
  host: "0.0.0.0"
  port: 8000
```

---

## 七、关键算法速查

| 算法 | 公式 | 参数 |
|------|------|------|
| 时间衰减 | $W = 1.0 \times 2^{-d/7}$ | 半衰期=7天, 下限=0.15 |
| 密度加成 | $B = 1.3^{\min(n-2,3)} \times (1+0.1 \times 30/W)$ | 窗口=[1,3,7,14,30]天 |
| 一致性加成 | $B = 1.2 \times ratio$ (ratio≥0.7) / $0.7$ (ratio<0.5) | 阈值=0.7/0.5 |
| 综合得分 | $Score = (\sum S_i \times W_i) \times B_{density} \times B_{consistency}$ | - |
| 买入信号 | $S = confidence \times 3.0$ | 未被否定词翻转 |
| 卖出信号 | $S = confidence \times 2.0$ | - |
| 板块权重(LLM-bullish) | $W = confidence \times 3$ | - |
| 板块权重(LLM-bearish) | $W = -confidence \times 1$ | 负权重 |
| 交叉验证加成 | 1博主=1.0, 2博主=1.5, ≥3博主=2.0 | - |
