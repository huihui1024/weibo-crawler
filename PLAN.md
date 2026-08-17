# Plan: ML预测器真正训练上线

## 目标
让 `MLPredictor` 用真实市场数据训练 RandomForest 模型，替代当前 `_rule_based_predict` 规则引擎，输出带置信区间的预测结果。

## 核心思路

1. **数据采集层**：通过腾讯K线API批量获取股票池中所有股票约200个交易日的历史数据（OHLCV）
2. **特征工程**：从K线数据中提取技术因子（均线、动量、波动率、RSI、MACD、换手率、量价背离等），构建训练特征矩阵
3. **标签构建**：用T日特征预测T+1日涨跌幅分类（涨/平/跌）和回归（涨跌幅绝对值）
4. **模型训练**：用 RandomForest 训练分类+回归双模型，保存到 `data/ml_model.pkl`
5. **集成到 Predictor**：修改 `Predictor.predict()` 优先使用训练好的ML模型，降级到规则引擎
6. **置信区间**：用 RandomForest 的预测方差估计置信度

## 文件修改计划

### 1. 新建 `app/selector/data_collector.py` — 历史数据采集
- `collect_training_data(stock_codes, days=200)` — 批量获取K线数据
- `build_feature_label_pairs(klines)` — 从K线构建(特征, 标签)对
- `save_training_data(features, labels, path)` — 保存为pkl供训练使用

### 2. 重写 `app/selector/ml_predictor.py` — ML预测器
- 删除未使用的 LSTM/Transformer 代码（依赖CUDA，环境不支持）
- 实现 `FeatureEngineer` 类：从K线数据提取技术因子
- 实现 `StockMLModel` 类：RandomForest 分类+回归双模型
- `train()` — 加载采集数据，训练模型，评估精度，保存模型
- `predict()` — 加载模型，输入当日特征，输出预测+置信区间
- 输出格式：`{change_pct, direction, confidence, confidence_interval, trend, hold_days}`

### 3. 修改 `app/selector/predictor.py` — 集成ML预测
- `predict()` 方法：优先调用 `MLPredictor.predict()`，失败则降级到规则引擎
- `batch_predict()` 方法：同上逻辑
- 预测结果中新增 `confidence` 和 `confidence_interval` 字段

### 4. 修改 `app/database.py` — 新增置信度字段
- predictions 表新增 `confidence` (REAL) 和 `confidence_interval` (TEXT) 列
- 自动迁移逻辑

### 5. 新建 `train_model.py` — 一键训练脚本
- `python train_model.py` 即可完成数据采集+特征工程+模型训练+保存
- 输出训练精度报告

## 特征工程设计（15个因子）

| # | 因子名 | 计算方式 | 预测力预期 |
|---|--------|---------|-----------|
| 1 | return_1d | 当日涨跌幅 | 高 |
| 2 | return_5d | 5日累计涨跌幅 | 中 |
| 3 | return_10d | 10日累计涨跌幅 | 中 |
| 4 | ma5_ratio | close/MA5 - 1 | 高 |
| 5 | ma10_ratio | close/MA10 - 1 | 高 |
| 6 | ma20_ratio | close/MA20 - 1 | 中 |
| 7 | ma_cross | MA5>MA10>MA20 布尔编码 | 高 |
| 8 | volatility_5d | 5日收益率标准差 | 中 |
| 9 | volatility_20d | 20日收益率标准差 | 中 |
| 10 | volume_ratio | 当日量/MA5量 | 中 |
| 11 | rsi_14 | 14日RSI | 高 |
| 12 | macd_signal | MACD信号线方向 | 中 |
| 13 | amplitude | 当日振幅 | 低 |
| 14 | turnover_rate | 换手率变化 | 低 |
| 15 | price_position | (close-low20)/(high20-low20) | 中 |

## 标签设计

- **分类标签**：next_day_return > 0.5% → "涨"；< -0.5% → "跌"；其余 → "平"
- **回归标签**：next_day_return 精确值

## 训练验证

- 时间序列切分：前70%训练，后30%测试（非随机，避免未来信息泄漏）
- 分类指标：accuracy, precision, recall, F1
- 回归指标：MAE, RMSE, R²
- 基线对比：与规则引擎预测比较

## 置信区间

- 用 RandomForest 的各棵树预测方差估计
- `confidence` = 分类投票一致性（如80%树投"涨"→置信度0.8）
- `confidence_interval` = 回归预测的 [mean-1.96*std, mean+1.96*std]
