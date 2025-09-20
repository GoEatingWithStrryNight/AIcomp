# copy前看一下：

## 1. 数据处理

- 原始数据位于 `data/raw/`，包括：
  - `inter_preliminary.csv`：借阅交互记录
  - `item.csv`：图书信息
  - `user.csv`：用户信息
- 预处理脚本（`scripts/preprocess_step*.py`）完成了：
  - 列名标准化、编码统一、缺失值与异常处理
  - 时间字段解析，生成借阅时长、借阅时间特征
  - 用户/图书ID连续映射，生成映射表
  - 统计分析（如热门书籍、用户活跃度分布）
- 处理后数据存放于 `data/processed/`，主要文件：
  - `interactions_clean.csv`、`user_mapping.csv`、`item_mapping.csv` 等

### 处理后数据文件说明（data/processed/）

- `interactions_step1.csv`：原始交互表的字段标准化与简单清洗版本。
  - 字段：`inter_id`（若原始存在）、`user_id`、`book_id`、`borrow_time`、`return_time`、`renew_time`、`renew_cnt`
  - 规则：`user_id`、`book_id` 去除空白并转为字符串；其余列按原始值保留（未解析时间）。

- `items_step1.csv`：原始图书表的字段标准化。
  - 字段：`book_id`、`title`、`author`、`publisher`、`cat_lvl1`、`cat_lvl2`
  - 规则：`book_id` 去除空白并转为字符串。

- `users_step1.csv`：原始用户表的字段标准化。
  - 字段：`user_id`、`gender`、`dept`、`grade`、`user_type`
  - 规则：`user_id` 去除空白并转为字符串。

- `interactions_step2_time.csv`：在 step1 基础上完成时间解析与时间特征生成。
  - 时间列解析：`borrow_time`、`return_time`、`renew_time` 支持多种混合格式，无法解析的置为缺失。
  - 新增时间特征：
    - `borrow_year`、`borrow_month`、`borrow_dow`（星期一=0…星期日=6）、`borrow_hour`
    - `borrow_duration_days` = `return_time` − `borrow_time` 的天数（负值置为缺失）
    - `renew_gap_days` = `renew_time` − `borrow_time` 的天数（负值置为缺失）

- `interactions_clean.csv`：用于建模与统计的清洗后交互明细（核心明细表）。
  - 来源：对 `interactions_step2_time.csv` 进一步清洗。
  - 清洗规则：
    - 删除 `user_id` 或 `book_id` 缺失的记录
    - `renew_cnt` 转为整数（无法解析的按 0）
    - `borrow_duration_days` 大于 400 天的异常值置为缺失
  - 字段：继承 `interactions_step2_time.csv` 的全部列（含时间特征与解析后的时间列）。

- `user_stats.csv`：按用户统计的聚合特征。
  - 字段：
    - `user_id`
    - `user_interactions`：该用户的交互总次数
    - `user_unique_books`：该用户借阅过的不重复图书数

- `item_stats.csv`：按图书统计的聚合特征。
  - 字段：
    - `book_id`
    - `item_interactions`：该图书被借阅的总次数
    - `item_unique_users`：借过该书的不重复用户数

- `user_mapping.csv`：用户离散 ID 到连续索引的映射表（0 起始）。
  - 字段：`user_id`、`user_idx`
  - 用途：为图模型/矩阵分解等构造稠密索引。

- `item_mapping.csv`：图书离散 ID 到连续索引的映射表（0 起始）。
  - 字段：`book_id`、`item_idx`
  - 用途：与 `user_mapping.csv` 配套使用。

- `interactions_slim.parquet`：仅保留稠密索引对的轻量交互明细。
  - 字段：`user_idx`、`item_idx`
  - 特点：Parquet 列式存储，便于高效载入训练（如 LightGCN）。

- `global_stats.json`：全局统计信息与热门图书 TopN。
  - 关键键：
    - `n_rows`、`n_users`、`n_items`
    - `time_span_days`：时间跨度（按 `borrow_time` 计算）
    - `user_unique_books_quantiles`、`item_unique_users_quantiles`：分位数统计
    - `top_items`：`book_id` 到计数的映射（前 20）

说明：CSV 中的缺失以空值表示；时间列在写回 CSV 后为可解析的字符串格式，读取时可再次用 `pd.to_datetime` 解析。



## 2. 评估脚本

- 评估脚本（`eval_holdout.py`）支持时间窗口与Leave-One-Out（LOO）评估

  （注意这是伪评估，实际结果按照网站上给的f1分数为准）


# 目前方案介绍：

## 第一阶段：序列模型预测 (seqrec_gru.py)
## 第二阶段：重新排序 (rerank_lgcn_seq.py)

配好环境之后，直接 python xx.py即可

## 输出的文件：submission_seq_rerank.csv

如果使用评估脚本：

```bash
python eval.py --submission data/ans/submission_seq_rerank.csv --mode loo
```

