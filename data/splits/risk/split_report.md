# Risk Split Freeze Report

## Source

- 源文件：`risk_nasdaq/risk_deepseek_cleaned_nasdaq_news_full.csv`
- SHA256：`33251916177feceebec6f2bec58e81cfbf10fb9115178504cb044498c20c7e73`
- logical CSV rows: 127176
- physical text lines including header: 2092987
- 行数口径：原始 Risk 数据包含 127176 条金融新闻记录；物理文本行数更多，因为 `Article` 字段的引号内嵌换行被计为文本行而非 CSV 记录。

## Cleaning

- raw rows: 127176
- after cleaning: 77748
- unique removed rows: 49428
- reason hits: 49445
- overlap: 17
- after dedup: 72275
- dropped reason counts: {"missing_label": 140, "missing_summary": 49305}

## Dedup

> 当前只进行了股票任务粒度的精确去重和时间切分，不声称完成事件级近似去重。

- URL stage: consistent_groups=0, conflicting_groups=0, rows_removed=0
- title+summary stage: consistent_groups=152, conflicting_groups=180, rows_removed=1361
- title stage: consistent_groups=1211, conflicting_groups=758, rows_removed=4112

## Split

| Split | Rows | Start Date | End Date |
| --- | ---: | --- | --- |
| Train | 50588 | 2009-07-07 | 2022-02-25 |
| Val | 10857 | 2022-02-26 | 2023-02-03 |
| Test | 10830 | 2023-02-04 | 2024-01-09 |
| eval_test | 500 | 2023-02-04 | 2023-12-16 |

## Label Distribution

### train
| label | count |
| ---: | ---: |
| 1 | 291 |
| 2 | 6672 |
| 3 | 39095 |
| 4 | 4494 |
| 5 | 36 |

### val
| label | count |
| ---: | ---: |
| 1 | 67 |
| 2 | 1783 |
| 3 | 7725 |
| 4 | 1270 |
| 5 | 12 |

### test
| label | count |
| ---: | ---: |
| 1 | 70 |
| 2 | 2221 |
| 3 | 7602 |
| 4 | 931 |
| 5 | 6 |

### eval_test
| label | count |
| ---: | ---: |
| 1 | 12 |
| 2 | 103 |
| 3 | 330 |
| 4 | 49 |
| 5 | 6 |

## Eval Quota

- eval_size: 500
- seed: 42
- allocation: minimum allocation + remaining-capacity proportional allocation + largest remainder + label ascending tie-break
- label 1: 12
- label 2: 103
- label 3: 330
- label 4: 49
- label 5: 6
- total: 500

## Integrity

- Train duplicate sample_id: 0
- Val duplicate sample_id: 0
- Test duplicate sample_id: 0
- eval_test duplicate sample_id: 0
- Train ∩ Val: 0
- Train ∩ Test: 0
- Val ∩ Test: 0
- eval_test outside Test: 0

## Reproducibility

- eval_test ordered sample_ids equal across two runs: True
- eval_test CSV SHA256 equal across two runs: True
- train.csv SHA256 equal across two runs: True
- val.csv SHA256 equal across two runs: True
- test.csv SHA256 equal across two runs: True
