# Data Format

## Training file

Supported formats:

- `.csv`
- `.tsv`
- `.xlsx`
- `.xls`
- `.jsonl`

The current readability setup expects:

| Column | Description |
|---|---|
| `Sentence` | input text |
| `Rating` | raw readability label |

Training label mapping:

```text
2, 3 -> simple
5    -> complex
```

## Prediction files

Required columns:

| Column | Description |
|---|---|
| `Sentence` or `text` | evaluation text |
| `True Label` or `Label` | gold label |
| `Lang` | dataset/language identifier |

Evaluation label mapping:

```text
0, 2, 3 -> simple
1, 5    -> complex
```

## Overlap removal

Training examples whose text appears in the evaluation subset are removed before fitting MD, LOF, and ISOF.
