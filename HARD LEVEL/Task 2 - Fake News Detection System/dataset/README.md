# Dataset Instructions

## Dataset Name

This project supports two standard, publicly available Fake and Real News datasets:

1. **Fake and Real News Dataset (McIntire / DataCamp Benchmark Dataset)** — `news.csv` (or `fake_or_real_news.csv`)
2. **ISOT Fake and Real News Dataset (University of Victoria / Kaggle)** — `Fake.csv` and `True.csv`

## Dataset Source

- **Option 1 (`news.csv` / `fake_or_real_news.csv`)**:
  - Source: Open-source Fake or Real News Dataset originally compiled by George McIntire and distributed via DataCamp / GitHub.
  - Direct Download URL: `https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv`
- **Option 2 (`Fake.csv` and `True.csv`)**:
  - Source: ISOT Research Lab (University of Victoria) / Kaggle "Fake and real news dataset" (`https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset`).

## How to Download It

### Method A: Download `news.csv` via PowerShell (Recommended)

Run the following command from the `Task 2 - Fake News Detection System` folder:

```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv" -OutFile "dataset/news.csv"
```

### Method B: Download `Fake.csv` and `True.csv` from Kaggle

1. Visit `https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset`.
2. Download the archive and extract `Fake.csv` and `True.csv`.
3. Place both files inside the `dataset/` directory.

## Expected File Names

Place one of the following inside the `dataset/` folder:

- `dataset/news.csv` (or `dataset/fake_or_real_news.csv`)
- OR both `dataset/Fake.csv` and `dataset/True.csv`

## Expected Columns

- For single-file datasets (`news.csv` or `fake_or_real_news.csv`):
  - `title` — Headline of the news article
  - `text` — Full body text of the news article
  - `label` — Ground-truth class (`FAKE` / `REAL`, or `0` / `1`)
  - Optional metadata columns: `id`, `subject`, `date`
- For two-file ISOT datasets (`Fake.csv` and `True.csv`):
  - `title` — Headline of the news article
  - `text` — Full body text of the news article
  - `subject` — News category or topic
  - `date` — Publication date
  - (`label` is assigned automatically as `FAKE` for `Fake.csv` and `REAL` for `True.csv`)

## Where to Place the Dataset

Place the CSV file(s) directly inside:

```text
HARD LEVEL/Task 2 - Fake News Detection System/dataset/
```

Note: Large `.csv` files are excluded from Git tracking via `.gitignore` so the repository remains lightweight.
