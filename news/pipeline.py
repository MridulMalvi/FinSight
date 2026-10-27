import json
from pathlib import Path
from news.ingest import load_news_csv
from news.features import analyze_articles, aggregate_features


def process_csv(path: str, output_path: str | None = None):
    articles = load_news_csv(path)
    df = analyze_articles(articles)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


def build_feature_json(news_csv: str, symbols=None, output_path=None):
    articles = load_news_csv(news_csv)
    symbols = symbols or sorted({a.symbol for a in articles if a.symbol})
    result = {s: aggregate_features(articles, symbol=s) for s in symbols}
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=2))
    return result
