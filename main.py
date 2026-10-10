"""CLI entrypoint for HF Dataset Toolkit mirroring Danbooru toolkit structure."""

import typer
from src.pipeline import HFDataPipeline

app = typer.Typer(
    help="CLI for streaming, curating, and uploading open multimodal datasets."
)


@app.command()
def build_dataset(
    config_path: str = typer.Option(
        "configs/default_config.yaml",
        help="Path to YAML configuration file.",
    ),
):
    """Streams datasets, creates balanced Parquet shards, and uploads to Hub."""
    pipeline = HFDataPipeline(config_path)
    pipeline.run_pipeline()


if __name__ == "__main__":
    app()
