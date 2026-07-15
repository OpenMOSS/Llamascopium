from pathlib import Path
from typing import Annotated

import typer

from llamascopium.backend.language_model import LanguageModelConfig
from llamascopium.config import DatasetConfig
from llamascopium.database import MongoClient, MongoDBConfig
from llamascopium.models.sparse_dictionary import SparseDictionaryConfig

from .common import (
    DEFAULT_MONGO_DB,
    DEFAULT_MONGO_URI,
    DEFAULT_SAE_SERIES,
    MongoDBOption,
    MongoURIOption,
    SAESeriesOption,
)
from .utils import load_config

app = typer.Typer(help="Create database records for datasets, models, or SAEs.")


@app.command("dataset")
def create_dataset(
    name: Annotated[str, typer.Argument(help="Name of the dataset.")],
    config: Annotated[Path, typer.Argument(help="Path to DatasetConfig configuration file.")],
    mongo_uri: MongoURIOption = DEFAULT_MONGO_URI,
    mongo_db: MongoDBOption = DEFAULT_MONGO_DB,
) -> None:
    """Create a dataset record in the database."""
    client = MongoClient(MongoDBConfig(mongo_uri=mongo_uri, mongo_db=mongo_db))
    cfg = load_config(config)
    client.add_dataset(name=name, cfg=DatasetConfig.model_validate(cfg))
    typer.echo(f"Dataset '{name}' created successfully.")


@app.command("model")
def create_model(
    name: Annotated[str, typer.Argument(help="Name of the model.")],
    config: Annotated[Path, typer.Argument(help="Path to LanguageModelConfig configuration file.")],
    mongo_uri: MongoURIOption = DEFAULT_MONGO_URI,
    mongo_db: MongoDBOption = DEFAULT_MONGO_DB,
) -> None:
    """Create a model record in the database."""
    client = MongoClient(MongoDBConfig(mongo_uri=mongo_uri, mongo_db=mongo_db))
    cfg = load_config(config)
    client.add_model(name=name, cfg=LanguageModelConfig.model_validate(cfg))
    typer.echo(f"Model '{name}' created successfully.")


@app.command("sae")
def create_sae(
    name: Annotated[str, typer.Argument(help="Name of the SAE.")],
    path: Annotated[Path, typer.Argument(help="Path to the pretrained SAE directory.")],
    series: SAESeriesOption = DEFAULT_SAE_SERIES,
    mongo_uri: MongoURIOption = DEFAULT_MONGO_URI,
    mongo_db: MongoDBOption = DEFAULT_MONGO_DB,
    model_name: Annotated[
        str | None,
        typer.Option("--model-name", help="Language model associated with this SAE."),
    ] = None,
    metadata_only: Annotated[
        bool,
        typer.Option(
            "--metadata-only",
            help="Upsert only the SAE record without creating or modifying feature documents.",
        ),
    ] = False,
) -> None:
    """Create an SAE record in the database."""
    client = MongoClient(MongoDBConfig(mongo_uri=mongo_uri, mongo_db=mongo_db))
    cfg = SparseDictionaryConfig.from_pretrained(str(path))
    if metadata_only:
        client.upsert_sae_metadata(
            name=name,
            series=series,
            path=str(path),
            cfg=cfg,
            model_name=model_name,
        )
        typer.echo(f"SAE metadata '{name}' (series: {series}) upserted successfully; features were not modified.")
    else:
        client.create_sae(
            name=name,
            series=series,
            path=str(path),
            cfg=cfg,
            model_name=model_name,
        )
        typer.echo(f"SAE '{name}' (series: {series}) created successfully.")
