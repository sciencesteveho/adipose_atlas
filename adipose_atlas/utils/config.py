"""Configuration structures and loaders AT reproduction specific pipelines."""

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Type, TypeVar, Union

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class InitConfig:
    """Data loading with optional subsampling.

    Attributes:
        h5ad_path: Path to the integrated AnnData object.
        xenium_h5ad_path: Path to the Xenium object.
        subsample_n: Number of cells to subsample for faster processing.
        seed: Random seed for reproducibility.
    """

    h5ad_path: Path | None = None
    xenium_h5ad_path: Path | None = None
    subsample_n: int | None = None
    seed: int = 42


@dataclass(frozen=True)
class EmbeddingConfig:
    """Parameters for dimensionality reduction and embeddings.

    Attributes:
        pca_key: PCA column in AnnData.obsm
        n_neighbors: Number of neighbors for graph construction in
            scanpy.pp.neighbors
        umap_min_dist: UMAP minimum distance
        umap_spread: UMAP spread
        umap_random_state: Seed for UMAP
        tsne_perplexity: t-SNE perplexity
        tsne_random_state: Seed for t-SNE
    """

    pca_key: str = "X_pca"
    n_neighbors: int = 30
    umap_min_dist: float = 0.3
    umap_spread: float = 1.0
    umap_random_state: int = 42
    tsne_perplexity: float = 30.0
    tsne_random_state: int = 42


@dataclass(frozen=True)
class SpatialConfig:
    """Parameters for spatial AnnData.

    Attributes:
        spatial_xy_key: Key in obsm containing spatial x/y coords.
        seed: Random seed.
    """

    spatial_xy_key: str = "spatial"
    seed: int = 42


@dataclass(frozen=True)
class GlobalEmbeddingConfig:
    """Config for embedding of the global AT atlas.

    Attributes:
        loader: Shared loader config.
        embedding: Shared embedding config.
        output_dir: Output directory.
        cell_type_key: Primary cell type label.
        condition_key: Condition label ("Lean" or "Obese").
    """

    loader: InitConfig
    embedding: EmbeddingConfig
    output_dir: Path

    cell_type_key: str = "cell_type_level2"
    condition_key: str = "condition"


@dataclass(frozen=True)
class ProximityMatrixConfig:
    """Configuration for spatial proximity matrices.

    Attributes:
        cell_state_key: obs key for spatial cell-state labels.
        sample_key: Optional obs key to compute within each sample separately.
        radius_um: Maximum distance threshold (µm).
        exclude_states: State labels to ignore.
        state_order: Optional explicit ordering of states for display.
    """

    cell_state_key: str = "cell_state"
    sample_key: Optional[str] = None
    radius_um: float = 300.0
    exclude_states: Tuple[str, ...] = ("Unassigned",)
    state_order: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class StressHeatmapConfig:
    """Configuration for the spatial stress heatmap analysis."""

    cell_state_key: str = "cell_state"
    condition_key: str = "condition"
    niche_key: str = "niche"

    exclude_states: Tuple[str, ...] = ("Unassigned",)
    state_order: Optional[Tuple[str, ...]] = None
    stressed_states: Tuple[str, ...] = ()

    bin_um: float = 50.0
    layer_for_raw: Optional[str] = None
    score_key: str = "Stress_Score_raw"
    save_csv: bool = True


@dataclass(frozen=True)
class SpatialAnalysisConfig:
    """Configuration for spatial analyses."""

    loader: InitConfig
    spatial: SpatialConfig
    proximity: ProximityMatrixConfig
    stress: StressHeatmapConfig
    output_dir: Path

    stress_markers: Tuple[str, ...]


@dataclass(frozen=True)
class MyeloidLineageConfig:
    """Config for myeloid lineage embedding + marker dotplot reproduction."""

    loader: InitConfig
    embedding: EmbeddingConfig
    output_dir: Path

    cell_type_key: str = "cell_type_level1"
    myeloid_label: str = "Myeloid"
    cell_state_key: str = "cell_state"

    dotplot_genes: Tuple[str, ...] = ()


def _safe_load_yaml(yaml_file: Path) -> Dict[str, Any]:
    """Load YAML into a dict."""
    if not yaml_file.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_file}")

    with open(yaml_file, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Config YAML must be a mapping, got {type(data)}")

    return data


def _write_resolved_yaml(payload: Any, output_path: Union[str, Path]) -> None:
    """Write config out for saving. Converts dataclasses to dict; Paths dumped as strings."""
    yaml.SafeDumper.add_multi_representer(
        Path,
        lambda dumper, data: dumper.represent_scalar(
            "tag:yaml.org,2002:str", str(data)
        ),
    )

    if is_dataclass(payload) and not isinstance(payload, type):
        payload = asdict(payload)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return data[key] as a mapping, raising if missing or not a mapping."""
    if key not in data:
        raise ValueError(f"Missing required key '{key}' in YAML config.")
    value = data[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected '{key}' to be a mapping/dict in YAML config.")
    return value


def _as_path_or_none(value: Any) -> Path | None:
    """Coerce value into Optional[Path]."""
    if value is None:
        return None
    return Path(str(value))


def _as_tuple_str(value: Any) -> Tuple[str, ...]:
    """Coerce scalars/lists/tuples into Tuple[str, ...]."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


def _pick_known_fields(cls: Type[T], mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """Filter mapping down to dataclass fields for cls."""
    allowed = {f.name for f in fields(cls)}  # type: ignore
    return {k: v for k, v in mapping.items() if k in allowed}


def _parse_init(mapping: Mapping[str, Any]) -> InitConfig:
    """Parse InitConfig from YAML mapping."""
    data = _pick_known_fields(InitConfig, mapping)
    if "h5ad_path" in data:
        data["h5ad_path"] = _as_path_or_none(data["h5ad_path"])
    if "xenium_h5ad_path" in data:
        data["xenium_h5ad_path"] = _as_path_or_none(data["xenium_h5ad_path"])
    return InitConfig(**data)


def _parse_embedding(mapping: Mapping[str, Any]) -> EmbeddingConfig:
    """Parse EmbeddingConfig from YAML mapping."""
    data = _pick_known_fields(EmbeddingConfig, mapping)
    return EmbeddingConfig(**data)


def _parse_spatial(mapping: Mapping[str, Any], *, seed: int) -> SpatialConfig:
    """Parse SpatialConfig from YAML mapping."""
    data = _pick_known_fields(SpatialConfig, mapping)
    data.setdefault("seed", seed)
    return SpatialConfig(**data)


def _parse_proximity(mapping: Mapping[str, Any]) -> ProximityMatrixConfig:
    """Parse ProximityMatrixConfig from YAML mapping."""
    data = _pick_known_fields(ProximityMatrixConfig, mapping)
    if "exclude_states" in data:
        data["exclude_states"] = _as_tuple_str(data["exclude_states"]) or (
            "Unassigned",
        )
    if "state_order" in data:
        data["state_order"] = (
            None if data["state_order"] is None else _as_tuple_str(data["state_order"])
        )
    return ProximityMatrixConfig(**data)


def _parse_stress(
    mapping: Mapping[str, Any], *, fallback_exclude: Tuple[str, ...]
) -> StressHeatmapConfig:
    """Parse StressHeatmapConfig from YAML mapping."""
    data = _pick_known_fields(StressHeatmapConfig, mapping)

    if "exclude_states" in data:
        data["exclude_states"] = (
            _as_tuple_str(data["exclude_states"]) or fallback_exclude
        )
    else:
        data["exclude_states"] = fallback_exclude

    if "state_order" in data:
        data["state_order"] = (
            None if data["state_order"] is None else _as_tuple_str(data["state_order"])
        )

    if "stressed_states" in data:
        data["stressed_states"] = _as_tuple_str(data["stressed_states"])

    return StressHeatmapConfig(**data)


def _apply_parameter_overrides(
    loader: InitConfig,
    output_dir: Path,
    *,
    override_output_dir: Optional[Path] = None,
    override_subsample_n: Optional[int] = None,
    override_seed: Optional[int] = None,
) -> Tuple[InitConfig, Path]:
    """Apply command-line parameter overrides."""
    out_dir = output_dir if override_output_dir is None else override_output_dir

    if override_subsample_n is None and override_seed is None:
        return loader, out_dir

    return (
        replace(
            loader,
            subsample_n=(
                loader.subsample_n
                if override_subsample_n is None
                else override_subsample_n
            ),
            seed=loader.seed if override_seed is None else int(override_seed),
        ),
        out_dir,
    )


def load_global_embedding_config(
    yaml_file: Union[str, Path],
    *,
    override_output_dir: Optional[Union[str, Path]] = None,
    override_subsample_n: Optional[int] = None,
    override_seed: Optional[int] = None,
) -> GlobalEmbeddingConfig:
    """Load config for global atlas embedding."""
    cfg = _safe_load_yaml(Path(yaml_file))

    loader = _parse_init(cfg.get("loader", {}))
    embedding = _parse_embedding(cfg.get("embedding", {}))
    output_dir = Path(str(cfg.get("output_dir", "./results/global_atlas_embedding")))

    loader, output_dir = _apply_parameter_overrides(
        loader=loader,
        output_dir=output_dir,
        override_output_dir=Path(override_output_dir) if override_output_dir else None,
        override_subsample_n=override_subsample_n,
        override_seed=override_seed,
    )

    return GlobalEmbeddingConfig(
        loader=loader,
        embedding=embedding,
        output_dir=output_dir,
        cell_type_key=str(cfg.get("cell_type_key", "cell_type_level2")),
        condition_key=str(cfg.get("condition_key", "condition")),
    )


def load_myeloid_lineage_config(
    yaml_file: Union[str, Path],
    *,
    override_output_dir: Optional[Union[str, Path]] = None,
    override_subsample_n: Optional[int] = None,
    override_seed: Optional[int] = None,
) -> MyeloidLineageConfig:
    """Load config for myeloid lineage embedding + dotplot reproduction."""
    cfg = _safe_load_yaml(Path(yaml_file))

    if "dotplot_genes" not in cfg:
        raise ValueError("Missing required key 'dotplot_genes' in YAML config.")
    dotplot_genes = _as_tuple_str(cfg["dotplot_genes"])

    loader = _parse_init(cfg.get("loader", {}))
    embedding = _parse_embedding(cfg.get("embedding", {}))
    output_dir = Path(str(cfg.get("output_dir", "./results/myeloid_lineage_analysis")))

    loader, output_dir = _apply_parameter_overrides(
        loader=loader,
        output_dir=output_dir,
        override_output_dir=Path(override_output_dir) if override_output_dir else None,
        override_subsample_n=override_subsample_n,
        override_seed=override_seed,
    )

    return MyeloidLineageConfig(
        loader=loader,
        embedding=embedding,
        output_dir=output_dir,
        cell_type_key=str(cfg.get("cell_type_key", "cell_type_level1")),
        myeloid_label=str(cfg.get("myeloid_label", "Myeloid")),
        cell_state_key=str(cfg.get("cell_state_key", "cell_state")),
        dotplot_genes=dotplot_genes,
    )


def load_spatial_analysis_config(
    yaml_file: Union[str, Path],
    *,
    override_output_dir: Optional[Union[str, Path]] = None,
    override_subsample_n: Optional[int] = None,
    override_seed: Optional[int] = None,
) -> SpatialAnalysisConfig:
    """Load spatial analysis config (proximity matrix + stress heatmaps)."""
    cfg = _safe_load_yaml(Path(yaml_file))

    if "stress_markers" not in cfg:
        raise ValueError("Missing required key 'stress_markers' in YAML config.")
    stress_markers = _as_tuple_str(cfg["stress_markers"])

    loader = _parse_init(cfg.get("loader", {}))
    output_dir = Path(str(cfg.get("output_dir", "results/spatial_analysis")))

    loader, output_dir = _apply_parameter_overrides(
        loader=loader,
        output_dir=output_dir,
        override_output_dir=Path(override_output_dir) if override_output_dir else None,
        override_subsample_n=override_subsample_n,
        override_seed=override_seed,
    )

    spatial_cfg = _require_mapping(cfg, "spatial")
    spatial = _parse_spatial(spatial_cfg, seed=loader.seed)

    proximity_cfg = cfg.get("proximity", {})
    if not isinstance(proximity_cfg, Mapping):
        raise TypeError("Expected 'proximity' to be a mapping/dict in YAML config.")
    proximity = _parse_proximity(proximity_cfg)

    stress_cfg = cfg.get("stress", {})
    if not isinstance(stress_cfg, Mapping):
        raise TypeError("Expected 'stress' to be a mapping/dict in YAML config.")
    stress = _parse_stress(stress_cfg, fallback_exclude=proximity.exclude_states)

    return SpatialAnalysisConfig(
        loader=loader,
        spatial=spatial,
        proximity=proximity,
        stress=stress,
        output_dir=output_dir,
        stress_markers=stress_markers,
    )
