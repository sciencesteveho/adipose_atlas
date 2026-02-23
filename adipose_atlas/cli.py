"""Command-line entry points for AT single-cell analyses."""

import argparse
import sys
from pathlib import Path

from loguru import logger

from adipose_atlas.pipelines import (
    run_global_atlas_embedding,
    run_myeloid_lineage_analysis,
    run_spatial_analysis,
)
from adipose_atlas.utils.config import (
    load_global_embedding_config,
    load_myeloid_lineage_config,
    load_spatial_analysis_config,
)
from adipose_atlas.utils.logger import configure_logging


def _prepare_output_dir(path: Path) -> None:
    """Ensure output directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def _run_global_atlas_embedding(args: argparse.Namespace) -> None:
    """CLI entry for global atlas embedding reproduction."""
    configure_logging(level=str(args.log_level))
    cfg = load_global_embedding_config(
        args.config,
        override_output_dir=args.output_dir,
        override_subsample_n=args.subsample_n,
        override_seed=args.seed,
    )
    _prepare_output_dir(cfg.output_dir)
    logger.info("Starting analysis: global_atlas_embedding")
    run_global_atlas_embedding(cfg)
    logger.info("Completed analysis: global_atlas_embedding")


def _run_myeloid_lineage_analysis(args: argparse.Namespace) -> None:
    """CLI entry for myeloid state analysis."""
    configure_logging(level=str(args.log_level))
    cfg = load_myeloid_lineage_config(
        args.config,
        override_output_dir=args.output_dir,
        override_subsample_n=args.subsample_n,
        override_seed=args.seed,
    )
    _prepare_output_dir(cfg.output_dir)
    logger.info("Starting analysis: myeloid_lineage_analysis")
    run_myeloid_lineage_analysis(cfg)
    logger.info("Completed analysis: myeloid_lineage_analysis")


def _run_spatial_analysis(args: argparse.Namespace) -> None:
    """CLI entry for spatial analysis"""
    configure_logging(level=str(args.log_level))
    cfg = load_spatial_analysis_config(
        args.config,
        override_output_dir=args.output_dir,
        override_subsample_n=args.subsample_n,
        override_seed=args.seed,
    )
    _prepare_output_dir(cfg.output_dir)
    run_spatial_analysis(cfg)


def _add_config_arg(p: argparse.ArgumentParser) -> None:
    """Add config argument"""
    p.add_argument("--config", type=str, required=False, help="Path to YAML config.")


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser with subcommands."""

    parser = argparse.ArgumentParser(
        prog="at_single_cell",
        description="Selected reproduction/reimplementation of analyses from Miranda et al. (Nature, 2025).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level for loguru (default: INFO).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (applies to any analysis).",
    )
    parser.add_argument(
        "--subsample-n",
        type=int,
        default=None,
        help="Override subsampling (n_obs) for rapid iteration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed for deterministic runs.",
    )
    parser.add_argument(
        "--no-save-h5ad",
        action="store_true",
        help="Disable writing computed .h5ad artifacts.",
    )

    sub = parser.add_subparsers(title="analyses", dest="command", required=True)

    p1 = sub.add_parser(
        "global_atlas_embedding",
        help="Reproduce global atlas embedding panels and recompute embeddings from PCA.",
    )
    _add_config_arg(p1)
    p1.set_defaults(func=_run_global_atlas_embedding)

    p2 = sub.add_parser(
        "myeloid_state_analysis",
        help="Myeloid lineage embedding + dotplot reproduction (Fig. 2A/2B).",
    )
    _add_config_arg(p2)
    p2.set_defaults(func=_run_myeloid_lineage_analysis)

    p3 = sub.add_parser(
        "spatial_analysis", help="Spatial analysis: proximity matrix + stress."
    )
    _add_config_arg(p3)
    p3.set_defaults(func=_run_spatial_analysis)

    return parser


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "config", None) is None:
        parser.error("--config is required.")

    try:
        args.func(args)
    except Exception as exc:
        logger.exception(f"Unhandled exception: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
