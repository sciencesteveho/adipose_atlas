"""Computes a symmetric cell-state x cell-state matrix."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import anndata as ad  # type: ignore
import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial import cKDTree  # type: ignore

from adipose_atlas.spatial_stress import _get_spatial_coords
from adipose_atlas.utils.config import SpatialConfig


@dataclass(frozen=True)
class ProximityMatrix:
    """Result of proximity matrix computation."""

    mean_distance_um: pd.DataFrame
    pair_counts: pd.DataFrame


class CellStateProximityMatrix:
    """Compute mean-distance proximity matrices between spatial cell states."""

    def __init__(
        self,
        adata: ad.AnnData,
        *,
        spatial: SpatialConfig,
        cell_state_key: str,
        radius_um: float = 300.0,
        sample_key: Optional[str] = None,
        exclude_states: Tuple[str, ...] = ("Unassigned",),
        state_order: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Args:
          spatial: Spatial coordinate configuration.
          cell_state_key: obs key containing spatial cell-state labels.
          radius_um: Maximum distance (µm) for including a cell-cell pair.
          sample_key: Optional obs key to compute within each sample separately.
          exclude_states: State labels to ignore entirely.
          state_order: Optional explicit order of states in the output matrix.
        """
        self.adata = adata
        self.spatial = spatial
        self.cell_state_key = cell_state_key
        self.radius_um = float(radius_um)
        self.sample_key = sample_key
        self.exclude_states = exclude_states
        self.state_order = list(state_order) if state_order is not None else None

        self.matrix = self._compute_proximity_matrix()

    def _compute_proximity_matrix(self) -> ProximityMatrix:
        """Build a symmetric state x state matrix of mean pairwise distances and
        pair counts.

        Find all cell pairs within radius_um accumulating distances
        symmetrically so each pair is counted once. When sample_key is provided,
        trees are built per sample to avoid measuring distances across sections.
        """
        if self.cell_state_key not in self.adata.obs.columns:
            raise KeyError(f"Required obs key '{self.cell_state_key}' not found.")

        if (
            self.sample_key is not None
            and self.sample_key not in self.adata.obs.columns
        ):
            raise KeyError(f"sample_key '{self.sample_key}' not found in adata.obs.")

        coordinates = _get_spatial_coords(self.adata, self.spatial.spatial_xy_key)
        states = self.adata.obs[self.cell_state_key].astype(str).to_numpy()

        keep = ~pd.Series(states).isin(self.exclude_states).to_numpy()
        coordinates = coordinates[keep]
        states = states[keep]

        if self.state_order is None:
            state_levels = sorted(pd.unique(states).tolist())
        else:
            present = set(pd.unique(states).tolist())
            state_levels = [s for s in self.state_order if s in present]

        if len(state_levels) == 0:
            raise ValueError("No states remain after exclusions / ordering.")

        state_to_idx: Dict[str, int] = {
            state: i for i, state in enumerate(state_levels)
        }
        state_indices = np.array([state_to_idx[s] for s in states], dtype=np.int32)

        n_states = len(state_levels)
        sum_dist = np.zeros((n_states, n_states), dtype=float)
        count = np.zeros((n_states, n_states), dtype=np.int64)

        if self.sample_key is None:
            logger.info(
                f"Computing proximity matrix on all cells (n={coordinates.shape[0]})."
            )
            self._accumulate_pairwise_distances(
                coordinates=coordinates,
                state_indices=state_indices,
                sum_dist=sum_dist,
                count=count,
            )
        else:
            sample_groups = self.adata.obs[self.sample_key].astype(str).to_numpy()[keep]
            for group in pd.unique(sample_groups):
                group_mask = np.where(sample_groups == group)[0]
                if group_mask.size == 0:
                    continue
                logger.info(
                    f"Computing proximity contributions for sample '{group}' (n={group_mask.size})."
                )
                self._accumulate_pairwise_distances(
                    coordinates=coordinates[group_mask],
                    state_indices=state_indices[group_mask],
                    sum_dist=sum_dist,
                    count=count,
                )

        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # account for no pairs observed
            mean_distances = sum_dist / count
            mean_distances[count == 0] = np.nan

        mean_df = pd.DataFrame(mean_distances, index=state_levels, columns=state_levels)
        count_df = pd.DataFrame(count, index=state_levels, columns=state_levels)

        return ProximityMatrix(mean_distance_um=mean_df, pair_counts=count_df)

    def _accumulate_pairwise_distances(
        self,
        coordinates: np.ndarray,
        state_indices: np.ndarray,
        sum_dist: np.ndarray,
        count: np.ndarray,
    ) -> None:
        """Add pairwise distances and counts for one spatial block into the
        running totals.

        Builds a KD-tree over the block, queries all pairs within radius_um,
        then accumulates symmetrically using only the upper triangle to avoid
        double-counting. Modifies sum_dist and count in place.
        """
        tree = cKDTree(coordinates)
        distance_matrix = tree.sparse_distance_matrix(
            tree,
            max_distance=self.radius_um,
            output_type="coo_matrix",
        )
        if distance_matrix.nnz == 0:
            return

        upper_triangle = distance_matrix.row < distance_matrix.col
        row_indices = distance_matrix.row[upper_triangle]
        col_indices = distance_matrix.col[upper_triangle]
        distances = distance_matrix.data[upper_triangle]

        state_a = state_indices[row_indices]
        state_b = state_indices[col_indices]

        np.add.at(sum_dist, (state_a, state_b), distances)
        np.add.at(sum_dist, (state_b, state_a), distances)
        np.add.at(count, (state_a, state_b), 1)
        np.add.at(count, (state_b, state_a), 1)
