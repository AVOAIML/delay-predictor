"""maxxflow_features — medallion (bronze/silver/gold) IO on fsspec.

minio (S3) locally == ADLS Gen2 in prod; only the URI protocol differs, set by
``LAKE_URI``. Shared feature transforms also live here so module feature code is
written once.
"""

from maxxflow_features.cleaning import (
    CleaningReport, apply_impute_stats, apply_outlier_bounds, clean_frame,
    drop_duplicate_rows, fit_impute_stats, fit_outlier_bounds,
)
from maxxflow_features.lake import LakeIO, get_lake

__all__ = ["LakeIO", "get_lake", "clean_frame", "CleaningReport", "drop_duplicate_rows",
           "fit_outlier_bounds", "apply_outlier_bounds", "fit_impute_stats", "apply_impute_stats"]
