import zipfile

import pandas as pd
import pytest

from src.utils import create_sample_dataset, deduplicate_column_names, extract_zip_file


def test_deduplicate_column_names_no_duplicates_returns_original():
    frame = pd.DataFrame({"a": [1], "b": [2]})
    result, renamed, has_duplicates = deduplicate_column_names(frame)
    assert result is frame
    assert renamed == {}
    assert has_duplicates is False


def test_deduplicate_column_names_is_generic_and_preserves_data():
    frame = pd.DataFrame(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
        columns=["col", "col", "col", "unique", "unique"],
    )
    result, renamed, has_duplicates = deduplicate_column_names(frame)
    assert list(result.columns) == ["col", "col_1", "col_2", "unique", "unique_1"]
    assert result.to_numpy().tolist() == frame.to_numpy().tolist()
    assert renamed == {"col": "col_2", "unique": "unique_1"}
    assert has_duplicates is True


def test_deduplicate_avoids_generated_name_collisions():
    frame = pd.DataFrame([[1, 2, 3]], columns=["A", "A", "A_1"])
    result, _, _ = deduplicate_column_names(frame)
    assert list(result.columns) == ["A", "A_1", "A_1_1"]
    assert result.columns.is_unique


def test_create_sample_dataset_is_exact_and_deterministic(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame({"value": range(100)}).to_csv(source, index=False)
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    create_sample_dataset(source, first, sample_size=10, random_seed=7)
    create_sample_dataset(source, second, sample_size=10, random_seed=7)
    a = pd.read_csv(first)
    b = pd.read_csv(second)
    assert len(a) == 10
    pd.testing.assert_frame_equal(a, b)


def test_extract_zip_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="Unsafe zip member"):
        extract_zip_file(archive, tmp_path / "out")
