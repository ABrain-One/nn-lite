"""Tests for choosing and reading the source of LEMUR models and transforms.

These run without nn-dataset, a network connection or PyTorch.
"""
import sqlite3

import pytest

from ab.lite.sources import CheckoutSource, PackageSource, find_checkout, is_checkout

MODEL = "class Net:\n    pass\n"
TRANSFORM = "def transform(norm):\n    return norm\n"


def make_checkout(root):
    (root / "ab" / "nn" / "nn").mkdir(parents=True)
    (root / "ab" / "nn" / "transform").mkdir(parents=True)
    (root / "ab" / "nn" / "nn" / "AirNet.py").write_text(MODEL)
    (root / "ab" / "nn" / "transform" / "norm_128.py").write_text(TRANSFORM)
    return root


# --- choosing the source -------------------------------------------------

def test_explicit_checkout_wins(tmp_path):
    a, b = make_checkout(tmp_path / "a"), make_checkout(tmp_path / "b")
    assert find_checkout(a, b, b) == a.resolve()


def test_environment_variable_is_used_without_option(tmp_path):
    b = make_checkout(tmp_path / "b")
    assert find_checkout(env=str(b)) == b.resolve()


def test_sibling_checkout_is_detected(tmp_path):
    sibling = make_checkout(tmp_path / "nn-dataset")
    assert find_checkout(None, None, sibling) == sibling.resolve()


def test_sibling_is_preferred_to_current_folder(tmp_path):
    sibling = make_checkout(tmp_path / "sibling" / "nn-dataset")
    here = make_checkout(tmp_path / "here" / "nn-dataset")
    assert find_checkout(None, None, sibling, here) == sibling.resolve()
    assert find_checkout(None, None, tmp_path / "missing", here) == here.resolve()


def test_package_is_used_without_checkout(tmp_path):
    assert find_checkout(None, None, tmp_path / "nn-dataset") is None
    (tmp_path / "nn-dataset").mkdir()  # an empty folder is not a checkout
    assert find_checkout(None, None, tmp_path / "nn-dataset") is None


def test_wrong_explicit_checkout_is_an_error(tmp_path):
    assert not is_checkout(tmp_path)
    with pytest.raises(ValueError, match="not an nn-dataset checkout"):
        find_checkout(explicit=tmp_path)
    with pytest.raises(ValueError, match="NN_DATASET_ROOT"):
        find_checkout(env=tmp_path)


# --- reading models and transforms ---------------------------------------

def test_checkout_source(tmp_path):
    source = CheckoutSource(make_checkout(tmp_path))
    assert source.names() == {"AirNet"}
    assert source.model_file("AirNet").read_text() == MODEL
    assert source.transform_file("norm_128").read_text() == TRANSFORM
    assert source.transform_file("missing") is None


def test_package_source_writes_code_from_the_database(tmp_path):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE nn (name TEXT, code TEXT)")
    db.execute("CREATE TABLE transform (name TEXT, code TEXT)")
    db.execute("INSERT INTO nn VALUES ('AirNet', ?)", [MODEL])
    db.execute("INSERT INTO transform VALUES ('norm_128', ?)", [TRANSFORM])
    source = PackageSource(lambda *q: db.execute(*q).fetchall(), tmp_path / "code")

    assert source.names() == {"AirNet"}
    assert source.model_file("AirNet").read_text() == MODEL
    assert source.transform_file("norm_128") == tmp_path / "code" / "transform" / "norm_128.py"
    assert source.transform_file("missing") is None
