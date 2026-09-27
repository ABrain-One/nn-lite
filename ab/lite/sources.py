"""Where NN-Lite reads the LEMUR models and their input transforms from.

* ``CheckoutSource``: a git checkout of nn-dataset. It is used by people who commit
  results back to the dataset, and results are then written into the same checkout.
* ``PackageSource``: the nn-dataset package installed together with NN-Lite. On first
  use it downloads the LEMUR database (about 1.2 GB unpacked) from Hugging Face.

Both return each model and each transform as a ``.py`` file, so the engine loads them
in exactly the same way whichever source is used.
"""
import os
from pathlib import Path

CLONE_HINT = "git clone https://github.com/ABrain-One/nn-dataset.git"


def is_checkout(path):
    """True if ``path`` is an nn-dataset checkout containing model definitions."""
    return (Path(path) / "ab" / "nn" / "nn").is_dir()


def find_checkout(explicit=None, env=None, *candidates):
    """Return the nn-dataset checkout to use, or None to use the installed package.

    ``explicit`` (--dataset-root) and ``env`` (NN_DATASET_ROOT) must point at a
    checkout. The ``candidates`` (an nn-dataset folder next to the nn-lite
    checkout, then one in the current folder) are used only if they are one.
    """
    for origin, path in (("--dataset-root", explicit), ("NN_DATASET_ROOT", env)):
        if path:
            path = Path(path).expanduser().resolve()
            if not is_checkout(path):
                raise ValueError(f"{origin} points to {path}, which is not an nn-dataset checkout "
                                 f"(no ab/nn/nn folder). Clone it with: {CLONE_HINT}")
            return path
    for path in candidates:
        if path and is_checkout(path):
            return Path(path).resolve()
    return None


class CheckoutSource:
    """Models and transforms read from the files of an nn-dataset checkout."""

    def __init__(self, root):
        root = Path(root)
        self._models = {p.stem: p for p in (root / "ab" / "nn" / "nn").rglob("*.py")}
        self._transforms = root / "ab" / "nn" / "transform"

    def names(self):
        return set(self._models)

    def model_file(self, name):
        return self._models.get(name)

    def transform_file(self, name):
        path = self._transforms / f"{name}.py"
        return path if path.exists() else None


class PackageSource:
    """Models and transforms read from the database of the installed nn-dataset package.

    ``query`` runs an SQL query on that database and returns the rows. The code of
    each requested model or transform is written to ``code_dir``.
    """

    def __init__(self, query, code_dir):
        self._query = query
        self._code_dir = Path(code_dir)

    @classmethod
    def open(cls, home, code_dir):
        """Open the LEMUR database kept in ``home``, downloading it on first use.

        nn-dataset keeps its database in ``db/`` of the first folder, from the current
        one upwards, that contains an ``ab`` folder and a README.md. Importing it from
        inside ``home``, which has both, keeps the database in one fixed place.
        """
        home = Path(home)
        (home / "ab").mkdir(parents=True, exist_ok=True)
        readme = home / "README.md"
        if not readme.exists():
            readme.write_text("Working folder of the nn-dataset package used by NN-Lite. "
                              "The LEMUR database is kept in db/.\n")
        cwd = os.getcwd()
        os.chdir(home)
        try:
            from ab.nn.util.db.Read import query_rows
        finally:
            os.chdir(cwd)
        return cls(query_rows, code_dir)

    def names(self):
        return {name for (name,) in self._query("SELECT name FROM nn")}

    def model_file(self, name):
        return self._write("nn", name)

    def transform_file(self, name):
        return self._write("transform", name)

    def _write(self, table, name):
        rows = self._query(f"SELECT code FROM {table} WHERE name = ?", [name])
        if not rows:
            return None
        path = self._code_dir / table / f"{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rows[0][0])
        return path
