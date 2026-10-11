"""Regression tests: an aliased Hermes-home spelling must not force a rebind.

Symptom this guards (seen in production):

    AttributeError: 'NoneType' object has no attribute 'execute'

raised from ``MessageStore.get_session_messages_after`` while context
compression ran on a background worker thread.

Mechanism (all three links are pinned below):

1. ``/home/arturj/.hermes`` is a symlink to ``/mnt/interop/.hermes``, so one
   directory reaches the engine under two spellings.
2. ``_rebind_storage_for_home`` compared those spellings as raw strings, read
   the alias as a profile switch, and ran ``_close_storage()`` -- which sets
   ``MessageStore._conn = None`` -- before re-binding.
3. The re-bind went through the *alias*, and ``_open_private_sqlite_directory``
   refuses a symlink as a parent directory (``parent is not a regular
   directory``), so the replacement bind raised after the live connection was
   already gone.  The engine kept a store whose ``_conn`` was ``None``, and the
   next compaction query on a worker thread hit ``self._conn.execute``.

The fix resolves both spellings before comparing, and resolves home-derived
database paths before binding, so an alias is never mistaken for a different
home and never handed to the store as a symlinked parent.
"""

from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.sqlite_util import _prepare_private_sqlite_file


@pytest.fixture
def aliased_homes(tmp_path):
    """A real Hermes home plus a symlink that names the same directory."""
    real = tmp_path / "real-hermes"
    real.mkdir()
    alias = tmp_path / "alias-hermes"
    alias.symlink_to(real, target_is_directory=True)
    return real, alias


def _literal_compare(left, right) -> bool:
    """The pre-fix comparison: raw strings only."""
    return bool(left) and bool(right) and str(left) == str(right)


def test_aliased_home_spelling_does_not_rebind_or_close_store(aliased_homes):
    """Two spellings of one home are not a profile switch."""
    real, alias = aliased_homes

    # database_path left empty so this exercises the branch that previously
    # called _close_storage() before re-binding.
    engine = LCMEngine(config=LCMConfig(), hermes_home=str(real))

    store_before = engine._store
    conn_before = engine._store.connection
    assert conn_before is not None, "engine should start with a live connection"

    rebound = engine._rebind_storage_for_home(str(alias))

    assert rebound is False, "an aliased spelling must not trigger a rebind"
    assert engine._store is store_before, "store object must be preserved"
    assert engine._store.connection is conn_before, "connection must not be reopened"
    assert engine._store.connection is not None, "connection must stay live"


def test_literal_comparison_reproduces_the_original_defect(aliased_homes, monkeypatch):
    """Pins the bug class: string comparison churns the store.

    With the resolved comparison disabled the alias reads as a different home
    and the live store is torn down and replaced -- the unnecessary
    close/reopen that raced the compaction worker.
    """
    real, alias = aliased_homes

    engine = LCMEngine(config=LCMConfig(), hermes_home=str(real))
    store_before = engine._store
    conn_before = engine._store.connection

    monkeypatch.setattr(LCMEngine, "_same_location", staticmethod(_literal_compare))

    assert engine._rebind_storage_for_home(str(alias)) is True
    assert engine._store is not store_before, "string compare re-binds the store"
    assert engine._store.connection is not conn_before, "string compare swaps the connection"


def test_symlink_parent_is_refused_by_private_storage(aliased_homes):
    """The hardening that turned a spurious rebind into a dead connection."""
    real, alias = aliased_homes

    with pytest.raises(PermissionError):
        _prepare_private_sqlite_file(alias / "lcm.db")

    # The resolved spelling names a real directory and is accepted.
    _prepare_private_sqlite_file((alias / "lcm.db").resolve())


def test_db_path_resolves_symlinked_home(aliased_homes):
    """Binding through an alias must succeed, not raise after closing the store."""
    real, alias = aliased_homes

    engine = LCMEngine(config=LCMConfig(), hermes_home=str(alias))

    assert engine._store.connection is not None
    assert Path(engine._store.db_path).resolve() == (real / "lcm.db").resolve()
    assert engine._store.db_path == (real / "lcm.db").resolve()


def test_distinct_home_still_rebinds(tmp_path):
    """A genuinely different home must still switch storage (no over-correction)."""
    home_a = tmp_path / "home-a"
    home_b = tmp_path / "home-b"
    home_a.mkdir()
    home_b.mkdir()

    engine = LCMEngine(config=LCMConfig(), hermes_home=str(home_a))
    store_before = engine._store

    assert engine._rebind_storage_for_home(str(home_b)) is True
    assert engine._store is not store_before
    assert engine._store.connection is not None
    assert Path(engine._store.db_path).resolve() == (home_b / "lcm.db").resolve()


def test_configured_database_path_branch_also_resolves(tmp_path):
    """The explicit database_path branch compares homes by location too."""
    real = tmp_path / "real-hermes"
    real.mkdir()
    alias = tmp_path / "alias-hermes"
    alias.symlink_to(real, target_is_directory=True)

    engine = LCMEngine(
        config=LCMConfig(database_path=str(real / "lcm.db")),
        hermes_home=str(real),
    )
    store_before = engine._store

    assert engine._rebind_storage_for_home(str(alias)) is False
    assert engine._store is store_before
    assert engine._store.connection is not None
