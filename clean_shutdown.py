"""Clean-shutdown hygiene for the plugin's long-lived SQLite connections.

Every store in this plugin (``store.MessageStore``, the DAG store, query-view,
rollup, vector, lifecycle, assertion and trajectory stores) opens its own
connection to the same profile ``lcm.db`` and holds it for the life of the
process.  ``db_bootstrap.configure_connection()`` documents that checkpoints
"only run during graceful shutdown (see ``MessageStore.close()`` etc.)" -- but
nothing in the plugin ever *performs* a graceful shutdown.  There is no
``atexit`` hook and no ``SIGTERM`` handler anywhere in the tree, and each
``close()`` issues only ``PRAGMA wal_checkpoint(PASSIVE)``, which by its own
docstring "does not run on crash/kill, and PASSIVE can leave frames behind when
another reader is active".

Measured consequence (2026-09): on an unclean stop the WAL is left
unmaterialised and the main database is left holding half-checkpointed pages,
so the next open reports ``database disk image is malformed``.  Observed three
times, each *discovered* at a startup boundary -- 2026-09-17 23:51,
2026-09-18 07:36 and 2026-09-20 11:57 (4 s after boot) -- and it reproduced
identically after the store was relocated from ntfs3 to btrfs, which rules the
filesystem out as the variable.

This module supplies the missing half.  Each connection registers the database
file behind it from ``configure_connection()``, and on normal interpreter exit
or ``SIGTERM`` every registered database is checkpointed with
``wal_checkpoint(TRUNCATE)`` -- which *does* materialise and shrink the WAL,
unlike PASSIVE.

Deliberately NOT hooked: ``SIGINT``.  Overriding it would change Ctrl-C
semantics for interactive sessions, and both systemd and the desktop stop
services with ``SIGTERM``, which is the signal that matters here.

A residual gap remains by construction: a ``SIGKILL`` (including systemd's
post-``TimeoutStopSec`` escalation, or the desktop reaping its own backend)
cannot run any Python, so no in-process hook can cover it.  Closing that case
needs a stop timeout long enough for ``SIGTERM`` to be honoured.

Implementation note -- why this tracks *paths* and not connections:
``sqlite3.Connection`` is **not weak-referenceable** (``weakref.ref(conn)``
raises ``TypeError``), so a ``WeakSet`` of connections silently discards every
registration and checkpoints nothing.  Holding strong references instead would
pin connections that the stores would otherwise let ``__del__`` close, turning
a bounded set of long-lived handles into an unbounded leak.  Recording the
*file* behind each connection sidesteps both problems, and it is the file whose
WAL is the thing that must be materialised.
"""

from __future__ import annotations

import atexit
import logging
import signal
import sqlite3
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_paths: "set[str]" = set()
_installed = False
_ran = False

_CHECKPOINT_TIMEOUT_S = 5.0


def register(conn: "sqlite3.Connection | None") -> None:
    """Record the database file(s) behind ``conn`` for shutdown checkpointing.

    Safe to call repeatedly, from any thread, and on an already-closed
    connection (in which case there is nothing to learn and it is ignored).
    In-memory and temporary databases are skipped: they have no WAL to
    materialise and no file to reopen.
    """
    if conn is None:
        return
    try:
        rows = conn.execute("PRAGMA database_list").fetchall()
    except Exception:
        # Closed, or mid-teardown. Nothing to record.
        return

    with _lock:
        for row in rows:
            # (seq, name, file); file is '' for in-memory/temp databases.
            path = row[2] if len(row) > 2 else None
            if path:
                _paths.add(str(path))


def checkpoint_all(reason: str = "shutdown") -> int:
    """Checkpoint (TRUNCATE) every registered database. Idempotent.

    Opens a short-lived connection per file, so it works regardless of which
    store objects are still alive and never keeps anything alive.  Each file is
    attempted independently and failures are logged, never raised: this runs
    from an ``atexit`` hook or a signal handler, where an exception would
    either be swallowed or mask the real exit status.  Returns the number of
    databases successfully checkpointed.
    """
    global _ran
    with _lock:
        if _ran:
            return 0
        _ran = True
        paths = sorted(_paths)
        _paths.clear()

    checkpointed = 0
    for path in paths:
        conn = None
        try:
            conn = sqlite3.connect(path, timeout=_CHECKPOINT_TIMEOUT_S)
            busy, log_pages, moved = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if busy:
                # Another connection (possibly a sibling process) holds a read
                # mark, so TRUNCATE could not finish. The WAL stays, but the
                # database is still consistent.
                logger.warning(
                    "lcm clean shutdown (%s): checkpoint of %s could not complete "
                    "(busy=1, wal=%s pages, checkpointed=%s) -- a sibling connection "
                    "is still reading; WAL left in place",
                    reason,
                    path,
                    log_pages,
                    moved,
                )
            else:
                checkpointed += 1
        except Exception as exc:  # noqa: BLE001 - shutdown must never raise
            logger.warning(
                "lcm clean shutdown (%s): checkpoint of %s failed: %r", reason, path, exc
            )
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    if paths:
        logger.info(
            "lcm clean shutdown (%s): checkpointed %d/%d database(s)",
            reason,
            checkpointed,
            len(paths),
        )
    return checkpointed


def _handle_signal(signum, _frame) -> None:
    checkpoint_all(f"signal {signum}")
    # Restore the default disposition and re-raise, so the exit status and any
    # outer supervisor observe an ordinary signal death.
    try:
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)
    except Exception:
        raise SystemExit(128 + signum)


def install() -> bool:
    """Install the ``atexit`` + ``SIGTERM`` hooks once.

    Returns ``True`` if this call installed them, ``False`` if they were
    already in place.  Called from ``configure_connection()``, so it runs on
    the first store opened in the process.
    """
    global _installed
    with _lock:
        if _installed:
            return False
        _installed = True

    atexit.register(checkpoint_all, "atexit")

    try:
        if threading.current_thread() is threading.main_thread():
            previous = signal.getsignal(signal.SIGTERM)
            # Only claim the slot when nobody else holds it: a host that
            # installed its own SIGTERM handler is better left alone than
            # clobbered, and it will still get the atexit path.
            if previous in (signal.SIG_DFL, None):
                signal.signal(signal.SIGTERM, _handle_signal)
    except (ValueError, OSError, RuntimeError, AttributeError):
        # Non-main thread, or a platform without SIGTERM.
        pass

    return True
