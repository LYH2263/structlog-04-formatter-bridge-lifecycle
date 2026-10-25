# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

"""
First-class lifecycle metadata for the bridge between *structlog* and
:class:`logging.Formatter` (that is, `structlog.stdlib.ProcessorFormatter`).

The bridge has an explicit, observable lifecycle:

1. A *structlog* event dictionary that has not reached
   `~structlog.stdlib.ProcessorFormatter.wrap_for_formatter` yet is
   :attr:`BridgePhase.UNWRAPPED`.
2. `wrap_for_formatter` attaches the ``_logger`` and ``_name`` sentinels to
   the `logging.LogRecord` via ``extra``.  Between that point and the call to
   `~structlog.stdlib.ProcessorFormatter.format`, the record is
   :attr:`BridgePhase.WRAPPED`.
3. Inside ``format()`` an immutable `BridgeContext` is put into the event
   dictionary under `BRIDGE_KEY`:

   - *structlog*-originated records get
     :attr:`BridgePhase.STRUCTLOG` and never run the ``foreign_pre_chain``.
   - Foreign records get :attr:`BridgePhase.FOREIGN_PRE_CHAIN` while the
     ``foreign_pre_chain`` runs and transition to
     :attr:`BridgePhase.FOREIGN` for the shared processor chain.

4. `~structlog.stdlib.ProcessorFormatter.remove_processors_meta` removes all
   bridge metadata.  Downstream processors can no longer take the
   structlog/foreign fork; use `require_bridge_context` for an explicit
   failure instead of relying on silently missing keys.

The module deliberately only depends on the standard library so that both
:mod:`structlog.stdlib` and :mod:`structlog.processors` can import it without
an import cycle.

.. versionadded:: 26.2.0
"""

from __future__ import annotations

import enum
import logging

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn


__all__ = [
    "BRIDGE_KEY",
    "FROM_STRUCTLOG_KEY",
    "RECORD_KEY",
    "BridgeContext",
    "BridgePhase",
    "IncompatibleBridgeChainError",
    "MissingBridgeMetadataError",
    "bridge_metadata_present",
    "get_bridge_context",
    "get_bridge_phase",
    "get_bridge_record",
    "raise_incompatible_bridge_chain",
    "require_bridge_context",
]


#: Key under which the `BridgeContext` is stored in event dictionaries.
BRIDGE_KEY = "_bridge"

#: Key under which the formatted `logging.LogRecord` copy is stored.
RECORD_KEY = "_record"

#: Key that says whether the record originated from a *structlog* logger.
FROM_STRUCTLOG_KEY = "_from_structlog"

#: Sentinel attribute attached to a record by ``wrap_for_formatter``.
_WRAPPED_LOGGER_KEY = "_logger"

#: Sentinel attribute attached to a record by ``wrap_for_formatter``.
_WRAPPED_NAME_KEY = "_name"

_UNSET = object()


class BridgePhase(enum.Enum):
    """
    The phase a record or event dictionary is in, relative to the
    `structlog.stdlib.ProcessorFormatter` bridge.

    .. versionadded:: 26.2.0
    """

    #: Never went through the bridge: a plain foreign `logging.LogRecord`,
    #: or an event dictionary that is not being formatted by
    #: ProcessorFormatter.
    UNWRAPPED = "unwrapped"

    #: ``wrap_for_formatter`` attached the bridge sentinels;
    #: ``ProcessorFormatter.format()`` has not run yet.
    WRAPPED = "wrapped"

    #: Inside ``ProcessorFormatter.format()``, structlog-source branch:
    #: ``record.msg`` was copied as a dictionary and ``foreign_pre_chain``
    #: is never run.
    STRUCTLOG = "structlog"

    #: Inside ``ProcessorFormatter.format()``, foreign-source branch, while
    #: ``foreign_pre_chain`` is running.
    FOREIGN_PRE_CHAIN = "foreign_pre_chain"

    #: Inside ``ProcessorFormatter.format()``, foreign-source branch, on
    #: the shared processor chain (``foreign_pre_chain`` has finished).
    FOREIGN = "foreign"


@dataclass(frozen=True)
class BridgeContext:
    """
    Immutable bridge metadata carried alongside an event dictionary.

    Do not instantiate this class yourself; ProcessorFormatter creates one
    per ``format()`` call.  Read it with `get_bridge_context` or
    `get_bridge_phase`.

    Args:
        phase:
            The current `BridgePhase`.
        from_structlog:
            Whether the record originated from a *structlog* logger.
        record:
            The `logging.LogRecord` *copy* that ``format()`` works on, or
            `None` for a legacy event dictionary that only carries the
            ``_from_structlog`` key.

    .. versionadded:: 26.2.0
    """

    phase: BridgePhase
    from_structlog: bool
    record: logging.LogRecord | None


class IncompatibleBridgeChainError(TypeError):
    """
    ``wrap_for_formatter`` was combined with ``render_to_log_kwargs`` or
    ``render_to_log_args_and_kwargs`` in one processor chain.

    These are three mutually exclusive terminal processors: ``wrap_for_formatter``
    hands the event dictionary to `structlog.stdlib.ProcessorFormatter`,
    while the renderers hand rendering off to plain :mod:`logging`.

    .. versionadded:: 26.2.0
    """


class MissingBridgeMetadataError(KeyError):
    """
    Bridge metadata was requested from an event dictionary that has none.

    This happens either outside of a `structlog.stdlib.ProcessorFormatter`
    processor chain or after
    `~structlog.stdlib.ProcessorFormatter.remove_processors_meta` removed the
    metadata.  Processors that need the originating `logging.LogRecord` must
    run before ``remove_processors_meta``.

    .. versionadded:: 26.2.0
    """


def raise_incompatible_bridge_chain(where: str) -> NoReturn:
    """
    Raise `IncompatibleBridgeChainError` for processor *where*.

    Args:
        where:
            Name of the processor that detected the incompatible chain.

    .. versionadded:: 26.2.0
    """
    msg = (
        f"{where} cannot be combined with render_to_log_kwargs or "
        "render_to_log_args_and_kwargs in the same structlog processor "
        "chain: ProcessorFormatter.wrap_for_formatter hands the event "
        "dictionary to ProcessorFormatter, while the render_to_log_* "
        "processors hand rendering off to plain logging. Configure one or "
        "the other: use wrap_for_formatter as the terminal processor with "
        "a ProcessorFormatter, or a render_to_log_* processor without a "
        "ProcessorFormatter."
    )

    raise IncompatibleBridgeChainError(msg)


def _context_from_legacy_keys(
    event_dict: Mapping[str, Any],
) -> BridgeContext | None:
    """
    Build a context from bare ``_record``/``_from_structlog`` keys.

    Event dictionaries constructed by third-party code (and by the library's
    own unit tests) may carry the two pin-compatible keys without the
    canonical context.  The synthesized phase can only distinguish the two
    sources, not the foreign pre-chain sub-phases.
    """
    record = event_dict.get(RECORD_KEY)
    if not isinstance(record, logging.LogRecord):
        record = None
    if record is None and FROM_STRUCTLOG_KEY not in event_dict:
        return None

    from_structlog = bool(event_dict.get(FROM_STRUCTLOG_KEY, False))

    return BridgeContext(
        phase=(
            BridgePhase.STRUCTLOG if from_structlog else BridgePhase.FOREIGN
        ),
        from_structlog=from_structlog,
        record=record,
    )


def get_bridge_context(event_dict: Mapping[str, Any]) -> BridgeContext | None:
    """
    Return the `BridgeContext` of *event_dict*, or `None`.

    Returns `None` for event dictionaries that are not (or no longer) part
    of a `structlog.stdlib.ProcessorFormatter` bridge -- for example in a
    plain *structlog* chain or after
    ``ProcessorFormatter.remove_processors_meta`` ran.

    This is a read-only query; it never mutates *event_dict*.

    Args:
        event_dict:
            The event dictionary passed to a processor.

    .. versionadded:: 26.2.0
    """
    context = event_dict.get(BRIDGE_KEY)
    if isinstance(context, BridgeContext):
        return context

    return _context_from_legacy_keys(event_dict)


def get_bridge_record(
    event_dict: Mapping[str, Any],
) -> logging.LogRecord | None:
    """
    Return the bridged `logging.LogRecord` copy, or `None`.

    This is the record-aware counterpart of reading ``event_dict["_record"]``
    directly: it works for both the canonical `BridgeContext` and for event
    dictionaries that only carry the legacy ``_record`` key.

    Args:
        event_dict:
            The event dictionary passed to a processor.

    .. versionadded:: 26.2.0
    """
    context = get_bridge_context(event_dict)
    if context is None:
        return None

    return context.record


def require_bridge_context(event_dict: Mapping[str, Any]) -> BridgeContext:
    """
    Return the `BridgeContext` of *event_dict*, or raise.

    Use this in processors that can only work inside a
    `structlog.stdlib.ProcessorFormatter` bridge.  It fails explicitly
    instead of silently observing missing ``_record`` /
    ``_from_structlog`` keys -- in particular after
    ``ProcessorFormatter.remove_processors_meta`` ran.

    Args:
        event_dict:
            The event dictionary passed to a processor.

    Raises:
        MissingBridgeMetadataError:
            If *event_dict* carries no bridge metadata.

    .. versionadded:: 26.2.0
    """
    context = get_bridge_context(event_dict)
    if context is None:
        msg = (
            "No ProcessorFormatter bridge metadata found in the event "
            "dictionary. Processors that need the originating "
            "logging.LogRecord or the structlog/foreign distinction must "
            "run inside a ProcessorFormatter chain and before "
            "ProcessorFormatter.remove_processors_meta."
        )

        raise MissingBridgeMetadataError(msg)

    return context


def bridge_metadata_present(event_dict: Mapping[str, Any]) -> bool:
    """
    Return whether *event_dict* still carries bridge metadata.

    This is `True` from the moment ``ProcessorFormatter.format()`` injects
    the `BridgeContext` until
    ``ProcessorFormatter.remove_processors_meta`` removes it.

    Args:
        event_dict:
            The event dictionary passed to a processor.

    .. versionadded:: 26.2.0
    """
    return (
        BRIDGE_KEY in event_dict
        or RECORD_KEY in event_dict
        or FROM_STRUCTLOG_KEY in event_dict
    )


def get_bridge_phase(obj: Any) -> BridgePhase:
    """
    Determine the `BridgePhase` of a record or event dictionary.

    Accepts either a `logging.LogRecord` or an event dictionary
    (`~collections.abc.Mapping`):

    - A plain foreign record (no bridge sentinels) is
      :attr:`BridgePhase.UNWRAPPED`.
    - A record produced by ``wrap_for_formatter`` that has not been
      formatted yet is :attr:`BridgePhase.WRAPPED`.
    - An event dictionary inside a ``ProcessorFormatter`` chain reports the
      source branch it is being processed in
      (:attr:`BridgePhase.STRUCTLOG`,
      :attr:`BridgePhase.FOREIGN_PRE_CHAIN`, or
      :attr:`BridgePhase.FOREIGN`).
    - An event dictionary without bridge metadata is
      :attr:`BridgePhase.UNWRAPPED`.  After
      ``remove_processors_meta`` this cannot be distinguished from a
      dictionary that never entered the bridge; use
      `bridge_metadata_present` or `require_bridge_context` for that.

    Args:
        obj:
            A `logging.LogRecord` or an event dictionary.

    Raises:
        TypeError:
            If *obj* is neither a `logging.LogRecord` nor a mapping.

    .. versionadded:: 26.2.0
    """
    if isinstance(obj, logging.LogRecord):
        if (
            getattr(obj, _WRAPPED_LOGGER_KEY, _UNSET) is not _UNSET
            and getattr(obj, _WRAPPED_NAME_KEY, _UNSET) is not _UNSET
        ):
            return BridgePhase.WRAPPED

        return BridgePhase.UNWRAPPED

    if isinstance(obj, Mapping):
        context = obj.get(BRIDGE_KEY)
        if isinstance(context, BridgeContext):
            return context.phase

        legacy = _context_from_legacy_keys(obj)
        if legacy is not None:
            return legacy.phase

        return BridgePhase.UNWRAPPED

    msg = (
        "Expected a logging.LogRecord or an event dictionary mapping, got "
        f"{type(obj).__name__} instead."
    )
    raise TypeError(msg)
