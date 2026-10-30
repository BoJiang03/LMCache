# SPDX-License-Identifier: Apache-2.0
"""
Class for distributed storage manager internal API data structures
"""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable
import enum

# First Party
from lmcache.v1.distributed.api import (
    L1BackendType,
    MemoryLayoutDesc,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.memory_management import MemoryObj

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.config import L1ManagerConfig

L1OperationResult = tuple[L1Error, "MemoryObj | None"]


@dataclass(frozen=True)
class L1MemoryDesc:
    """Describe the contiguous L1 memory arena exposed to external backends.

    Attributes:
        ptr: Base address of the L1 arena.
        size: Final size of the L1 arena in bytes.
        align_bytes: Allocation alignment within the arena.
        stable_registration_size: Stable size of the L1 arena. For a lazy
            allocator, this is a snapshot of the currently pinned prefix and
            does not change as the allocator grows. ``None`` means that no
            stable size is exposed.
    """

    ptr: int
    size: int
    align_bytes: int
    stable_registration_size: int | None = None


@dataclass(frozen=True)
class L1ObjectMeta:
    """Per-object metadata published alongside keys in L1 cache events.

    Attributes:
        size_bytes: Logical byte size of the object.
        backend: The medium of the L1 that holds the object (its manager's
            ``backend_type``).
    """

    size_bytes: int
    backend: L1BackendType


#: Lifecycle state of one L1 object in :class:`L1ObjectListing`: ``committed``
#: (resident, readable), ``temporary`` (resident, deleted after its read),
#: or ``staging`` (write-reserved, its write not finished, unreadable).
L1ObjectStateName = Literal["committed", "temporary", "staging"]


@dataclass(frozen=True)
class L1ObjectListing:
    """Every L1 object, as parallel lists (one entry per object).

    Parallel lists rather than one record per object, so listing a large L1
    under its lock allocates one list per field.

    Attributes:
        keys: Each object's key.
        size_bytes: Its logical size (``MemoryObj.get_size``).
        states: Its lifecycle state (see :data:`L1ObjectStateName`).
    """

    keys: list[ObjectKey]
    size_bytes: list[int]
    states: list[L1ObjectStateName]


@dataclass(frozen=True)
class L1ManagerSnapshot:
    """One L1 manager's objects and eviction order, for offline analysis.

    The objects and the order are read under two different locks (the L1
    manager's, then the eviction policy's), one right after the other, so
    an object written or evicted in between appears in one but not both.

    Attributes:
        l1_manager_id: The manager's process-local id.
        tag: The manager's configured tag.
        objects: Every resident and staging object (see
            :meth:`L1ManagerInterface.list_objects`).
        eviction_order: The policy's keys, next victim first, or ``None``
            when the policy keeps no single order
            (``EvictionPolicy.eviction_order``).
        list_ms: Wall time of ``list_objects``: the L1 manager lock's hold,
            plus any wait for it.
        order_ms: Wall time of reading the eviction order, under the
            policy's own lock.
    """

    l1_manager_id: int
    tag: str
    objects: L1ObjectListing
    eviction_order: list[ObjectKey] | None
    list_ms: float
    order_ms: float


class EventListener(ABC):  # noqa: B024
    pass


# For L1 manager event notifications
class L1ManagerListener(EventListener):
    """
    Listener for L1 manager events
    """

    @abstractmethod
    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]):
        """
        Notify the listener that new keys have been reserved for read on L1.

        Args:
            keys (list[ObjectKey]): The keys that have been successfully reserved
        """
        pass

    @abstractmethod
    def on_l1_keys_read_finished(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been accessed on L1.

        Args:
            keys (list[ObjectKey]): The keys that have been successfully read
        """
        pass

    @abstractmethod
    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been reserved for write on L1.

        Args:
            keys (list[ObjectKey]): The keys that have been successfully reserved
        """
        pass

    @abstractmethod
    def on_l1_keys_write_finished(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been finished for writing on L1.

        Args:
            keys (list[ObjectKey]): The keys that have been successfully written
        """
        pass

    @abstractmethod
    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been finished for writing
        and reserved for read on L1.

        This will only be trigger by the prefetch operation now.

        Args:
            keys (list[ObjectKey]): The keys that have been successfully
                finished for writing and reserved for read
        """
        # NOTE (ApostaC): may consider renaming this to `on_l1_keys_finish_prefetch`
        # for better clarity
        pass

    @abstractmethod
    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been deleted from L1.

        Args:
            keys (list[ObjectKey]): The keys that have been deleted
        """
        pass

    @abstractmethod
    def on_l1_keys_accessed(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been accessed on L1.

        Args:
            keys (list[ObjectKey]): The keys that have been accessed
        """
        pass


class L2AdapterListener(EventListener):
    """Listener for L2 adapter events, analogous to L1ManagerListener."""

    @abstractmethod
    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]):
        """
        Notify the listener that keys have been successfully stored in L2.

        Args:
            keys (list[ObjectKey]): The keys that have been stored.
            sizes (list[int]): The byte size of each stored key.
        """
        pass

    @abstractmethod
    def on_l2_keys_accessed(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been accessed (lookup hit) in L2.

        Args:
            keys (list[ObjectKey]): The keys that have been accessed.
        """
        pass

    @abstractmethod
    def on_l2_keys_deleted(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been deleted from L2.

        Args:
            keys (list[ObjectKey]): The keys that have been deleted.
        """
        pass


# For Eviction
class EvictionDestination(enum.Enum):
    """
    The destination of evicted objects
    """

    DISCARD = enum.auto()
    """Discard the evicted objects"""

    L2_CACHE = enum.auto()
    """Evict to L2 storage"""


@dataclass(frozen=True)
class EvictionAction:
    """
    An action to be taken for eviction
    """

    destination: EvictionDestination
    """The destination of the evicted object"""

    keys: list[ObjectKey] = field(default_factory=list)
    """The key of the object to be evicted"""


class L2StoreResult(int):
    """Immutable result of a completed L2 store task.

    Encodes both the success flag and bytes transferred in the int
    value: ``>= 0`` means success (value = bytes transferred);
    ``-1`` means failure.

    Args:
        success: Whether the store task succeeded.
        bytes_transferred: Bytes actually written to L2. Must be >= 0.

    Raises:
        ValueError: If ``bytes_transferred`` is negative.
    """

    def __new__(cls, success: bool, bytes_transferred: int) -> "L2StoreResult":
        if bytes_transferred < 0:
            raise ValueError(f"bytes_transferred must be >= 0, got {bytes_transferred}")
        return super().__new__(cls, bytes_transferred if success else -1)

    def is_successful(self) -> bool:
        """Return ``True`` when the store task succeeded."""
        return int(self) >= 0

    def bytes_transferred(self) -> int:
        """Return the number of bytes actually written, or 0 on failure."""
        value = int(self)
        return value if value >= 0 else 0


@dataclass(frozen=True)
class QuotaEntry:
    """Snapshot of a single quota registration."""

    cache_salt: str
    limit_bytes: int


@runtime_checkable
class L1ManagerInterface(Protocol):
    """The L1 surface ``StorageManager`` and the storage controllers call.

    ``L1Manager`` implements it with the allocator, index, locks and
    eviction in this process. An implementation that cannot honour a call
    answers with the documented ``L1Error`` codes instead of raising, so
    the callers keep one code path.
    """

    @property
    def l1_manager_id(self) -> int:
        """Process-local identity, also the owner tag on memory objects."""
        ...

    @property
    def config(self) -> "L1ManagerConfig":
        """The configuration this L1 was built from."""
        ...

    @property
    def backend_type(self) -> L1BackendType:
        """The medium backing this L1.

        Its wire value is the ``backend`` of capacity declarations, cache
        events and the Prometheus gauges; usage telemetry's ``l1_medium``
        uses the same vocabulary.
        """
        ...

    def register_listener(self, listener: L1ManagerListener) -> None:
        """Register a listener for this L1's lifecycle notifications."""
        ...

    def reserve_read(
        self, keys: list[ObjectKey], read_locks: int = 1
    ) -> dict[ObjectKey, L1OperationResult]:
        """Take ``read_locks`` read locks on each resident key."""
        ...

    def unsafe_read(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1OperationResult]:
        """Return already read-locked objects without new locks."""
        ...

    def finish_read(
        self, keys: list[ObjectKey], read_locks: int = 1
    ) -> dict[ObjectKey, L1Error]:
        """Release ``read_locks`` read locks on each key."""
        ...

    def reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        tag: str = "",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve a write buffer per key for the writer ``tag``."""
        ...

    def finish_write(
        self, keys: list[ObjectKey], tag: str = ""
    ) -> dict[ObjectKey, L1Error]:
        """Commit ``tag``'s written buffers so readers can see them."""
        ...

    def finish_write_and_reserve_read(
        self, keys: list[ObjectKey], read_locks: int = 1, tag: str = ""
    ) -> dict[ObjectKey, L1OperationResult]:
        """Commit ``tag``'s buffers and read-lock the resident objects."""
        ...

    def finish_write_and_delete(
        self, keys: list[ObjectKey], tag: str = ""
    ) -> dict[ObjectKey, L1Error]:
        """Discard ``tag``'s write reservations without committing them."""
        ...

    def delete(
        self, keys: list[ObjectKey], force: bool = False
    ) -> dict[ObjectKey, L1Error]:
        """Delete resident objects."""
        ...

    def touch_keys(self, keys: list[ObjectKey]) -> None:
        """Record an access to the keys; takes no lock."""
        ...

    def clear(self, force: bool = False) -> None:
        """Drop every object this L1 is allowed to drop."""
        ...

    def is_key_evictable(self, key: ObjectKey) -> bool:
        """Whether eviction may delete the key now."""
        ...

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(used_bytes, total_bytes)``."""
        ...

    def get_staging_memory_usage(self) -> int:
        """Return bytes held by uncommitted write reservations."""
        ...

    def get_capacity_bytes(self) -> int:
        """Return the currently usable capacity in bytes."""
        ...

    def get_l1_memory_desc(self) -> L1MemoryDesc | None:
        """Describe a registerable L1 buffer, or None when there is none."""
        ...

    def get_memory_tags(self) -> MemoryTags:
        """Return the memory tags of this L1's primary pool."""
        ...

    def report_status(self) -> dict:
        """Return a status dictionary for the status endpoints."""
        ...

    def list_objects(self) -> L1ObjectListing:
        """List every resident and staging object, under one lock hold.

        No side effects: no read lock, recency update, listener call or
        event.

        Returns:
            One entry per resident object and one per writer's staging
            object, in no particular order.
        """
        ...

    def memcheck(self) -> bool:
        """Check bookkeeping consistency."""
        ...

    def close(self) -> None:
        """Release every resource held by this L1."""
        ...
