"""Strict history transfer for the Anthony structured-Q2 remeshing path.

The geometry layer supplies every destination-node correspondence explicitly.
This module never invokes pyoomph's generic locator or a nearest-node fallback.
It is initially serial because the explicit plan is local to one mesh/rank.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from pyoomph.meshes.interpolator import BaseMeshToMeshInterpolator

from .q2_locator import SourceLocation


POSITION_SLOT_COUNT = 7
POSITION_COORDINATE_LIKE_SLOTS = frozenset((1, 2, 6))
POSITION_DERIVATIVE_SLOTS = frozenset((3, 4, 5))


class RemapTransferError(RuntimeError):
    """Raised when an explicit remap contract is incomplete or incompatible."""


@dataclass(frozen=True)
class PlannedNodeTransfer:
    """One destination node and its already-resolved old-bulk location."""

    destination_node: Any
    source: SourceLocation


@dataclass(frozen=True)
class InterfaceNodeFields:
    """Interface-only value indices present on one destination node."""

    destination_node: Any
    field_indices: Mapping[str, int]


InterfaceInitializer = float | Callable[[Any, SourceLocation, int], float]


def transform_position_history(
    source_slots: Sequence[Sequence[float]],
    displacement: Sequence[float],
    destination_current: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """Apply a reconstruction displacement with MultiTimeStepper semantics.

    Slot 0 is the prescribed destination geometry.  The displacement is added
    only to coordinate-like slots 1, 2 and 6; velocity/acceleration slots 3,
    4 and 5 are derivatives and therefore remain untranslated.
    """
    if len(source_slots) != POSITION_SLOT_COUNT:
        raise RemapTransferError(
            f"position history must contain exactly {POSITION_SLOT_COUNT} slots, "
            f"got {len(source_slots)}"
        )
    current = tuple(float(value) for value in destination_current)
    shift = tuple(float(value) for value in displacement)
    if not current or len(shift) != len(current):
        raise RemapTransferError("position displacement and destination geometry dimensions differ")
    if not all(math.isfinite(value) for value in current + shift):
        raise RemapTransferError("position transfer received a non-finite coordinate")

    result: list[tuple[float, ...]] = [current]
    for slot, values in enumerate(source_slots[1:], start=1):
        source = tuple(float(value) for value in values)
        if len(source) != len(current):
            raise RemapTransferError(
                f"position slot {slot} has dimension {len(source)}, expected {len(current)}"
            )
        if not all(math.isfinite(value) for value in source):
            raise RemapTransferError(f"position slot {slot} contains a non-finite value")
        if slot in POSITION_COORDINATE_LIKE_SLOTS:
            result.append(tuple(value + delta for value, delta in zip(source, shift)))
        elif slot in POSITION_DERIVATIVE_SLOTS:
            result.append(source)
        else:  # pragma: no cover - guarded by the explicit seven-slot contract
            raise RemapTransferError(f"unclassified position history slot {slot}")
    return tuple(result)


def validate_field_history_contract(
    source_fields: Mapping[str, int],
    destination_fields: Mapping[str, int],
    source_storage: int,
    destination_storage: int,
) -> tuple[str, ...]:
    """Validate exact bulk-field names, indices and history storage."""
    source_names, destination_names = set(source_fields), set(destination_fields)
    if source_names != destination_names:
        missing = sorted(destination_names - source_names)
        extra = sorted(source_names - destination_names)
        raise RemapTransferError(
            f"bulk nodal field mismatch: missing in source={missing}, source-only={extra}"
        )
    if source_storage != destination_storage or source_storage <= 0:
        raise RemapTransferError(
            f"bulk field history storage mismatch: source={source_storage}, "
            f"destination={destination_storage}"
        )
    for owner, fields in (("source", source_fields), ("destination", destination_fields)):
        indices = list(fields.values())
        if any(not isinstance(index, int) or index < 0 for index in indices):
            raise RemapTransferError(f"{owner} bulk field indices must be non-negative integers")
        if len(set(indices)) != len(indices):
            raise RemapTransferError(f"{owner} bulk field indices are not unique")
    return tuple(sorted(source_names))


def pyoomph_local_coordinates(location: SourceLocation) -> tuple[float, float]:
    """Convert graph coordinates on ``[0,1]^2`` to pyoomph's ``[-1,1]^2``."""
    return 2.0 * location.u - 1.0, 2.0 * location.v - 1.0


class AnthonyQ2Interpolator(BaseMeshToMeshInterpolator):
    """Serial, explicit-plan remapper for Anthony structured Q2 meshes.

    ``source_elements`` binds graph element IDs to old pyoomph bulk elements.
    ``interface_fields`` lists interface-only values on destination nodes.
    Every listed interface field must have an explicit initializer; such fields
    are never read from the old mesh as though they were transported fluid data.
    """

    distributed_limitation = (
        "its explicit destination-node/source-element plan is rank-local and "
        "has no distributed ownership or halo exchange"
    )

    def __init__(
        self,
        old: Any,
        new: Any,
        plan: Iterable[PlannedNodeTransfer],
        source_elements: Mapping[tuple[int, int, int], Any],
        *,
        interface_fields: Iterable[InterfaceNodeFields] = (),
        interface_initializers: Mapping[str, InterfaceInitializer] | None = None,
    ) -> None:
        super().__init__(old, new)
        self.source_elements = dict(source_elements)
        self.interface_initializers = dict(interface_initializers or {})
        self._plan_by_node_id: dict[int, PlannedNodeTransfer] = {}
        for item in plan:
            key = id(item.destination_node)
            if key in self._plan_by_node_id:
                raise RemapTransferError("destination node occurs more than once in transfer plan")
            self._plan_by_node_id[key] = item
        self._interface_by_node_id: dict[int, InterfaceNodeFields] = {}
        for item in interface_fields:
            key = id(item.destination_node)
            if key in self._interface_by_node_id:
                raise RemapTransferError("destination node has duplicate interface-field declarations")
            unknown = set(item.field_indices) - set(self.interface_initializers)
            if unknown:
                raise RemapTransferError(
                    f"unknown interface-only fields without explicit initializer: {sorted(unknown)}"
                )
            self._interface_by_node_id[key] = item

    @staticmethod
    def _uniform_storage(nodes: Sequence[Any], *, position: bool, owner: str) -> int:
        if not nodes:
            raise RemapTransferError(f"{owner} mesh has no nodes")
        values = [
            int(node.variable_position_pt().ntstorage()) if position else int(node.ntstorage())
            for node in nodes
        ]
        if len(set(values)) != 1:
            kind = "position" if position else "field"
            raise RemapTransferError(f"{owner} nodes disagree on {kind} history storage: {values}")
        return values[0]

    @staticmethod
    def _initializer_value(
        initializer: InterfaceInitializer,
        node: Any,
        location: SourceLocation,
        history_index: int,
    ) -> float:
        value = initializer(node, location, history_index) if callable(initializer) else initializer
        value = float(value)
        if not math.isfinite(value):
            raise RemapTransferError("interface-field initializer returned a non-finite value")
        return value

    def interpolate(self) -> None:
        old_nodes, new_nodes = list(self.old.nodes()), list(self.new.nodes())
        old_position_storage = self._uniform_storage(old_nodes, position=True, owner="source")
        new_position_storage = self._uniform_storage(new_nodes, position=True, owner="destination")
        if old_position_storage != POSITION_SLOT_COUNT or new_position_storage != POSITION_SLOT_COUNT:
            raise RemapTransferError(
                f"Anthony position transfer requires seven MultiTimeStepper slots, got "
                f"source={old_position_storage}, destination={new_position_storage}"
            )
        old_field_storage = self._uniform_storage(old_nodes, position=False, owner="source")
        new_field_storage = self._uniform_storage(new_nodes, position=False, owner="destination")
        source_fields = dict(self.old.get_nodal_field_indices())
        destination_fields = dict(self.new.get_nodal_field_indices())
        field_names = validate_field_history_contract(
            source_fields, destination_fields, old_field_storage, new_field_storage
        )
        planned_ids = set(self._plan_by_node_id)
        destination_ids = {id(node) for node in new_nodes}
        missing = destination_ids - planned_ids
        extra = planned_ids - destination_ids
        if missing or extra:
            raise RemapTransferError(
                f"explicit source plan does not cover destination nodes: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        extra_interface = set(self._interface_by_node_id) - destination_ids
        if extra_interface:
            raise RemapTransferError(
                f"interface-field policies include {len(extra_interface)} non-destination nodes"
            )

        for node in new_nodes:
            item = self._plan_by_node_id[id(node)]
            interface = self._interface_by_node_id.get(id(node))
            interface_indices = set(interface.field_indices.values()) if interface is not None else set()
            bulk_indices = set(destination_fields.values())
            if bulk_indices.intersection(interface_indices):
                raise RemapTransferError("bulk and interface-only field indices overlap")
            declared_indices = bulk_indices | interface_indices
            actual_indices = set(range(node.nvalue()))
            if actual_indices != declared_indices:
                raise RemapTransferError(
                    f"destination nodal values are not fully described by transfer policies: "
                    f"unknown interface indices={sorted(actual_indices - declared_indices)}, "
                    f"missing indices={sorted(declared_indices - actual_indices)}"
                )
            element = self.source_elements.get(item.source.element_id)
            if element is None:
                raise RemapTransferError(
                    f"no old bulk element for source ID {item.source.element_id!r}"
                )
            local = pyoomph_local_coordinates(item.source)
            current = tuple(float(node.x(component)) for component in range(node.ndim()))
            source_positions = tuple(
                tuple(float(value) for value in element.get_interpolated_position_at_s(slot, local, False))
                for slot in range(POSITION_SLOT_COUNT)
            )
            if len(source_positions[0]) != len(current):
                raise RemapTransferError("source and destination nodal dimensions differ")
            displacement = tuple(
                destination - source
                for destination, source in zip(current, source_positions[0])
            )
            transformed = transform_position_history(source_positions, displacement, current)
            for component, value in enumerate(current):
                node.set_x_lagr(component, value)
            for slot in range(1, POSITION_SLOT_COUNT):
                for component, value in enumerate(transformed[slot]):
                    node.set_x_at_t(slot, component, value)

            for history_index in range(new_field_storage):
                values = element.get_interpolated_nodal_values_at_s(history_index, local)
                for name in field_names:
                    source_index = source_fields[name]
                    destination_index = destination_fields[name]
                    if source_index >= len(values) or destination_index >= node.nvalue():
                        raise RemapTransferError(
                            f"field {name!r} index is outside source interpolation or destination node"
                        )
                    value = float(values[source_index])
                    if not math.isfinite(value):
                        raise RemapTransferError(
                            f"field {name!r} has a non-finite interpolated value at history "
                            f"slot {history_index}"
                        )
                    node.set_value_at_t(history_index, destination_index, value)

            if interface is not None:
                for name, index in interface.field_indices.items():
                    if index < 0 or index >= node.nvalue():
                        raise RemapTransferError(
                            f"interface field {name!r} index {index} is outside destination node"
                        )
                    initializer = self.interface_initializers[name]
                    for history_index in range(new_field_storage):
                        node.set_value_at_t(
                            history_index,
                            index,
                            self._initializer_value(initializer, node, item.source, history_index),
                        )


__all__ = [
    "AnthonyQ2Interpolator",
    "InterfaceNodeFields",
    "POSITION_COORDINATE_LIKE_SLOTS",
    "POSITION_DERIVATIVE_SLOTS",
    "POSITION_SLOT_COUNT",
    "PlannedNodeTransfer",
    "RemapTransferError",
    "pyoomph_local_coordinates",
    "transform_position_history",
    "validate_field_history_contract",
]
