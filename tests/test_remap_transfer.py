"""Software tests for explicit-plan Anthony remap history transfer."""

from __future__ import annotations

import unittest

from problems.newtonian.q2_locator import LocationCandidate, SourceLocation
from problems.newtonian.remap_transfer import (
    AnthonyQ2Interpolator,
    InterfaceNodeFields,
    PlannedNodeTransfer,
    RemapTransferError,
    transform_position_history,
    validate_field_history_contract,
)


def _location(element_id=(0, 0, 0), u=0.25, v=0.75):
    candidate = LocationCandidate(element_id, u, v, 0.0, 0.0)
    return SourceLocation(element_id, u, v, 0.0, 0.0, False, (candidate,))


class _Storage:
    def __init__(self, count):
        self.count = count

    def ntstorage(self):
        return self.count


class _Node:
    def __init__(self, positions, values, lagrangian=None):
        self.positions = [list(slot) for slot in positions]
        self.values = [list(slot) for slot in values]
        self.lagrangian = list(lagrangian if lagrangian is not None else positions[0])
        self._position_storage = _Storage(len(self.positions))

    def variable_position_pt(self):
        return self._position_storage

    def ntstorage(self):
        return len(self.values)

    def ndim(self):
        return len(self.positions[0])

    def nvalue(self):
        return len(self.values[0])

    def x(self, component):
        return self.positions[0][component]

    def set_x_at_t(self, slot, component, value):
        self.positions[slot][component] = value

    def set_x_lagr(self, component, value):
        self.lagrangian[component] = value

    def set_value_at_t(self, slot, index, value):
        self.values[slot][index] = value


class _Mesh:
    def __init__(self, nodes, fields):
        self._nodes = list(nodes)
        self._fields = dict(fields)

    def nodes(self):
        return iter(self._nodes)

    def get_nodal_field_indices(self):
        return dict(self._fields)


class _Element:
    def __init__(self, positions, values):
        self.positions = [list(slot) for slot in positions]
        self.values = [list(slot) for slot in values]
        self.position_calls = []
        self.value_calls = []

    def get_interpolated_position_at_s(self, slot, local, lagrangian):
        self.position_calls.append((slot, tuple(local), lagrangian))
        return list(self.positions[slot])

    def get_interpolated_nodal_values_at_s(self, slot, local):
        self.value_calls.append((slot, tuple(local)))
        return list(self.values[slot])


def _positions(base_x=10.0, base_y=20.0):
    return [[base_x + 10 * slot, base_y + 10 * slot] for slot in range(7)]


def _values(offset=0.0):
    # Q2 velocity components and Q1 pressure are already evaluated by the old
    # element binding; distinct sentinels prove that every history slot moves.
    return [
        [offset + 100 * slot + 1, offset + 100 * slot + 2, offset + 100 * slot + 3]
        for slot in range(7)
    ]


class PositionHistoryTests(unittest.TestCase):
    def test_only_coordinate_like_slots_receive_reconstruction_displacement(self):
        source = _positions()
        transformed = transform_position_history(source, (3.0, -4.0), (101.0, 202.0))
        self.assertEqual(transformed[0], (101.0, 202.0))
        for slot in (1, 2, 6):
            self.assertEqual(
                transformed[slot],
                (source[slot][0] + 3.0, source[slot][1] - 4.0),
            )
        for slot in (3, 4, 5):
            self.assertEqual(transformed[slot], tuple(source[slot]))

    def test_position_storage_must_be_exactly_seven_slots(self):
        with self.assertRaisesRegex(RemapTransferError, "exactly 7"):
            transform_position_history(_positions()[:-1], (0.0, 0.0), (0.0, 0.0))


class FieldContractTests(unittest.TestCase):
    def test_field_names_and_storage_must_match(self):
        self.assertEqual(
            validate_field_history_contract(
                {"velocity_x": 0, "velocity_y": 1, "pressure": 2},
                {"pressure": 2, "velocity_y": 1, "velocity_x": 0},
                7,
                7,
            ),
            ("pressure", "velocity_x", "velocity_y"),
        )
        with self.assertRaisesRegex(RemapTransferError, "field mismatch"):
            validate_field_history_contract({"velocity_x": 0}, {"pressure": 0}, 7, 7)
        with self.assertRaisesRegex(RemapTransferError, "storage mismatch"):
            validate_field_history_contract({"pressure": 0}, {"pressure": 0}, 7, 6)


class InterpolatorTests(unittest.TestCase):
    fields = {"velocity_x": 0, "velocity_y": 1, "pressure": 2}

    def _make(self, *, destination_positions=None, source_positions=None, source_values=None,
              destination_values=None, plan=True, source_elements=True,
              interface_fields=(), interface_initializers=None, old_fields=None, new_fields=None):
        source_positions = source_positions or _positions()
        source_values = source_values or _values()
        destination_positions = destination_positions or [[12.0, 17.0]] + [[-1.0, -1.0] for _ in range(6)]
        destination_values = destination_values or [[-1.0, -1.0, -1.0] for _ in range(7)]
        old_node = _Node(source_positions, source_values)
        new_node = _Node(destination_positions, destination_values, lagrangian=(-99.0, -98.0))
        element = _Element(source_positions, source_values)
        location = _location()
        interpolator = AnthonyQ2Interpolator(
            _Mesh([old_node], old_fields or self.fields),
            _Mesh([new_node], new_fields or self.fields),
            [PlannedNodeTransfer(new_node, location)] if plan else [],
            {location.element_id: element} if source_elements else {},
            interface_fields=interface_fields(new_node) if callable(interface_fields) else interface_fields,
            interface_initializers=interface_initializers,
        )
        return interpolator, new_node, element, location

    def test_all_field_slots_and_position_semantics_transfer_from_one_location(self):
        interpolator, node, element, _ = self._make()
        current_before = tuple(node.positions[0])
        interpolator.interpolate()

        self.assertEqual(tuple(node.positions[0]), current_before)
        displacement = (2.0, -3.0)
        for slot in (1, 2, 6):
            self.assertEqual(
                tuple(node.positions[slot]),
                tuple(value + delta for value, delta in zip(element.positions[slot], displacement)),
            )
        for slot in (3, 4, 5):
            self.assertEqual(node.positions[slot], element.positions[slot])
        self.assertEqual(node.lagrangian, list(current_before))
        for slot in range(7):
            self.assertEqual(node.values[slot][:3], element.values[slot])
        self.assertEqual(
            element.position_calls,
            [(slot, (-0.5, 0.5), False) for slot in range(7)],
        )
        self.assertEqual(
            element.value_calls,
            [(slot, (-0.5, 0.5)) for slot in range(7)],
        )

    def test_identity_transfer_is_idempotent(self):
        positions, values = _positions(), _values()
        interpolator, node, _, _ = self._make(
            destination_positions=[list(slot) for slot in positions],
            source_positions=positions,
            source_values=values,
        )
        interpolator.interpolate()
        first_positions = [list(slot) for slot in node.positions]
        first_values = [list(slot) for slot in node.values]
        interpolator.interpolate()
        self.assertEqual(node.positions, first_positions)
        self.assertEqual(node.values, first_values)

    def test_known_divider_multiplier_is_initialized_not_transported(self):
        interpolator, node, element, location = self._make(
            destination_values=[[-1.0, -1.0, -1.0, -1.0] for _ in range(7)],
            interface_fields=lambda target: [InterfaceNodeFields(target, {"divider_lambda": 3})],
            interface_initializers={
                "divider_lambda": lambda _node, source, slot: source.u + source.v + slot
            },
        )
        interpolator.interpolate()
        for slot in range(7):
            self.assertEqual(node.values[slot][3], location.u + location.v + slot)
            self.assertEqual(node.values[slot][:3], element.values[slot])

    def test_missing_source_plan_and_element_fail_loudly(self):
        interpolator, _, _, _ = self._make(plan=False)
        with self.assertRaisesRegex(RemapTransferError, "does not cover"):
            interpolator.interpolate()
        interpolator, _, _, _ = self._make(source_elements=False)
        with self.assertRaisesRegex(RemapTransferError, "no old bulk element"):
            interpolator.interpolate()

    def test_field_and_storage_mismatches_fail_loudly(self):
        interpolator, _, _, _ = self._make(new_fields={"velocity_x": 0, "pressure": 2})
        with self.assertRaisesRegex(RemapTransferError, "field mismatch"):
            interpolator.interpolate()

        destination_values = [[-1.0] * 4 for _ in range(6)]
        interpolator, _, _, _ = self._make(destination_values=destination_values)
        with self.assertRaisesRegex(RemapTransferError, "storage mismatch"):
            interpolator.interpolate()

    def test_unknown_interface_only_field_is_rejected_at_construction(self):
        with self.assertRaisesRegex(RemapTransferError, "unknown interface-only"):
            self._make(
                interface_fields=lambda target: [InterfaceNodeFields(target, {"mystery": 3})],
                interface_initializers={"divider_lambda": 0.0},
            )

    def test_undeclared_interface_value_slot_is_rejected(self):
        interpolator, _, _, _ = self._make(
            destination_values=[[-1.0, -1.0, -1.0, 0.0] for _ in range(7)]
        )
        with self.assertRaisesRegex(RemapTransferError, "unknown interface indices"):
            interpolator.interpolate()


if __name__ == "__main__":
    unittest.main()
