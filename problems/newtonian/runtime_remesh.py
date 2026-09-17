"""Prepared serial remeshing transaction for the Anthony mapped Q2 mesh.

The transaction freezes every old-mesh identity before
``RemesherViaRecreation`` resets the template.  New nodes are paired with
graph keys by template index and located in the old full-Q2 graph; no spatial
node matching or generic pyoomph interpolation fallback is used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from .evolved_geometry import (
    OrientedQ2Segment,
    PiecewiseQ2Curve,
    build_evolved_four_patch_graph,
)
from .q2_locator import Q2PointLocator, SourceLocation
from .remap_transfer import (
    AnthonyQ2Interpolator,
    InterfaceNodeFields,
    PlannedNodeTransfer,
)
from .structured_blocks import MeshNode, StructuredQ2Graph


KNOWN_INTERFACE_FIELDS = frozenset(("_kin_bc", "_lagr_enf_bc_mesh_x"))


class RuntimeRemeshError(RuntimeError):
    """Raised when a prepared remesh cannot be made exact and explicit."""


@dataclass(frozen=True)
class GeometryDiagnostics:
    interface_elements: int
    old_elements: int
    old_nodes: int
    new_elements: int
    new_nodes: int
    old_minimum_scaled_jacobian: float
    new_minimum_scaled_jacobian: float


def _coordinates(node: Any) -> tuple[float, float]:
    point = (float(node.x(0)), float(node.x(1)))
    if not all(math.isfinite(value) for value in point):
        raise RuntimeRemeshError("runtime mesh contains a non-finite coordinate")
    return point


def _runtime_nodes_by_template_index(template: Any, mesh: Any) -> tuple[Any, ...]:
    """Recover template-node identity from ordered element connectivity.

    pyoomph may reorder ``mesh.nodes()`` while assigning equations.  Element
    creation order and each C2 element's nine template indices remain the
    explicit, coordinate-free identity carried by this template.
    """
    elements = tuple(mesh.elements())
    graph = template.graph
    if graph is None or len(elements) != len(graph.elements):
        raise RuntimeRemeshError("runtime elements do not match the template graph")
    by_index: dict[int, Any] = {}
    for graph_element, runtime_element in zip(graph.elements, elements):
        indices = template.template_element_indices.get(graph_element.element_id)
        if indices is None or runtime_element.nnode() != len(indices):
            raise RuntimeRemeshError(
                f"runtime C2 connectivity is missing for {graph_element.element_id!r}"
            )
        for index, node_number in enumerate(indices):
            node = runtime_element.node_pt(index)
            incumbent = by_index.setdefault(node_number, node)
            if incumbent is not node:
                raise RuntimeRemeshError(
                    f"template node {node_number} maps to inconsistent runtime nodes"
                )
    expected = set(range(len(graph.nodes)))
    if set(by_index) != expected:
        raise RuntimeRemeshError("element connectivity does not cover every template node")
    if len({id(node) for node in by_index.values()}) != len(by_index):
        raise RuntimeRemeshError("distinct template indices collapse onto one runtime node")
    return tuple(by_index[index] for index in range(len(by_index)))


def capture_interface_q2_chain(interface_mesh: Any, expected_faces: int) -> PiecewiseQ2Curve:
    """Walk endpoint adjacency and return a neck-to-pole Q2 chain."""
    elements = tuple(interface_mesh.elements())
    if len(elements) != expected_faces:
        raise RuntimeRemeshError(
            f"interface face count mismatch: runtime={len(elements)}, expected={expected_faces}"
        )
    faces: dict[frozenset[int], tuple[Any, Any, Any]] = {}
    incident: dict[int, list[frozenset[int]]] = {}
    endpoint_nodes: dict[int, Any] = {}
    for element in elements:
        if int(element.nnode()) != 3:
            raise RuntimeRemeshError("the live interface contains a non-Q2 face")
        nodes = tuple(element.node_pt(index) for index in range(3))
        ids = tuple(id(node) for node in nodes)
        if len(set(ids)) != 3:
            raise RuntimeRemeshError("an interface face repeats one of its nodes")
        identity = frozenset((ids[0], ids[2]))
        if identity in faces:
            raise RuntimeRemeshError("duplicate interface face endpoints")
        faces[identity] = nodes
        for endpoint in (nodes[0], nodes[2]):
            key = id(endpoint)
            endpoint_nodes[key] = endpoint
            incident.setdefault(key, []).append(identity)

    if any(len(entries) > 2 for entries in incident.values()):
        raise RuntimeRemeshError("the interface endpoint graph branches")
    terminals = [key for key, entries in incident.items() if len(entries) == 1]
    if len(terminals) != 2 or any(len(entries) not in (1, 2) for entries in incident.values()):
        raise RuntimeRemeshError("the interface faces do not form one open chain")

    scale = max(1.0, *(
        abs(value) for node in endpoint_nodes.values() for value in _coordinates(node)
    ))
    tolerance = 128.0 * math.ulp(scale)
    necks = [key for key in terminals if abs(_coordinates(endpoint_nodes[key])[1]) <= tolerance]
    poles = [key for key in terminals if abs(_coordinates(endpoint_nodes[key])[0]) <= tolerance]
    if len(necks) != 1 or len(poles) != 1 or necks[0] == poles[0]:
        raise RuntimeRemeshError("interface terminals are not a unique plane neck and axis pole")

    current = necks[0]
    previous: frozenset[int] | None = None
    segments: list[OrientedQ2Segment] = []
    visited: set[frozenset[int]] = set()
    while True:
        available = [face for face in incident[current] if face != previous]
        if not available:
            break
        if len(available) != 1:
            raise RuntimeRemeshError("ambiguous interface-chain continuation")
        identity = available[0]
        if identity in visited:
            raise RuntimeRemeshError("interface chain contains a cycle or duplicate face")
        visited.add(identity)
        first, middle, last = faces[identity]
        if id(first) == current:
            oriented = (first, middle, last)
        elif id(last) == current:
            oriented = (last, middle, first)
        else:  # pragma: no cover - protected by the adjacency construction
            raise RuntimeRemeshError("interface adjacency lost an endpoint")
        segments.append(OrientedQ2Segment(*map(_coordinates, oriented)))
        previous, current = identity, id(oriented[2])
    if current != poles[0] or len(visited) != len(faces):
        raise RuntimeRemeshError("interface chain is disconnected or misses a face")
    return PiecewiseQ2Curve(tuple(segments))


def runtime_graph_from_template(template: Any, mesh: Any) -> StructuredQ2Graph:
    """Replace graph coordinates from runtime nodes by explicit template index."""
    graph = template.graph
    if graph is None:
        raise RuntimeRemeshError("mapped template has no graph")
    runtime_nodes = _runtime_nodes_by_template_index(template, mesh)
    indices = template.node_key_to_template_index
    if set(indices) != set(graph.nodes) or set(indices.values()) != set(range(len(runtime_nodes))):
        raise RuntimeRemeshError("template node indices do not bijectively cover the runtime mesh")
    nodes: dict[tuple[int, ...], MeshNode] = {}
    for key, graph_node in graph.nodes.items():
        nodes[key] = replace(graph_node, coordinates=_coordinates(runtime_nodes[indices[key]]))
    return replace(graph, nodes=nodes)


def bind_runtime_elements(template: Any, mesh: Any) -> dict[tuple[int, int, int], Any]:
    """Bind graph element IDs to bulk elements through template node indices."""
    runtime_nodes = _runtime_nodes_by_template_index(template, mesh)
    runtime_index = {id(node): index for index, node in enumerate(runtime_nodes)}
    by_connectivity: dict[tuple[int, ...], Any] = {}
    for element in mesh.elements():
        connectivity = tuple(runtime_index[id(element.node_pt(i))] for i in range(element.nnode()))
        if connectivity in by_connectivity:
            raise RuntimeRemeshError("runtime bulk mesh has duplicate element connectivity")
        by_connectivity[connectivity] = element
    result: dict[tuple[int, int, int], Any] = {}
    for element_id, connectivity in template.template_element_indices.items():
        element = by_connectivity.get(tuple(connectivity))
        if element is None:
            raise RuntimeRemeshError(f"no runtime bulk element for template ID {element_id!r}")
        result[element_id] = element
    if len(result) != len(by_connectivity):
        raise RuntimeRemeshError("template element IDs do not cover the runtime bulk mesh")
    return result


@dataclass(frozen=True)
class FieldSchema:
    bulk_fields: tuple[tuple[str, int], ...]
    interface_fields: tuple[str, ...]
    position_storage: int
    field_storage: int


def preflight_old_field_schema(mesh: Any) -> FieldSchema:
    """Validate every old nodal value slot describable before recreation."""
    nodes = tuple(mesh.nodes())
    if not nodes:
        raise RuntimeRemeshError("old bulk mesh has no nodes")
    position_storage = {int(node.variable_position_pt().ntstorage()) for node in nodes}
    field_storage = {int(node.ntstorage()) for node in nodes}
    if position_storage != {7}:
        raise RuntimeRemeshError(
            f"old position history must have seven slots, got {sorted(position_storage)}"
        )
    if len(field_storage) != 1 or next(iter(field_storage)) <= 0:
        raise RuntimeRemeshError(
            f"old field history storage is not uniform and positive: {sorted(field_storage)}"
        )
    bulk_fields = dict(mesh.get_nodal_field_indices())
    if len(set(bulk_fields.values())) != len(bulk_fields):
        raise RuntimeRemeshError("old bulk nodal field indices are not unique")
    declared: set[str] = set()
    for interface_mesh in mesh._interfacemeshes.values():
        declared.update(set(interface_mesh.get_nodal_field_indices()) - set(bulk_fields))
    unknown = declared - KNOWN_INTERFACE_FIELDS
    if unknown:
        raise RuntimeRemeshError(f"undeclared old interface nodal fields: {sorted(unknown)}")
    field_ids = {name: int(mesh.has_interface_dof_id(name)) for name in declared}
    if any(value < 0 for value in field_ids.values()):
        raise RuntimeRemeshError("old interface field has no bulk-mesh additional-value ID")
    bulk_indices = set(bulk_fields.values())
    for node in nodes:
        interface_indices = {
            int(node.additional_value_index(field_id))
            for field_id in field_ids.values()
            if int(node.additional_value_index(field_id)) >= 0
        }
        described = bulk_indices | interface_indices
        actual = set(range(node.nvalue()))
        if described != actual:
            raise RuntimeRemeshError(
                "old nodal values are not fully described by the bulk/interface schema: "
                f"unknown={sorted(actual - described)}, missing={sorted(described - actual)}"
            )
    return FieldSchema(
        tuple(sorted(bulk_fields.items())), tuple(sorted(declared)),
        next(iter(position_storage)), next(iter(field_storage)),
    )


@dataclass
class PreparedMappedRemesh:
    template: Any
    captured_interface: PiecewiseQ2Curve
    old_graph: StructuredQ2Graph
    new_graph: StructuredQ2Graph
    locator: Q2PointLocator
    source_locations: dict[tuple[int, ...], SourceLocation]
    source_elements: dict[tuple[int, int, int], Any]
    source_schema: FieldSchema
    diagnostics: GeometryDiagnostics
    accepted_rmin: float
    accepted_abs2h: float | None
    destination_count: int = 0
    missing_count: int = 0
    fallback_count: int = 0
    completed: bool = False
    _transferred_state: dict[int, tuple[Any, tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]] | None = None

    def _record_transferred_state(self, mesh: Any) -> None:
        state = {}
        for node in mesh.nodes():
            positions = tuple(
                tuple(float(node.x_at_t(slot, component)) for component in range(node.ndim()))
                for slot in range(7)
            )
            values = tuple(
                tuple(float(node.value_at_t(slot, index)) for index in range(node.nvalue()))
                for slot in range(node.ntstorage())
            )
            state[id(node)] = (node, positions, values)
        self._transferred_state = state

    def restore_after_remesh_hooks(self, mesh: Any) -> None:
        """Undo boundary-condition history rewrites performed by base hooks."""
        if self._transferred_state is None:
            raise RuntimeRemeshError("interpolator did not record transferred destination state")
        live = {id(node): node for node in mesh.nodes()}
        if set(live) != set(self._transferred_state):
            raise RuntimeRemeshError("destination nodes changed after interpolation")
        for key, (node, positions, values) in self._transferred_state.items():
            if live[key] is not node:
                raise RuntimeRemeshError("destination node identity changed after interpolation")
            for slot in range(1, 7):
                for component, value in enumerate(positions[slot]):
                    node.set_x_at_t(slot, component, value)
            for slot, history in enumerate(values):
                for index, value in enumerate(history):
                    node.set_value_at_t(slot, index, value)

    def interpolator(self, old: Any, new: Any) -> AnthonyQ2Interpolator:
        """Construct the sole interpolator accepted for this transaction."""
        if old.get_name() != "drop" or new.get_name() != "drop":
            raise RuntimeRemeshError("mapped remesh transaction only covers the drop bulk mesh")
        runtime_nodes = _runtime_nodes_by_template_index(self.template, new)
        indices = self.template.node_key_to_template_index
        if self.template.graph is not self.new_graph:
            raise RuntimeRemeshError("recreated template did not install the prepared graph")
        if set(indices) != set(self.new_graph.nodes) or set(indices.values()) != set(range(len(runtime_nodes))):
            raise RuntimeRemeshError("new template indices do not bijectively cover destination nodes")

        plan: list[PlannedNodeTransfer] = []
        for key in sorted(self.new_graph.nodes):
            node = runtime_nodes[indices[key]]
            expected = self.new_graph.nodes[key].coordinates
            if _coordinates(node) != expected:
                raise RuntimeRemeshError(f"destination node {key!r} does not match its template index")
            location = self.source_locations.get(key)
            if location is None:
                raise RuntimeRemeshError(f"prepared source plan has no location for {key!r}")
            plan.append(PlannedNodeTransfer(node, location))

        bulk_fields = set(new.get_nodal_field_indices())
        declared = set()
        for interface_mesh in new._interfacemeshes.values():
            declared.update(set(interface_mesh.get_nodal_field_indices()) - bulk_fields)
        unknown = declared - KNOWN_INTERFACE_FIELDS
        if unknown:
            raise RuntimeRemeshError(f"undeclared interface nodal fields: {sorted(unknown)}")
        field_ids = {name: new.has_interface_dof_id(name) for name in sorted(declared)}
        if any(value < 0 for value in field_ids.values()):
            raise RuntimeRemeshError("declared interface field has no bulk-mesh additional-value ID")
        policies = []
        for node in new.nodes():
            fields = {
                name: int(node.additional_value_index(field_id))
                for name, field_id in field_ids.items()
                if int(node.additional_value_index(field_id)) >= 0
            }
            if fields:
                policies.append(InterfaceNodeFields(node, fields))

        self.destination_count = len(plan)
        transaction = self

        class RecordingAnthonyQ2Interpolator(AnthonyQ2Interpolator):
            def interpolate(inner_self) -> None:
                super().interpolate()
                transaction._record_transferred_state(inner_self.new)

        return RecordingAnthonyQ2Interpolator(
            old,
            new,
            plan,
            self.source_elements,
            interface_fields=policies,
            interface_initializers={name: 0.0 for name in KNOWN_INTERFACE_FIELDS},
        )

    def interpolator_factory(self) -> type:
        transaction = self

        class TransactionBoundInterpolator:
            distributed_limitation = AnthonyQ2Interpolator.distributed_limitation

            def __new__(cls, old: Any, new: Any) -> AnthonyQ2Interpolator:
                return transaction.interpolator(old, new)

        TransactionBoundInterpolator.__name__ = "AnthonyPreparedQ2Interpolator"
        return TransactionBoundInterpolator


def prepare_mapped_transaction(
    template: Any,
    mesh: Any,
    interface_mesh: Any,
    accepted_rmin: float,
    *,
    cap_resolution: int,
    bridge_normal_resolution: int,
    accepted_abs2h: float | None = None,
) -> PreparedMappedRemesh:
    if template.pending_remesh_graph is not None:
        raise RuntimeRemeshError("a template remesh graph is already pending")
    accepted_rmin = float(accepted_rmin)
    if not math.isfinite(accepted_rmin) or accepted_rmin <= 0.0:
        raise ValueError("accepted R_min must be positive and finite")
    if accepted_abs2h is not None:
        accepted_abs2h = float(accepted_abs2h)
        if not math.isfinite(accepted_abs2h) or accepted_abs2h <= 0.0:
            raise ValueError("accepted |2H| must be positive and finite when supplied")
    # Everything below this point remains read-only until the complete source
    # plan and schema have been proved usable.
    old_graph = runtime_graph_from_template(template, mesh)
    interface = capture_interface_q2_chain(
        interface_mesh, len(template.graph.boundary_facets["interface"])
    )
    scale = max(1.0, abs(accepted_rmin), abs(interface.start[0]))
    if abs(interface.start[0] - accepted_rmin) > 128.0 * math.ulp(scale):
        raise RuntimeRemeshError(
            f"accepted R_min {accepted_rmin!r} disagrees with captured neck {interface.start[0]!r}"
        )
    new_graph, _intersection = build_evolved_four_patch_graph(
        interface,
        cap_resolution=cap_resolution,
        bridge_normal_resolution=bridge_normal_resolution,
        grading_u=template.grading_u,
        grading_v=template.grading_v,
    )
    locator = Q2PointLocator(old_graph)
    source_locations = {
        key: locator.locate(new_graph.nodes[key].coordinates)
        for key in sorted(new_graph.nodes)
    }
    source_elements = bind_runtime_elements(template, mesh)
    source_schema = preflight_old_field_schema(mesh)
    diagnostics = GeometryDiagnostics(
        interface_elements=len(interface.segments),
        old_elements=len(old_graph.elements),
        old_nodes=len(old_graph.nodes),
        new_elements=len(new_graph.elements),
        new_nodes=len(new_graph.nodes),
        old_minimum_scaled_jacobian=old_graph.minimum_scaled_jacobian(),
        new_minimum_scaled_jacobian=new_graph.minimum_scaled_jacobian(),
    )
    if diagnostics.old_minimum_scaled_jacobian <= 0.0 or diagnostics.new_minimum_scaled_jacobian <= 0.0:
        raise RuntimeRemeshError("old or prepared Q2 graph is folded")
    transaction = PreparedMappedRemesh(
        template, interface, old_graph, new_graph, locator, source_locations,
        source_elements, source_schema, diagnostics, accepted_rmin, accepted_abs2h,
        destination_count=len(source_locations),
    )
    template.prepare_remesh_graph(new_graph)
    return transaction


__all__ = [
    "GeometryDiagnostics",
    "FieldSchema",
    "KNOWN_INTERFACE_FIELDS",
    "PreparedMappedRemesh",
    "RuntimeRemeshError",
    "bind_runtime_elements",
    "capture_interface_q2_chain",
    "prepare_mapped_transaction",
    "preflight_old_field_schema",
    "runtime_graph_from_template",
]
