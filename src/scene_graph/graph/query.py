from typing import List, Optional, Tuple, Dict, Any

from scene_graph.graph.temporal_graph import TemporalSceneGraph
from scene_graph.graph.node import GraphNode
from scene_graph.graph.edge import GraphEdge
from scene_graph.graph.event import GraphEvent


INVERSE_PREDICATES: Dict[str, str] = {
    "RIGHT_OF": "LEFT_OF",
    "LEFT_OF": "RIGHT_OF",
    "BELOW": "ABOVE",
    "ABOVE": "BELOW",
    "BEHIND": "IN_FRONT_OF",
    "IN_FRONT_OF": "BEHIND",
    "UNDER": "ON",
    "ON": "UNDER",
    "CONTAINING": "INSIDE",
    "INSIDE": "CONTAINING",
    "OCCLUDED_BY": "OCCLUDING",
    "OCCLUDING": "OCCLUDED_BY",
    "NEAR": "NEAR",
    "FAR": "FAR",
}

CANONICAL_PREDICATES = frozenset([
    "ON", "INSIDE", "LEFT_OF", "ABOVE", "IN_FRONT_OF", "OCCLUDING", "NEAR", "FAR"
])


class QueryEngine:
    """
    API for querying the temporal scene graph.
    Supports dynamic inverse predicate derivation over canonical edge storage.
    """
    def __init__(self, graph: TemporalSceneGraph):
        self.graph = graph

    def get_objects(self, class_name: Optional[str] = None) -> List[GraphNode]:
        """Return all active nodes, optionally filtered by class."""
        nodes = self.graph.get_active_nodes()
        if class_name:
            nodes = [n for n in nodes if n.class_name == class_name]
        return nodes

    def get_object_by_id(self, object_id: str) -> Optional[GraphNode]:
        """Return a specific active node."""
        node = self.graph.nodes.get(object_id)
        if node and node.is_active:
            return node
        return None

    def get_objects_in_context(self, context_id: str) -> List[GraphNode]:
        """Return all active nodes belonging to a specific spatial context."""
        return [
            node for node in self.graph.get_active_nodes()
            if getattr(node, "spatial_context_id", "world") == context_id
        ]

    def get_context_hierarchy(self) -> Dict[str, Any]:
        """Return the spatial context hierarchy tree."""
        return getattr(self.graph, "spatial_contexts", {})

    def what_is(self, predicate: str, object_id: str) -> List[GraphNode]:
        """
        Query: What is [predicate] [object_id]?
        e.g., what_is("ON", "table_1") -> [cup_1, monitor_1]
        Supports dynamic inverse derivation.
        """
        edges = self.graph.get_active_edges()
        subjects = []

        # 1. Direct canonical match: (sub, predicate, object_id)
        for edge in edges:
            if edge.predicate == predicate and edge.object_id == object_id:
                if edge.subject_id in self.graph.nodes:
                    subjects.append(self.graph.nodes[edge.subject_id])

        # 2. Dynamic inverse match: if predicate is non-canonical or has an inverse
        inv_pred = INVERSE_PREDICATES.get(predicate)
        if inv_pred and inv_pred != predicate:
            # (object_id, inv_pred, sub)
            for edge in edges:
                if edge.predicate == inv_pred and edge.subject_id == object_id:
                    if edge.object_id in self.graph.nodes and self.graph.nodes[edge.object_id] not in subjects:
                        subjects.append(self.graph.nodes[edge.object_id])

        return subjects

    def what_does(self, subject_id: str, predicate: str) -> List[GraphNode]:
        """
        Query: What does [subject_id] [predicate]?
        e.g., what_does("cup_1", "ON") -> [table_1]
        Supports dynamic inverse derivation.
        """
        edges = self.graph.get_active_edges()
        objects = []

        # 1. Direct canonical match: (subject_id, predicate, obj)
        for edge in edges:
            if edge.subject_id == subject_id and edge.predicate == predicate:
                if edge.object_id in self.graph.nodes:
                    objects.append(self.graph.nodes[edge.object_id])

        # 2. Dynamic inverse match
        inv_pred = INVERSE_PREDICATES.get(predicate)
        if inv_pred and inv_pred != predicate:
            # (obj, inv_pred, subject_id)
            for edge in edges:
                if edge.predicate == inv_pred and edge.object_id == subject_id:
                    if edge.subject_id in self.graph.nodes and self.graph.nodes[edge.subject_id] not in objects:
                        objects.append(self.graph.nodes[edge.subject_id])

        return objects

    def get_relations_between(self, subject_id: str, object_id: str) -> List[Tuple[str, GraphEdge]]:
        """
        Get all active relations between two specific objects, returning (predicate, edge).
        Derives inverse predicates dynamically when canonical edges exist.
        """
        edges = self.graph.get_active_edges()
        results: List[Tuple[str, GraphEdge]] = []

        for edge in edges:
            if edge.subject_id == subject_id and edge.object_id == object_id:
                results.append((edge.predicate, edge))
            elif edge.subject_id == object_id and edge.object_id == subject_id:
                # Derived inverse relation
                inv_pred = INVERSE_PREDICATES.get(edge.predicate)
                if inv_pred:
                    results.append((inv_pred, edge))

        return results

    def get_event_history(self, object_id: str) -> List[GraphEvent]:
        """Get all structural changes involving a specific object."""
        return self.graph.history.get_events_for_object(object_id)
        
    def query_past_relations(self, subject_id: str, predicate: str, time_window: Optional[tuple[float, float]] = None) -> List[GraphEvent]:
        """Find historical occurrences of a specific relation."""
        events = self.graph.history.get_events_for_object(subject_id)
        
        relation_events = [
            e for e in events 
            if e.subject_id == subject_id and e.predicate == predicate and e.event_type.name == "EDGE_ADDED"
        ]
        
        if time_window:
            start_t, end_t = time_window
            relation_events = [e for e in relation_events if start_t <= e.timestamp <= end_t]
            
        return relation_events
