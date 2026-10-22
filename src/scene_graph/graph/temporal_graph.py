from typing import Dict, List, Tuple, Optional, Any

from scene_graph.graph.node import GraphNode
from scene_graph.graph.edge import GraphEdge
from scene_graph.temporal.object_state import ObjectState
from scene_graph.temporal.relation_state import RelationState
from scene_graph.graph.participation_state import GraphParticipationState
from scene_graph.graph.event import GraphEvent, GraphEventType
from scene_graph.graph.graph_history import GraphHistory
from scene_graph.tracking.track import TrackState


class TemporalSceneGraph:
    """
    Maintains the stateful, dynamic network of objects and relations over time.
    
    Key invariant: The graph represents the CURRENT belief about the physical scene,
    not an accumulation of everything ever observed. Dead edges are removed, not just
    marked as REMOVED.
    """
    
    def __init__(self, history_maxlen: int = 1000, history_dump_file: Optional[str] = None):
        self.nodes: Dict[str, GraphNode] = {}
        # Key: (subject_id, object_id, predicate) -> GraphEdge
        self.edges: Dict[Tuple[str, str, str], GraphEdge] = {}
        self.spatial_contexts: Dict[str, Any] = {}
        
        self.history = GraphHistory(maxlen=history_maxlen, dump_file=history_dump_file)
        
        self.current_frame_index: int = -1
        self.current_timestamp: float = -1.0

    def update_contexts(self, contexts: Dict[str, Any], membership: Dict[str, str]):
        """Update spatial contexts and node context membership."""
        self.spatial_contexts = contexts
        anchor_ids = {
            ctx.anchor_track_id
            for ctx in contexts.values()
            if getattr(ctx, "anchor_track_id", None) is not None
        }
        for obj_id, node in self.nodes.items():
            if obj_id in membership:
                node.spatial_context_id = membership[obj_id]
            node.is_spatial_anchor = (obj_id in anchor_ids)

    def update_nodes(self, tracks: list, object_states: Dict[str, ObjectState]):
        """Update the set of nodes in the graph based on tracks and states."""
        current_active_ids = set()

        # Add or update nodes based on explicit tracker lifecycle policy
        for track in tracks:
            # CANDIDATE tracks are unconfirmed and do not enter the world model
            if track.state == TrackState.CANDIDATE:
                continue

            # Invariant 3: Camera-relative fallback may produce local geometry, but must never update
            # global centroids, global bounding boxes, global relations, or global-map state.
            if getattr(track, "coordinate_frame", "world") != "world" or track.centroid_world is None:
                continue

            state = object_states.get(track.object_id, ObjectState.UNKNOWN)

            if track.state in (TrackState.ACTIVE, TrackState.TEMPORARILY_UNOBSERVED):
                current_active_ids.add(track.object_id)
                is_obs = (track.state == TrackState.ACTIVE)
                in_frustum = getattr(track, "is_in_frustum", True)
                vis_str = "observed" if is_obs else ("predicted" if in_frustum else "out_of_view")
                vis_state_str = "observed" if is_obs else ("occluded" if in_frustum else "out_of_view")

                if track.object_id in self.nodes:
                    node = self.nodes[track.object_id]
                    node.track = track
                    node.state = state
                    node.participation = GraphParticipationState.ACTIVE
                    node.attributes["observed"] = is_obs
                    node.attributes["in_frustum"] = in_frustum
                    node.attributes["visibility"] = vis_str
                    node.attributes["visibility_state"] = vis_state_str
                else:
                    self.nodes[track.object_id] = GraphNode(
                        object_id=track.object_id,
                        class_name=track.class_name,
                        track=track,
                        state=state,
                        participation=GraphParticipationState.ACTIVE,
                        attributes={
                            "observed": is_obs,
                            "in_frustum": in_frustum,
                            "visibility": vis_str,
                            "visibility_state": vis_state_str,
                        },
                    )
                    self.history.add_event(GraphEvent(
                        event_type=GraphEventType.NODE_ADDED,
                        timestamp=self.current_timestamp,
                        frame_index=self.current_frame_index,
                        subject_id=track.object_id,
                    ))

            elif track.state == TrackState.LOST:
                if track.object_id in self.nodes:
                    node = self.nodes[track.object_id]
                    node.track = track
                    node.state = state
                    if node.participation != GraphParticipationState.REMOVED:
                        node.participation = GraphParticipationState.REMOVED
                        self.history.add_event(GraphEvent(
                            event_type=GraphEventType.NODE_REMOVED,
                            timestamp=self.current_timestamp,
                            frame_index=self.current_frame_index,
                            subject_id=track.object_id,
                        ))
                    self._remove_edges_for_node(track.object_id)

        # Retire any nodes that are no longer in tracks or marked LOST
        for obj_id, node in self.nodes.items():
            if obj_id not in current_active_ids:
                if node.participation != GraphParticipationState.REMOVED:
                    node.participation = GraphParticipationState.REMOVED
                    self.history.add_event(GraphEvent(
                        event_type=GraphEventType.NODE_REMOVED,
                        timestamp=self.current_timestamp,
                        frame_index=self.current_frame_index,
                        subject_id=obj_id,
                    ))
                self._remove_edges_for_node(obj_id)

    def update_edges(self, evidences: list, relation_states: Dict[Tuple[str, str, str], RelationState]):
        """Update edges to reflect only the current relation state machine output.
        
        Edges are added when SUPPORTED, updated when state changes, and DELETED
        when the state machine garbage-collects a key or the state is CONTRADICTED/UNKNOWN.
        """
        
        # Build lookup for latest evidence
        latest_ev = {}
        for ev in evidences:
            key = (ev.subject_id, ev.object_id, ev.predicate)
            latest_ev[key] = ev
        
        # 1. Remove edges whose keys no longer exist in the state machine
        #    (they were garbage-collected)
        stale_keys = [k for k in self.edges if k not in relation_states]
        for key in stale_keys:
            self._remove_edge(key)
            
        # 2. Update or create edges based on current state
        for key, state in relation_states.items():
            subject_id, object_id, predicate = key
            
            # Edges can only exist if both nodes exist and are active
            if subject_id not in self.nodes or object_id not in self.nodes:
                # Clean up orphaned edges
                if key in self.edges:
                    self._remove_edge(key)
                continue
            
            if state in (RelationState.SUPPORTED, RelationState.ACTIVE):
                if key in self.edges:
                    # Update existing edge
                    edge = self.edges[key]
                    edge.state = state
                    edge.participation = GraphParticipationState.ACTIVE
                    if key in latest_ev:
                        edge.latest_evidence = latest_ev[key]
                elif key in latest_ev:
                    # Create new edge
                    self.edges[key] = GraphEdge(
                        predicate=predicate,
                        subject_id=subject_id,
                        object_id=object_id,
                        state=state,
                        latest_evidence=latest_ev[key],
                        participation=GraphParticipationState.ACTIVE,
                        start_time=self.current_timestamp,
                        start_frame=self.current_frame_index,
                    )
                    self.history.add_event(GraphEvent(
                        event_type=GraphEventType.EDGE_ADDED,
                        timestamp=self.current_timestamp,
                        frame_index=self.current_frame_index,
                        subject_id=subject_id,
                        object_id=object_id,
                        predicate=predicate
                    ))
            elif state in (RelationState.DECAYING, RelationState.WEAKENING, RelationState.OCCLUDED):
                if key in self.edges:
                    edge = self.edges[key]
                    edge.state = state
                    edge.participation = GraphParticipationState.TEMPORARILY_UNOBSERVED
                    if key in latest_ev:
                        edge.latest_evidence = latest_ev[key]
            elif state == RelationState.UNKNOWN:
                if key in self.edges:
                    edge = self.edges[key]
                    edge.state = state
                    edge.participation = GraphParticipationState.TEMPORARILY_UNOBSERVED
            elif state in (RelationState.CONTRADICTED, RelationState.TERMINATED):
                # Contradicted or terminated relations are removed
                if key in self.edges:
                    self._remove_edge(key)
            else:
                # Hypothesized / proposed relations have not yet earned an active graph edge
                if key in self.edges:
                    self._remove_edge(key)

    def _remove_edges_for_node(self, node_id: str):
        """Remove all edges connected to a given node."""
        keys_to_remove = [
            k for k in self.edges
            if k[0] == node_id or k[1] == node_id
        ]
        for k in keys_to_remove:
            self._remove_edge(k)
            
    def _remove_edge(self, key: Tuple[str, str, str]):
        """Helper to remove edge and log event."""
        if key in self.edges:
            edge = self.edges[key]
            edge.end_time = self.current_timestamp
            edge.end_frame = self.current_frame_index
            self.history.add_event(GraphEvent(
                event_type=GraphEventType.EDGE_REMOVED,
                timestamp=self.current_timestamp,
                frame_index=self.current_frame_index,
                subject_id=key[0],
                object_id=key[1],
                predicate=key[2]
            ))
            del self.edges[key]

    def get_active_nodes(self) -> List[GraphNode]:
        """Return all STABLE nodes."""
        return [node for node in self.nodes.values() if node.is_active]
        
    def get_active_edges(self) -> List[GraphEdge]:
        """Return all CONFIRMED edges connecting STABLE nodes."""
        return [
            edge for edge in self.edges.values() 
            if edge.is_active and 
               edge.subject_id in self.nodes and
               edge.object_id in self.nodes and
               self.nodes[edge.subject_id].is_active and 
               self.nodes[edge.object_id].is_active
        ]
