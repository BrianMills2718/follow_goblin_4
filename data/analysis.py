"""
Network analysis utilities for X Network Visualization.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any

import networkx as nx

from config import DEFAULT_CLOUTRANK_DAMPING, DEFAULT_CLOUTRANK_EPSILON, DEFAULT_CLOUTRANK_MAX_ITER

logger = logging.getLogger(__name__)

def compute_in_degree(nodes: Dict[str, Dict], edges: List[Tuple[str, str]]) -> Dict[str, int]:
    """
    Compute in-degree (number of incoming connections) for each node.
    Edges point from follower to followed, so in-degree = number of followers.
    
    Args:
        nodes: Dictionary mapping node IDs to node data
        edges: List of (source, target) edge tuples
    
    Returns:
        Dictionary mapping node IDs to in-degree values
    """
    in_degrees = {node_id: 0 for node_id in nodes.keys()}
    
    for src, tgt in edges:
        if tgt in in_degrees:
            in_degrees[tgt] += 1
            
    return in_degrees

def compute_cloutrank(nodes: Dict[str, Dict], 
                     edges: List[Tuple[str, str]], 
                     damping: float = DEFAULT_CLOUTRANK_DAMPING, 
                     epsilon: float = DEFAULT_CLOUTRANK_EPSILON, 
                     max_iter: int = DEFAULT_CLOUTRANK_MAX_ITER, 
                     return_contributors: bool = False) -> Dict[str, float] or Tuple[Dict[str, float], Dict[str, Dict], Dict[str, Dict]]:
    """
    Compute CloutRank (PageRank) using the network structure with proper contribution tracking.
    
    Args:
        nodes: Dictionary mapping node IDs to node data
        edges: List of (source, target) edge tuples
        damping: Damping factor
        epsilon: Convergence threshold
        max_iter: Maximum iterations
        return_contributors: Whether to track and return contribution data
        
    Returns:
        Dictionary mapping node IDs to CloutRank scores, or tuple with scores and contributions
    """
    # Construct the directed graph - edges point from follower to followed
    # Influence flows from follower to followed nodes
    G = nx.DiGraph()
    G.add_nodes_from(nodes.keys())
    
    # Add edges, ensuring all node IDs are strings
    formatted_edges = [(str(src), str(tgt)) for src, tgt in edges]
    G.add_edges_from(formatted_edges)
    
    # Calculate in-degrees and out-degrees
    in_degrees = {node: G.in_degree(node) for node in G.nodes()}
    out_degrees = {node: G.out_degree(node) for node in G.nodes()}
    
    # Identify dangling nodes (nodes with no outgoing edges)
    dangling_nodes = [node for node, out_degree in out_degrees.items() if out_degree == 0]
    
    # Compute PageRank using NetworkX's implementation
    cloutrank_scores = nx.pagerank(G, alpha=damping, max_iter=max_iter, tol=epsilon)
    
    # If we don't need to track contributors, just return scores
    if not return_contributors:
        return cloutrank_scores
    
    # Initialize contribution tracking
    incoming_contributions = {node: {} for node in G.nodes()}  # Who contributes to me
    outgoing_contributions = {node: {} for node in G.nodes()}  # Who I contribute to
    
    # First, handle normal nodes (non-dangling)
    for node in G.nodes():
        # Skip dangling nodes, we'll handle them separately
        if node in dangling_nodes:
            continue
            
        # Get the node's PageRank score
        node_score = cloutrank_scores.get(node, 0)
        
        # Get all accounts this node follows
        followed_accounts = list(G.successors(node))
        
        # Calculate contribution per followed account
        contribution_per_followed = (damping * node_score) / len(followed_accounts) if followed_accounts else 0
        
        # Record contributions
        for followed in followed_accounts:
            incoming_contributions[followed][node] = contribution_per_followed
            outgoing_contributions[node][followed] = contribution_per_followed
    
    # Then handle dangling nodes - their PageRank is distributed to all nodes
    for node in dangling_nodes:
        node_score = cloutrank_scores.get(node, 0)
        
        # Dangling nodes distribute to all nodes equally
        contribution_per_node = (damping * node_score) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        
        # Record these contributions
        for target in G.nodes():
            # Skip self-contributions for clarity
            if target == node:
                continue
                
            # Add dangling contribution
            if 'dangling' not in incoming_contributions[target]:
                incoming_contributions[target]['dangling'] = 0
            incoming_contributions[target]['dangling'] += contribution_per_node
            
            # Track outgoing from dangling node
            if 'global' not in outgoing_contributions[node]:
                outgoing_contributions[node]['global'] = 0
            outgoing_contributions[node]['global'] += contribution_per_node
    
    # Finally add random teleportation component
    teleport_weight = (1 - damping) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    for node in G.nodes():
        # Each node gets teleportation weight from every node (including itself)
        for source in G.nodes():
            if 'teleport' not in incoming_contributions[node]:
                incoming_contributions[node]['teleport'] = 0
            incoming_contributions[node]['teleport'] += teleport_weight * cloutrank_scores.get(source, 0)
    
    return cloutrank_scores, incoming_contributions, outgoing_contributions


def select_top_nodes_for_visualization(network_data, importance_scores, max_nodes=50):
    """
    Select nodes for visualization based on importance scores.
    
    Args:
        network_data: NetworkData instance
        importance_scores: Dictionary mapping node IDs to importance scores
        max_nodes: Maximum number of nodes to display
        
    Returns:
        Set of selected node IDs
    """
    # Always include original node
    selected_nodes = {network_data.original_id} if network_data.original_id else set()
    
    # Get nodes directly followed by original
    first_degree_nodes = network_data.get_first_degree_nodes()
    
    # Get top first-degree connections (prioritize showing these)
    top_first_degree = []
    if first_degree_nodes:
        # Create list of (node_id, score) tuples for first-degree nodes
        first_degree_scores = [(node_id, importance_scores.get(node_id, 0)) 
                              for node_id in first_degree_nodes]
        
        # Sort by importance score
        sorted_first_degree = sorted(first_degree_scores, key=lambda x: x[1], reverse=True)
        
        # Take the top third (at least 5, but no more than max_nodes/4)
        num_first_degree = max(min(len(sorted_first_degree), max_nodes // 4), 5)
        top_first_degree = sorted_first_degree[:num_first_degree]
        
        # Add these to the selected nodes
        for node_id, _ in top_first_degree:
            selected_nodes.add(node_id)
    
    # Calculate how many slots are left for other nodes
    remaining_slots = max_nodes - len(selected_nodes)
    
    # Split the remaining slots between top overall and independent nodes
    slots_overall = remaining_slots // 2
    slots_independent = remaining_slots - slots_overall
    
    # Get top nodes overall (excluding those already selected)
    top_overall = []
    for node_id, score, _ in network_data.get_top_nodes_by_importance(
        importance_scores, 
        max_nodes=max_nodes  # Get more than we need to filter out already selected
    ):
        if node_id not in selected_nodes:
            top_overall.append(node_id)
            if len(top_overall) >= slots_overall:
                break
    
    # Get top independent nodes (not directly connected to original)
    top_independent = []
    for node_id, score, _ in network_data.get_top_independent_nodes(
        importance_scores, 
        max_nodes=max_nodes  # Get more than we need to filter out already selected
    ):
        if node_id not in selected_nodes:
            top_independent.append(node_id)
            if len(top_independent) >= slots_independent:
                break
    
    # Add the selected nodes
    selected_nodes.update(top_overall)
    selected_nodes.update(top_independent)
    
    # Debug info
    logger.info(f"Selected nodes breakdown: Original (1), First-degree ({len(top_first_degree)}), " 
                f"Top overall ({len(top_overall)}), Independent ({len(top_independent)})")
    
    return selected_nodes