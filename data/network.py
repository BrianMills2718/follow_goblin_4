"""
Network data structures for X Network Visualization.
"""
import logging
from typing import Dict, List, Tuple, Set, Any, Optional
from datetime import datetime

import streamlit as st

from config import DEFAULT_FILTERS
from utils.logging_config import get_logger

logger = get_logger(__name__)

def compute_ratio(followers_count: int, friends_count: int) -> float:
    """
    Compute follower/following ratio.
    
    Args:
        followers_count: Number of followers
        friends_count: Number of friends/following
        
    Returns:
        Follower/following ratio or 0 if friends_count is 0
    """
    return followers_count / friends_count if friends_count else 0

class NetworkData:
    """
    Class for storing and managing Twitter network data.
    """
    
    def __init__(self):
        """Initialize an empty network."""
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Tuple[str, str]] = []
        self.original_id: Optional[str] = None
        self.original_username: Optional[str] = None
        self.topic_nodes: Dict[str, Dict] = {}  # Store topic nodes separately
    
    def add_original_node(self, username: str) -> str:
        """
        Add the original user node.
        
        Args:
            username: Twitter username
            
        Returns:
            Node ID for the original user
        """
        node_id = f"orig_{username}"
        self.nodes[node_id] = {
            "screen_name": username,
            "name": username,
            "followers_count": None,
            "friends_count": None,
            "statuses_count": None,
            "media_count": None,
            "created_at": None,
            "location": None,
            "blue_verified": None,
            "verified": None,
            "website": None,
            "business_account": None,
            "ratio": None,
            "description": "",
            "direct": True,
            "tweets": [],
            "tweet_summary": ""
        }
        self.original_id = node_id
        self.original_username = username
        return node_id
    
    def add_node(self, node_id: str, attributes: Dict) -> None:
        """
        Add a node to the network.
        
        Args:
            node_id: Unique identifier for the node
            attributes: Node attributes
        """
        # Calculate ratio
        ratio = compute_ratio(
            attributes.get("followers_count", 0), 
            attributes.get("friends_count", 0)
        )
        
        # Create node with all required attributes
        node = {
            "screen_name": attributes.get("screen_name", ""),
            "name": attributes.get("name", ""),
            "followers_count": attributes.get("followers_count", 0),
            "friends_count": attributes.get("friends_count", 0),
            "statuses_count": attributes.get("statuses_count", 0),
            "media_count": attributes.get("media_count", 0),
            "created_at": attributes.get("created_at", ""),
            "location": attributes.get("location", ""),
            "blue_verified": attributes.get("blue_verified", False),
            "verified": attributes.get("verified", False),
            "website": attributes.get("website", ""),
            "business_account": attributes.get("business_account", False),
            "description": attributes.get("description", ""),
            "ratio": ratio,
            "direct": attributes.get("direct", False),
            "tweets": [],
            "tweet_summary": ""
        }
        
        self.nodes[node_id] = node
    
    def add_edge(self, source_id: str, target_id: str) -> None:
        """
        Add an edge to the network.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
        """
        if source_id in self.nodes and target_id in self.nodes:
            self.edges.append((source_id, target_id))
        else:
            logger.warning(f"Cannot add edge between non-existent nodes: {source_id} -> {target_id}")
    
    def get_first_degree_nodes(self) -> Set[str]:
        """
        Get nodes directly connected to the original node.
        
        Returns:
            Set of node IDs for first-degree connections
        """
        if not self.original_id:
            return set()
        
        return {tgt for src, tgt in self.edges if src == self.original_id}
    
    def get_second_degree_nodes(self) -> Set[str]:
        """
        Get nodes that are second-degree connections.
        
        Returns:
            Set of node IDs for second-degree connections
        """
        if not self.original_id:
            return set()
        
        first_degree = self.get_first_degree_nodes()
        second_degree = set()
        
        for src, tgt in self.edges:
            if src in first_degree and tgt != self.original_id and tgt not in first_degree:
                second_degree.add(tgt)
        
        return second_degree
    
    def filter_nodes(self, filters: Dict = None) -> Dict[str, Dict]:
        """
        Filter nodes based on provided criteria.
        
        Args:
            filters: Dictionary of filter criteria
            
        Returns:
            Dictionary of filtered nodes
        """
        if filters is None:
            filters = DEFAULT_FILTERS
            
        filtered = {}
        
        for node_id, node in self.nodes.items():
            # Always include the original node
            if node_id.startswith("orig_"):
                filtered[node_id] = node
                continue

            # Helper function to safely compare values that might be None
            def is_in_range(value, min_val, max_val):
                if value is None:
                    return False
                return min_val <= value <= max_val

            # Numeric filters with None handling
            if not is_in_range(node.get("statuses_count"), filters["statuses_range"][0], filters["statuses_range"][1]):
                continue
            if not is_in_range(node.get("followers_count"), filters["followers_range"][0], filters["followers_range"][1]):
                continue
            if not is_in_range(node.get("friends_count"), filters["friends_range"][0], filters["friends_range"][1]):
                continue
            if not is_in_range(node.get("media_count"), filters["media_range"][0], filters["media_range"][1]):
                continue

            # Location filters
            location = node.get("location")
            if filters["selected_locations"]:
                if location is not None and isinstance(location, str) and location.strip():
                    location = location.strip().lower()
                    if not any(loc.lower() in location for loc in filters["selected_locations"]):
                        continue
                else:
                    continue
            elif filters["require_location"]:
                if not location or not isinstance(location, str) or not location.strip():
                    continue

            # Blue verified filter
            if filters["require_blue_verified"]:
                if not node.get("blue_verified", False):
                    continue

            # Verified filter
            if filters["verified_option"] == "Only Verified":
                if not node.get("verified", False):
                    continue
            elif filters["verified_option"] == "Only Not Verified":
                if node.get("verified", False):
                    continue

            # Website filter
            if filters["require_website"]:
                if not node.get("website", "").strip():
                    continue

            # Business account filter
            if filters["business_account_option"] == "Only Business Accounts":
                if not node.get("business_account", False):
                    continue
            elif filters["business_account_option"] == "Only Non-Business Accounts":
                if node.get("business_account", False):
                    continue
            
            filtered[node_id] = node
            
        return filtered
    
    def filter_by_degree(self, filtered_nodes: Dict[str, Dict], 
                        show_original: bool = True,
                        show_first_degree: bool = True,
                        show_second_degree: bool = True) -> Dict[str, Dict]:
        """
        Filter nodes by their degree of connection to the original node.
        
        Args:
            filtered_nodes: Pre-filtered nodes
            show_original: Whether to include the original node
            show_first_degree: Whether to include first-degree connections
            show_second_degree: Whether to include second-degree connections
            
        Returns:
            Dictionary of filtered nodes
        """
        if not self.original_id:
            return {}
            
        first_degree = self.get_first_degree_nodes()
        second_degree = self.get_second_degree_nodes()
        
        degree_filtered = {}
        
        for node_id, node in filtered_nodes.items():
            if node_id == self.original_id and show_original:
                degree_filtered[node_id] = node
            elif node_id in first_degree and show_first_degree:
                degree_filtered[node_id] = node
            elif node_id in second_degree and show_second_degree:
                degree_filtered[node_id] = node
                
        return degree_filtered
    
    def filter_by_communities(self, nodes: Dict[str, Dict], 
                             node_communities: Dict[str, str],
                             selected_communities: Dict[str, bool]) -> Dict[str, Dict]:
        """
        Filter nodes by their community assignments.
        
        Args:
            nodes: Pre-filtered nodes
            node_communities: Dictionary mapping usernames to community IDs
            selected_communities: Dictionary mapping community IDs to selection state
            
        Returns:
            Dictionary of filtered nodes
        """
        community_filtered = {}
        
        for node_id, node in nodes.items():
            # Always include original node
            if node_id.startswith("orig_"):
                community_filtered[node_id] = node
                continue
                
            username = node["screen_name"]
            if username in node_communities:
                community = node_communities[username]
                if selected_communities.get(community, True):
                    community_filtered[node_id] = node
            
        return community_filtered
    
    def update_node_community_colors(self, community_colors: Dict[str, str], 
                                   node_communities: Dict[str, str]) -> None:
        """
        Update node colors based on community assignments.
        
        Args:
            community_colors: Dictionary mapping community IDs to colors
            node_communities: Dictionary mapping usernames to community IDs
        """
        for node_id, node in self.nodes.items():
            username = node["screen_name"]
            if username in node_communities:
                community = node_communities[username]
                # Add safety check for community ID
                if community in community_colors:
                    self.nodes[node_id]["community_color"] = community_colors[community]
                else:
                    # Assign default color for unknown communities
                    self.nodes[node_id]["community_color"] = "#cccccc"  # Default gray color
    
    def update_node_tweet_data(self, node_id: str, tweets: List[Dict], summary: str) -> None:
        """
        Update a node with tweet data and summary.
        
        Args:
            node_id: Node ID
            tweets: List of tweet dictionaries
            summary: Tweet summary
        """
        if node_id in self.nodes:
            self.nodes[node_id]["tweets"] = tweets
            self.nodes[node_id]["tweet_summary"] = summary
    
    def get_top_nodes_by_importance(self, importance_scores: Dict[str, float], 
                                  max_nodes: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        Get top nodes by importance score.
        
        Args:
            importance_scores: Dictionary mapping node IDs to importance scores
            max_nodes: Maximum number of nodes to return
            
        Returns:
            List of (node_id, score, node_data) tuples
        """
        # Skip original node
        nodes_with_scores = [
            (nid, importance_scores.get(nid, 0), self.nodes[nid]) 
            for nid in self.nodes 
            if not nid.startswith("orig_")
        ]
        
        # Sort by importance score (descending)
        sorted_nodes = sorted(nodes_with_scores, key=lambda x: x[1], reverse=True)
        
        return sorted_nodes[:max_nodes]
    
    def get_top_independent_nodes(self, importance_scores: Dict[str, float], 
                                max_nodes: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        Get top nodes not directly connected to the original node.
        
        Args:
            importance_scores: Dictionary mapping node IDs to importance scores
            max_nodes: Maximum number of nodes to return
            
        Returns:
            List of (node_id, score, node_data) tuples
        """
        if not self.original_id:
            return []
            
        first_degree = self.get_first_degree_nodes()
        
        # Get nodes not directly connected to original
        independent_nodes = [
            (nid, importance_scores.get(nid, 0), self.nodes[nid]) 
            for nid in self.nodes 
            if not nid.startswith("orig_") and nid not in first_degree
        ]
        
        # Sort by importance score (descending)
        sorted_nodes = sorted(independent_nodes, key=lambda x: x[1], reverse=True)
        
        return sorted_nodes[:max_nodes]

    def add_topic_nodes(self, topic_to_accounts: Dict[str, List[str]], 
                       account_to_topics: Dict[str, List[str]],
                       topic_influence: Dict[str, float]) -> None:
        """
        Add topic nodes to the network and create edges between topics and accounts.
        
        Args:
            topic_to_accounts: Dictionary mapping topic names to lists of usernames
            account_to_topics: Dictionary mapping usernames to lists of topics
            topic_influence: Dictionary mapping topic names to influence scores
        """
        logger.info(f"Adding {len(topic_to_accounts)} topic nodes to the network")
        
        # Create a mapping from usernames to node IDs
        username_to_node_id = {}
        for node_id, node in self.nodes.items():
            username = node.get("screen_name", "").lower()
            if username:
                username_to_node_id[username] = node_id
        
        # Select top topics by influence score to avoid overcrowding
        top_topics = sorted(
            [(topic, score) for topic, score in topic_influence.items()],
            key=lambda x: x[1],
            reverse=True
        )[:20]  # Limit to top 20 topics
        
        # Add topic nodes
        for topic_name, influence_score in top_topics:
            topic_id = f"topic_{topic_name.replace(' ', '_').lower()}"
            
            # Create the topic node
            self.nodes[topic_id] = {
                "screen_name": topic_name,
                "name": topic_name,
                "followers_count": 0,
                "friends_count": 0,
                "statuses_count": 0,
                "media_count": 0,
                "created_at": "",
                "location": "",
                "blue_verified": False,
                "verified": False,
                "website": "",
                "business_account": False,
                "description": f"Topic: {topic_name}",
                "ratio": 0,
                "direct": False,
                "tweets": [],
                "tweet_summary": "",
                "is_topic": True,  # Flag to identify topic nodes
                "topic_influence": influence_score,
                "community_color": "#FFA500"  # Default color for topics (orange)
            }
            
            # Create edges from accounts to topics
            usernames = topic_to_accounts.get(topic_name, [])
            for username in usernames:
                node_id = username_to_node_id.get(username.lower())
                if node_id and node_id in self.nodes:
                    # Add edge from account to topic
                    self.add_edge(node_id, topic_id)
        
        logger.info(f"Added {sum(1 for n in self.nodes if n.startswith('topic_'))} topic nodes with connections")