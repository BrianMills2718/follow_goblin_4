"""
Helper utilities for X Network Visualization.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime

import streamlit as st

logger = logging.getLogger(__name__)

class StateManager:
    """
    Manager for Streamlit session state.
    """
    
    # Session state keys
    NETWORK_DATA = "network_data"
    COMMUNITY_LABELS = "community_labels"
    COMMUNITY_COLORS = "community_colors"
    NODE_COMMUNITIES = "node_communities"
    IMPORTANCE_SCORES = "importance_scores"
    CLOUTRANK_SCORES = "full_cloutrank"
    INDEGREE_SCORES = "in_degrees"
    CLOUTRANK_CONTRIBUTIONS = "cloutrank_contributions"
    USE_3D = "use_3d"
    MAX_ACCOUNTS_DISPLAY = "max_accounts_display"
    SHOW_TWEET_SUMMARIES = "show_tweet_summaries"
    ALL_NODES_WITH_TWEETS = "all_nodes_with_tweets"
    IMPORTANCE_METRIC_MODE = "importance_metric_mode"
    NODE_SPACING = "node_spacing"
    TOPIC_TO_ACCOUNTS = "topic_to_accounts"
    ACCOUNT_TO_TOPICS = "account_to_topics"
    DATA_PROCESSING_COMPLETE = "data_processing_complete"
    
    @staticmethod
    def initialize() -> None:
        """Initialize session state variables if they don't exist."""
        if StateManager.NETWORK_DATA not in st.session_state:
            st.session_state[StateManager.NETWORK_DATA] = None
        if StateManager.COMMUNITY_LABELS not in st.session_state:
            st.session_state[StateManager.COMMUNITY_LABELS] = None
        if StateManager.COMMUNITY_COLORS not in st.session_state:
            st.session_state[StateManager.COMMUNITY_COLORS] = None
        if StateManager.NODE_COMMUNITIES not in st.session_state:
            st.session_state[StateManager.NODE_COMMUNITIES] = None
        if StateManager.USE_3D not in st.session_state:
            st.session_state[StateManager.USE_3D] = True
        if StateManager.MAX_ACCOUNTS_DISPLAY not in st.session_state:
            st.session_state[StateManager.MAX_ACCOUNTS_DISPLAY] = 50
        if StateManager.SHOW_TWEET_SUMMARIES not in st.session_state:
            st.session_state[StateManager.SHOW_TWEET_SUMMARIES] = False
        if StateManager.TOPIC_TO_ACCOUNTS not in st.session_state:
            st.session_state[StateManager.TOPIC_TO_ACCOUNTS] = None
        if StateManager.ACCOUNT_TO_TOPICS not in st.session_state:
            st.session_state[StateManager.ACCOUNT_TO_TOPICS] = None
        if StateManager.DATA_PROCESSING_COMPLETE not in st.session_state:
            st.session_state[StateManager.DATA_PROCESSING_COMPLETE] = False
    
    @staticmethod
    def get_network_data() -> Tuple[Dict[str, Dict], List[Tuple[str, str]]] or None:
        """
        Get stored network data.
        
        Returns:
            Tuple of (nodes, edges) if available, None otherwise
        """
        return st.session_state.get(StateManager.NETWORK_DATA)
    
    @staticmethod
    def store_network_data(nodes: Dict[str, Dict], edges: List[Tuple[str, str]]) -> None:
        """
        Store network data in session state.
        
        Args:
            nodes: Dictionary mapping node IDs to node data
            edges: List of (source, target) edge tuples
        """
        st.session_state[StateManager.NETWORK_DATA] = (nodes, edges)
    
    @staticmethod
    def store_importance_scores(scores: Dict[str, float], metric_type: str) -> None:
        """
        Store importance scores.
        
        Args:
            scores: Dictionary mapping node IDs to importance scores
            metric_type: Type of importance metric ("CloutRank" or "In-Degree")
        """
        st.session_state[StateManager.IMPORTANCE_SCORES] = scores
        st.session_state[StateManager.IMPORTANCE_METRIC_MODE] = metric_type
        
        # Also store by specific type
        if metric_type == "CloutRank":
            st.session_state[StateManager.CLOUTRANK_SCORES] = scores
        elif metric_type == "In-Degree":
            st.session_state[StateManager.INDEGREE_SCORES] = scores
    
    @staticmethod
    def store_cloutrank_contributions(incoming: Dict[str, Dict], outgoing: Dict[str, Dict]) -> None:
        """
        Store CloutRank contribution data.
        
        Args:
            incoming: Dictionary mapping node IDs to incoming contribution dictionaries
            outgoing: Dictionary mapping node IDs to outgoing contribution dictionaries
        """
        st.session_state[StateManager.CLOUTRANK_CONTRIBUTIONS] = (incoming, outgoing)
    
    @staticmethod
    def store_community_data(community_labels: Dict[str, str], 
                           community_colors: Dict[str, str], 
                           node_communities: Dict[str, str]) -> None:
        """
        Store community detection data.
        
        Args:
            community_labels: Dictionary mapping community IDs to labels
            community_colors: Dictionary mapping community IDs to colors
            node_communities: Dictionary mapping usernames to community IDs
        """
        st.session_state[StateManager.COMMUNITY_LABELS] = community_labels
        st.session_state[StateManager.COMMUNITY_COLORS] = community_colors
        st.session_state[StateManager.NODE_COMMUNITIES] = node_communities
    
    @staticmethod
    def get_community_data() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]] or Tuple[None, None, None]:
        """
        Get stored community data.
        
        Returns:
            Tuple of (community_labels, community_colors, node_communities) if available, (None, None, None) otherwise
        """
        community_labels = st.session_state.get(StateManager.COMMUNITY_LABELS)
        community_colors = st.session_state.get(StateManager.COMMUNITY_COLORS)
        node_communities = st.session_state.get(StateManager.NODE_COMMUNITIES)
        
        if community_labels and community_colors and node_communities:
            return community_labels, community_colors, node_communities
        else:
            return None, None, None
    
    @staticmethod
    def get_importance_metric() -> str:
        """
        Get current importance metric type.
        
        Returns:
            Importance metric name ("CloutRank" or "In-Degree")
        """
        return st.session_state.get(StateManager.IMPORTANCE_METRIC_MODE, "CloutRank")
    
    @staticmethod
    def update_nodes_with_tweets(nodes_with_tweets: Dict[str, Dict]) -> None:
        """
        Update stored nodes with tweet data.
        
        Args:
            nodes_with_tweets: Dictionary mapping node IDs to node data with tweets
        """
        if StateManager.ALL_NODES_WITH_TWEETS not in st.session_state:
            st.session_state[StateManager.ALL_NODES_WITH_TWEETS] = {}
            
        for node_id, node_data in nodes_with_tweets.items():
            if "tweet_summary" in node_data and node_data["tweet_summary"]:
                if node_id not in st.session_state[StateManager.ALL_NODES_WITH_TWEETS]:
                    st.session_state[StateManager.ALL_NODES_WITH_TWEETS][node_id] = {}
                
                # Copy only the necessary fields
                st.session_state[StateManager.ALL_NODES_WITH_TWEETS][node_id] = node_data.copy()
    
    @staticmethod
    def store_topic_data(topic_to_accounts: Dict[str, List[str]], 
                        account_to_topics: Dict[str, List[str]]) -> None:
        """
        Store topic data in session state.
        
        Args:
            topic_to_accounts: Dictionary mapping topic names to lists of usernames
            account_to_topics: Dictionary mapping usernames to lists of topics
        """
        st.session_state[StateManager.TOPIC_TO_ACCOUNTS] = topic_to_accounts
        st.session_state[StateManager.ACCOUNT_TO_TOPICS] = account_to_topics
    
    @staticmethod
    def get_topic_data() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]] or Tuple[None, None]:
        """
        Get stored topic data.
        
        Returns:
            Tuple of (topic_to_accounts, account_to_topics) if available, (None, None) otherwise
        """
        topic_to_accounts = st.session_state.get(StateManager.TOPIC_TO_ACCOUNTS)
        account_to_topics = st.session_state.get(StateManager.ACCOUNT_TO_TOPICS)
        
        if topic_to_accounts and account_to_topics:
            return topic_to_accounts, account_to_topics
        else:
            return None, None

def format_date_for_display(date_str: str) -> str:
    """
    Format a date string for display.
    
    Args:
        date_str: Date string in Twitter format
        
    Returns:
        Formatted date string
    """
    try:
        # Twitter date format: "Wed Apr 07 22:52:51 +0000 2021"
        date_obj = datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
        return date_obj.strftime("%Y-%m-%d")
    except:
        return date_str

def get_selected_communities(community_labels: Dict[str, str]) -> Dict[str, bool]:
    """
    Get user selection state for each community.
    
    Args:
        community_labels: Dictionary mapping community IDs to labels
        
    Returns:
        Dictionary mapping community IDs to selection state
    """
    selected_communities = {}
    
    # Count accounts in each community
    community_counts = {}
    if StateManager.NODE_COMMUNITIES in st.session_state:
        for username, comm_id in st.session_state[StateManager.NODE_COMMUNITIES].items():
            if comm_id not in community_counts:
                community_counts[comm_id] = 0
            community_counts[comm_id] += 1
    
    # Create checkboxes for each community
    for comm_id, label in community_labels.items():
        count = community_counts.get(comm_id, 0)
        comm_label = f"{label} ({count} accounts)"
        selected_communities[comm_id] = st.sidebar.checkbox(
            comm_label,
            value=True,
            key=f"community_{comm_id}"
        )
    
    return selected_communities