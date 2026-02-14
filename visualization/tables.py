"""
Table visualization utilities for X Network Visualization.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any

import streamlit as st
import pandas as pd

from data.network import NetworkData

logger = logging.getLogger(__name__)

class TableVisualizer:
    """
    Visualizer for data tables.
    """
    
    @staticmethod
    def format_text_with_line_breaks(text: str, max_line_length: int = 80) -> str:
        """
        Format text with line breaks at word boundaries for better readability in tables.
        
        Args:
            text: Text to format
            max_line_length: Maximum length for each line
            
        Returns:
            Formatted text with line breaks
        """
        if not text or len(text) <= max_line_length:
            return text
            
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) > max_line_length:
                lines.append(current_line)
                current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return "\n".join(lines)
    
    @staticmethod
    def apply_table_styles():
        """
        Apply custom CSS styling to make tables more readable, particularly in dark mode.
        """
        # Apply custom CSS to make tables more readable, especially in dark mode
        table_styles = """
        <style>
            /* Container styling for horizontal scrolling */
            .stTable {
                width: 100% !important;
                overflow-x: auto !important;
            }
            
            /* Base table styles */
            .stTable table {
                min-width: 1000px;
                padding: 1em;
                font-size: 14px;
            }
            
            /* Header styling */
            .stTable thead tr {
                background-color: rgba(108, 166, 205, 0.3);
            }
            
            .stTable thead th {
                padding: 12px !important;
                font-weight: bold !important;
                color: white !important;
                text-align: left !important;
                position: sticky !important;
                top: 0 !important;
                z-index: 1 !important;
                background-color: rgb(38, 39, 48) !important; /* Solid background color for headers */
                border-bottom: 2px solid rgba(108, 166, 205, 0.7) !important;
            }
            
            /* Row styling */
            .stTable tbody tr {
                border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
            }
            
            .stTable tbody tr:nth-child(even) {
                background-color: rgba(255, 255, 255, 0.05) !important;
            }
            
            .stTable tbody tr:hover {
                background-color: rgba(108, 166, 205, 0.2) !important;
            }
            
            /* Cell styling */
            .stTable tbody td {
                padding: 10px !important;
                text-align: left !important;
                white-space: pre-line !important; /* Allow line breaks within cells */
                word-wrap: break-word !important;
                max-width: 300px !important;
                line-height: 1.4 !important;
            }
            
            /* Description column - allow multiline text */
            .stTable tbody td:nth-child(8) {
                white-space: pre-line !important;
                max-height: 150px !important;
                overflow-y: auto !important;
            }
            
            /* Tweet summary column - allow multiline text */
            .stTable tbody td:nth-child(9) {
                white-space: pre-line !important;
                max-height: 150px !important;
                overflow-y: auto !important;
            }
            
            /* Username column styling */
            .stTable tbody td:nth-child(2) {
                font-weight: bold !important;
                white-space: nowrap !important;
            }
            
            /* Connection column styling - color based on value */
            .connection-original {
                color: #ff9d00 !important;
                font-weight: bold !important;
            }
            
            .connection-first {
                color: #00c3ff !important;
                font-weight: bold !important;
            }
            
            .connection-second {
                color: #8bc34a !important;
                font-weight: bold !important;
            }
        </style>
        """
        st.markdown(table_styles, unsafe_allow_html=True)
        
        # Add JavaScript to color the connection cells based on their content
        js = """
        <script>
            // Function to apply styling to connection cells
            function styleConnectionCells() {
                // Get all tables
                const tables = document.querySelectorAll('.stTable table');
                if (!tables.length) return;
                
                tables.forEach(table => {
                    // Find Connection column index
                    let connectionIdx = -1;
                    const headers = table.querySelectorAll('thead th');
                    
                    headers.forEach((header, idx) => {
                        if (header.textContent.includes('Connection')) {
                            connectionIdx = idx;
                        }
                    });
                    
                    if (connectionIdx === -1) return;
                    
                    // Style each connection cell in the table body
                    const rows = table.querySelectorAll('tbody tr');
                    rows.forEach(row => {
                        const cell = row.cells[connectionIdx];
                        if (!cell) return;
                        
                        const text = cell.textContent.trim();
                        
                        if (text.includes('Original')) {
                            cell.classList.add('connection-original');
                        } else if (text.includes('1st Degree')) {
                            cell.classList.add('connection-first');
                        } else if (text.includes('2nd Degree')) {
                            cell.classList.add('connection-second');
                        }
                    });
                });
            }
            
            // Run initially and observe for changes
            document.addEventListener("DOMContentLoaded", function() {
                styleConnectionCells();
                
                // Set up a MutationObserver to watch for changes
                const observer = new MutationObserver(function(mutations) {
                    styleConnectionCells();
                });
                
                // Start observing the document body for DOM changes
                observer.observe(document.body, { childList: true, subtree: true });
            });
        </script>
        """
        st.markdown(js, unsafe_allow_html=True)
    
    @staticmethod
    def display_top_accounts_table(nodes: Dict[str, Dict],
                                  importance_scores: Dict[str, float],
                                  cloutrank_scores: Dict[str, float],
                                  in_degrees: Dict[str, int],
                                  top_accounts: List[Tuple[str, float, Dict]],
                                  original_id: str,
                                  show_tweet_summaries: bool = False,
                                  importance_metric: str = "CloutRank") -> None:
        """
        Display table of top accounts based on importance scores.
        
        Args:
            nodes: Dictionary mapping node IDs to node data
            importance_scores: Dictionary mapping node IDs to importance scores
            cloutrank_scores: Dictionary mapping node IDs to CloutRank scores
            in_degrees: Dictionary mapping node IDs to in-degree values
            top_accounts: List of (node_id, score, node_data) tuples
            original_id: ID of the original node
            show_tweet_summaries: Whether to show tweet summaries
            importance_metric: Name of the importance metric
        """
        st.subheader(f"Top Accounts by {importance_metric}")
        
        # Apply table styles
        TableVisualizer.apply_table_styles()
        
        # Create table data with both metrics always included
        table_data = {
            "Rank": [],
            "Username": [],
            "Connection": [],  # New column for connection degree
            "CloutRank": [],        # Always include CloutRank
            "In-Degree": [],        # Always include In-Degree
            "Followers": [],
            "Following": [],
            "Description": []
        }

        # Contribution data (incoming/outgoing)
        table_data["Contributors"] = []            # List accounts that give clout and amount
        table_data["Total In"] = []                # Total clout received
        table_data["Distributions"] = []           # List accounts this node gives clout to
        table_data["Total Out"] = []               # Total clout distributed
        
        # Only add community column if communities exist
        communities_exist = ('node_communities' in st.session_state and st.session_state.node_communities)
        if communities_exist:
            table_data["Community"] = []
            communities = st.session_state.node_communities
            community_labels = st.session_state.community_labels
        
        # Add tweet summary column if requested
        if show_tweet_summaries:
            table_data["Tweet Summary"] = []
        
        # Get original username to ensure proper identification
        original_username = None
        if original_id in nodes:
            original_username = nodes[original_id].get("screen_name", "").lower()
        
        # Get first and second degree nodes
        network = NetworkData()
        network.nodes = nodes
        network.edges = []  # Not needed for this display
        network.original_id = original_id
        
        first_degree_nodes = network.get_first_degree_nodes()
        second_degree_nodes = network.get_second_degree_nodes()
        
        # Create expander for top accounts table
        with st.expander(f"Top {len(top_accounts)} Accounts by {importance_metric}", expanded=False):
            # Populate table
            for idx, (node_id, score, node) in enumerate(top_accounts, 1):
                table_data["Rank"].append(idx)
                
                # Get username for this node
                username = node.get("screen_name", "")
                table_data["Username"].append(f"@{username}")
                
                # Determine connection degree
                # Check both node_id and username to identify original account
                is_original = (
                    node_id == original_id or 
                    (original_username and username.lower() == original_username.lower()) or
                    node_id.startswith("orig_")
                )
                
                if is_original:
                    table_data["Connection"].append("Original")
                    logger.info(f"Identified original account: {username} (ID: {node_id})")
                elif node_id in first_degree_nodes:
                    table_data["Connection"].append("1st Degree")
                elif node_id in second_degree_nodes:
                    table_data["Connection"].append("2nd Degree")
                else:
                    table_data["Connection"].append("Other")
                
                # Add community data if communities exist
                if communities_exist:
                    community_id = communities.get(node["screen_name"], "0")
                    community_label = community_labels.get(community_id, f"Community {community_id}")
                    table_data["Community"].append(community_label)
                
                # Always add both metrics separately
                cr_value = cloutrank_scores.get(node_id, 0)
                table_data["CloutRank"].append(f"{cr_value:.4f}")
                
                id_value = in_degrees.get(node_id, 0)
                table_data["In-Degree"].append(str(id_value))
                
                # Add other account data
                table_data["Followers"].append(f"{node.get('followers_count', 0):,}")
                table_data["Following"].append(f"{node.get('friends_count', 0):,}")
                
                # Format description for better readability
                description = node.get("description", "")
                if len(description) > 100:
                    formatted_description = TableVisualizer.format_text_with_line_breaks(description)
                    table_data["Description"].append(formatted_description)
                else:
                    table_data["Description"].append(description)
                
                # Add tweet summary if requested
                if show_tweet_summaries:
                    tweet_summary = node.get("tweet_summary", "")
                    if tweet_summary and len(tweet_summary) > 0:
                        # Format tweet summary with line breaks
                        if len(tweet_summary) > 150:
                            formatted_summary = TableVisualizer.format_text_with_line_breaks(tweet_summary)
                            table_data["Tweet Summary"].append(formatted_summary)
                        else:
                            table_data["Tweet Summary"].append(tweet_summary)
                    else:
                        table_data["Tweet Summary"].append("No tweet summary available")

                # ----- Contribution columns -----
                incoming_outgoing = st.session_state.get("cloutrank_contributions")
                if incoming_outgoing:
                    incoming_contribs, outgoing_contribs = incoming_outgoing
                else:
                    incoming_contribs, outgoing_contribs = ({}, {})

                # Incoming contributions for this node
                node_incoming = incoming_contribs.get(node_id, {})
                # Sort by amount desc
                sorted_incoming = sorted(node_incoming.items(), key=lambda x: x[1], reverse=True)
                contributors_list = []
                total_in = 0.0
                for src_id, amt in sorted_incoming:
                    if amt <= 0:
                        continue
                    total_in += amt
                    name = nodes.get(src_id, {}).get("screen_name", src_id)
                    contributors_list.append(f"@{name}: {amt:.4f}")
                table_data["Contributors"].append("; ".join(contributors_list) if contributors_list else "")
                table_data["Total In"].append(f"{total_in:.4f}")

                # Outgoing contributions from this node
                node_outgoing = outgoing_contribs.get(node_id, {})
                sorted_outgoing = sorted(node_outgoing.items(), key=lambda x: x[1], reverse=True)
                distributions_list = []
                total_out = 0.0
                for tgt_id, amt in sorted_outgoing:
                    if amt <= 0:
                        continue
                    total_out += amt
                    name = nodes.get(tgt_id, {}).get("screen_name", tgt_id)
                    distributions_list.append(f"@{name}: {amt:.4f}")
                table_data["Distributions"].append("; ".join(distributions_list) if distributions_list else "")
                table_data["Total Out"].append(f"{total_out:.4f}")

            # Convert to DataFrame for better styling
            df = pd.DataFrame(table_data)
            
            # Use st.table for display
            st.table(df)
    
    @staticmethod
    def display_community_tables(network_nodes: Dict[str, Dict],
                               edges: List[Tuple[str, str]],
                               top_accounts_by_community: Dict[str, List],
                               community_colors: Dict[str, str],
                               community_labels: Dict[str, str],
                               cloutrank_scores: Dict[str, float],
                               in_degrees: Dict[str, int],
                               show_tweet_summaries: bool = False) -> None:
        """
        Display tables of top accounts for each community.
        
        Args:
            network_nodes: Dictionary mapping node IDs to node data
            edges: List of (source, target) edge tuples
            top_accounts_by_community: Dictionary mapping community IDs to lists of (node_id, node_data) tuples
            community_colors: Dictionary mapping community IDs to color strings
            community_labels: Dictionary mapping community IDs to label strings
            cloutrank_scores: Dictionary mapping node IDs to CloutRank scores
            in_degrees: Dictionary mapping node IDs to in-degree values
            show_tweet_summaries: Whether to show tweet summaries
        """
        st.header("Community Analysis")
        
        # Apply table styles
        TableVisualizer.apply_table_styles()
        
        # Get original node ID
        original_id = next((id for id in network_nodes.keys() if id.startswith("orig_")), None)
        
        # Get original username to ensure proper identification
        original_username = None
        if original_id in network_nodes:
            original_username = network_nodes[original_id].get("screen_name", "").lower()
        
        # Get first and second degree nodes
        network = NetworkData()
        network.nodes = network_nodes
        network.edges = edges
        network.original_id = original_id
        
        first_degree_nodes = network.get_first_degree_nodes()
        second_degree_nodes = network.get_second_degree_nodes()
        
        # Sort communities by size (number of accounts)
        sorted_communities = sorted(
            top_accounts_by_community.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        # Display a table for each community
        for community_id, accounts in sorted_communities:
            # Get community label and color
            label = community_labels.get(community_id, f"Community {community_id}")
            color = community_colors.get(community_id, "#6ca6cd")
            
            # Create expander for community
            with st.expander(f"{label} ({len(accounts)} accounts)", expanded=False):
                st.markdown(f"<div style='width:100%; height:3px; background-color:{color}'></div>", 
                            unsafe_allow_html=True)
                
                # Create table data
                table_data = {
                    "Username": [],
                    "Connection": [],  # New connection column
                    "CloutRank": [],
                    "In-Degree": [],
                    "Followers": [],
                    "Following": [],
                    "Description": []
                }
                
                # Add tweet summary column if requested
                if show_tweet_summaries:
                    table_data["Tweet Summary"] = []
                
                # Populate table
                for node_id, node_data, score in accounts:
                    # Get username for this node
                    username = node_data.get("screen_name", "")
                    table_data["Username"].append(f"@{username}")
                    
                    # Determine connection degree
                    # Check both node_id and username to identify original account
                    is_original = (
                        node_id == original_id or 
                        (original_username and username.lower() == original_username.lower()) or
                        node_id.startswith("orig_")
                    )
                    
                    if is_original:
                        table_data["Connection"].append("Original")
                    elif node_id in first_degree_nodes:
                        table_data["Connection"].append("1st Degree")
                    elif node_id in second_degree_nodes:
                        table_data["Connection"].append("2nd Degree")
                    else:
                        table_data["Connection"].append("Other")
                    
                    # Add importance metrics
                    cr_value = cloutrank_scores.get(node_id, 0)
                    table_data["CloutRank"].append(f"{cr_value:.4f}")
                    
                    id_value = in_degrees.get(node_id, 0)
                    table_data["In-Degree"].append(str(id_value))
                    
                    # Add other account data
                    table_data["Followers"].append(f"{node_data.get('followers_count', 0):,}")
                    table_data["Following"].append(f"{node_data.get('friends_count', 0):,}")
                    
                    # Format description for better readability
                    description = node_data.get("description", "")
                    if len(description) > 100:
                        formatted_description = TableVisualizer.format_text_with_line_breaks(description)
                        table_data["Description"].append(formatted_description)
                    else:
                        table_data["Description"].append(description)
                    
                    # Add tweet summary if requested
                    if show_tweet_summaries:
                        tweet_summary = node_data.get("tweet_summary", "")
                        if tweet_summary and len(tweet_summary) > 0:
                            # Format tweet summary with line breaks
                            if len(tweet_summary) > 150:
                                formatted_summary = TableVisualizer.format_text_with_line_breaks(tweet_summary)
                                table_data["Tweet Summary"].append(formatted_summary)
                            else:
                                table_data["Tweet Summary"].append(tweet_summary)
                        else:
                            table_data["Tweet Summary"].append("No tweet summary available")
                
                # Convert to DataFrame for better styling
                df = pd.DataFrame(table_data)
                
                # Display table with custom formatting
                st.table(df)
    
    @staticmethod
    def display_community_color_key(community_labels: Dict[str, str],
                                   community_colors: Dict[str, str],
                                   node_communities: Dict[str, str]) -> None:
        """
        Display a color key for communities.
        
        Args:
            community_labels: Dictionary mapping community IDs to labels
            community_colors: Dictionary mapping community IDs to colors
            node_communities: Dictionary mapping usernames to community IDs
        """
        st.subheader("Community Color Key")
        
        # Count accounts in each community
        community_counts = {}
        for username, comm_id in node_communities.items():
            if comm_id not in community_counts:
                community_counts[comm_id] = 0
            community_counts[comm_id] += 1
        
        # Convert community data to sorted list by labels with account counts
        community_data = []
        for comm_id, color in community_colors.items():
            label = community_labels.get(comm_id, f"Community {comm_id}")
            count = community_counts.get(comm_id, 0)
            label_with_count = f"{label} ({count} accounts)"
            community_data.append((label_with_count, color, comm_id))
        
        # Sort alphabetically by label
        community_data.sort(key=lambda x: x[0])
        
        # Calculate number of columns based on total communities
        num_communities = len(community_data)
        num_cols = min(4, max(2, 5 - (num_communities // 15)))
        
        # Create a container with fixed height and scrolling
        st.markdown("""
        <style>
        .community-grid {
            max-height: 400px;
            overflow-y: auto;
            padding-right: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create a grid using Streamlit columns
        with st.container():
            st.markdown('<div class="community-grid">', unsafe_allow_html=True)
            
            # Create rows of communities instead of columns
            for i in range(0, num_communities, num_cols):
                # Create a row of columns
                cols = st.columns(num_cols)
                
                # Add communities to this row
                for j in range(num_cols):
                    idx = i + j
                    if idx < num_communities:
                        label, color, _ = community_data[idx]
                        with cols[j]:
                            # Use a simple layout with colored text
                            st.markdown(
                                f'<div style="display:flex; align-items:center">'
                                f'<div style="width:15px; height:15px; background-color:{color}; '
                                f'border-radius:3px; margin-right:8px;"></div>'
                                f'<span style="font-size:0.9em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{label}</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def display_topics_table(
        network_nodes: Dict[str, Dict],
        topic_to_accounts: Dict[str, List[str]],
        account_to_topics: Dict[str, List[str]],
        cloutrank_scores: Dict[str, float]
    ) -> None:
        """
        Display a table of topics and the accounts that discuss them.
        
        Args:
            network_nodes: Dictionary mapping node IDs to node data
            topic_to_accounts: Dictionary mapping topic names to lists of usernames
            account_to_topics: Dictionary mapping usernames to lists of topics
            cloutrank_scores: Dictionary mapping node IDs to CloutRank scores
        """
        if not topic_to_accounts or not account_to_topics:
            logger.warning("No topic data available for displaying topic table")
            st.info("🔍 No topic data available. Use the 'Summarize Tweets & Generate Communities' button to generate topics.")
            return
        
        logger.info(f"Generating topic table with {len(topic_to_accounts)} topics across {len(account_to_topics)} accounts")
        
        try:
            # Apply table styles first
            TableVisualizer.apply_table_styles()
            
            # Add Topics label
            st.header("Topics")
            
            # Create expander for topics table
            with st.expander(f"🔍 Topics from Twitter Accounts ({len(topic_to_accounts)} topics found)", expanded=False):
                st.write("Topics extracted from tweet content and account descriptions, showing connections between accounts and topics they discuss.")
                
                # Get usernames to node IDs mapping for looking up CloutRank scores
                username_to_node_id = {}
                for node_id, node in network_nodes.items():
                    username = node.get("screen_name", "").lower()
                    if username:
                        username_to_node_id[username] = node_id
                
                logger.info(f"Found {len(username_to_node_id)} username to node ID mappings for topic influence calculation")
                
                # Sort topics by influence (combined CloutRank of accounts discussing each topic)
                topic_influence = {}
                for topic, usernames in topic_to_accounts.items():
                    topic_score = 0
                    matched_accounts = 0
                    for username in usernames:
                        # Look up node ID for this username to get CloutRank
                        node_id = username_to_node_id.get(username.lower())
                        if node_id and node_id in cloutrank_scores:
                            topic_score += cloutrank_scores[node_id]
                            matched_accounts += 1
                    topic_influence[topic] = topic_score
                    logger.debug(f"Topic '{topic}': score {topic_score:.4f}, matched {matched_accounts}/{len(usernames)} accounts")
                
                # Get top 20 topics by influence
                top_topics = sorted(
                    [(topic, score) for topic, score in topic_influence.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:20]
                
                logger.info(f"Selected top {len(top_topics)} topics by influence score")
                
                # Prepare table data
                table_data = {
                    "Topic": [],
                    "Accounts": [],
                    "Key Statements": []
                }
                
                # Process each topic
                if not top_topics:
                    st.warning("No topics with influence score were found. This may happen if account usernames don't match between tables.")
                    return
                    
                for topic_name, influence_score in top_topics:
                    usernames = topic_to_accounts.get(topic_name, [])
                    logger.debug(f"Processing topic '{topic_name}' with {len(usernames)} accounts, influence score: {influence_score:.4f}")
                    
                    # Add the topic
                    table_data["Topic"].append(topic_name)
                    
                    # Format accounts with their CloutRank - show all accounts
                    accounts_list = []
                    for username in usernames:  # No limit on number of accounts
                        accounts_list.append(f"@{username}")
                    
                    accounts_text = ", ".join(accounts_list)  # Use commas instead of newlines
                    table_data["Accounts"].append(accounts_text)
                    
                    # Extract key statements about this topic
                    key_statements = []
                    statements_found = 0
                    for username in usernames[:3]:  # Keep top 3 for key statements to avoid too much text
                        # Find node data for this username
                        node_id = username_to_node_id.get(username.lower())
                        if node_id and node_id in network_nodes:
                            node = network_nodes[node_id]
                            tweet_summary = node.get("tweet_summary", "")
                            
                            # If there's a tweet summary, extract a relevant snippet
                            if tweet_summary:
                                # Check if the topic is mentioned in the tweet summary
                                if topic_name.lower() in tweet_summary.lower():
                                    # Extract a snippet containing the topic
                                    sentences = tweet_summary.split(". ")
                                    relevant_sentences = [s for s in sentences if topic_name.lower() in s.lower()]
                                    if relevant_sentences:
                                        key_statements.append(f"@{username}: {relevant_sentences[0]}")
                                        statements_found += 1
                                    else:
                                        # If no direct mention, just use the first sentence
                                        key_statements.append(f"@{username}: {sentences[0]}")
                                else:
                                    # Use the first sentence if topic not explicitly mentioned
                                    first_sentence = tweet_summary.split(". ")[0]
                                    key_statements.append(f"@{username}: {first_sentence}")
                    
                    logger.debug(f"Found {statements_found} relevant statements for topic '{topic_name}'")
                    
                    # Format key statements
                    statements_text = "\n\n".join(key_statements)
                    formatted_statements = TableVisualizer.format_text_with_line_breaks(statements_text, max_line_length=100)
                    table_data["Key Statements"].append(formatted_statements)
                
                # Convert to DataFrame
                df = pd.DataFrame(table_data)
                
                # Display table
                logger.info("Displaying topic table with data rows: " + str(len(df)))
                st.table(df)
                
        except Exception as e:
            error_msg = f"Error generating topic table: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)