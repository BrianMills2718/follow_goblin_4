"""
Main application for X Network Visualization.
"""
import asyncio
import logging
import io
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime
import time
import platform

import streamlit as st
import pandas as pd

from api.twitter_client import TwitterClient
from api.ai_client import AIClient
from data.network import NetworkData
from data.analysis import compute_in_degree, compute_cloutrank, select_top_nodes_for_visualization
from data.processing import DataProcessor
from data.communities import CommunityManager
from visualization.network_3d import Network3DVisualizer
from visualization.tables import TableVisualizer
from utils.helpers import StateManager, format_date_for_display, get_selected_communities
from config import (
    RAPIDAPI_KEY, RAPIDAPI_HOST, GEMINI_API_KEY,
    DEFAULT_MAX_ACCOUNTS, DEFAULT_ACCOUNT_SIZE_FACTOR, DEFAULT_LABEL_SIZE_FACTOR,
    DEFAULT_FILTERS, DEFAULT_NUM_COMMUNITIES, MIN_ACCOUNTS_FOR_COMMUNITIES,
    DEFAULT_NODE_SPACING, DEFAULT_BASE_SIZE
)

# Set up logging with a string IO handler to capture logs for display
log_stream = io.StringIO()
stream_handler = logging.StreamHandler(log_stream)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)

# Set up default logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Add the stream handler
logger.addHandler(stream_handler)

# Add stream handler to other loggers
for module in ['api.ai_client', 'data.processing', 'data.communities', 'visualization.tables']:
    mod_logger = logging.getLogger(module)
    mod_logger.addHandler(stream_handler)

# Fix asyncio on Windows - only apply on Windows platform
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Set page config - must be the first Streamlit command
st.set_page_config(
    layout="wide",
    page_title="X Network Analysis",
    page_icon="🔍",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/followgoblin',
        'Report a bug': 'https://github.com/your-repo/followgoblin/issues',
        'About': 'X Network Visualization Tool',
    }
)

def main():
    """Main application."""
    # Initialize session state
    StateManager.initialize()
    
    # Force dark theme
    st.markdown("""
        <style>
        :root {
            --background-color: #0e1117;
            --secondary-background-color: #262730;
            --text-color: #fafafa;
            --font: "Source Sans Pro", sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Page title and description
    st.title("X Account Following Network Visualization")
    st.markdown("Enter an X (formerly Twitter) username to retrieve its following network.")

    # Ensure event loop is created for this thread
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Initialize API clients
    twitter_client = TwitterClient(RAPIDAPI_KEY, RAPIDAPI_HOST)
    ai_client = AIClient(GEMINI_API_KEY)
    
    # Initialize processors
    data_processor = DataProcessor(twitter_client, ai_client)
    community_manager = CommunityManager(ai_client)
    
    # Input for username
    input_username = st.text_input("X Username (without @):", value="elonmusk")
    
    # Sidebar: Display Options
    st.sidebar.header("Display Options")
    
    # Dropdown menu for importance metric
    importance_metric = st.sidebar.selectbox(
        "Importance Metric", 
        options=["In-Degree", "CloutRank"],
        index=1,
        help="In-Degree measures importance by how many accounts follow this account in the network. "
             "CloutRank considers both quantity and quality of connections."
    )
    use_pagerank = (importance_metric == "CloutRank")
    
    # Account size factor
    account_size_factor = st.sidebar.number_input(
        "Account Size Factor",
        min_value=0.1,
        max_value=10.0,
        value=DEFAULT_ACCOUNT_SIZE_FACTOR,
        step=0.1,
        format="%.1f",
        help="Controls how much account importance affects node size in the visualization. "
             "Higher values make important accounts appear larger."
    )
    
    # Node spacing factor - controls the space between nodes
    node_spacing = st.sidebar.number_input(
        "Node Spacing",
        min_value=1.0,
        max_value=20.0,
        value=st.session_state.get(StateManager.NODE_SPACING, DEFAULT_NODE_SPACING),
        step=1.0,
        format="%.1f",
        help="Controls the space between nodes in the visualization. "
             "Higher values create more distance between nodes."
    )
    
    # Store in session state for persistence
    st.session_state[StateManager.NODE_SPACING] = node_spacing
    
    # Label size control - only shown if using 3D visualization
    label_size = DEFAULT_LABEL_SIZE_FACTOR
    if StateManager.get_network_data() is not None:
        use_3d = st.checkbox(
            "Use 3D Visualization", 
            value=st.session_state.get(StateManager.USE_3D, True),
            help="Toggle for 3D network visualization. 3D offers more interactive features."
        )
        st.session_state[StateManager.USE_3D] = use_3d
        
        if use_3d:
            label_size = st.sidebar.number_input(
                "Label Size",
                min_value=0.1,
                max_value=5.0,
                value=DEFAULT_LABEL_SIZE_FACTOR,
                step=0.1,
                format="%.1f",
                help="Controls the size of the account name labels in the 3D visualization."
            )
    
    # Number input for max accounts
    max_accounts_display = st.sidebar.number_input(
        "Max Accounts to Display",
        min_value=5,
        max_value=1000,
        value=st.session_state.get(StateManager.MAX_ACCOUNTS_DISPLAY, DEFAULT_MAX_ACCOUNTS),
        step=5,
        help="Maximum number of accounts to show in the visualization. Lower values improve performance."
    )
    
    # Store in session state for persistence
    st.session_state[StateManager.MAX_ACCOUNTS_DISPLAY] = max_accounts_display

    # Add a button to update the visualization
    update_visualization = st.sidebar.button(
        "Update Visualization", 
        help="Force update the visualization with current settings",
        key="update_viz_button"
    )
    
    # Sidebar: Filter Criteria
    st.sidebar.header("Filter Criteria")
    
    # Numeric ranges with separate min/max inputs
    st.sidebar.subheader("Numeric Ranges")
    
    # Total Tweets Range
    st.sidebar.markdown("**Total Tweets Range**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        tweets_min = st.number_input(
            "Min Tweets",
            min_value=0,
            max_value=1000000,
            value=0,
            step=1000,
            help="Minimum number of tweets an account must have"
        )
    with col2:
        tweets_max = st.number_input(
            "Max Tweets",
            min_value=0,
            max_value=1000000,
            value=1000000,
            step=1000,
            help="Maximum number of tweets an account can have"
        )
    total_tweets_range = (tweets_min, tweets_max)
    
    # Followers Range
    st.sidebar.markdown("**Followers Range**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        followers_min = st.number_input(
            "Min Followers",
            min_value=0,
            max_value=10000000,
            value=0,
            step=1000,
            help="Minimum number of followers an account must have"
        )
    with col2:
        followers_max = st.number_input(
            "Max Followers",
            min_value=0,
            max_value=10000000,
            value=10000000,
            step=1000,
            help="Maximum number of followers an account can have"
        )
    followers_range = (followers_min, followers_max)
    
    # Following Range
    st.sidebar.markdown("**Following Range**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        following_min = st.number_input(
            "Min Following",
            min_value=0,
            max_value=10000000,
            value=0,
            step=1000,
            help="Minimum number of accounts this account must follow"
        )
    with col2:
        following_max = st.number_input(
            "Max Following",
            min_value=0,
            max_value=10000000,
            value=10000000,
            step=1000,
            help="Maximum number of accounts this account can follow"
        )
    following_range = (following_min, following_max)
    
    # Create filters dictionary
    filters = {
        "statuses_range": total_tweets_range,
        "followers_range": followers_range,
        "friends_range": following_range,
        "media_range": (0, 10000),  # Default value
        "created_range": (datetime(2000, 1, 1).date(), datetime(2100, 1, 1).date()),
        "require_location": False,
        "selected_locations": [],
        "require_blue_verified": False,
        "verified_option": "Any",
        "require_website": False,
        "business_account_option": "Any"
    }
    
    # Data Fetch Options
    st.sidebar.header("Data Fetch Options")

    following_pages = st.sidebar.number_input(
        "Pages of Following for Original Account",
        min_value=1,
        max_value=10,
        value=2,
        help="How many pages of following accounts to fetch for the original account (20 accounts per page)"
    )

    second_degree_pages = st.sidebar.number_input(
        "Pages of Following for Second Degree",
        min_value=1,
        max_value=5,
        value=1,
        help="How many pages of following accounts to fetch for each first-degree connection"
    )
    
    # Helper function to run async code without creating new event loops
    async def run_async_network_collection(username, following_pages, second_degree_pages):
        return await data_processor.collect_network_data(
            username, 
            following_pages=following_pages,
            second_degree_pages=second_degree_pages
        )
    
    async def run_async_tweet_processing(network, selected_nodes):
        return await data_processor.process_tweet_data(
            network,
            selected_nodes
        )
    
    async def run_async_community_label_generation(nodes_list, num_communities):
        return await community_manager.generate_community_labels(
            nodes_list,
            num_communities=num_communities
        )
    
    async def run_async_account_classification(nodes_list):
        return await community_manager.classify_accounts(
            nodes_list
    )
    
    # Generate Network button
    if st.button("Generate Network"):
        with st.spinner("Collecting network data..."):
            # Get current loop and run collection
            loop = asyncio.get_event_loop()
            network = loop.run_until_complete(run_async_network_collection(
                input_username, 
                following_pages,
                second_degree_pages
            ))
            
            # Debug information
            st.write(f"Retrieved {len(network.nodes)} nodes and {len(network.edges)} edges from API.")
            
            # Compute importance scores
            with st.spinner("Computing network influence scores..."):
                # Calculate in-degrees
                in_degrees = compute_in_degree(network.nodes, network.edges)
                
                # Calculate CloutRank with contributions
                cloutrank_scores, incoming_contribs, outgoing_contribs = compute_cloutrank(
                    network.nodes, 
                    network.edges, 
                    return_contributors=True
                )
                
                # Store results in session state
                StateManager.store_network_data(network.nodes, network.edges)
                StateManager.store_importance_scores(
                    cloutrank_scores if use_pagerank else in_degrees,
                    "CloutRank" if use_pagerank else "In-Degree"
                )
                StateManager.store_cloutrank_contributions(incoming_contribs, outgoing_contribs)
                
                # Store separately for reference
                st.session_state["in_degrees"] = in_degrees
                st.session_state["full_cloutrank"] = cloutrank_scores
                
                st.success("Network data collection and analysis complete!")
    
    # Process existing network data if available
    if StateManager.get_network_data() is not None:
        nodes, edges = StateManager.get_network_data()
        
        # Create NetworkData instance from stored data
        network = NetworkData()
        network.nodes = nodes
        network.edges = edges
        network.original_id = next((id for id in nodes.keys() if id.startswith("orig_")), None)
        network.original_username = nodes[network.original_id]["screen_name"] if network.original_id else None
        
        # Apply filters
        filtered_nodes = network.filter_nodes(filters)
        
        # Add degree filtering
        st.sidebar.subheader("Node Degree Filtering")
        show_original = st.sidebar.checkbox(
            "Show Original Node", 
            value=True,
            help="Include the original account (the one you entered) in the visualization."
        )
        show_first_degree = st.sidebar.checkbox(
            "Show First Degree Connections", 
            value=True,
            help="Include accounts that are directly followed by the original account."
        )
        show_second_degree = st.sidebar.checkbox(
            "Show Second Degree Connections", 
            value=True,
            help="Include accounts that are followed by the first-degree connections."
        )
        
        # Filter by degree
        degree_filtered_nodes = network.filter_by_degree(
            filtered_nodes,
            show_original=show_original,
            show_first_degree=show_first_degree,
            show_second_degree=show_second_degree
        )
        
        # Get community data if available
        community_labels, community_colors, node_communities = StateManager.get_community_data()
        
        # Add community filtering if available
        community_filtered_nodes = degree_filtered_nodes
        if community_labels and community_colors and node_communities:
            st.sidebar.subheader("Community Filtering")
            
            # Get community selection state
            selected_communities = get_selected_communities(community_labels)
            
            # Filter by communities
            community_filtered_nodes = network.filter_by_communities(
                degree_filtered_nodes,
                node_communities,
                selected_communities
            )
            
            # Update node colors based on communities
            network.update_node_community_colors(community_colors, node_communities)
        
        # Calculate importance scores
        importance_metric_name = importance_metric  # Store the name for display
        
        # Use appropriate importance metric
        if use_pagerank:
            if "full_cloutrank" in st.session_state:
                importance_scores = st.session_state["full_cloutrank"]
            else:
                importance_scores = compute_cloutrank(network.nodes, network.edges)
                st.session_state["full_cloutrank"] = importance_scores
        else:
            if "in_degrees" in st.session_state:
                importance_scores = st.session_state["in_degrees"]
            else:
                importance_scores = compute_in_degree(network.nodes, network.edges)
                st.session_state["in_degrees"] = importance_scores
        
        # Store current importance metric
        StateManager.store_importance_scores(importance_scores, importance_metric)
        
        # Ensure we have both metrics available
        if "in_degrees" not in st.session_state:
            st.session_state["in_degrees"] = compute_in_degree(network.nodes, network.edges)
        in_degrees = st.session_state["in_degrees"]
        
        if "full_cloutrank" not in st.session_state:
            st.session_state["full_cloutrank"] = compute_cloutrank(network.nodes, network.edges)
        cloutrank_scores = st.session_state["full_cloutrank"]
        
        # Select nodes for visualization
        selected_nodes = select_top_nodes_for_visualization(
            network,
            importance_scores,
            max_nodes=max_accounts_display
        )
        
        # Make sure to add first-degree connections to selected nodes
        # This ensures direct connections to the original account are visible
        if network.original_id:
            first_degree_nodes = network.get_first_degree_nodes()
            # Include connections with higher importance scores
            prioritized_first_degree = sorted(
                [(node_id, importance_scores.get(node_id, 0)) for node_id in first_degree_nodes],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Add top first-degree connections if they're not already included
            # Limit to 10 to avoid overcrowding
            for node_id, _ in prioritized_first_degree[:10]:
                if node_id not in selected_nodes:
                    selected_nodes.add(node_id)
                    
            # Make sure original node is included
            selected_nodes.add(network.original_id)
        
        # Create size factors dictionary
        size_factors = {
            'base_size': DEFAULT_BASE_SIZE,
            'importance_factor': account_size_factor,
            'label_size_factor': label_size,
            'node_spacing': node_spacing
        }
        
        # Display the graph
        if st.session_state.get(StateManager.USE_3D, True):
            # Show visual feedback
            st.info(f"Building 3D visualization with up to {max_accounts_display} accounts")
            
            # Keep the original ID for reference
            original_id = network.original_id
            
            # Create a placeholder for the visualization
            viz_placeholder = st.empty()

            # Build initial visualization
            visualizer = Network3DVisualizer(size_factors)
            html_code = visualizer.build_visualization(
                community_filtered_nodes,
                edges,
                list(selected_nodes),
                importance_scores,
                cloutrank_scores,
                in_degrees,
                use_pagerank=use_pagerank
            )
            
            # Render visualization in the placeholder
            with viz_placeholder:
                visualizer.render(html_code, height=750)

            # Add back the node count caption
            st.caption(f"Selected {len(selected_nodes)} nodes based on importance")
            
            # Add topic node toggle in the main area
            show_topic_nodes = st.checkbox(
                "Show Topic Nodes in Graph",
                value=True,  # Set to True to show topics by default
                help="Add topic nodes to the network graph with connections to accounts that discuss them"
            )

            # After retrieving topic data and before building the visualization
            topic_to_accounts, account_to_topics = StateManager.get_topic_data()

            # Add topic nodes to the network if requested
            if show_topic_nodes and topic_to_accounts and account_to_topics:
                # Calculate topic influence for use in visualization sizing
                username_to_node_id = {}
                for node_id, node in network.nodes.items():
                    username = node.get("screen_name", "").lower()
                    if username:
                        username_to_node_id[username] = node_id
                
                # Calculate influence scores for topics
                topic_influence = {}
                for topic, usernames in topic_to_accounts.items():
                    topic_score = 0
                    matched_accounts = 0
                    for username in usernames:
                        node_id = username_to_node_id.get(username.lower())
                        if node_id and node_id in cloutrank_scores:
                            topic_score += cloutrank_scores[node_id]
                            matched_accounts += 1
                    topic_influence[topic] = topic_score
                    logger.debug(f"Topic '{topic}': score {topic_score:.4f}, matched {matched_accounts}/{len(usernames)} accounts")
                
                # Add topic nodes to network
                network.add_topic_nodes(topic_to_accounts, account_to_topics, topic_influence)
                
                # Add topic nodes to selected nodes for visualization
                # Get top topics by influence
                top_topics = sorted(
                    [(topic, score) for topic, score in topic_influence.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]  # Limit to top 10 for better visualization
                
                # Add topic nodes to selected nodes
                topic_node_ids = []
                for topic_name, _ in top_topics:
                    topic_id = f"topic_{topic_name.replace(' ', '_').lower()}"
                    if topic_id in network.nodes:
                        selected_nodes.add(topic_id)
                        topic_node_ids.append(topic_id)
                
                # Rebuild the visualization with topic nodes
                html_code = visualizer.build_visualization(
                    network.nodes,
                    network.edges,
                    list(selected_nodes),
                    importance_scores,
                    cloutrank_scores,
                    in_degrees,
                    use_pagerank=use_pagerank
                )
                
                # Re-render the visualization with topic nodes
                with viz_placeholder:
                    visualizer.render(html_code, height=750)
                
                # Update node count
                st.caption(f"Selected {len(selected_nodes)} nodes based on importance (including {len(topic_node_ids)} topic nodes)")

                st.success(f"Added {len(topic_node_ids)} topic nodes to the visualization")
        else:
            st.error("3D visualization is required in this version")
            st.session_state[StateManager.USE_3D] = True
            st.rerun()

        # Move community color key here, right after the graph
        if community_labels and community_colors and node_communities:
            TableVisualizer.display_community_color_key(
                community_labels,
                community_colors,
                node_communities
            )

        # Add community detection controls
        st.header("Community Detection")
        col1, col2 = st.columns([3, 2])  # Two columns
        
        with col1:
            # Button for summarizing tweets and generating communities
            if st.button("Summarize Tweets to Generate Communities & Topics", use_container_width=True):
                # Step 1: Process tweets for selected nodes including original account
                with st.spinner("Step 1: Summarizing tweets for displayed accounts and original account..."):
                    # First get tweets for original account
                    async def run_async_original_tweet_processing(network):
                        return await data_processor.process_original_account_tweets(network)
                    
                    # Process original account tweets first
                    loop = asyncio.get_event_loop()
                    network = loop.run_until_complete(run_async_original_tweet_processing(network))
                    
                    # Then process tweets for other selected nodes
                    network = loop.run_until_complete(run_async_tweet_processing(
                        network,
                        selected_nodes
                    ))
                    
                    # Update session state with tweet data
                    StateManager.update_nodes_with_tweets(network.nodes)
                    StateManager.store_network_data(network.nodes, network.edges)
                    
                    # Enable tweet summaries checkbox by default
                    st.session_state[StateManager.SHOW_TWEET_SUMMARIES] = True
                
                # Step 2: Extract topics from tweets
                with st.spinner("Step 2: Extracting topics from tweets..."):
                    # Get nodes with tweet data for topic extraction
                    nodes_for_topics = {
                        node_id: node_data for node_id, node_data in network.nodes.items()
                        if node_id in selected_nodes and "tweet_summary" in node_data and node_data["tweet_summary"]
                    }
                    
                    # Print debug information
                    st.text(f"Processing {len(nodes_for_topics)} accounts with tweet summaries for topic extraction...")
                    logger.info(f"Starting topic extraction from {len(nodes_for_topics)} accounts")
                    
                    debug_container = st.empty()
                    status_container = st.empty()
                    debug_text = []
                    
                    status_container.info("Sending request to AI for topic extraction...")
                    
                    # Extract topics from tweets
                    try:
                        loop = asyncio.get_event_loop()
                        topic_to_accounts, account_to_topics = loop.run_until_complete(
                            ai_client.extract_topics_from_tweets(list(nodes_for_topics.values()))
                        )
                        
                        # Print debug information about topics found
                        if topic_to_accounts:
                            debug_text.append(f"✅ Successfully found {len(topic_to_accounts)} topics across {len(account_to_topics)} accounts")
                            debug_text.append("\n📊 Top topics by account count:")
                            
                            # Count accounts per topic and sort by popularity
                            topic_counts = [(topic, len(accounts)) for topic, accounts in topic_to_accounts.items()]
                            sorted_topics = sorted(topic_counts, key=lambda x: x[1], reverse=True)
                            
                            # Show top 10 topics with account counts
                            for idx, (topic, count) in enumerate(sorted_topics[:10], 1):
                                debug_text.append(f"{idx}. {topic} ({count} accounts)")
                                # List up to 3 accounts for each topic
                                accounts = topic_to_accounts[topic][:3]
                                debug_text.append(f"   Accounts: {', '.join('@' + acc for acc in accounts)}")
                                if len(topic_to_accounts[topic]) > 3:
                                    debug_text.append(f"   + {len(topic_to_accounts[topic]) - 3} more accounts")
                            
                            status_container.success("✅ Topic extraction complete!")
                        else:
                            debug_text.append("❌ No topics were extracted. Check logs for details.")
                            status_container.error("❌ Failed to extract topics")
                        
                        # Show the debug text
                        debug_container.text("\n".join(debug_text))
                        
                    except Exception as e:
                        error_msg = f"Error during topic extraction: {str(e)}"
                        logger.error(error_msg)
                        status_container.error(error_msg)
                        debug_container.text(error_msg)
                    
                    # Store results in session state
                    if topic_to_accounts and account_to_topics:
                        StateManager.store_topic_data(topic_to_accounts, account_to_topics)
                        st.success(f"✅ Successfully extracted {len(topic_to_accounts)} topics from tweets!")
                    else:
                        st.error("❌ Failed to extract topics from tweets. Check logs for details.")
                
                # Step 3: Generate communities
                with st.spinner("Step 3: Generating community labels and classifying accounts..."):
                    # Get nodes with tweet data for community generation
                    nodes_for_communities = {
                        node_id: node_data for node_id, node_data in network.nodes.items()
                        if node_id in selected_nodes
                    }
                    
                    # Generate community labels
                    loop = asyncio.get_event_loop()
                    community_labels = loop.run_until_complete(run_async_community_label_generation(
                        list(nodes_for_communities.values()),
                        DEFAULT_NUM_COMMUNITIES
                    ))
                    
                    if community_labels:
                        # Classify accounts into communities
                        node_communities = loop.run_until_complete(run_async_account_classification(
                            list(nodes_for_communities.values())
                        ))
                        
                        # Store results in session state
                        StateManager.store_community_data(
                            community_manager.community_labels,
                            community_manager.community_colors,
                            community_manager.node_communities
                        )
                        
                        # Force a rerun to update the visualization
                        st.success("Community detection and topic extraction complete!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to generate community labels. Please try again.")

        with col2:
            st.info("Communities will be automatically determined based on account descriptions and tweets")

        # Network Analysis section
        st.header("Network Analysis")
        
        # Tweet summaries toggle
        has_tweet_data = any(
            "tweet_summary" in node and node["tweet_summary"] 
            for node_id, node in network.nodes.items()
        )
        
        # Use the session state flag if available, otherwise use has_tweet_data
        default_show_summaries = st.session_state.get(StateManager.SHOW_TWEET_SUMMARIES, False) or has_tweet_data
        
        show_tweet_summaries = st.checkbox(
            "Show Tweet Summaries in Tables", 
            value=default_show_summaries,
            help="Include AI-generated summaries of tweets in the account tables"
        ) if has_tweet_data else False
        
        # Store the current preference back to session state
        st.session_state[StateManager.SHOW_TWEET_SUMMARIES] = show_tweet_summaries
        
        # Add toggle for excluding first-degree follows
        exclude_first_degree = st.checkbox(
            "Exclude First Degree Follows from Top Accounts", 
            value=False,
            help="When checked, accounts directly followed by the original account won't appear in the top accounts table."
        )
        
        # Get top accounts
        if exclude_first_degree:
            # Exclude directly followed accounts
            first_degree_follows = network.get_first_degree_nodes()
            top_accounts = [
                (nid, score, network.nodes[nid]) 
                for nid, score in sorted(
                    [(nid, score) for nid, score in importance_scores.items() 
                     if not nid.startswith("orig_") and nid not in first_degree_follows],
                    key=lambda x: x[1],
                    reverse=True
                )[:20]
            ]
        else:
            # Include all accounts
            top_accounts = [
                (nid, score, network.nodes[nid]) 
                for nid, score in sorted(
                    [(nid, score) for nid, score in importance_scores.items() 
                     if not nid.startswith("orig_")],
                    key=lambda x: x[1],
                    reverse=True
                )[:20]
            ]
        
        # Display top accounts table
        TableVisualizer.display_top_accounts_table(
            network.nodes,
            importance_scores,
            cloutrank_scores,
            in_degrees,
            top_accounts,
            network.original_id,
            show_tweet_summaries=show_tweet_summaries,
            importance_metric=importance_metric_name
        )
        
        # Display topics table if topic data is available
        topic_to_accounts, account_to_topics = StateManager.get_topic_data()
        if topic_to_accounts and account_to_topics:
            TableVisualizer.display_topics_table(
                network.nodes,
                topic_to_accounts,
                account_to_topics,
                cloutrank_scores
        )
        
        # Display community tables if communities have been assigned
        if community_labels and community_colors and node_communities:
            # Ensure community manager has access to the community data
            community_manager.community_labels = community_labels
            community_manager.community_colors = community_colors
            community_manager.node_communities = node_communities
            
            # Get top accounts by community
            top_accounts_by_community = community_manager.get_top_accounts_by_community(
                network.nodes,  # Use full unfiltered list of nodes
                importance_scores,
                top_n=20  # Show top 20 accounts per community
            )
            
            # Display community tables
            TableVisualizer.display_community_tables(
                network.nodes,
                network.edges,
                top_accounts_by_community,
                community_colors,
                community_labels,
                cloutrank_scores,
                in_degrees,
                show_tweet_summaries=show_tweet_summaries
            )
        
        # Add a download button for the full account data
        st.sidebar.markdown("---")  # Separator
        st.sidebar.header("Export Data")
        
        # Create downloadable data
        df = data_processor.create_downloadable_data(
            network,
            importance_scores,
            in_degrees,
            cloutrank_scores,
            node_communities if community_labels else None,
            community_labels if community_labels else None
        )
        
        # Download button
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download All Account Data (CSV)",
            data=csv,
            file_name=f"{input_username}_account_data.csv",
            mime="text/csv",
        )
        
        # Option to view the full table in the app
        if st.sidebar.checkbox("Show Full Account Table"):
            st.header("Complete Account Data")
            st.dataframe(df, use_container_width=True)
            
        # Debug logs section
        st.sidebar.markdown("---")  # Separator
        st.sidebar.header("Debug Options")
        
        if st.sidebar.checkbox("Show Debug Logs"):
            st.header("Debug Logs")
            st.text_area("Log Output", log_stream.getvalue(), height=300)
            if st.button("Clear Logs"):
                log_stream.truncate(0)
                log_stream.seek(0)
                st.rerun()

if __name__ == "__main__":
    main()