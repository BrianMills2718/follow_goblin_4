"""
Data processing utilities for X Network Visualization.
"""
import logging
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Set

import aiohttp
import pandas as pd
import streamlit as st

from api.twitter_client import TwitterClient
from api.ai_client import AIClient
from data.network import NetworkData
from config import DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Processor for network data collection and processing.
    """
    
    def __init__(self, twitter_client: TwitterClient, ai_client: AIClient):
        """
        Initialize the data processor.
        
        Args:
            twitter_client: TwitterClient instance
            ai_client: AIClient instance
        """
        self.twitter_client = twitter_client
        self.ai_client = ai_client
    
    async def collect_network_data(self, 
                                 username: str, 
                                 following_pages: int = 2, 
                                 second_degree_pages: int = 1) -> NetworkData:
        """
        Collect network data for a given username.
        
        Args:
            username: Twitter username
            following_pages: Number of pages of following accounts to fetch
            second_degree_pages: Number of pages of following for second-degree connections
            
        Returns:
            NetworkData object with collected data
        """
        network = NetworkData()
        original_id = network.add_original_node(username)
        
        # Create progress indicators
        progress = st.progress(0)
        status_text = st.empty()
        
        try:
            async with TwitterClient.create_session() as session:
                # Step 1: Get following for original account with pagination
                status_text.text("Fetching accounts followed by original user...")
                first_hop_accounts = []
                cursor = None
                
                for page in range(following_pages):
                    accounts, cursor = await self.twitter_client.get_following(username, session, cursor)
                    first_hop_accounts.extend(accounts)
                    
                    progress.progress((page + 1) / (following_pages + 1))
                    status_text.text(f"Fetched page {page+1}/{following_pages} of following accounts for original user")
                    
                    if not cursor or cursor == "-1":
                        break
                
                # Add first hop accounts to nodes and create edges
                for account in first_hop_accounts:
                    uid = str(account.get("user_id"))
                    if not uid:
                        continue
                    
                    # Add "direct" attribute to indicate first-degree connection
                    account["direct"] = True
                    network.add_node(uid, account)
                    network.add_edge(original_id, uid)
                
                # Step 2: Get following for each first-degree account IN PARALLEL
                status_text.text(f"Fetching following for {len(first_hop_accounts)} first-degree accounts in parallel...")
                
                # Create tasks for all first-degree accounts
                second_degree_tasks = []
                for account in first_hop_accounts:
                    source_id = str(account.get("user_id"))
                    source_name = account.get("screen_name", "")
                    
                    # Create task to fetch second-degree connections
                    task = self._fetch_second_degree_connections(
                        source_id, 
                        source_name, 
                        session, 
                        second_degree_pages
                    )
                    second_degree_tasks.append(task)
                
                # Run all tasks concurrently with progress reporting
                total_tasks = len(second_degree_tasks)
                second_degree_results = []
                for i, task_coroutine in enumerate(asyncio.as_completed(second_degree_tasks), 1):
                    result = await task_coroutine
                    second_degree_results.append(result)
                    progress.progress((following_pages + i) / (following_pages + total_tasks + 1))
                    status_text.text(f"Processed {i}/{total_tasks} first-degree accounts")
                
                # Process all second-degree connections
                for source_id, connections in second_degree_results:
                    for sid, node_data in connections:
                        if sid not in network.nodes:
                            network.add_node(sid, node_data)
                        network.add_edge(source_id, sid)
                
                # Complete progress
                progress.progress(1.0)
                status_text.text("Network data collection complete!")
                
        except Exception as e:
            logger.error(f"Error in network collection: {str(e)}")
            status_text.text(f"Error: {str(e)}")
        
        return network
    
    async def _fetch_second_degree_connections(self, 
                                             source_id: str, 
                                             source_name: str, 
                                             session: aiohttp.ClientSession, 
                                             max_pages: int) -> Tuple[str, List[Tuple[str, Dict]]]:
        """
        Fetch all second-degree connections for a given account with parallel page fetching.
        
        Args:
            source_id: Source account ID
            source_name: Source account username
            session: aiohttp ClientSession
            max_pages: Maximum number of pages to fetch
            
        Returns:
            Tuple containing source ID and list of (node_id, node_data) tuples
        """
        connections = []
        
        # Function to fetch a single page
        async def fetch_page(cursor=None):
            page_connections = []
            accounts, next_cursor = await self.twitter_client.get_following(source_name, session, cursor)
            
            # Process accounts
            for account in accounts:
                sid = str(account.get("user_id"))
                if sid:
                    # Add "direct" attribute (set to False for second-degree connections)
                    account["direct"] = False
                    page_connections.append((sid, account))
            
            return page_connections, next_cursor
        
        # First, fetch the initial page to get the cursor
        initial_connections, cursor = await fetch_page()
        connections.extend(initial_connections)
        
        # If we need more pages and have a valid cursor
        if max_pages > 1 and cursor and cursor != "-1":
            # Create tasks for remaining pages
            page_tasks = []
            current_cursor = cursor
            
            # Create up to max_pages-1 additional page fetching tasks
            for _ in range(max_pages - 1):
                if not current_cursor or current_cursor == "-1":
                    break
                    
                # Create task to fetch the next page with current cursor
                task = fetch_page(current_cursor)
                page_tasks.append(task)
                
                # We need to fetch the first result to get the next cursor
                # This is sequential, but we're still creating parallel tasks
                page_connections, current_cursor = await task
                connections.extend(page_connections)
                
                if not current_cursor or current_cursor == "-1":
                    break
        
        return (source_id, connections)
    
    async def process_tweet_data(self, 
                               network: NetworkData, 
                               selected_nodes: Set[str], 
                               batch_size: int = DEFAULT_BATCH_SIZE) -> NetworkData:
        """
        Fetch and summarize tweets for selected nodes with parallel batch processing.
        
        Args:
            network: NetworkData instance
            selected_nodes: Set of node IDs to process
            batch_size: Batch size for API calls
            
        Returns:
            Updated NetworkData instance
        """
        # Create UI elements for progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"Processing tweets for {len(selected_nodes)} accounts...")
        
        # Skip if no nodes to process
        if not selected_nodes:
            status_text.text("No nodes selected for tweet processing")
            return network
            
        # Get only nodes from selected_nodes set
        nodes_to_process = [
            (node_id, network.nodes[node_id]) 
            for node_id in selected_nodes 
            if node_id in network.nodes
        ]
        
        try:
            async with TwitterClient.create_session() as session:
                # Process in batches
                total_processed = 0
                
                # Create batches
                batches = [nodes_to_process[i:i+batch_size] 
                          for i in range(0, len(nodes_to_process), batch_size)]
                
                # Define function to process a single batch
                async def process_batch(batch_idx, batch):
                    batch_results = []
                    
                    # Create tasks to fetch tweets for each account in the batch
                    tweet_tasks = []
                    for node_id, node in batch:
                        task = self._fetch_tweets_for_node(node_id, node, session)
                        tweet_tasks.append(task)
                    
                    # Wait for all tasks to complete
                    results = await asyncio.gather(*tweet_tasks)
                    
                    # Process results
                    for node_id, tweets, status in results:
                        if tweets:
                            # Generate summary
                            username = network.nodes[node_id]["screen_name"]
                            summary = await self.ai_client.generate_tweet_summary(tweets, username)
                            
                            # Update network
                            network.update_node_tweet_data(node_id, tweets, summary)
                            
                            # Set success status
                            network.nodes[node_id]["tweet_fetch_status"] = ""
                        else:
                            # Set error status
                            network.nodes[node_id]["tweet_fetch_status"] = status
                    
                    return batch_idx, len(batch)
                
                # Process batches in parallel with a concurrency limit
                # Using up to 5 concurrent batches (adjust based on your rate limits)
                concurrency_limit = min(5, len(batches))
                
                # Create a semaphore to limit concurrency
                semaphore = asyncio.Semaphore(concurrency_limit)
                
                async def process_batch_with_semaphore(batch_idx, batch):
                    async with semaphore:
                        return await process_batch(batch_idx, batch)
                
                # Create tasks for all batches
                batch_tasks = [
                    process_batch_with_semaphore(idx, batch)
                    for idx, batch in enumerate(batches)
                ]
                
                # Process all batches with progress tracking
                for completed_task in asyncio.as_completed(batch_tasks):
                    batch_idx, batch_size = await completed_task
                    total_processed += batch_size
                    progress_bar.progress(total_processed / len(nodes_to_process))
                    status_text.text(f"Processed batch {batch_idx+1}/{len(batches)} " 
                                    f"({total_processed}/{len(nodes_to_process)} accounts)")
                
                # Complete progress
                progress_bar.progress(1.0)
                status_text.text(f"Tweet processing complete for {total_processed} accounts")
                
        except Exception as e:
            logger.error(f"Error processing tweets: {str(e)}")
            status_text.text(f"Error processing tweets: {str(e)}")
        
        return network
    
    async def _fetch_tweets_for_node(self, 
                                  node_id: str, 
                                  node: Dict, 
                                  session: aiohttp.ClientSession) -> Tuple[str, List[Dict], str]:
        """
        Fetch tweets for a single node.
        
        Args:
            node_id: Node ID
            node: Node data
            session: aiohttp ClientSession
            
        Returns:
            Tuple containing node ID, list of tweets, and status message
        """
        try:
            tweets, _ = await self.twitter_client.get_user_tweets(node_id, session)
            if tweets:
                return node_id, tweets, ""
            else:
                return node_id, [], "No tweets available"
        except Exception as e:
            logger.error(f"Error fetching tweets for {node['screen_name']}: {str(e)}")
            return node_id, [], f"Error: {str(e)}"
    
    def create_downloadable_data(self, 
                               network: NetworkData, 
                               importance_scores: Dict[str, float],
                               in_degrees: Dict[str, int],
                               cloutrank_scores: Dict[str, float], 
                               node_communities: Dict[str, str] = None,
                               community_labels: Dict[str, str] = None) -> pd.DataFrame:
        """
        Create a comprehensive downloadable table with all account information.
        
        Args:
            network: NetworkData instance
            importance_scores: Dictionary mapping node IDs to importance scores
            in_degrees: Dictionary mapping node IDs to in-degree values
            cloutrank_scores: Dictionary mapping node IDs to CloutRank scores
            node_communities: Dictionary mapping usernames to community IDs
            community_labels: Dictionary mapping community IDs to labels
            
        Returns:
            Pandas DataFrame with account data
        """
        # Create DataFrame for all accounts
        data = []
        
        for node_id, node in network.nodes.items():
            # Skip nodes that might not be complete accounts
            if not isinstance(node, dict) or "screen_name" not in node:
                continue
            
            # Get community info if available
            community_id = ""
            community_label = ""
            
            if node_communities and community_labels:
                username = node["screen_name"]
                if username in node_communities:
                    community_id = node_communities[username]
                    community_label = community_labels.get(community_id, "")
            
            # Get tweet summary
            tweet_summary = node.get("tweet_summary", "")
            
            # Add row to data
            row = {
                "Screen Name": node["screen_name"],
                "Name": node.get("name", ""),
                "CloutRank": cloutrank_scores.get(node_id, 0),
                "In-Degree": in_degrees.get(node_id, 0),
                "Importance": importance_scores.get(node_id, 0),
                "Followers": node.get("followers_count", 0),
                "Following": node.get("friends_count", 0),
                "Ratio": node.get("ratio", 0),
                "Tweets": node.get("statuses_count", 0),
                "Media": node.get("media_count", 0),
                "Created At": node.get("created_at", ""),
                "Verified": node.get("verified", False),
                "Blue Verified": node.get("blue_verified", False),
                "Business Account": node.get("business_account", False),
                "Website": node.get("website", ""),
                "Location": node.get("location", ""),
                "Community ID": community_id,
                "Community": community_label,
                "Tweet Summary": tweet_summary,
                "Description": node.get("description", "")
            }
            data.append(row)
        
        # Convert to DataFrame and sort by CloutRank
        df = pd.DataFrame(data)
        df = df.sort_values("CloutRank", ascending=False)
        
        return df
    
    async def process_original_account_tweets(self, network: NetworkData) -> NetworkData:
        """
        Fetch and process tweets specifically for the original account.
        
        Args:
            network: NetworkData instance
            
        Returns:
            Updated NetworkData instance with original account tweets processed
        """
        if not network.original_id or not network.original_username:
            logger.warning("Cannot process original account tweets: No original account found")
            return network
            
        status_text = st.empty()
        status_text.text(f"Retrieving tweets for original account @{network.original_username}...")
        
        try:
            async with TwitterClient.create_session() as session:
                # First get the Twitter user ID from the username
                user_id = await self.twitter_client.get_user_id_from_username(network.original_username, session)
                
                if not user_id:
                    status_text.text(f"Could not find Twitter ID for @{network.original_username}")
                    return network
                
                # Then fetch tweets for this user ID
                tweets, _ = await self.twitter_client.get_user_tweets(user_id, session)
                
                if not tweets:
                    status_text.text(f"No tweets available for @{network.original_username}")
                    return network
                
                # Update the original node with the tweets
                if tweets:
                    # Process the tweets with AI
                    tweet_text = "\n\n".join([t.get("full_text", t.get("text", "")) for t in tweets[:10]])
                    
                    # Get AI summary
                    summary = await self.ai_client.summarize_user_tweets(
                        network.original_username, 
                        tweet_text
                    )
                    
                    # Store the result
                    network.nodes[network.original_id]["tweets"] = tweets
                    network.nodes[network.original_id]["tweet_summary"] = summary
                    network.nodes[network.original_id]["tweet_fetch_status"] = ""
                
                status_text.text(f"Successfully processed tweets for @{network.original_username}")
                
        except Exception as e:
            logger.error(f"Error processing tweets for original account: {str(e)}")
            status_text.text(f"Error processing tweets for original account: {str(e)}")
        
        return network