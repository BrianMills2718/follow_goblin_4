"""
AI client for Gemini API interactions.
"""
import json
import re
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple

import google.generativeai as genai
import streamlit as st

from config import GEMINI_API_KEY, MAX_CONCURRENT_REQUESTS, DEFAULT_BATCH_SIZE

# Set up logger
logger = logging.getLogger(__name__)

class AIClient:
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self, api_key: str = GEMINI_API_KEY):
        """
        Initialize the Gemini AI client.
        
        Args:
            api_key: Gemini API key
        """
        self.api_key = api_key
        self.model_name = "gemini-2.0-flash"
        self._max_concurrent = MAX_CONCURRENT_REQUESTS
        self._semaphores = {}
        
    def _get_semaphore(self):
        """
        Get or create a semaphore for the current event loop.
        
        Returns:
            asyncio.Semaphore: A semaphore attached to the current event loop
        """
        loop = asyncio.get_event_loop()
        loop_id = id(loop)
        
        if loop_id not in self._semaphores:
            self._semaphores[loop_id] = asyncio.Semaphore(self._max_concurrent)
            
        return self._semaphores[loop_id]
    
    def _initialize_client(self) -> Any:
        """
        Initialize Gemini client.
        
        Returns:
            Gemini GenerativeModel
        """
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(self.model_name)
        
    async def generate_tweet_summary(self, tweets: List[Dict], username: str) -> str:
        """
        Generate an AI summary of tweets using Gemini API.
        
        Args:
            tweets: List of tweet dictionaries
            username: Twitter username
            
        Returns:
            Summary of tweets
        """
        async with self._get_semaphore():
            if not tweets:
                return "No tweets available"
            
            # Prepare tweet texts for the prompt
            tweet_texts = [f"- {tweet['date']}: {tweet['text']}" for tweet in tweets[:20]]  # Limit to 20 tweets
            tweet_content = "\n".join(tweet_texts)
            
            prompt = f"""Analyze these recent tweets from @{username} and provide a brief summary (max 100 words) of their main topics, interests, and tone:

{tweet_content}

Summary:"""

            try:
                # Initialize Gemini client
                client = self._initialize_client()
                
                # Generate summary
                response = client.generate_content(prompt)
                summary = response.text.strip()
                return summary
            
            except Exception as e:
                logger.error(f"Error generating tweet summary for {username}: {str(e)}")
                return f"Error generating summary: {str(e)[:100]}..."
    
    async def generate_batch_tweet_summaries(self, batch_data: List[tuple], 
                                           batch_size: int = DEFAULT_BATCH_SIZE,
                                           max_retries: int = 3) -> Dict[str, str]:
        """
        Generate summaries for multiple accounts in a single API call.
        
        Args:
            batch_data: List of (username, tweets) tuples
            batch_size: Maximum batch size
            max_retries: Maximum number of retries for API failures
            
        Returns:
            Dictionary mapping usernames to summaries
        """
        async with self._get_semaphore():
            if not batch_data:
                return {}
            
            # Prepare batch content for the prompt
            batch_content = []
            usernames = []
            account_tweets_map = {}  # Track number of tweets per account
            
            # Format each account's tweets
            for username, tweets in batch_data:
                usernames.append(username)
                account_tweets_map[username] = len(tweets)
                
                # Check if there are any tweets for this account
                if not tweets:
                    continue
                    
                account_tweets = tweets[:10]  # Limit to 10 tweets per account for brevity
                tweet_texts = [f"@{username} {tweet['date']}: {tweet['text']}" for tweet in account_tweets]
                batch_content.extend(tweet_texts)
                # Add separator between accounts
                if len(batch_content) > 0:
                    batch_content.append("---")  # Separator between accounts
            
            # Remove the last separator
            if batch_content and batch_content[-1] == "---":
                batch_content.pop()
            
            # Initialize results with specific error messages for accounts with no tweets
            summaries = {}
            for username in usernames:
                if account_tweets_map[username] == 0:
                    summaries[username] = "No tweets available for this account"
            
            # If no accounts have tweets, return early
            if not batch_content:
                return summaries
            
            prompt = f"""Analyze the following tweets from {len([u for u in usernames if account_tweets_map[u] > 0])} different Twitter accounts.
For EACH account, provide a brief summary (max 50 words) of their main topics, interests, and tone.

Tweets:
{chr(10).join(batch_content)}

Response format:
@username1: [brief summary of account 1]
@username2: [brief summary of account 2]
...and so on for all accounts
"""

            for retry_count in range(max_retries):
                try:
                    # Initialize Gemini client
                    client = self._initialize_client()
                    
                    logger.info(f"Generating batch summaries (attempt {retry_count + 1}/{max_retries})")
                    
                    # Generate batch summaries
                    response = client.generate_content(prompt)
                    
                    # Parse the response to extract summaries for each account
                    text = response.text.strip()
                    
                    # Check if the response contains valid summaries
                    valid_response = False
                    parsed_summaries = {}
                    
                    for line in text.split('\n'):
                        if line.startswith('@') and ':' in line:
                            username, summary = line.split(':', 1)
                            username = username.strip('@').strip()
                            if username in usernames:
                                parsed_summaries[username] = summary.strip()
                                valid_response = True
                    
                    if valid_response:
                        # Merge with existing summaries
                        summaries.update(parsed_summaries)
                        
                        # Ensure all accounts with tweets have summaries
                        for username in usernames:
                            if username not in summaries and account_tweets_map[username] > 0:
                                summaries[username] = "API generated no summary despite tweets being available"
                        
                        return summaries
                    else:
                        logger.warning(f"API response didn't contain valid summaries (attempt {retry_count + 1}/{max_retries})")
                        logger.warning(f"Response text (first 200 chars): {text[:200]}...")
                        # Continue to next retry
                
                except Exception as e:
                    error_message = str(e)
                    logger.error(f"Error in attempt {retry_count + 1}/{max_retries}: {error_message}")
                    
                    # Continue to next retry if not the last attempt
                    if retry_count < max_retries - 1:
                        continue
                    
                    # On the last attempt, prepare error messages
                    error_type = "API error"
                    if "rate limit" in error_message.lower():
                        error_type = "Rate limit exceeded"
                    elif "timeout" in error_message.lower():
                        error_type = "API timeout"
                    elif "auth" in error_message.lower() or "key" in error_message.lower():
                        error_type = "API authentication error"
                    
                    # Set error message for all accounts that don't already have summaries
                    for username in usernames:
                        if username not in summaries:
                            summaries[username] = f"Error: {error_type} after {max_retries} attempts - {error_message[:50]}..."
            
            logger.error(f"Failed to generate batch summaries after {max_retries} attempts")
            return summaries
    
    async def generate_community_labels(self, accounts: List[Dict], 
                                      num_communities: int,
                                      max_retries: int = 3) -> Dict[str, str]:
        """
        Get community labels from Gemini using account descriptions.
        
        Args:
            accounts: List of account dictionaries
            num_communities: Number of communities to generate
            max_retries: Maximum number of retries for API failures
            
        Returns:
            Dictionary mapping community IDs to labels
        """
        async with self._get_semaphore():
            # Prepare account information for the prompt
            account_info = []
            for acc in accounts:
                desc = acc.get('description', '').strip()
                tweet_summary = acc.get('tweet_summary', '').strip()
                
                if desc or tweet_summary:
                    line = f"Username: {acc['screen_name']}, Description: {desc}"
                    if tweet_summary:
                        line += f", Tweet Summary: {tweet_summary}"
                    account_info.append(line)
            
            if not account_info:
                logger.warning("No account information available for community label generation")
                return {}
                
            # Join account info with line breaks
            accounts_text = "\n".join(account_info)
            
            # If text is very long, truncate it
            if len(accounts_text) > 15000:
                accounts_text = accounts_text[:15000] + "\n...[additional accounts omitted for brevity]"
            
            prompt = f"""Analyze these X/Twitter accounts and their descriptions. Create {num_communities} community labels that provide good coverage of the different types of accounts present. Include an "Other" category for accounts that don't fit well into specific groups.

Accounts to analyze:
{accounts_text}

Return your response as a JSON object mapping community IDs to labels, like:
{{
  "0": "Tech Entrepreneurs",
  "1": "Political Commentators", 
  "2": "Other"
}}"""

            for retry_count in range(max_retries):
                try:
                    # Initialize Gemini client
                    client = self._initialize_client()
                    
                    logger.info(f"Generating community labels (attempt {retry_count + 1}/{max_retries})")
                    
                    # Generate community labels
                    response = client.generate_content(prompt)
                    
                    # Try to extract JSON
                    text = response.text.strip()
                    
                    try:
                        # First attempt direct JSON parsing
                        community_labels = json.loads(text)
                        logger.info("Successfully parsed community labels JSON directly")
                        
                        # Validate structure (should be string keys and string values)
                        valid_labels = {}
                        for k, v in community_labels.items():
                            if isinstance(v, str):
                                valid_labels[str(k)] = v
                            
                        if valid_labels:
                            return valid_labels
                            
                    except json.JSONDecodeError:
                        # Try to extract JSON using regex
                        logger.info("Direct JSON parsing failed, trying to extract JSON block")
                        json_match = re.search(r'({[\s\S]*})', text)
                        
                        if json_match:
                            json_str = json_match.group(1)
                            
                            # Clean up common JSON formatting issues
                            json_str = re.sub(r',\s*}', '}', json_str)
                            json_str = re.sub(r',\s*]', ']', json_str)
                            json_str = json_str.replace("'", '"')
                            
                            try:
                                community_labels = json.loads(json_str)
                                logger.info("Successfully parsed community labels JSON after extraction")
                                
                                # Validate structure
                                valid_labels = {}
                                for k, v in community_labels.items():
                                    if isinstance(v, str):
                                        valid_labels[str(k)] = v
                                
                                if valid_labels:
                                    return valid_labels
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse JSON after extraction (attempt {retry_count + 1}/{max_retries})")
                    
                    # If we got here, parsing failed, try manual extraction of key-value pairs
                    logger.info("JSON parsing failed, trying manual extraction of key-value pairs")
                    labels = {}
                    
                    # Try to extract "key": "value" patterns
                    pattern = r'"(\d+)":\s*"([^"]+)"'
                    matches = re.findall(pattern, text)
                    
                    if matches:
                        for k, v in matches:
                            labels[k] = v
                            
                        if labels:
                            logger.info(f"Extracted {len(labels)} community labels through regex pattern matching")
                            return labels
                    
                    # If still no luck, continue to next retry
                    logger.warning(f"Could not extract valid community labels (attempt {retry_count + 1}/{max_retries})")
                    
                except Exception as e:
                    logger.error(f"Error in attempt {retry_count + 1}/{max_retries}: {str(e)}")
                    # Continue to next retry
            
            # If all retries fail, return a default set of labels
            logger.error(f"Failed to generate community labels after {max_retries} attempts, using defaults")
            default_labels = {
                "0": "Other",
                "1": "Topic 1",
                "2": "Topic 2"
            }
            # Add more default labels if needed
            for i in range(3, num_communities):
                default_labels[str(i)] = f"Topic {i}"
                
            return default_labels
    
    async def classify_accounts(self, accounts: List[Dict], 
                              community_labels: Dict[str, str],
                              max_retries: int = 3) -> Dict[str, str]:
        """
        Classify accounts into communities.
        
        Args:
            accounts: List of account dictionaries
            community_labels: Dictionary mapping community IDs to labels
            max_retries: Maximum number of retries for API failures
            
        Returns:
            Dictionary mapping usernames to community IDs
        """
        async with self._get_semaphore():
            # Create formatted community label string
            labels_str = "\n".join([f"{comm_id}: {label}" for comm_id, label in community_labels.items()])
            
            # Find the "Other" category ID
            other_community_id = next((cid for cid, label in community_labels.items() 
                                   if label.lower() == "other"), list(community_labels.keys())[0])
            
            # Convert accounts to required format
            accounts_info = []
            for acc in accounts:
                desc = acc.get('description', '').strip()
                tweet_summary = acc.get('tweet_summary', '').strip()
                
                if desc or tweet_summary:
                    line = f"Username: {acc['screen_name']}, Description: {desc}"
                    if tweet_summary:
                        line += f", Tweet Summary: {tweet_summary}"
                    accounts_info.append(line)
            
            if not accounts_info:
                logger.warning("No account information available for classification")
                return {}
                
            # Format account info into a string
            accounts_text = "\n".join(accounts_info)
            
            # If text is very long, truncate
            if len(accounts_text) > 15000:
                accounts_text = accounts_text[:15000] + "\n...[additional accounts omitted for brevity]"
            
            prompt = f"""Given these community labels:
{labels_str}

Classify each of these accounts into exactly one of the above communities.
Only use the exact community ID keys provided above (e.g., "0", "1", etc.), do not create new ones.
If unsure, use the "{other_community_id}" category (Other).

Accounts to classify:
{accounts_text}

Return in format:
username: community_id"""

            for retry_count in range(max_retries):
                try:
                    # Initialize Gemini client
                    client = self._initialize_client()
                    
                    logger.info(f"Classifying accounts (attempt {retry_count + 1}/{max_retries})")
                    
                    # Generate classifications
                    response = client.generate_content(prompt)
                    
                    # Parse the response into dictionary
                    classifications = {}
                    valid_community_ids = set(community_labels.keys())
                    valid_response = False
                    
                    for line in response.text.split('\n'):
                        if ':' in line:
                            username, community = line.split(':', 1)
                            username = username.strip()
                            community = community.strip()
                            
                            # Clean up username if it has @ symbol
                            if username.startswith('@'):
                                username = username[1:]
                            
                            # Validate community ID
                            if community in valid_community_ids:
                                classifications[username] = community
                                valid_response = True
                            else:
                                # If invalid community, use "Other"
                                classifications[username] = other_community_id
                                valid_response = True
                                logger.warning(f"Invalid community '{community}' assigned to {username}, using 'Other' instead")
                    
                    # Check if we got any valid classifications
                    if valid_response and classifications:
                        return classifications
                    else:
                        logger.warning(f"No valid classifications found in response (attempt {retry_count + 1}/{max_retries})")
                        logger.warning(f"Response text (first 200 chars): {response.text[:200]}...")
                    
                except Exception as e:
                    logger.error(f"Error in attempt {retry_count + 1}/{max_retries}: {str(e)}")
            
            # If all retries fail, return a default classification
            logger.error(f"Failed to classify accounts after {max_retries} attempts")
            
            # As a fallback, assign all accounts to the "Other" category
            default_classifications = {}
            for acc in accounts:
                username = acc.get('screen_name', '')
                if username:
                    default_classifications[username] = other_community_id
            
            return default_classifications
    
    async def extract_topics_from_tweets(self, accounts: List[Dict], max_retries: int = 3) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Extract topics from tweet summaries and descriptions of accounts.
        
        Args:
            accounts: List of account dictionaries with tweet_summary and description fields
            max_retries: Maximum number of retries for JSON parsing failures
            
        Returns:
            Tuple containing:
            - Dictionary mapping topics to list of usernames
            - Dictionary mapping usernames to list of topics
        """
        async with self._get_semaphore():
            # Filter accounts with tweet summaries or descriptions
            accounts_with_content = []
            for acc in accounts:
                username = acc.get('screen_name', '')
                summary = acc.get('tweet_summary', '')
                description = acc.get('description', '')
                
                if (summary or description) and username:
                    accounts_with_content.append({
                        'username': username,
                        'summary': summary,
                        'description': description
                    })
            
            if not accounts_with_content:
                logger.warning("No accounts with content found for topic extraction")
                return {}, {}
                
            accounts_json = json.dumps(accounts_with_content, indent=2)
            
            # Create prompt for topic extraction
            prompt = f"""Analyze these Twitter accounts and their content to identify common topics they discuss.
Group accounts by topic based on their tweet summaries and descriptions.

Accounts data:
{accounts_json}

Return a JSON object with this exact format:
{{
  "topics": [
    {{
      "name": "Topic Name",
      "accounts": ["username1", "username2", "username3"]
    }},
    {{
      "name": "Another Topic",
      "accounts": ["username1", "username4", "username5"]
    }}
  ]
}}

Ensure each topic has at least 2 accounts associated with it. Don't include username '@' symbols in the JSON arrays."""

            for retry_count in range(max_retries):
                try:
                    # Initialize Gemini client
                    client = self._initialize_client()
                    
                    logger.info(f"Sending topic extraction request to Gemini API (attempt {retry_count + 1}/{max_retries})")
                    
                    # Generate topic analysis
                    response = client.generate_content(prompt)
                    response_text = response.text.strip()
                    
                    # Extract JSON from response - look for JSON block
                    data = None
                    parsing_success = False
                    
                    try:
                        # First try direct JSON parsing
                        data = json.loads(response_text)
                        logger.info("Successfully parsed JSON response directly")
                        parsing_success = True
                    except json.JSONDecodeError:
                        # If direct parsing fails, try to extract JSON block using regex
                        logger.info("Direct JSON parsing failed, trying to extract JSON block")
                        json_match = re.search(r'({[\s\S]*})', response_text)
                        
                        if not json_match:
                            logger.warning("Could not extract JSON from topic extraction response")
                            logger.warning(f"Response text (first 200 chars): {response_text[:200]}...")
                            # Continue to next retry
                            continue
                        
                        json_str = json_match.group(1)
                        logger.info(f"Extracted JSON block (length: {len(json_str)})")
                        
                        # Clean up common JSON formatting issues
                        # Remove trailing commas
                        json_str = re.sub(r',\s*}', '}', json_str)
                        json_str = re.sub(r',\s*]', ']', json_str)
                        # Replace single quotes with double quotes
                        json_str = json_str.replace("'", '"')
                        # Remove comments
                        json_str = re.sub(r'//.*?\n', '\n', json_str)
                        
                        try:
                            data = json.loads(json_str)
                            logger.info("JSON parsed successfully after cleaning")
                            parsing_success = True
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON parsing error after cleaning: {str(e)}")
                            logger.error(f"Cleaned JSON string (first 200 chars): {json_str[:200]}...")
                            # Continue to next retry
                            continue
                    
                    if parsing_success and data:
                        logger.info(f"JSON parsed successfully, found {len(data.get('topics', []))} topics")
                        
                        # Create topic to usernames mapping
                        topic_to_accounts = {}
                        account_to_topics = {}
                        
                        for topic_entry in data.get("topics", []):
                            topic_name = topic_entry.get("name", "").strip()
                            accounts = topic_entry.get("accounts", [])
                            
                            if topic_name and accounts:
                                logger.info(f"Topic '{topic_name}' has {len(accounts)} accounts")
                                topic_to_accounts[topic_name] = accounts
                                
                                # Also build the reverse mapping of usernames to topics
                                for username in accounts:
                                    if username not in account_to_topics:
                                        account_to_topics[username] = []
                                    account_to_topics[username].append(topic_name)
                        
                        logger.info(f"Extraction complete: {len(topic_to_accounts)} topics across {len(account_to_topics)} accounts")
                        return topic_to_accounts, account_to_topics
                
                except Exception as e:
                    logger.error(f"Error in attempt {retry_count + 1}/{max_retries}: {str(e)}")
                    # Continue to next retry
                    continue
            
            # If we've exhausted all retries
            logger.error(f"Failed to extract topics after {max_retries} attempts")
            return {}, {}
    
    async def summarize_user_tweets(self, username: str, tweet_text: str) -> str:
        """
        Generate an AI summary of tweets using Gemini API specifically for the original user.
        
        Args:
            username: Twitter username
            tweet_text: Concatenated tweet texts
            
        Returns:
            Summary of tweets
        """
        async with self._get_semaphore():
            if not tweet_text:
                return "No tweets available"
            
            prompt = f"""Analyze these recent tweets from @{username} and provide a brief summary (max 150 words) of their main topics, interests, and communication style:

{tweet_text}

Summary:"""

            try:
                # Initialize Gemini client
                client = self._initialize_client()
                
                # Generate summary
                response = client.generate_content(prompt)
                summary = response.text.strip()
                return summary
            
            except Exception as e:
                logger.error(f"Error generating tweet summary for original user {username}: {str(e)}")
                return f"Error generating summary: {str(e)[:100]}..."