"""
Twitter API client for the X Network Visualization application.
"""
import json
import asyncio
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any

import aiohttp
import streamlit as st

from config import RAPIDAPI_KEY, RAPIDAPI_HOST, MAX_CONCURRENT_REQUESTS, DEFAULT_FETCH_TIMEOUT

logger = logging.getLogger(__name__)

class TwitterClient:
    """Client for interacting with Twitter API through RapidAPI."""
    
    def __init__(self, api_key: str = RAPIDAPI_KEY, api_host: str = RAPIDAPI_HOST):
        """
        Initialize the Twitter API client.
        
        Args:
            api_key: RapidAPI key
            api_host: RapidAPI host
        """
        self.api_key = api_key
        self.api_host = api_host
        self.headers = {
            "x-rapidapi-key": self.api_key, 
            "x-rapidapi-host": self.api_host
        }
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
        
    async def get_following(self, screenname: str, session: aiohttp.ClientSession, 
                           cursor: Optional[str] = None) -> Tuple[List[Dict], Optional[str]]:
        """
        Asynchronously retrieve accounts that the given user is following.
        
        Args:
            screenname: Twitter username
            session: aiohttp ClientSession
            cursor: Pagination cursor
            
        Returns:
            Tuple containing list of account dictionaries and next cursor
        """
        semaphore = self._get_semaphore()
        async with semaphore:
            endpoint = f"/FollowingLight?username={screenname}&count=20"
            if cursor and cursor != "-1":
                endpoint += f"&cursor={cursor}"
                
            url = f"https://{self.api_host}{endpoint}"
            
            try:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        logger.error(f"Failed to get following for {screenname}: {response.status}")
                        return [], None
                    data = await response.text()
                    return self._parse_following_response(data)
            except Exception as e:
                logger.error(f"Error fetching following for {screenname}: {str(e)}")
                return [], None
    
    async def get_user_tweets(self, user_id: str, session: aiohttp.ClientSession, 
                             cursor: Optional[str] = None) -> Tuple[List[Dict], Optional[str]]:
        """
        Asynchronously fetch tweets from a specific user.
        
        Args:
            user_id: Twitter user ID
            session: aiohttp ClientSession
            cursor: Pagination cursor
            
        Returns:
            Tuple containing list of tweet dictionaries and next cursor
        """
        async with self._get_semaphore():
            endpoint = f"/UserTweets?user_id={user_id}"
            if cursor:
                endpoint += f"&cursor={cursor}"
            
            url = f"https://{self.api_host}{endpoint}"
            
            # Add exponential backoff for rate limiting
            max_retries = 3
            retry_delay = 1.0
            
            for retry in range(max_retries):
                try:
                    async with session.get(url, headers=self.headers) as response:
                        if response.status == 429:  # Too Many Requests
                            # Rate limited, exponential backoff
                            wait_time = retry_delay * (2 ** retry)
                            logger.warning(f"Rate limited when getting tweets for {user_id}. Retrying in {wait_time:.1f}s (attempt {retry+1}/{max_retries})")
                            await asyncio.sleep(wait_time)
                            continue
                            
                        if response.status != 200:
                            logger.error(f"Failed to get tweets for {user_id}: {response.status}")
                            if retry < max_retries - 1:
                                # For other errors, also retry with backoff
                                wait_time = retry_delay * (2 ** retry)
                                logger.warning(f"Retrying in {wait_time:.1f}s (attempt {retry+1}/{max_retries})")
                                await asyncio.sleep(wait_time)
                                continue
                            return [], None
                            
                        data = await response.text()
                        return self._parse_tweet_data(data)
                except Exception as e:
                    logger.error(f"Error fetching tweets for {user_id}: {str(e)}")
                    if retry < max_retries - 1:
                        wait_time = retry_delay * (2 ** retry)
                        logger.warning(f"Retrying after error in {wait_time:.1f}s (attempt {retry+1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                    else:
                        return [], None
            
            # If we exhausted all retries
            return [], None
    
    def _parse_following_response(self, json_str: str) -> Tuple[List[Dict], Optional[str]]:
        """
        Parse the JSON response from Twitter API for following accounts.
        
        Args:
            json_str: JSON response string
            
        Returns:
            Tuple containing list of account dictionaries and next cursor
        """
        try:
            data = json.loads(json_str)
            
            accounts = []
            next_cursor = data.get("next_cursor_str")
            
            # Extract users from the FollowingLight response format
            users = data.get("users", [])
            
            for user in users:
                account = {
                    "user_id": user.get("id_str"),
                    "screen_name": user.get("screen_name", ""),
                    "name": user.get("name", ""),
                    "followers_count": user.get("followers_count", 0),
                    "friends_count": user.get("friends_count", 0),
                    "statuses_count": user.get("statuses_count", 0),
                    "media_count": user.get("media_count", 0),
                    "created_at": user.get("created_at", ""),
                    "location": user.get("location", ""),
                    "blue_verified": user.get("verified", False),
                    "verified": user.get("verified", False),
                    "website": user.get("url", ""),
                    "business_account": False,  # Not available in this API
                    "description": user.get("description", ""),
                }
                accounts.append(account)
            
            return accounts, next_cursor
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing following response: {str(e)}")
            return [], None
        except Exception as e:
            logger.error(f"Unexpected error parsing following response: {str(e)}")
            return [], None
    
    def _parse_tweet_data(self, json_str: str) -> Tuple[List[Dict], Optional[str]]:
        """
        Parse the tweet data from the API response.
        
        Args:
            json_str: JSON response string
            
        Returns:
            Tuple containing list of tweet dictionaries and next cursor
        """
        try:
            # First check if the response is valid JSON
            if not json_str or len(json_str.strip()) == 0:
                return [], None
                
            tweet_data = json.loads(json_str)
            tweets = []
            next_cursor = None
            
            # Basic validation of API response format
            if not isinstance(tweet_data, dict):
                return [], None
                
            # Check for error messages in the API response
            if "errors" in tweet_data:
                error_msgs = [error.get("message", "Unknown error") for error in tweet_data.get("errors", [])]
                if error_msgs:
                    logger.error(f"API returned errors: {', '.join(error_msgs)}")
                return [], None
                
            # Handle case where API returns empty data
            if "data" not in tweet_data:
                return [], None
                
            # Navigate to the timeline instructions
            timeline = tweet_data.get("data", {}).get("user_result_by_rest_id", {}).get("result", {}).get("profile_timeline_v2", {}).get("timeline", {})
            
            # Check for suspended or protected accounts
            if not timeline:
                user_result = tweet_data.get("data", {}).get("user_result_by_rest_id", {}).get("result", {})
                if user_result:
                    if user_result.get("__typename") == "UserUnavailable":
                        reason = user_result.get("reason", "Account unavailable")
                        logger.warning(f"User unavailable: {reason}")
                return [], None
            
            # Get all entries from the timeline
            all_entries = []
            
            # First check for pinned entry
            for instruction in timeline.get("instructions", []):
                if instruction.get("__typename") == "TimelinePinEntry":
                    entry = instruction.get("entry", {})
                    if entry:
                        all_entries.append(entry)
                
                # Then get regular entries
                elif instruction.get("__typename") == "TimelineAddEntries":
                    entries = instruction.get("entries", [])
                    for entry in entries:
                        # Skip cursor entries
                        if entry.get("content", {}).get("__typename") == "TimelineTimelineCursor":
                            cursor_type = entry.get("content", {}).get("cursor_type")
                            if cursor_type == "Bottom":
                                next_cursor = entry.get("content", {}).get("value")
                            continue
                        all_entries.append(entry)
            
            # Process each entry
            for entry in all_entries:
                tweet_content = entry.get("content", {}).get("content", {}).get("tweet_results", {}).get("result", {})
                
                # Skip if no tweet content
                if not tweet_content:
                    continue
                
                # Check if it's a retweet
                is_retweet = False
                tweet_to_parse = tweet_content
                
                # For retweets, get the original tweet
                if "retweeted_status_results" in tweet_content.get("legacy", {}):
                    is_retweet = True
                    tweet_to_parse = tweet_content.get("legacy", {}).get("retweeted_status_results", {}).get("result", {})
                
                # Extract tweet data
                legacy = tweet_to_parse.get("legacy", {})
                
                # Skip if no legacy data
                if not legacy:
                    continue
                
                # Get tweet text
                text = legacy.get("full_text", "")
                
                # Clean URLs from text
                urls = legacy.get("entities", {}).get("urls", [])
                for url in urls:
                    text = text.replace(url.get("url", ""), url.get("display_url", ""))
                
                # Get engagement metrics
                likes = legacy.get("favorite_count", 0)
                retweets = legacy.get("retweet_count", 0)
                replies = legacy.get("reply_count", 0)
                quotes = legacy.get("quote_count", 0)
                
                # Get tweet date
                created_at = legacy.get("created_at", "")
                try:
                    date_obj = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
                    date_str = date_obj.strftime("%Y-%m-%d %H:%M")
                except:
                    date_str = created_at
                
                # Add to tweets list
                tweets.append({
                    "text": text,
                    "date": date_str,
                    "likes": likes,
                    "retweets": retweets,
                    "replies": replies,
                    "quotes": quotes,
                    "total_engagement": likes + retweets + replies + quotes,
                    "is_retweet": is_retweet
                })
            
            return tweets, next_cursor
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from API: {str(e)}")
            return [], None
        except Exception as e:
            logger.error(f"Error parsing tweet data: {str(e)}")
            return [], None

    async def get_user_id_from_username(self, username: str, session: aiohttp.ClientSession) -> Optional[str]:
        """
        Get the Twitter user ID from a username.
        
        Args:
            username: Twitter username (without @)
            session: aiohttp ClientSession
            
        Returns:
            User ID string or None if not found
        """
        semaphore = self._get_semaphore()
        async with semaphore:
            endpoint = f"/UsernameToUserId?username={username}"
            url = f"https://{self.api_host}{endpoint}"
            
            try:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        logger.error(f"Error fetching user ID for {username}: {response.status}")
                        return None
                        
                    json_str = await response.text()
                    data = json.loads(json_str)
                    
                    # Extract the ID string
                    return data.get("id_str")
            except Exception as e:
                logger.error(f"Error fetching user ID for {username}: {str(e)}")
                return None

    @staticmethod
    def create_session(connector_limit: int = MAX_CONCURRENT_REQUESTS, 
                       timeout: int = DEFAULT_FETCH_TIMEOUT) -> aiohttp.ClientSession:
        """
        Create a properly configured aiohttp ClientSession with optimized connection settings.
        
        Args:
            connector_limit: Maximum number of concurrent connections
            timeout: Timeout in seconds
            
        Returns:
            Configured aiohttp ClientSession
        """
        # Using a higher connection limit that matches our MAX_CONCURRENT_REQUESTS setting
        conn = aiohttp.TCPConnector(
            limit=connector_limit, 
            force_close=False,  # Changed to False for connection reuse
            ttl_dns_cache=600,  # Increased DNS cache TTL for better performance
            use_dns_cache=True,
            ssl=False  # Disable SSL verification for better performance (only if your API endpoint doesn't require it)
        )
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        return aiohttp.ClientSession(connector=conn, timeout=timeout_obj)