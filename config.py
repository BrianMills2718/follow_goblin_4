"""
Configuration settings for the X Network Visualization application.
"""
import os
import streamlit as st
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# API Configuration
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = "twitter283.p.rapidapi.com"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Network Analysis Parameters
DEFAULT_CLOUTRANK_DAMPING = 0.85
DEFAULT_CLOUTRANK_EPSILON = 1e-8
DEFAULT_CLOUTRANK_MAX_ITER = 100

# UI Configuration
DEFAULT_MAX_ACCOUNTS = 200
DEFAULT_ACCOUNT_SIZE_FACTOR = 3.0
DEFAULT_LABEL_SIZE_FACTOR = 1.0
DEFAULT_BASE_SIZE = 100
DEFAULT_NODE_SPACING = 5.0  # Default node spacing factor for collision radius

# API Request Limits
MAX_CONCURRENT_REQUESTS = 50
DEFAULT_FETCH_TIMEOUT = 300  # seconds
DEFAULT_BATCH_SIZE = 20

# Default Filters
DEFAULT_FILTERS = {
    "statuses_range": (0, 1000000),
    "followers_range": (0, 10000000),
    "friends_range": (0, 10000000),
    "media_range": (0, 10000),
    "created_range": None,  # Will be set dynamically
    "require_location": False,
    "selected_locations": [],
    "require_blue_verified": False,
    "verified_option": "Any",
    "require_website": False,
    "business_account_option": "Any"
}

# Community Detection
DEFAULT_NUM_COMMUNITIES = 5
MIN_ACCOUNTS_FOR_COMMUNITIES = 3