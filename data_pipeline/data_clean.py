import pandas as pd
import re
from collections import defaultdict

class DataCleaner:
    def __init__(self):
        pass
        
    def clean_data(self, df):
        """Clean the raw CSV data following ML best practices"""
        df = df.copy()
        
        # Convert data types
        df['tweet_id'] = df['tweet_id'].astype(str)
        df['author_id'] = df['author_id'].astype(str) 
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['inbound'] = df['inbound'].fillna(True).astype(bool)
        
        # Handle response fields - convert decimals to integers then strings
        df['response_tweet_id'] = df['response_tweet_id'].fillna('').apply(self._clean_tweet_id)
        df['in_response_to_tweet_id'] = df['in_response_to_tweet_id'].fillna('').apply(self._clean_tweet_id)
        
        # Clean text field
        df['text'] = df['text'].fillna('')
        df['text'] = df['text'].apply(self.clean_text)
        
        # Remove duplicates and empty texts
        df = df.drop_duplicates(subset=['tweet_id'])
        df = df[df['text'].str.len() > 0]
        
        # Sort by created_at for proper conversation flow
        df = df.sort_values('created_at')
        
        # Debug: Check response relationships
        has_responses = df[df['in_response_to_tweet_id'] != '']['in_response_to_tweet_id'].nunique()
        total_with_responses = len(df[df['in_response_to_tweet_id'] != ''])
        
        print(f"Data cleaning complete. Removed {len(pd.read_csv('./dataset/twcs.csv')) - len(df)} invalid records")
        print(f"Tweets with response relationships: {total_with_responses}")
        print(f"Unique parent tweets referenced: {has_responses}")
        
        # Show sample of cleaned response IDs
        sample_responses = df[df['in_response_to_tweet_id'] != ''][['tweet_id', 'in_response_to_tweet_id']].head()
        print("Sample response relationships after cleaning:")
        print(sample_responses)
        
        return df
    
    def _clean_tweet_id(self, value):
        """Convert tweet ID from decimal format to integer string"""
        if pd.isna(value) or value == '' or value == 'nan':
            return ''
        
        try:
            # Convert to float first (handles decimals), then to int, then to string
            return str(int(float(value)))
        except (ValueError, TypeError):
            return str(value)
    
    def clean_text(self, text):
        """Clean text data using ML best practices"""
        if not isinstance(text, str):
            return ''
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Normalize mentions (keep @USER format for consistency)
        text = re.sub(r'@\w+', '@USER', text)
        
        # Remove hashtags (optional - keep if needed for analysis)
        text = re.sub(r'#\w+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Trim and return
        return text.strip()
    
    def build_conversations(self, df):
        """Group tweets into conversations based on response relationships"""
        conversations = defaultdict(list)
        
        # Create lookup dictionaries for faster access - keep tweet_id as column
        tweet_lookup = {}
        for _, row in df.iterrows():
            tweet_lookup[str(row['tweet_id'])] = row.to_dict()
        
        # Track which tweets we've already assigned to conversations
        assigned_tweets = set()
        
        print("Building conversation threads...")
        print(f"Sample tweet_lookup keys: {list(tweet_lookup.keys())[:5]}")
        
        for _, row in df.iterrows():
            tweet_id = str(row['tweet_id'])
            
            # Skip if already assigned to a conversation
            if tweet_id in assigned_tweets:
                continue
            
            # Find the root of this conversation thread
            root_id = self._find_conversation_root(row, tweet_lookup)
            
            # Get all tweets in this conversation thread
            thread_tweets = self._get_conversation_thread(root_id, df, tweet_lookup)
            
            # Add all tweets in this thread to the conversation
            for thread_tweet_id in thread_tweets:
                thread_tweet_id_str = str(thread_tweet_id)
                if thread_tweet_id_str in tweet_lookup and thread_tweet_id_str not in assigned_tweets:
                    tweet_data = tweet_lookup[thread_tweet_id_str]
                    conversations[root_id].append({
                        'tweet_id': str(tweet_data['tweet_id']),
                        'author_id': str(tweet_data['author_id']),
                        'is_customer': bool(tweet_data['inbound']),
                        'message_text': str(tweet_data['text']),
                        'created_at': tweet_data['created_at'],
                        'response_tweet_id': str(tweet_data['response_tweet_id']),
                        'in_response_to_tweet_id': str(tweet_data['in_response_to_tweet_id'])
                    })
                    assigned_tweets.add(thread_tweet_id_str)
        
        # Sort messages within each conversation by timestamp
        for conv_id in conversations:
            conversations[conv_id] = sorted(
                conversations[conv_id], 
                key=lambda x: x['created_at']
            )
        
        print(f"Grouped {len(df)} tweets into {len(conversations)} conversations")
        
        return dict(conversations)
    
    def _get_conversation_thread(self, root_id, df, tweet_lookup):
        """Get all tweets that belong to a conversation thread starting from root"""
        thread_tweets = set()
        root_id_str = str(root_id)
        
        def add_tweet_and_children(tweet_id):
            tweet_id_str = str(tweet_id)
            if tweet_id_str in thread_tweets:
                return
            
            thread_tweets.add(tweet_id_str)
            
            # Find all tweets that respond to this tweet
            children = df[
                (df['in_response_to_tweet_id'].astype(str) == tweet_id_str)
            ]
            
            for _, child in children.iterrows():
                add_tweet_and_children(str(child['tweet_id']))
        
        # Start from root and recursively add all children
        add_tweet_and_children(root_id_str)
        
        return thread_tweets
    
    def _find_conversation_root(self, tweet, tweet_lookup, visited=None):
        """Find the root tweet ID for a conversation thread"""
        if visited is None:
            visited = set()
        
        # Handle both dict and Series input
        if hasattr(tweet, 'get'):
            current_tweet_id = str(tweet.get('tweet_id', ''))
            in_response_to = str(tweet.get('in_response_to_tweet_id', ''))
        else:
            current_tweet_id = str(tweet['tweet_id'])
            in_response_to = str(tweet['in_response_to_tweet_id'])
        
        # Prevent infinite loops
        if current_tweet_id in visited:
            return current_tweet_id
        
        visited.add(current_tweet_id)
        
        # If not responding to anything, this is the root
        if not in_response_to or in_response_to in ['', 'nan', 'None']:
            return current_tweet_id
        
        # Look for parent tweet
        if in_response_to in tweet_lookup:
            parent_tweet = tweet_lookup[in_response_to]
            return self._find_conversation_root(parent_tweet, tweet_lookup, visited)
        else:
            # Parent not found, current tweet becomes root
            return current_tweet_id