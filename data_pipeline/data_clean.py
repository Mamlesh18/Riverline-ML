import pandas as pd
import numpy as np
import re
import emoji
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from collections import defaultdict
import hashlib

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

class DataCleaner:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.company_accounts = set()
        
    def clean_raw_data(self, df):
        df = df.copy()
        
        df['tweet_id'] = df['tweet_id'].astype(str)
        df['author_id'] = df['author_id'].astype(str)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['inbound'] = df['inbound'].fillna(True)
        
        df['response_tweet_id'] = df['response_tweet_id'].fillna('').astype(str)
        df['in_response_to_tweet_id'] = df['in_response_to_tweet_id'].fillna('').astype(str)
        
        df['text'] = df['text'].fillna('')
        df['text'] = df['text'].apply(self.clean_text)
        
        df = df.drop_duplicates(subset=['tweet_id'])
        df = df[df['text'].str.len() > 0]
        
        self._identify_company_accounts(df)
        
        return df
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ''
        
        text = emoji.demojize(text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'@\w+', '@USER', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _identify_company_accounts(self, df):
        outbound_authors = df[df['inbound'] == False]['author_id'].unique()
        self.company_accounts.update(outbound_authors)
    
    def build_conversations(self, df):
        conversations = defaultdict(list)
        conversation_metadata = {}
        
        # Create a mapping for faster lookups
        tweet_lookup = df.set_index('tweet_id').to_dict('index')
        
        for _, row in df.iterrows():
            conv_id = self._find_conversation_id_fixed(row, tweet_lookup)
            conversations[conv_id].append(row)
        
        processed_conversations = {}
        
        for conv_id, messages in conversations.items():
            if len(messages) > 0:
                sorted_messages = sorted(messages, key=lambda x: x['created_at'])
                processed_conversations[conv_id] = sorted_messages
                
                metadata = self._extract_conversation_metadata(conv_id, sorted_messages)
                conversation_metadata[conv_id] = metadata
        
        return processed_conversations, conversation_metadata
    
    def _find_conversation_id_fixed(self, tweet, tweet_lookup, visited=None):
        """
        Fixed version that prevents infinite recursion and handles circular references
        """
        if visited is None:
            visited = set()
        
        current_tweet_id = tweet['tweet_id']
        
        # Prevent infinite recursion by tracking visited tweets
        if current_tweet_id in visited:
            return current_tweet_id
        
        visited.add(current_tweet_id)
        
        # If this tweet is not responding to anything, it's the root
        in_response_to = tweet.get('in_response_to_tweet_id', '')
        if not in_response_to or in_response_to == '' or in_response_to == 'nan':
            return current_tweet_id
        
        # Look for the parent tweet
        if in_response_to in tweet_lookup:
            parent_tweet = tweet_lookup[in_response_to]
            return self._find_conversation_id_fixed(parent_tweet, tweet_lookup, visited)
        else:
            # Parent tweet not found in this chunk, current tweet becomes root
            return current_tweet_id
    
    def _extract_conversation_metadata(self, conv_id, messages):
        customer_msgs = [m for m in messages if m['inbound']]
        agent_msgs = [m for m in messages if not m['inbound']]
        
        all_participants = set([m['author_id'] for m in messages])
        customer_text = ' '.join([m['text'] for m in customer_msgs])
        
        metadata = {
            'conversation_id': conv_id,
            'root_tweet_id': messages[0]['tweet_id'],
            'participant_ids': ','.join(all_participants),
            'start_time': messages[0]['created_at'],
            'last_update_time': messages[-1]['created_at'],
            'message_count': len(messages),
            'is_resolved': int(self._check_if_resolved(messages)),
            'sentiment_score': float(self._calculate_sentiment(customer_text)),
            'urgency_score': float(self._calculate_urgency(customer_text)),
            'complexity_score': float(self._calculate_complexity(messages, customer_text))
        }
        
        return metadata
    
    def _check_if_resolved(self, messages):
        if len(messages) < 2:
            return False
        
        last_message = messages[-1]
        if not last_message['inbound']:
            resolution_patterns = [
                r'resolved', r'fixed', r'solved', r'thank you', r'thanks',
                r'glad.*help', r'happy.*help', r'issue.*closed'
            ]
            last_text = last_message['text'].lower()
            if any(re.search(pattern, last_text) for pattern in resolution_patterns):
                return True
        
        time_diff = messages[-1]['created_at'] - messages[0]['created_at']
        if time_diff.total_seconds() / 3600 > 24 and not messages[-1]['inbound']:
            return True
        
        return False
    
    def _calculate_sentiment(self, text):
        if not text:
            return 0.0
        scores = self.sia.polarity_scores(text)
        return scores['compound']
    
    def _calculate_urgency(self, text):
        if not text:
            return 0.0
        
        text_lower = text.lower()
        urgent_keywords = [
            'urgent', 'asap', 'immediately', 'emergency', 'critical',
            'now', 'help', 'please help', 'need help', 'right now'
        ]
        
        urgency_score = sum(1 for keyword in urgent_keywords if keyword in text_lower)
        return min(urgency_score / 3, 1.0)
    
    def _calculate_complexity(self, messages, customer_text):
        if not messages:
            return 0.0
        
        factors = []
        
        factors.append(min(len(messages) / 10, 1.0))
        
        avg_msg_length = np.mean([len(m['text']) for m in messages])
        factors.append(min(avg_msg_length / 200, 1.0))
        
        technical_terms = [
            'error', 'bug', 'system', 'server', 'database', 'api',
            'integration', 'technical', 'issue', 'problem', 'not working'
        ]
        tech_score = sum(1 for term in technical_terms if term in customer_text.lower())
        factors.append(min(tech_score / 5, 1.0))
        
        return np.mean(factors)
    
    def prepare_messages_for_storage(self, conversation_id, messages):
        prepared_messages = []
        
        for idx, msg in enumerate(messages):
            prepared_msg = {
                'conversation_id': conversation_id,
                'tweet_id': msg['tweet_id'],
                'author_id': msg['author_id'],
                'is_customer': msg['inbound'],
                'message_text': msg['text'],
                'created_at': msg['created_at'],
                'sequence_number': idx
            }
            prepared_messages.append(prepared_msg)
        
        return pd.DataFrame(prepared_messages)