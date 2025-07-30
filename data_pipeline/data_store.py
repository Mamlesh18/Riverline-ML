import sqlite3
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataStore:
    def __init__(self, db_name='riverline.db'):
        self.db_path = db_name
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and create tables"""
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()
        
    def _create_tables(self):
        """Create minimal database tables"""
        cursor = self.conn.cursor()
        
        # Raw tweets table - exact copy of CSV structure
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_tweets (
                tweet_id TEXT PRIMARY KEY,
                author_id TEXT NOT NULL,
                inbound BOOLEAN,
                created_at TIMESTAMP,
                text TEXT,
                response_tweet_id TEXT,
                in_response_to_tweet_id TEXT,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Grouped conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS grouped_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                tweet_id TEXT,
                author_id TEXT,
                is_customer BOOLEAN,
                message_text TEXT,
                created_at TIMESTAMP,
                response_tweet_id TEXT,
                in_response_to_tweet_id TEXT,
                sequence_number INTEGER,
                FOREIGN KEY (tweet_id) REFERENCES raw_tweets(tweet_id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversation_id ON grouped_conversations(conversation_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_author_id ON raw_tweets(author_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON raw_tweets(created_at)')
        
        self.conn.commit()
        print("Database tables created successfully")
    
    def insert_raw_tweets(self, df):
        """Insert cleaned tweets into raw_tweets table"""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy['created_at'] = df_copy['created_at'].astype(str)
            df_copy['created_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Insert using pandas to_sql
            df_copy.to_sql('raw_tweets', self.conn, if_exists='replace', index=False)
            self.conn.commit()
            
            print(f"Inserted {len(df)} tweets into raw_tweets table")
            
        except Exception as e:
            logger.error(f"Error inserting raw tweets: {str(e)}")
            raise
    
    def insert_conversations(self, conversations):
        """Insert grouped conversations with sequential conversation IDs"""
        try:
            cursor = self.conn.cursor()
            
            # Clear existing conversation data
            cursor.execute('DELETE FROM grouped_conversations')
            
            # Assign sequential conversation IDs starting from 1
            conversation_id = 1
            
            for original_conv_id, messages in conversations.items():
                # Skip single-message conversations that don't have responses
                if len(messages) == 1:
                    # Check if this single message has any children
                    msg = messages[0]
                    has_children = any(
                        other_messages for other_conv_id, other_messages in conversations.items() 
                        if other_conv_id != original_conv_id and 
                        any(m['in_response_to_tweet_id'] == msg['tweet_id'] for m in other_messages)
                    )
                    if not has_children:
                        continue
                
                for seq_num, message in enumerate(messages):
                    cursor.execute('''
                        INSERT INTO grouped_conversations 
                        (conversation_id, tweet_id, author_id, is_customer, message_text, 
                         created_at, response_tweet_id, in_response_to_tweet_id, sequence_number)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        conversation_id,  # Use sequential ID instead of tweet ID
                        message['tweet_id'],
                        message['author_id'],
                        message['is_customer'],
                        message['message_text'],
                        message['created_at'].strftime('%Y-%m-%d %H:%M:%S'),
                        message['response_tweet_id'],
                        message['in_response_to_tweet_id'],
                        seq_num
                    ))
                
                conversation_id += 1  # Increment for next conversation
            
            self.conn.commit()
            print(f"Inserted {conversation_id - 1} multi-message conversations into grouped_conversations table")
            
        except Exception as e:
            logger.error(f"Error inserting conversations: {str(e)}")
            raise
    
    def get_conversations_with_messages(self, limit=10, order_by='row'):
        """Retrieve full conversations with their messages for display"""
        
        if order_by == 'row':
            # Simple row order - first conversations in database
            query = '''
                SELECT conversation_id, tweet_id, author_id, is_customer, 
                       message_text, created_at, sequence_number,
                       response_tweet_id, in_response_to_tweet_id
                FROM grouped_conversations
                WHERE conversation_id IN (
                    SELECT DISTINCT conversation_id 
                    FROM grouped_conversations 
                    ORDER BY conversation_id
                    LIMIT ?
                )
                ORDER BY conversation_id, sequence_number
            '''
        elif order_by == 'size':
            # Order by conversation size (most messages first)
            query = '''
                SELECT conversation_id, tweet_id, author_id, is_customer, 
                       message_text, created_at, sequence_number,
                       response_tweet_id, in_response_to_tweet_id
                FROM grouped_conversations
                WHERE conversation_id IN (
                    SELECT conversation_id 
                    FROM grouped_conversations 
                    GROUP BY conversation_id
                    ORDER BY COUNT(*) DESC
                    LIMIT ?
                )
                ORDER BY conversation_id, sequence_number
            '''
        else:
            # Default to row order
            query = '''
                SELECT conversation_id, tweet_id, author_id, is_customer, 
                       message_text, created_at, sequence_number,
                       response_tweet_id, in_response_to_tweet_id
                FROM grouped_conversations
                WHERE conversation_id IN (
                    SELECT DISTINCT conversation_id 
                    FROM grouped_conversations 
                    ORDER BY conversation_id
                    LIMIT ?
                )
                ORDER BY conversation_id, sequence_number
            '''
        
        df = pd.read_sql_query(query, self.conn, params=(limit,))
        
        # Group by conversation_id and maintain order
        conversations = {}
        for conv_id, group in df.groupby('conversation_id'):
            conversations[conv_id] = group.to_dict('records')
        
        return conversations
    
    def get_conversation_stats(self):
        """Get detailed statistics about conversation sizes"""
        query = '''
            SELECT 
                conversation_id,
                COUNT(*) as message_count,
                MIN(created_at) as start_time,
                MAX(created_at) as end_time,
                COUNT(DISTINCT author_id) as participants
            FROM grouped_conversations
            GROUP BY conversation_id
            ORDER BY message_count DESC
        '''
        
        stats_df = pd.read_sql_query(query, self.conn)
        
        print("\n" + "="*60)
        print("CONVERSATION SIZE DISTRIBUTION")
        print("="*60)
        
        # Show size distribution
        size_counts = stats_df['message_count'].value_counts().sort_index()
        print("Messages per conversation:")
        for size, count in size_counts.items():
            print(f"  {size} messages: {count} conversations")
        
        print(f"\nTop 10 largest conversations:")
        print(stats_df.head(10)[['conversation_id', 'message_count', 'participants']].to_string(index=False))
        
        return stats_df
    
    def get_basic_stats(self):
        """Get basic statistics about the processed data"""
        stats = {}
        
        # Raw tweets stats
        cursor = self.conn.cursor()
        
        stats['total_tweets'] = cursor.execute('SELECT COUNT(*) FROM raw_tweets').fetchone()[0]
        stats['total_conversations'] = cursor.execute('SELECT COUNT(DISTINCT conversation_id) FROM grouped_conversations').fetchone()[0]
        stats['customer_messages'] = cursor.execute('SELECT COUNT(*) FROM grouped_conversations WHERE is_customer = 1').fetchone()[0]
        stats['agent_messages'] = cursor.execute('SELECT COUNT(*) FROM grouped_conversations WHERE is_customer = 0').fetchone()[0]
        
        # Average messages per conversation
        avg_messages = cursor.execute('''
            SELECT AVG(msg_count) FROM (
                SELECT COUNT(*) as msg_count 
                FROM grouped_conversations 
                GROUP BY conversation_id
            )
        ''').fetchone()[0]
        stats['avg_messages_per_conversation'] = round(avg_messages, 2) if avg_messages else 0
        
        return stats
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("Database connection closed")