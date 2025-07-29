import sqlite3
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataStore:
    def __init__(self, db_name='riverline.db'):
        self.db_path = f'{db_name}'
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()
    
    def _create_tables(self):
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_tweets (
                tweet_id TEXT PRIMARY KEY,
                author_id TEXT NOT NULL,
                inbound BOOLEAN,
                created_at TIMESTAMP,
                text TEXT,
                response_tweet_id TEXT,
                in_response_to_tweet_id TEXT,
                ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                root_tweet_id TEXT,
                participant_ids TEXT,
                start_time TIMESTAMP,
                last_update_time TIMESTAMP,
                message_count INTEGER,
                is_resolved BOOLEAN DEFAULT FALSE,
                sentiment_score REAL,
                urgency_score REAL,
                complexity_score REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                tweet_id TEXT,
                author_id TEXT,
                is_customer BOOLEAN,
                message_text TEXT,
                created_at TIMESTAMP,
                sequence_number INTEGER,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id),
                FOREIGN KEY (tweet_id) REFERENCES raw_tweets(tweet_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ingestion_log (
                batch_id TEXT PRIMARY KEY,
                file_hash TEXT UNIQUE,
                records_processed INTEGER,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_author_id ON raw_tweets(author_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON raw_tweets(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversation_id ON conversation_messages(conversation_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_response_tweet ON raw_tweets(response_tweet_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_in_response_to ON raw_tweets(in_response_to_tweet_id)')
        
        self.conn.commit()
    
    def check_if_processed(self, file_hash):
        cursor = self.conn.cursor()
        result = cursor.execute(
            'SELECT batch_id FROM ingestion_log WHERE file_hash = ? AND status = ?',
            (file_hash, 'completed')
        ).fetchone()
        return result is not None
    
    def log_ingestion_start(self, batch_id, file_hash):
        cursor = self.conn.cursor()
        cursor.execute(
            '''INSERT INTO ingestion_log (batch_id, file_hash, start_time, status) 
               VALUES (?, ?, ?, ?)''',
            (batch_id, file_hash, datetime.now(), 'in_progress')
        )
        self.conn.commit()
    
    def log_ingestion_complete(self, batch_id, records_processed):
        cursor = self.conn.cursor()
        cursor.execute(
            '''UPDATE ingestion_log 
               SET end_time = ?, records_processed = ?, status = ?
               WHERE batch_id = ?''',
            (datetime.now(), records_processed, 'completed', batch_id)
        )
        self.conn.commit()
    
    def insert_tweets_batch(self, tweets_df):
        tweets_df = tweets_df.copy()
        tweets_df['created_at'] = tweets_df['created_at'].astype(str)
        tweets_df['ingestion_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            tweets_df.to_sql('raw_tweets', self.conn, if_exists='append', index=False, method='multi', chunksize=1000)
        except Exception as e:
            for _, row in tweets_df.iterrows():
                try:
                    self.conn.execute('''
                        INSERT OR IGNORE INTO raw_tweets 
                        (tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (row['tweet_id'], row['author_id'], row['inbound'], 
                         row['created_at'], row['text'], row['response_tweet_id'], 
                         row['in_response_to_tweet_id']))
                except:
                    continue
            self.conn.commit()
    
    def insert_conversation(self, conversation_data):
        cursor = self.conn.cursor()
        
        # Convert timestamps to strings
        conv_data = conversation_data.copy()
        if 'start_time' in conv_data:
            conv_data['start_time'] = str(conv_data['start_time'])
        if 'last_update_time' in conv_data:
            conv_data['last_update_time'] = str(conv_data['last_update_time'])
            
        cursor.execute(
            '''INSERT OR REPLACE INTO conversations 
               (conversation_id, root_tweet_id, participant_ids, start_time, 
                last_update_time, message_count, is_resolved, sentiment_score, 
                urgency_score, complexity_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            tuple(conv_data.values())
        )
        self.conn.commit()
    
    def insert_conversation_messages(self, messages_df):
        messages_df = messages_df.copy()
        messages_df['created_at'] = messages_df['created_at'].astype(str)
        
        try:
            messages_df.to_sql('conversation_messages', self.conn, if_exists='append', index=False, method='multi', chunksize=500)
        except Exception as e:
            for _, row in messages_df.iterrows():
                try:
                    cursor = self.conn.cursor()
                    cursor.execute('''
                        INSERT INTO conversation_messages 
                        (conversation_id, tweet_id, author_id, is_customer, message_text, created_at, sequence_number)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (row['conversation_id'], row['tweet_id'], row['author_id'], 
                         row['is_customer'], row['message_text'], row['created_at'], 
                         row['sequence_number']))
                except:
                    continue
            self.conn.commit()
    
    def get_conversation_by_id(self, conversation_id):
        query = '''
            SELECT cm.*, rt.text, rt.created_at
            FROM conversation_messages cm
            JOIN raw_tweets rt ON cm.tweet_id = rt.tweet_id
            WHERE cm.conversation_id = ?
            ORDER BY cm.sequence_number
        '''
        return pd.read_sql_query(query, self.conn, params=(conversation_id,))
    
    def get_unresolved_conversations(self, limit=None):
        query = '''
            SELECT * FROM conversations 
            WHERE is_resolved = FALSE
        '''
        if limit:
            query += f' LIMIT {limit}'
        return pd.read_sql_query(query, self.conn)
    
    def update_conversation_status(self, conversation_id, is_resolved):
        cursor = self.conn.cursor()
        cursor.execute(
            'UPDATE conversations SET is_resolved = ? WHERE conversation_id = ?',
            (is_resolved, conversation_id)
        )
        self.conn.commit()
    
    def close(self):
        if self.conn:
            self.conn.close()