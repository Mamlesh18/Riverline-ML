import pandas as pd
import sqlite3
import json
from datetime import datetime
from observe_user_behavior.conversation_analyzer import ConversationAnalyzer
from observe_user_behavior.cohort_builder import CohortBuilder
from observe_user_behavior.visualizer import BehaviorVisualizer

class UserBehaviorAnalyzer:
    def __init__(self, db_path='riverline.db'):
        self.db_path = db_path
        self.analyzer = ConversationAnalyzer(db_path)
        self.cohort_builder = CohortBuilder()
        self.visualizer = BehaviorVisualizer()
        self.analysis_results = {}
        
    def run_analysis(self):
        print("Loading conversations from database...")
        conversations = self.analyzer.load_conversations()
        print(f"Loaded {len(conversations)} conversations")
        
        resolved_count = conversations['is_resolved'].sum()
        open_count = len(conversations) - resolved_count
        print(f"\nResolved conversations: {resolved_count}")
        print(f"Open conversations: {open_count}")
        
        print("\nExtracting advanced features...")
        features_df = self.analyzer.extract_advanced_features(conversations)
        
        if features_df.empty:
            print("No conversation features extracted. Check if conversation_messages table has data.")
            return None
        
        print(f"Extracted features for {len(features_df)} conversations")
        
        print("Identifying conversation patterns...")
        features_df, variance_explained = self.analyzer.identify_conversation_patterns(features_df)
        
        print("\nAssigning customer cohorts...")
        features_df = self.cohort_builder.assign_cohorts(features_df)
        
        print("Analyzing cohort statistics...")
        cohort_stats = self.cohort_builder.analyze_cohort_statistics(features_df)
        
        print("\nIdentifying resolution patterns...")
        resolution_patterns = self.cohort_builder.identify_resolution_patterns(features_df)
        
        print("\nCreating ML-based cohorts...")
        features_df, ml_cohort_profiles, silhouette_score = self.cohort_builder.create_ml_cohorts(features_df)
        
        self._save_analysis_results(features_df, cohort_stats, resolution_patterns, ml_cohort_profiles)
        
        print("\nGenerating visualizations...")
        try:
            self.visualizer.create_comprehensive_report(features_df, cohort_stats)
            print("Visualizations created successfully!")
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
        
        try:
            self._update_database_with_cohorts(features_df)
            print("Database updated with cohort assignments")
        except Exception as e:
            print(f"Warning: Could not update database with cohorts: {e}")
        
        return {
            'total_conversations': len(conversations),
            'resolved_conversations': resolved_count,
            'open_conversations': open_count,
            'cohort_distribution': cohort_stats.to_dict(),
            'resolution_patterns': resolution_patterns,
            'ml_cohort_profiles': ml_cohort_profiles,
            'features_dataframe': features_df
        }
    
    def _save_analysis_results(self, features_df, cohort_stats, resolution_patterns, ml_cohort_profiles):
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_conversations_analyzed': len(features_df),
            'resolved_conversations': int(features_df['is_resolved'].sum()),
            'open_conversations': int((features_df['is_resolved'] == 0).sum()),
            'cohort_statistics': cohort_stats.to_dict(),
            'resolution_patterns': resolution_patterns,
            'ml_cohort_profiles': ml_cohort_profiles
        }
        
        with open('user_behavior_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        features_df.to_csv('conversation_features.csv', index=False)
        print("Analysis results saved to files")
        
    def _update_database_with_cohorts(self, features_df):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(conversations)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'cohort' not in columns:
            cursor.execute('''
                ALTER TABLE conversations 
                ADD COLUMN cohort TEXT DEFAULT 'unassigned'
            ''')
        
        if 'ml_cohort' not in columns:
            cursor.execute('''
                ALTER TABLE conversations 
                ADD COLUMN ml_cohort INTEGER DEFAULT -1
            ''')
        
        # Update cohort assignments
        for _, row in features_df.iterrows():
            cursor.execute('''
                UPDATE conversations 
                SET cohort = ?, ml_cohort = ?
                WHERE conversation_id = ?
            ''', (row['cohort'], int(row['ml_cohort']), row['conversation_id']))
        
        conn.commit()
        conn.close()
        
    def get_cohort_recommendations(self):
        recommendations = {
            'quick_resolution': {
                'description': 'Simple queries resolved quickly',
                'recommendation': 'Use automated responses or chatbots for these cases'
            },
            'frustrated_customer': {
                'description': 'Customers showing negative sentiment or frustration',
                'recommendation': 'Prioritize with senior agents, consider phone call escalation'
            },
            'urgent_technical': {
                'description': 'Urgent issues with technical complexity',
                'recommendation': 'Route to technical specialists immediately'
            },
            'patient_detailed': {
                'description': 'Customers providing detailed information patiently',
                'recommendation': 'Provide comprehensive email responses with documentation'
            },
            'abandoned_conversation': {
                'description': 'Conversations where customer stopped responding',
                'recommendation': 'Send follow-up message or email to re-engage'
            },
            'ping_pong': {
                'description': 'Long back-and-forth conversations',
                'recommendation': 'Consider phone call to resolve more efficiently'
            },
            'escalation_needed': {
                'description': 'Complex issues with frustrated customers',
                'recommendation': 'Immediate escalation to senior support or management'
            },
            'standard_support': {
                'description': 'Standard support queries',
                'recommendation': 'Follow standard support procedures'
            }
        }
        
        return recommendations
    
    def export_cohort_summary(self):
        conn = sqlite3.connect(self.db_path)
        
        # Check if cohort column exists
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(conversations)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'cohort' not in columns:
            print("Cohort column not found in database. Running basic summary...")
            summary_query = '''
                SELECT 
                    'all_conversations' as cohort,
                    COUNT(*) as count,
                    SUM(CASE WHEN is_resolved = 1 THEN 1 ELSE 0 END) as resolved_count,
                    AVG(sentiment_score) as avg_sentiment,
                    AVG(urgency_score) as avg_urgency,
                    AVG(complexity_score) as avg_complexity,
                    AVG(message_count) as avg_messages
                FROM conversations
            '''
        else:
            summary_query = '''
                SELECT 
                    cohort,
                    COUNT(*) as count,
                    SUM(CASE WHEN is_resolved = 1 THEN 1 ELSE 0 END) as resolved_count,
                    AVG(sentiment_score) as avg_sentiment,
                    AVG(urgency_score) as avg_urgency,
                    AVG(complexity_score) as avg_complexity,
                    AVG(message_count) as avg_messages
                FROM conversations
                WHERE cohort IS NOT NULL
                GROUP BY cohort
                ORDER BY count DESC
            '''
        
        summary_df = pd.read_sql_query(summary_query, conn)
        summary_df['resolution_rate'] = summary_df['resolved_count'] / summary_df['count'] * 100
        
        summary_df.to_csv('cohort_summary.csv', index=False)
        conn.close()
        
        return summary_df
    
    def close(self):
        """Close all database connections"""
        try:
            self.analyzer.close()
        except:
            pass