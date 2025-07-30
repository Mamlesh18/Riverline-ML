import sqlite3
import pandas as pd
import json
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class CohortBuilder:
    def __init__(self):
        self.cohort_definitions = self._define_cohorts()
    
    def _define_cohorts(self):
        """Define cohort criteria"""
        return {
            "frustrated_billing_customers": {
                "name": "Frustrated Billing Customers",
                "description": "Customers with negative sentiment about billing issues",
                "criteria": {
                    "nature_of_request": "billing",
                    "customer_sentiment": ["negative", "frustrated"]
                }
            },
            "technical_power_users": {
                "name": "Technical Power Users", 
                "description": "Customers with technical issues showing technical behavior",
                "criteria": {
                    "nature_of_request": "technical",
                    "customer_behavior": "technical"
                }
            },
            "urgent_unresolved": {
                "name": "Urgent Unresolved Issues",
                "description": "High urgency conversations that remain unresolved",
                "criteria": {
                    "urgency_level": ["high", "critical"],
                    "is_resolved": False
                }
            },
            "satisfied_customers": {
                "name": "Satisfied Customers",
                "description": "Customers with positive sentiment and resolved issues",
                "criteria": {
                    "customer_sentiment": ["positive", "satisfied"],
                    "is_resolved": True
                }
            },
            "impatient_customers": {
                "name": "Impatient Customers",
                "description": "Customers showing impatient behavior patterns",
                "criteria": {
                    "customer_behavior": ["impatient", "angry"]
                }
            },
            "account_access_issues": {
                "name": "Account Access Issues",
                "description": "Customers having trouble accessing their accounts",
                "criteria": {
                    "nature_of_request": "account"
                }
            },
            "complex_escalations": {
                "name": "Complex Escalations",
                "description": "Complex issues that required escalation",
                "criteria": {
                    "conversation_type": "escalation",
                    "urgency_level": ["high", "critical"]
                }
            },
            "polite_inquiries": {
                "name": "Polite General Inquiries", 
                "description": "Polite customers with general questions",
                "criteria": {
                    "customer_behavior": ["polite", "cooperative"],
                    "nature_of_request": "general"
                }
            }
        }
    
    def create_cohorts(self, conn):
        """Create cohorts based on conversation analysis"""
        try:
            # Get all analyzed conversations
            analysis_df = pd.read_sql_query('''
                SELECT conversation_id, is_resolved, tags, nature_of_request, 
                       customer_sentiment, urgency_level, conversation_type, customer_behavior
                FROM conversation_analysis
            ''', conn)
            
            if len(analysis_df) == 0:
                logger.warning("No analyzed conversations found")
                return 0
            
            cursor = conn.cursor()
            
            # Clear existing cohorts
            cursor.execute('DELETE FROM customer_cohorts')
            cursor.execute('DELETE FROM conversation_cohort_mapping')
            
            cohorts_created = 0
            
            # Create each cohort
            for cohort_key, cohort_def in self.cohort_definitions.items():
                matching_conversations = self._find_matching_conversations(analysis_df, cohort_def['criteria'])
                
                if len(matching_conversations) > 0:
                    # Insert cohort
                    cursor.execute('''
                        INSERT INTO customer_cohorts (cohort_name, cohort_description, cohort_criteria, conversation_count)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        cohort_def['name'],
                        cohort_def['description'],
                        json.dumps(cohort_def['criteria']),
                        len(matching_conversations)
                    ))
                    
                    cohort_id = cursor.lastrowid
                    
                    # Insert conversation mappings
                    for conv_id in matching_conversations:
                        cursor.execute('''
                            INSERT INTO conversation_cohort_mapping (conversation_id, cohort_id)
                            VALUES (?, ?)
                        ''', (conv_id, cohort_id))
                    
                    cohorts_created += 1
                    print(f"Created cohort '{cohort_def['name']}' with {len(matching_conversations)} conversations")
            
            # Create dynamic cohorts based on tags
            tag_cohorts = self._create_tag_based_cohorts(analysis_df, cursor)
            cohorts_created += tag_cohorts
            
            conn.commit()
            print(f"Total cohorts created: {cohorts_created}")
            return cohorts_created
            
        except Exception as e:
            logger.error(f"Error creating cohorts: {str(e)}")
            return 0
    
    def _find_matching_conversations(self, df, criteria):
        """Find conversations matching cohort criteria"""
        mask = pd.Series([True] * len(df))
        
        for field, expected_values in criteria.items():
            if isinstance(expected_values, list):
                mask &= df[field].isin(expected_values)
            elif isinstance(expected_values, bool):
                mask &= (df[field] == expected_values)
            else:
                mask &= (df[field] == expected_values)
        
        return df[mask]['conversation_id'].tolist()
    
    def _create_tag_based_cohorts(self, analysis_df, cursor):
        """Create cohorts based on common tag patterns"""
        try:
            # Collect all tags
            all_tags = []
            for _, row in analysis_df.iterrows():
                try:
                    tags = json.loads(row['tags']) if row['tags'] else []
                    all_tags.extend(tags)
                except:
                    continue
            
            # Find most common tags
            tag_counts = Counter(all_tags)
            common_tags = [tag for tag, count in tag_counts.most_common(10) if count >= 3]
            
            cohorts_created = 0
            
            # Create cohorts for common tags
            for tag in common_tags:
                # Find conversations with this tag
                matching_convs = []
                for _, row in analysis_df.iterrows():
                    try:
                        tags = json.loads(row['tags']) if row['tags'] else []
                        if tag in tags:
                            matching_convs.append(row['conversation_id'])
                    except:
                        continue
                
                if len(matching_convs) >= 3:  # Only create cohort if at least 3 conversations
                    cohort_name = f"Tag: {tag.replace('_', ' ').title()}"
                    cohort_description = f"Conversations tagged with '{tag}'"
                    
                    cursor.execute('''
                        INSERT INTO customer_cohorts (cohort_name, cohort_description, cohort_criteria, conversation_count)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        cohort_name,
                        cohort_description,
                        json.dumps({"tag": tag}),
                        len(matching_convs)
                    ))
                    
                    cohort_id = cursor.lastrowid
                    
                    # Insert mappings
                    for conv_id in matching_convs:
                        cursor.execute('''
                            INSERT INTO conversation_cohort_mapping (conversation_id, cohort_id)
                            VALUES (?, ?)
                        ''', (conv_id, cohort_id))
                    
                    cohorts_created += 1
                    print(f"Created tag-based cohort '{cohort_name}' with {len(matching_convs)} conversations")
            
            return cohorts_created
            
        except Exception as e:
            logger.error(f"Error creating tag-based cohorts: {str(e)}")
            return 0
    
    def get_cohort_analysis(self, conn):
        """Get detailed cohort analysis"""
        try:
            # Get cohort summary
            cohorts_df = pd.read_sql_query('''
                SELECT c.*, COUNT(ccm.conversation_id) as actual_count
                FROM customer_cohorts c
                LEFT JOIN conversation_cohort_mapping ccm ON c.id = ccm.cohort_id
                GROUP BY c.id
                ORDER BY actual_count DESC
            ''', conn)
            
            print("\n" + "="*80)
            print("COHORT ANALYSIS SUMMARY")
            print("="*80)
            
            for _, cohort in cohorts_df.iterrows():
                print(f"\n🎯 {cohort['cohort_name']}")
                print(f"   📝 {cohort['cohort_description']}")
                print(f"   📊 Conversations: {cohort['actual_count']}")
                
                # Get sample conversations for this cohort
                sample_query = '''
                    SELECT ca.conversation_id, ca.customer_sentiment, ca.nature_of_request
                    FROM conversation_analysis ca
                    JOIN conversation_cohort_mapping ccm ON ca.conversation_id = ccm.conversation_id
                    WHERE ccm.cohort_id = ?
                    LIMIT 3
                '''
                
                sample_df = pd.read_sql_query(sample_query, conn, params=(cohort['id'],))
                if len(sample_df) > 0:
                    print(f"   🔍 Sample conversations: {list(sample_df['conversation_id'])}")
            
            return cohorts_df
            
        except Exception as e:
            logger.error(f"Error getting cohort analysis: {str(e)}")
            return pd.DataFrame()