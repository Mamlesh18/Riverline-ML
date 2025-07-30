import re
import logging
from typing import Dict, Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from observe_user_behavior.gemini_analyzer import GeminiAnalyzer

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception:
    pass

logger = logging.getLogger(__name__)

class HybridAnalyzer:
    """
    Hybrid Conversation Analyzer that combines:
    1. Rule-Based Analysis: For resolution detection, urgency, and basic classification
    2. ML Models: For sentiment analysis, nature classification, and behavior prediction
    3. Gemini LLM: For complex reasoning, tags generation, and edge cases
    
    This approach is:
    - Cost-effective (fewer API calls)
    - Faster (local ML models)
    - More reliable (rule-based fallbacks)
    - Accurate (LLM for complex cases)
    """
    
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.gemini_analyzer = GeminiAnalyzer()
        
        # Initialize ML models
        self.nature_classifier = None
        self.behavior_classifier = None
        self.urgency_classifier = None
        
        # Rule-based patterns
        self.resolution_patterns = self._init_resolution_patterns()
        self.urgency_patterns = self._init_urgency_patterns()
        self.nature_patterns = self._init_nature_patterns()
        self.behavior_patterns = self._init_behavior_patterns()
        
        # Initialize and train ML models
        self._initialize_ml_models()
    
    def _init_resolution_patterns(self):
        """Initialize rule-based patterns for resolution detection"""
        return {
            'resolved_positive': [
                r'thank(?:s| you)', r'resolved', r'fixed', r'solved', r'working now',
                r'perfect', r'great', r'awesome', r'excellent', r'appreciate',
                r'got it working', r'all set', r'no longer', r'issue.*closed'
            ],
            'resolved_confirmation': [
                r'that.*work(?:s|ed)', r'problem.*gone', r'issue.*resolved',
                r'everything.*fine', r'all.*good', r'sorted.*out'
            ],
            'unresolved_indicators': [
                r'still.*(?:not|broken|problem)', r'doesn\'t work', r'not working',
                r'same.*(?:issue|problem)', r'still.*(?:issue|problem)',
                r'need.*help', r'frustrated', r'angry'
            ]
        }
    
    def _init_urgency_patterns(self):
        """Initialize patterns for urgency detection"""
        return {
            'critical': [
                r'emergency', r'urgent', r'asap', r'immediately', r'critical',
                r'broken.*down', r'can\'t.*access', r'system.*down'
            ],
            'high': [
                r'need.*soon', r'important', r'priority', r'quickly',
                r'losing.*money', r'deadline', r'right.*now'
            ],
            'medium': [
                r'when.*possible', r'convenient', r'sometime',
                r'help.*when', r'get.*back'
            ],
            'low': [
                r'no.*rush', r'whenever', r'eventually', r'curious',
                r'wondering', r'question'
            ]
        }
    
    def _init_nature_patterns(self):
        """Initialize patterns for request nature classification"""
        return {
            'billing': [
                r'bill', r'charge', r'payment', r'invoice', r'refund',
                r'subscription', r'credit', r'cost', r'price', r'fee'
            ],
            'technical': [
                r'error', r'bug', r'api', r'code', r'server', r'database',
                r'integration', r'not.*work', r'broken', r'crash'
            ],
            'account': [
                r'login', r'password', r'access', r'account', r'profile',
                r'settings', r'permissions', r'username'
            ],
            'product_info': [
                r'how.*to', r'what.*is', r'feature', r'documentation',
                r'tutorial', r'guide', r'explain', r'understand'
            ],
            'complaint': [
                r'terrible', r'worst', r'awful', r'disappointed',
                r'frustrated', r'angry', r'complaint', r'unacceptable'
            ]
        }
    
    def _init_behavior_patterns(self):
        """Initialize patterns for customer behavior classification"""
        return {
            'polite': [
                r'please', r'thank', r'sorry', r'excuse', r'appreciate',
                r'kindly', r'could.*you', r'would.*you'
            ],
            'impatient': [
                r'how.*long', r'still.*waiting', r'hours.*ago', r'urgent',
                r'asap', r'immediately', r'right.*now'
            ],
            'technical': [
                r'api', r'code', r'server', r'database', r'endpoint',
                r'json', r'xml', r'sql', r'http', r'ssl'
            ],
            'confused': [
                r'don\'t.*understand', r'confused', r'not.*sure',
                r'what.*mean', r'explain', r'unclear'
            ],
            'angry': [
                r'terrible', r'worst', r'awful', r'furious', r'outraged',
                r'unacceptable', r'ridiculous', r'pathetic'
            ],
            'cooperative': [
                r'understand', r'makes.*sense', r'will.*try', r'happy.*to',
                r'of.*course', r'absolutely', r'certainly'
            ]
        }
    
    def _initialize_ml_models(self):
        """Initialize and train ML models with synthetic data"""
        # For demo purposes, we'll create simple models
        # In production, these would be trained on real historical data
        
        # Nature classifier
        self.nature_classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        
        # Behavior classifier  
        self.behavior_classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=500, stop_words='english')),
            ('classifier', LogisticRegression())
        ])
        
        # Train with synthetic data (in production, use real data)
        self._train_synthetic_models()
    
    def _train_synthetic_models(self):
        """Train models with synthetic data for demo purposes"""
        # Synthetic training data for nature classification
        nature_data = [
            ("I can't login to my account", "account"),
            ("My bill seems incorrect", "billing"),
            ("The API is returning errors", "technical"),
            ("How do I use this feature", "product_info"),
            ("Your service is terrible", "complaint"),
            ("Password reset not working", "account"),
            ("Charged twice for subscription", "billing"),
            ("Server timeout errors", "technical"),
            ("What does this button do", "product_info"),
            ("Very disappointed with support", "complaint")
        ]
        
        nature_texts, nature_labels = zip(*nature_data)
        self.nature_classifier.fit(nature_texts, nature_labels)
        
        # Synthetic training data for behavior classification
        behavior_data = [
            ("Please help me with this issue", "polite"),
            ("I need this fixed RIGHT NOW", "impatient"),
            ("The REST API endpoint is failing", "technical"),
            ("I don't understand what you mean", "confused"),
            ("This is absolutely terrible service", "angry"),
            ("I understand, I'll try that solution", "cooperative"),
            ("Could you please assist me", "polite"),
            ("How much longer will this take", "impatient"),
            ("JSON response format is invalid", "technical"),
            ("What exactly should I do", "confused")
        ]
        
        behavior_texts, behavior_labels = zip(*behavior_data)
        self.behavior_classifier.fit(behavior_texts, behavior_labels)
    
    def analyze_conversation(self, conversation_text: str) -> Optional[Dict]:
        """Main analysis function that combines all approaches"""
        try:
            # Extract customer messages for analysis
            customer_messages = self._extract_customer_messages(conversation_text)
            
            if not customer_messages:
                return self._create_fallback_analysis()
            
            # 1. RULE-BASED ANALYSIS (Fast and reliable)
            resolution_analysis = self._analyze_resolution_rules(conversation_text)
            urgency_analysis = self._analyze_urgency_rules(customer_messages)
            
            # 2. ML MODEL ANALYSIS (Accurate and fast)
            nature_analysis = self._analyze_nature_ml(customer_messages)
            behavior_analysis = self._analyze_behavior_ml(customer_messages)
            sentiment_analysis = self._analyze_sentiment_ml(customer_messages)
            
            # 3. HYBRID APPROACH FOR TAGS (Rule-based + ML)
            tags = self._generate_tags_hybrid(conversation_text, nature_analysis, 
                                            sentiment_analysis, urgency_analysis)
            
            # 4. GEMINI LLM FOR COMPLEX CASES (Only when needed)
            conversation_type = self._determine_conversation_type(conversation_text)
            
            # Use Gemini only for complex cases or when confidence is low
            use_gemini = self._should_use_gemini(resolution_analysis, nature_analysis, 
                                               behavior_analysis, sentiment_analysis)
            
            if use_gemini:
                logger.info("Using Gemini LLM for complex analysis")
                gemini_result = self.gemini_analyzer.analyze_conversation(conversation_text)
                if gemini_result:
                    # Merge Gemini insights with rule-based/ML results
                    return self._merge_analyses(resolution_analysis, urgency_analysis,
                                              nature_analysis, behavior_analysis,
                                              sentiment_analysis, tags, conversation_type,
                                              gemini_result)
            
            # 5. COMBINE ALL ANALYSES
            final_analysis = {
                'is_resolved': resolution_analysis['is_resolved'],
                'resolution_confidence': resolution_analysis['confidence'],
                'tags': tags,
                'nature_of_request': nature_analysis['nature'],
                'customer_sentiment': sentiment_analysis['sentiment'],
                'urgency_level': urgency_analysis['level'],
                'conversation_type': conversation_type,
                'customer_behavior': behavior_analysis['behavior'],
                'resolution_summary': self._generate_summary(resolution_analysis, nature_analysis),
                'key_issues': self._extract_key_issues(customer_messages),
                'agent_performance': self._assess_agent_performance(conversation_text)
            }
            
            return final_analysis
            
        except Exception as e:
            logger.error(f"Error in hybrid analysis: {str(e)}")
            return self._create_fallback_analysis()
    
    def _extract_customer_messages(self, conversation_text: str) -> List[str]:
        """Extract only customer messages from conversation"""
        customer_messages = []
        lines = conversation_text.split(' | ')
        
        for line in lines:
            if 'CUSTOMER:' in line:
                # Extract message after "CUSTOMER: author_id:"
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    message = parts[2].strip()
                    customer_messages.append(message)
        
        return customer_messages
    
    def _analyze_resolution_rules(self, conversation_text: str) -> Dict:
        """Rule-based resolution analysis"""
        text_lower = conversation_text.lower()
        
        # Count resolution indicators
        resolved_score = 0
        unresolved_score = 0
        
        for pattern in self.resolution_patterns['resolved_positive']:
            resolved_score += len(re.findall(pattern, text_lower))
        
        for pattern in self.resolution_patterns['resolved_confirmation']:
            resolved_score += len(re.findall(pattern, text_lower)) * 0.8
        
        for pattern in self.resolution_patterns['unresolved_indicators']:
            unresolved_score += len(re.findall(pattern, text_lower))
        
        # Determine resolution status
        if resolved_score > unresolved_score and resolved_score > 0:
            is_resolved = True
            confidence = min(resolved_score / (resolved_score + unresolved_score + 1), 0.95)
        else:
            is_resolved = False
            confidence = min(unresolved_score / (resolved_score + unresolved_score + 1), 0.85)
        
        return {
            'is_resolved': is_resolved,
            'confidence': confidence,
            'resolved_indicators': resolved_score,
            'unresolved_indicators': unresolved_score
        }
    
    def _analyze_urgency_rules(self, customer_messages: List[str]) -> Dict:
        """Rule-based urgency analysis"""
        all_text = ' '.join(customer_messages).lower()
        
        urgency_scores = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for level, patterns in self.urgency_patterns.items():
            for pattern in patterns:
                urgency_scores[level] += len(re.findall(pattern, all_text))
        
        # Determine urgency level
        if urgency_scores['critical'] > 0:
            level = 'critical'
        elif urgency_scores['high'] > 0:
            level = 'high'
        elif urgency_scores['low'] > urgency_scores['medium']:
            level = 'low'
        else:
            level = 'medium'
        
        return {'level': level, 'scores': urgency_scores}
    
    def _analyze_nature_ml(self, customer_messages: List[str]) -> Dict:
        """ML-based nature classification"""
        if not customer_messages:
            return {'nature': 'general', 'confidence': 0.5}
        
        # Combine all customer messages
        text = ' '.join(customer_messages)
        
        try:
            # Use trained ML model
            nature_pred = self.nature_classifier.predict([text])[0]
            
            # Get prediction confidence (if available)
            try:
                confidence = max(self.nature_classifier.predict_proba([text])[0])
            except Exception:
                confidence = 0.7
            
            return {'nature': nature_pred, 'confidence': confidence}
            
        except Exception as e:
            logger.warning(f"ML nature classification failed: {e}")
            # Fallback to rule-based
            return self._analyze_nature_rules(text)
    
    def _analyze_nature_rules(self, text: str) -> Dict:
        """Rule-based fallback for nature classification"""
        text_lower = text.lower()
        nature_scores = {}
        
        for nature, patterns in self.nature_patterns.items():
            score = sum(len(re.findall(pattern, text_lower)) for pattern in patterns)
            nature_scores[nature] = score
        
        if nature_scores:
            best_nature = max(nature_scores, key=nature_scores.get)
            confidence = nature_scores[best_nature] / (sum(nature_scores.values()) + 1)
        else:
            best_nature = 'general'
            confidence = 0.5
        
        return {'nature': best_nature, 'confidence': confidence}
    
    def _analyze_behavior_ml(self, customer_messages: List[str]) -> Dict:
        """ML-based behavior classification"""
        if not customer_messages:
            return {'behavior': 'neutral', 'confidence': 0.5}
        
        text = ' '.join(customer_messages)
        
        try:
            behavior_pred = self.behavior_classifier.predict([text])[0]
            
            try:
                confidence = max(self.behavior_classifier.predict_proba([text])[0])
            except Exception:
                confidence = 0.7
            
            return {'behavior': behavior_pred, 'confidence': confidence}
            
        except Exception as e:
            logger.warning(f"ML behavior classification failed: {e}")
            return self._analyze_behavior_rules(text)
    
    def _analyze_behavior_rules(self, text: str) -> Dict:
        """Rule-based fallback for behavior classification"""
        text_lower = text.lower()
        behavior_scores = {}
        
        for behavior, patterns in self.behavior_patterns.items():
            score = sum(len(re.findall(pattern, text_lower)) for pattern in patterns)
            behavior_scores[behavior] = score
        
        if behavior_scores:
            best_behavior = max(behavior_scores, key=behavior_scores.get)
            confidence = behavior_scores[best_behavior] / (sum(behavior_scores.values()) + 1)
        else:
            best_behavior = 'neutral'
            confidence = 0.5
        
        return {'behavior': best_behavior, 'confidence': confidence}
    
    def _analyze_sentiment_ml(self, customer_messages: List[str]) -> Dict:
        """ML-based sentiment analysis using NLTK VADER"""
        if not customer_messages:
            return {'sentiment': 'neutral', 'confidence': 0.5}
        
        text = ' '.join(customer_messages)
        
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            compound = scores['compound']
            
            if compound >= 0.05:
                sentiment = 'positive'
                confidence = abs(compound)
            elif compound <= -0.05:
                if compound <= -0.5:
                    sentiment = 'frustrated'
                else:
                    sentiment = 'negative'
                confidence = abs(compound)
            else:
                sentiment = 'neutral'
                confidence = 1 - abs(compound)
            
            return {'sentiment': sentiment, 'confidence': confidence, 'scores': scores}
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5}
    
    def _generate_tags_hybrid(self, conversation_text: str, nature_analysis: Dict,
                             sentiment_analysis: Dict, urgency_analysis: Dict) -> List[str]:
        """Generate tags using hybrid approach"""
        tags = []
        
        # Add nature-based tags
        nature = nature_analysis['nature']
        if nature != 'general':
            tags.append(f"{nature}_issue")
        
        # Add sentiment-based tags
        sentiment = sentiment_analysis['sentiment']
        if sentiment != 'neutral':
            tags.append(f"{sentiment}_customer")
        
        # Add urgency-based tags
        urgency = urgency_analysis['level']
        if urgency in ['critical', 'high']:
            tags.append(f"{urgency}_urgency")
        
        # Add conversation pattern tags
        text_lower = conversation_text.lower()
        
        if 'escalat' in text_lower:
            tags.append('escalation')
        if any(word in text_lower for word in ['first', 'new', 'never']):
            tags.append('first_time_customer')
        if any(word in text_lower for word in ['again', 'before', 'previous']):
            tags.append('repeat_customer')
        if 'thank' in text_lower or 'appreciate' in text_lower:
            tags.append('polite_customer')
        
        return tags[:5]  # Limit to 5 tags
    
    def _should_use_gemini(self, resolution_analysis: Dict, nature_analysis: Dict,
                          behavior_analysis: Dict, sentiment_analysis: Dict) -> bool:
        """Decide whether to use Gemini LLM for additional analysis"""
        
        # Use Gemini if confidence is low in any key area
        low_confidence_threshold = 0.6
        
        if (resolution_analysis['confidence'] < low_confidence_threshold or
            nature_analysis.get('confidence', 1.0) < low_confidence_threshold or
            behavior_analysis.get('confidence', 1.0) < low_confidence_threshold):
            return True
        
        # Use Gemini for complex cases
        if (nature_analysis['nature'] in ['complaint', 'technical'] and
            sentiment_analysis['sentiment'] in ['frustrated', 'angry']):
            return True
        
        return False
    
    def _determine_conversation_type(self, conversation_text: str) -> str:
        """Determine conversation type using rules"""
        message_count = len(conversation_text.split(' | '))
        text_lower = conversation_text.lower()
        
        if message_count <= 2:
            return 'simple_inquiry'
        elif 'escalat' in text_lower or 'manager' in text_lower:
            return 'escalation'
        elif message_count > 6:
            return 'complex_issue'
        else:
            return 'follow_up'
    
    def _generate_summary(self, resolution_analysis: Dict, nature_analysis: Dict) -> str:
        """Generate resolution summary"""
        if resolution_analysis['is_resolved']:
            return f"Customer's {nature_analysis['nature']} issue was successfully resolved"
        else:
            return f"Customer's {nature_analysis['nature']} issue remains unresolved"
    
    def _extract_key_issues(self, customer_messages: List[str]) -> List[str]:
        """Extract key issues mentioned by customer"""
        all_text = ' '.join(customer_messages).lower()
        
        issue_keywords = [
            'error', 'problem', 'issue', 'bug', 'broken', 'fail', 'wrong',
            'charge', 'bill', 'payment', 'login', 'access', 'slow', 'down'
        ]
        
        found_issues = []
        for keyword in issue_keywords:
            if keyword in all_text:
                found_issues.append(keyword)
        
        return found_issues[:3]  # Return top 3 issues
    
    def _assess_agent_performance(self, conversation_text: str) -> str:
        """Assess agent performance based on conversation patterns"""
        text_lower = conversation_text.lower()
        
        positive_indicators = ['resolved', 'helped', 'thank', 'great', 'excellent']
        negative_indicators = ['frustrated', 'angry', 'terrible', 'awful', 'disappointed']
        
        positive_score = sum(1 for word in positive_indicators if word in text_lower)
        negative_score = sum(1 for word in negative_indicators if word in text_lower)
        
        if positive_score > negative_score and positive_score > 0:
            return 'good'
        elif negative_score > positive_score:
            return 'poor'
        else:
            return 'fair'
    
    def _merge_analyses(self, resolution_analysis: Dict, urgency_analysis: Dict,
                       nature_analysis: Dict, behavior_analysis: Dict,
                       sentiment_analysis: Dict, tags: List[str], 
                       conversation_type: str, gemini_result: Dict) -> Dict:
        """Merge rule-based/ML results with Gemini insights"""
        
        # Use Gemini for complex fields, keep rule-based for reliable ones
        merged = {
            'is_resolved': resolution_analysis['is_resolved'],  # Trust rule-based
            'resolution_confidence': resolution_analysis['confidence'],
            'tags': gemini_result.get('tags', tags),  # Use Gemini tags if available
            'nature_of_request': nature_analysis['nature'],  # Trust ML model
            'customer_sentiment': sentiment_analysis['sentiment'],  # Trust sentiment model
            'urgency_level': urgency_analysis['level'],  # Trust rule-based
            'conversation_type': conversation_type,
            'customer_behavior': behavior_analysis['behavior'],  # Trust ML model
            'resolution_summary': gemini_result.get('resolution_summary', 
                                                   self._generate_summary(resolution_analysis, nature_analysis)),
            'key_issues': gemini_result.get('key_issues', self._extract_key_issues([])),
            'agent_performance': gemini_result.get('agent_performance', 'fair')
        }
        
        return merged
    
    def _create_fallback_analysis(self) -> Dict:
        """Create fallback analysis when all methods fail"""
        return {
            'is_resolved': False,
            'resolution_confidence': 0.5,
            'tags': ['analysis_failed'],
            'nature_of_request': 'general',
            'customer_sentiment': 'neutral',
            'urgency_level': 'medium',
            'conversation_type': 'simple_inquiry',
            'customer_behavior': 'neutral',
            'resolution_summary': 'Unable to analyze conversation',
            'key_issues': ['unknown'],
            'agent_performance': 'fair'
        }