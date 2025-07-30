import pandas as pd
import numpy as np
import re
from datetime import datetime

class MessageGenerator:
    """
    Intelligent message generation for NBA actions
    
    Features:
    - Channel-specific message templates
    - Sentiment-aware tone adjustment
    - Personalization based on customer behavior
    - Escalation-appropriate messaging
    """
    
    def __init__(self):
        self.message_templates = self._define_message_templates()
        self.tone_adjustments = self._define_tone_adjustments()
        self.personalization_rules = self._define_personalization_rules()
    
    def _define_message_templates(self):
        """Define base message templates for each channel and scenario"""
        return {
            'twitter_dm_reply': {
                'quick_resolution': {
                    'positive': "Hi! Thanks for reaching out. We're here to help resolve this quickly. Could you share more details about what you're experiencing?",
                    'neutral': "Hello! We've received your message and want to help. Can you provide more information about the issue you're facing?",
                    'negative': "Hi there. We understand this is frustrating, and we're committed to making it right. Let's work together to resolve this issue."
                },
                'standard_support': {
                    'positive': "Thanks for contacting us! We're looking into your request and will have an update for you shortly.",
                    'neutral': "Hello! We've received your inquiry and our team is reviewing it. We'll get back to you with next steps.",
                    'negative': "We hear you and want to help. Our team is prioritizing your case and will respond with a solution soon."
                },
                'frustrated_customer': {
                    'positive': "We appreciate your patience. Let's get this sorted out for you right away.",
                    'neutral': "We understand your concern and are here to help. What specific issue can we address for you?",
                    'negative': "We sincerely apologize for the frustration. Your experience matters to us, and we're taking immediate action to resolve this."
                },
                'urgent_technical': {
                    'positive': "We've flagged this as a priority technical issue. Our specialists are investigating and will update you within the hour.",
                    'neutral': "This has been escalated to our technical team for immediate attention. We'll have an update shortly.",
                    'negative': "We understand the urgency and have our top technical specialists working on this now. You'll hear from us within 30 minutes."
                }
            },
            
            'email_reply': {
                'complex_inquiry': {
                    'positive': "Thank you for your detailed inquiry. We're preparing a comprehensive response with step-by-step guidance and will send it within the next 2 hours.",
                    'neutral': "We've received your detailed request and appreciate the information provided. Our team is researching this thoroughly to give you the most accurate response.",
                    'negative': "We understand this is a complex situation that's been causing frustration. We're dedicating focused attention to provide you with a complete solution via email within 2 hours."
                },
                'technical_documentation': {
                    'positive': "We'll send you detailed technical documentation and troubleshooting steps that should resolve this issue. Expected delivery: within 1 hour.",
                    'neutral': "Our technical team is preparing comprehensive documentation for your specific case. You'll receive detailed instructions via email shortly.",
                    'negative': "We know technical issues can be incredibly frustrating. We're preparing detailed, easy-to-follow documentation that will be sent to your email within the hour."
                },
                'patient_detailed': {
                    'positive': "Thank you for the comprehensive information you've provided. This helps us give you the most accurate solution. A detailed response is being prepared.",
                    'neutral': "We appreciate the thorough details you've shared. Our team is using this information to craft a complete solution for your situation.",
                    'negative': "Thank you for your patience and detailed explanation. We're using all this information to ensure we provide you with the right solution the first time."
                }
            },
            
            'scheduling_phone_call': {
                'escalation_needed': {
                    'positive': "To ensure we address all your concerns thoroughly, we'd like to schedule a call with one of our senior specialists within the next 2 hours.",
                    'neutral': "Given the complexity of your situation, a phone conversation would be most effective. A senior support specialist will call you within the next business hour.",
                    'negative': "We want to address your concerns personally and make this right. A senior team member will call you within the next hour to discuss a comprehensive solution."
                },
                'frustrated_customer': {
                    'positive': "We value your business and want to ensure you're completely satisfied. Let's schedule a call to discuss this directly.",
                    'neutral': "To better understand and resolve your concerns, we'd like to speak with you directly. A team lead will call you shortly.",
                    'negative': "We hear your frustration and take this seriously. Our customer success manager will call you within 30 minutes to personally address your concerns."
                },
                'complex_resolution': {
                    'positive': "This appears to require detailed discussion. A specialist will call you to walk through the solution step-by-step.",
                    'neutral': "Given the complexity, a phone conversation will be more efficient. Our technical specialist will call you within 2 hours.",
                    'negative': "We understand this has been complicated and frustrating. A senior specialist will call you shortly to provide a complete resolution."
                },
                'ping_pong': {
                    'positive': "To resolve this more efficiently, let's have a quick call to discuss your needs directly. A specialist will reach out within the hour.",
                    'neutral': "Multiple back-and-forth messages can be time-consuming. A brief call will help us resolve this faster. Expect a call within 2 hours.",
                    'negative': "We know the back-and-forth has been frustrating. A direct conversation will be much more efficient. Our team will call you within 30 minutes."
                }
            }
        }
    
    def _define_tone_adjustments(self):
        """Define tone adjustments based on sentiment"""
        return {
            'very_negative': {  # sentiment < -0.6
                'empathy_phrases': [
                    "We sincerely apologize", "We understand your frustration", 
                    "This is clearly unacceptable", "We take full responsibility"
                ],
                'urgency_words': ["immediately", "right away", "prioritizing", "urgent attention"],
                'reassurance': ["We're committed to making this right", "Your experience matters to us"]
            },
            'negative': {  # sentiment -0.6 to -0.2
                'empathy_phrases': [
                    "We understand your concern", "We hear you", 
                    "We appreciate your patience", "We want to help"
                ],
                'urgency_words': ["quickly", "promptly", "soon", "priority"],
                'reassurance': ["We're here to help", "Let's get this resolved"]
            },
            'neutral': {  # sentiment -0.2 to 0.2
                'professional_phrases': [
                    "Thank you for contacting us", "We've received your request", 
                    "We're looking into this", "Our team is reviewing"
                ],
                'action_words': ["reviewing", "investigating", "working on", "addressing"]
            },
            'positive': {  # sentiment > 0.2
                'appreciation_phrases': [
                    "Thank you for reaching out", "We appreciate your message", 
                    "Thanks for bringing this to our attention"
                ],
                'collaborative_words': ["together", "partnership", "support", "assist"]
            }
        }
    
    def _define_personalization_rules(self):
        """Define rules for personalizing messages based on customer behavior"""
        return {
            'high_urgency': {
                'add_phrases': ["We understand this is urgent", "Immediate attention"],
                'time_commitment': True
            },
            'technical_customer': {
                'add_phrases': ["Technical details", "Comprehensive documentation", "Step-by-step"],
                'formal_tone': True
            },
            'repeat_customer': {
                'add_phrases': ["We value your continued trust", "As a valued customer"],
                'prioritize': True
            },
            'long_conversation': {
                'add_phrases': ["We know this has been ongoing", "To move this forward efficiently"],
                'escalate_tone': True
            }
        }
    
    def generate_message(self, customer_data, channel):
        """
        Generate personalized message for the customer and channel
        
        Args:
            customer_data: dict with customer features and conversation history
            channel: selected communication channel
            
        Returns:
            str: personalized message
        """
        # Determine customer scenario
        scenario = self._classify_customer_scenario(customer_data)
        
        # Determine sentiment tone
        sentiment_tone = self._classify_sentiment_tone(customer_data.get('sentiment_score', 0))
        
        # Get base template
        base_message = self._get_base_template(channel, scenario, sentiment_tone)
        
        # Apply personalization
        personalized_message = self._apply_personalization(base_message, customer_data)
        
        # Apply tone adjustments
        final_message = self._apply_tone_adjustments(personalized_message, sentiment_tone, customer_data)
        
        return final_message
    
    def _classify_customer_scenario(self, customer_data):
        """Classify customer scenario based on behavior patterns"""
        urgency = customer_data.get('urgency_score', 0)
        complexity = customer_data.get('complexity_score', 0)
        message_count = customer_data.get('message_count', 1)
        sentiment = customer_data.get('sentiment_score', 0)
        cohort = customer_data.get('cohort', 'standard_support')
        
        # High-level scenario classification
        if sentiment < -0.4 or cohort == 'frustrated_customer':
            return 'frustrated_customer'
        elif urgency > 0.7 or cohort == 'urgent_technical':
            return 'urgent_technical'
        elif complexity > 0.6 or cohort == 'patient_detailed':
            return 'complex_inquiry'
        elif message_count > 6 or cohort == 'ping_pong':
            return 'ping_pong'
        elif cohort == 'escalation_needed':
            return 'escalation_needed'
        elif message_count <= 3 and complexity < 0.4:
            return 'quick_resolution'
        else:
            return 'standard_support'
    
    def _classify_sentiment_tone(self, sentiment_score):
        """Classify sentiment for tone selection"""
        if sentiment_score < -0.6:
            return 'very_negative'
        elif sentiment_score < -0.2:
            return 'negative'
        elif sentiment_score > 0.2:
            return 'positive'
        else:
            return 'neutral'
    
    def _get_base_template(self, channel, scenario, sentiment_tone):
        """Get base message template"""
        channel_templates = self.message_templates.get(channel, {})
        
        # Try exact scenario match first
        if scenario in channel_templates:
            scenario_templates = channel_templates[scenario]
            if sentiment_tone in scenario_templates:
                return scenario_templates[sentiment_tone]
            else:
                # Fallback to neutral tone for scenario
                return scenario_templates.get('neutral', scenario_templates.get('positive', ''))
        
        # Fallback to standard support
        standard_templates = channel_templates.get('standard_support', {})
        return standard_templates.get(sentiment_tone, standard_templates.get('neutral', 
            "Thank you for contacting us. We're here to help and will respond shortly."))
    
    def _apply_personalization(self, base_message, customer_data):
        """Apply customer-specific personalization"""
        message = base_message
        
        # Check personalization conditions
        urgency = customer_data.get('urgency_score', 0)
        complexity = customer_data.get('complexity_score', 0)
        message_count = customer_data.get('message_count', 1)
        
        # High urgency personalization
        if urgency > 0.7:
            rules = self.personalization_rules['high_urgency']
            for phrase in rules['add_phrases']:
                if phrase.lower() not in message.lower():
                    message = f"{phrase}. {message}"
                    break
        
        # Technical customer personalization
        if complexity > 0.6:
            rules = self.personalization_rules['technical_customer']
            message = message.replace("solution", "technical solution")
            message = message.replace("help", "provide detailed assistance")
        
        # Long conversation personalization
        if message_count > 6:
            rules = self.personalization_rules['long_conversation']
            for phrase in rules['add_phrases']:
                if phrase.lower() not in message.lower():
                    message = f"{phrase}. {message}"
                    break
        
        return message
    
    def _apply_tone_adjustments(self, message, sentiment_tone, customer_data):
        """Apply tone adjustments based on sentiment and situation"""
        if sentiment_tone in ['negative', 'very_negative']:
            tone_rules = self.tone_adjustments[sentiment_tone]
            
            # Add empathy phrases for negative sentiment
            empathy_phrases = tone_rules.get('empathy_phrases', [])
            if empathy_phrases and not any(phrase.lower() in message.lower() for phrase in empathy_phrases):
                message = f"{empathy_phrases[0]}. {message}"
            
            # Add reassurance for very negative sentiment
            if sentiment_tone == 'very_negative':
                reassurance = tone_rules.get('reassurance', [])
                if reassurance:
                    message = f"{message} {reassurance[0]}."
        
        # Add time commitments for urgent cases
        urgency = customer_data.get('urgency_score', 0)
        if urgency > 0.7 and 'within' not in message.lower():
            if 'twitter_dm_reply' in str(customer_data.get('channel', '')):
                message = message.replace('shortly', 'within 30 minutes')
            elif 'email_reply' in str(customer_data.get('channel', '')):
                message = message.replace('shortly', 'within 2 hours')
        
        return message
    
    def generate_follow_up_message(self, original_message, customer_response=None, escalation_level=1):
        """Generate follow-up message for continued conversations"""
        follow_up_templates = {
            1: "Thank you for the additional information. We're reviewing this and will have an update for you shortly.",
            2: "We appreciate your patience as we work on this. A senior specialist is now handling your case.",
            3: "We understand this has taken longer than expected. A manager will personally review and respond to your case immediately."
        }
        
        base_followup = follow_up_templates.get(escalation_level, follow_up_templates[1])
        
        # Customize based on customer response sentiment if provided
        if customer_response:
            if any(word in customer_response.lower() for word in ['frustrated', 'angry', 'upset', 'terrible']):
                base_followup = f"We sincerely apologize for the continued difficulty. {base_followup}"
        
        return base_followup
    
    def get_message_analytics(self, message):
        """Analyze generated message for quality metrics"""
        return {
            'word_count': len(message.split()),
            'character_count': len(message),
            'sentiment_keywords': self._count_sentiment_keywords(message),
            'urgency_indicators': self._count_urgency_indicators(message),
            'personalization_score': self._calculate_personalization_score(message)
        }
    
    def _count_sentiment_keywords(self, message):
        """Count positive/negative sentiment keywords in message"""
        positive_words = ['thank', 'appreciate', 'help', 'resolve', 'solution', 'support']
        negative_words = ['sorry', 'apologize', 'frustration', 'issue', 'problem', 'difficulty']
        
        message_lower = message.lower()
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        return {
            'positive': positive_count,
            'negative': negative_count,
            'net_sentiment': positive_count - negative_count
        }
    
    def _count_urgency_indicators(self, message):
        """Count urgency indicators in message"""
        urgency_words = ['immediate', 'urgent', 'priority', 'quickly', 'soon', 'within', 'right away']
        message_lower = message.lower()
        return sum(1 for word in urgency_words if word in message_lower)
    
    def _calculate_personalization_score(self, message):
        """Calculate how personalized the message appears (0-1 scale)"""
        personalization_indicators = [
            'your', 'you', 'specific', 'detailed', 'comprehensive', 
            'understand', 'situation', 'case', 'experience'
        ]
        
        message_lower = message.lower()
        matches = sum(1 for indicator in personalization_indicators if indicator in message_lower)
        return min(matches / len(personalization_indicators), 1.0)