import os
import json
import requests
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DecisionEngine:
    """
    NBA Decision Engine using Gemini LLM for intelligent channel selection
    
    This engine follows NBA (Next Best Action) principles:
    1. Customer Context: Uses chat history, sentiment, and behavior patterns
    2. Channel Optimization: Selects best channel based on customer preferences and urgency
    3. Timing Intelligence: Determines optimal send time for maximum engagement
    4. Message Personalization: Crafts messages tailored to customer situation
    """
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set!")
        
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        # NBA Channel Rules based on customer behavior research
        self.channel_rules = {
            'twitter_dm_reply': {
                'best_for': ['quick_responses', 'public_complaints', 'social_presence'],
                'urgency_match': ['low', 'medium'],
                'customer_types': ['impatient', 'public_complainers']
            },
            'email_reply': {
                'best_for': ['detailed_explanations', 'documentation', 'follow_ups'],
                'urgency_match': ['low', 'medium'],
                'customer_types': ['polite', 'technical', 'detailed_oriented']
            },
            'scheduling_phone_call': {
                'best_for': ['complex_issues', 'high_value_customers', 'escalations'],
                'urgency_match': ['high', 'critical'],
                'customer_types': ['frustrated', 'confused', 'escalated']
            }
        }
    
    def generate_recommendation(self, customer_data):
        """Generate NBA recommendation using Gemini AI"""
        try:
            # Create NBA analysis prompt
            prompt = self._create_nba_prompt(customer_data)
            
            # Call Gemini API
            response = self._call_gemini_api(prompt)
            
            if response:
                # Parse and validate response
                recommendation = self._parse_gemini_response(response)
                
                # Add customer context
                if recommendation:
                    recommendation['customer_id'] = customer_data['customer_id']
                    recommendation['conversation_id'] = customer_data['conversation_id']
                    recommendation['chat_history'] = customer_data.get('chat_history', '')
                
                return recommendation
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating recommendation for customer {customer_data['customer_id']}: {str(e)}")
            return None

    
    def _create_nba_prompt(self, customer_data):
        """Create NBA analysis prompt for Gemini"""
        
        # Parse tags for additional context
        # Parse tags for additional context
        tags = []
        try:
            tags = json.loads(customer_data['tags']) if customer_data['tags'] else []
        except Exception:
            tags = []
        
        # Format chat history for better readability in prompt
        chat_history = customer_data.get('chat_history', 'No chat history available')
        if chat_history and ' ### ' in chat_history:
            # Convert database format to readable format for the LLM
            formatted_history = chat_history.replace(' ### ', '\n---\n')
        else:
            formatted_history = chat_history
            print(formatted_history)
        
        prompt = f"""
    You are an expert Next Best Action (NBA) engine for customer support. Your goal is to maximize issue resolution by selecting the optimal channel, timing, and message for each customer.

    CUSTOMER CONTEXT:
    - Customer ID: {customer_data['customer_id']}
    - Conversation ID: {customer_data['conversation_id']}
    - Current Status: UNRESOLVED
    - Nature of Request: {customer_data['nature_of_request']}
    - Customer Sentiment: {customer_data['customer_sentiment']}
    - Urgency Level: {customer_data['urgency_level']}
    - Conversation Type: {customer_data['conversation_type']}
    - Customer Behavior: {customer_data['customer_behavior']}
    - Customer Cohorts: {customer_data['customer_cohorts']}
    - Tags: {tags}
    - Latest Message: {customer_data['latest_message']}
    - Total Messages: {customer_data['total_messages']}
    - Last Interaction: {customer_data['latest_interaction']}

    COMPLETE CHAT HISTORY:
    {chat_history}

    AVAILABLE CHANNELS:
    1. twitter_dm_reply: Best for quick responses, public visibility, immediate acknowledgment
    2. email_reply: Best for detailed explanations, documentation, formal communication
    3. scheduling_phone_call: Best for complex issues, high urgency, personal touch

    NBA DECISION CRITERIA:
    - Urgency Level: High/Critical → Phone, Medium → Email/Twitter, Low → Email
    - Customer Sentiment: Frustrated/Angry → Phone (personal touch), Neutral/Positive → Email/Twitter
    - Issue Complexity: Technical/Complex → Phone, Simple → Email/Twitter
    - Customer Behavior: Impatient → Twitter (fast), Technical → Email (detailed), Confused → Phone (guidance)
    - Chat History Analysis: Look for patterns, escalation signs, previous failed attempts

    TIMING RULES:
    - Critical/High Urgency: Within 1-2 hours
    - Medium Urgency: Within 4-8 hours
    - Low Urgency: Within 24 hours
    - Consider business hours: 9 AM - 6 PM local time

    MESSAGE PRINCIPLES:
    - Acknowledge the customer's frustration/concern
    - Show empathy and understanding
    - Provide clear next steps
    - Set realistic expectations
    - Match the tone to the channel and customer sentiment
    - Reference specific points from chat history when relevant

    ISSUE STATUS DETERMINATION:
    - "resolved": Use when your message provides a complete solution that fully addresses the customer's issue without requiring further customer response
    - "pending_customer_reply": Use when your message requires customer action, scheduling confirmation, additional information, or when the issue needs follow-up

    Please analyze this customer situation and provide your NBA recommendation in the following JSON format:

    {{
        "channel": "twitter_dm_reply | email_reply | scheduling_phone_call",
        "send_time": "2025-01-15T14:30:00Z",
        "message": "Your personalized message here",
        "reasoning": "Provide a detailed, specific analysis of this particular customer's situation. Reference specific elements from their chat history, sentiment, behavior patterns, and why this exact combination of channel/timing/message is optimal for THIS customer. Avoid generic templates - make it specific to their case.",
        "issue_status": "resolved | pending_customer_reply"
    }}

    IMPORTANT: 
    1. Your reasoning should be specific to this customer's actual situation, not a generic template
    2. Reference specific elements from their chat history and behavior
    3. Explain why this solution is optimal for their particular case
    4. Respond ONLY with the JSON object, no additional text.
    """
        return prompt

    
    def _call_gemini_api(self, prompt):
        """Call Gemini API with NBA prompt"""
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,  # Slightly higher for creative messaging
                "maxOutputTokens": 800
            }
        }
        
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': self.api_key
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    generated_text = result['candidates'][0]['content']['parts'][0]['text']
                    return generated_text
                else:
                    logger.error("No candidates in Gemini response")
                    return None
            else:
                logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            return None
    
    def _parse_gemini_response(self, response_text):
        """Parse Gemini NBA response"""
        try:
        # Clean response
            response_text = response_text.strip()
            
            # Remove markdown if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            recommendation = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['channel', 'send_time', 'message', 'reasoning', 'issue_status']
            for field in required_fields:
                if field not in recommendation:
                    logger.warning(f"Missing required field: {field}")
                    return None
            
            # Validate channel
            valid_channels = ['twitter_dm_reply', 'email_reply', 'scheduling_phone_call']
            if recommendation['channel'] not in valid_channels:
                logger.warning(f"Invalid channel: {recommendation['channel']}")
                return None
            
            # Validate issue status
            valid_statuses = ['resolved', 'pending_customer_reply']
            if recommendation['issue_status'] not in valid_statuses:
                logger.warning(f"Invalid issue status: {recommendation['issue_status']}")
                recommendation['issue_status'] = 'pending_customer_reply'  # Default fallback
            
            return recommendation
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse NBA JSON response: {e}")
            logger.error(f"Response text: {response_text}")
            return None
        
        except Exception as e:
            logger.error(f"Error parsing NBA response: {e}")
            return None
    
    def _create_fallback_recommendation(self, customer_data):
        """Create fallback recommendation when Gemini fails"""
        # Simple rule-based fallback
        urgency = customer_data['urgency_level']
        sentiment = customer_data['customer_sentiment']
        
        if urgency in ['high', 'critical'] or sentiment in ['frustrated', 'angry']:
            channel = 'scheduling_phone_call'
            reasoning = f"Phone call recommended due to {urgency} urgency and {sentiment} sentiment requiring immediate personal attention."
            issue_status = 'pending_customer_reply'
        elif sentiment == 'technical' or customer_data['customer_behavior'] == 'technical':
            channel = 'email_reply'
            reasoning = "Email selected for technical customer requiring detailed written solution and documentation."
            issue_status = 'pending_customer_reply'
        else:
            channel = 'twitter_dm_reply'
            reasoning = f"Twitter DM chosen for quick acknowledgment of {urgency} priority {customer_data['nature_of_request']} issue."
            issue_status = 'resolved'
        
        # Simple timing: 2 hours from now
        send_time = (datetime.now() + timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        return {
            'channel': channel,
            'send_time': send_time,
            'message': f"Hi! We're sorry for the inconvenience. Let us help resolve your {customer_data['nature_of_request']} issue.",
            'reasoning': reasoning,
            'issue_status': issue_status
        }