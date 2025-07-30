import requests
import json
import logging
from typing import Dict, Optional
GEMINI_API_KEY = "AIzaSyCvXhAn1JcpTVHEgntW-tqUnVeP1q8Y3Fc"
logger = logging.getLogger(__name__)

class GeminiAnalyzer:
    def __init__(self):
        # Get API key from environment variable
        self.api_key = GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set!")
        
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
    def analyze_conversation(self, conversation_text: str) -> Optional[Dict]:
        """Analyze a conversation using Gemini AI"""
        
        # Create the prompt for analysis
        prompt = self._create_analysis_prompt(conversation_text)
        
        # Prepare the request payload
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
                "temperature": 0.1,  # Low temperature for consistent analysis
                "maxOutputTokens": 1000
            }
        }
        
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': self.api_key
        }
        
        try:
            # Make request to Gemini API
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract the generated text
                if 'candidates' in result and len(result['candidates']) > 0:
                    generated_text = result['candidates'][0]['content']['parts'][0]['text']
                    
                    # Parse the structured response
                    return self._parse_gemini_response(generated_text)
                else:
                    logger.error("No candidates in Gemini response")
                    return None
            else:
                logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            return None
    
    def _create_analysis_prompt(self, conversation_text: str) -> str:
        """Create a structured prompt for conversation analysis"""
        
        prompt = f"""
You are an expert customer support analyst. Analyze the following customer support conversation and provide a structured analysis.

CONVERSATION:
{conversation_text}

Please analyze this conversation and provide your response in the following JSON format:

{{
    "is_resolved": true/false,
    "resolution_confidence": 0.0-1.0,
    "tags": ["tag1", "tag2", "tag3"],
    "nature_of_request": "billing|technical|account|general|product_info|complaint",
    "customer_sentiment": "positive|negative|neutral|frustrated|satisfied",
    "urgency_level": "low|medium|high|critical",
    "conversation_type": "simple_inquiry|complex_issue|escalation|follow_up",
    "customer_behavior": "polite|impatient|technical|confused|angry|cooperative",
    "resolution_summary": "brief description of what happened",
    "key_issues": ["issue1", "issue2"],
    "agent_performance": "poor|fair|good|excellent"
}}

ANALYSIS GUIDELINES:
1. **is_resolved**: True if the customer's issue was clearly solved or customer expressed satisfaction
2. **resolution_confidence**: How confident you are in the resolution status (0.0 = not sure, 1.0 = very sure)
3. **tags**: 3-5 relevant tags describing the conversation (e.g., "billing_error", "frustrated_customer", "quick_resolution")
4. **nature_of_request**: Primary category of the customer's request
5. **customer_sentiment**: Overall emotional tone of the customer
6. **urgency_level**: How urgent the customer's issue seems
7. **conversation_type**: Type of interaction pattern
8. **customer_behavior**: How the customer communicated
9. **resolution_summary**: Brief explanation of what happened
10. **key_issues**: Main problems mentioned
11. **agent_performance**: How well the agent handled the situation

Respond ONLY with the JSON object, no additional text.
"""
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Optional[Dict]:
        """Parse the JSON response from Gemini"""
        try:
            # Clean the response text
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith('```'):
                response_text = response_text[3:]   # Remove ```
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove ending ```
            
            response_text = response_text.strip()
            
            # Parse JSON
            analysis = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['is_resolved', 'tags', 'nature_of_request', 'customer_sentiment']
            for field in required_fields:
                if field not in analysis:
                    logger.warning(f"Missing required field: {field}")
                    analysis[field] = self._get_default_value(field)
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text}")
            
            # Return a basic analysis if JSON parsing fails
            return self._create_fallback_analysis()
        
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
            return self._create_fallback_analysis()
    
    def _get_default_value(self, field: str):
        """Get default value for missing fields"""
        defaults = {
            'is_resolved': False,
            'resolution_confidence': 0.5,
            'tags': ['unanalyzed'],
            'nature_of_request': 'general',
            'customer_sentiment': 'neutral',
            'urgency_level': 'medium',
            'conversation_type': 'simple_inquiry',
            'customer_behavior': 'neutral',
            'resolution_summary': 'Unable to analyze',
            'key_issues': ['unknown'],
            'agent_performance': 'fair'
        }
        return defaults.get(field, 'unknown')
    
    def _create_fallback_analysis(self) -> Dict:
        """Create a basic fallback analysis when Gemini fails"""
        return {
            'is_resolved': False,
            'resolution_confidence': 0.0,
            'tags': ['gemini_analysis_failed'],
            'nature_of_request': 'unknown',
            'customer_sentiment': 'neutral',
            'urgency_level': 'medium',
            'conversation_type': 'unknown',
            'customer_behavior': 'unknown',
            'resolution_summary': 'Analysis failed',
            'key_issues': ['analysis_error'],
            'agent_performance': 'unknown'
        }