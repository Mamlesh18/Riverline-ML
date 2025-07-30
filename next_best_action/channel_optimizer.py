import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class ChannelOptimizer:
    """
    Channel Optimizer that fine-tunes recommendations based on NBA best practices
    
    This optimizer applies business rules and constraints to ensure recommendations
    are practical and follow customer service best practices.
    """
    
    def __init__(self):
        # Business hours for different urgency levels
        self.business_hours = {
            'start': 9,  # 9 AM
            'end': 18    # 6 PM
        }
        
        # Channel-specific optimization rules
        self.channel_constraints = {
            'twitter_dm_reply': {
                'max_delay_hours': 2,     # Twitter users expect quick responses
                'preferred_hours': (9, 20), # 9 AM - 8 PM
                'message_max_length': 500,  # Twitter character limit
                'tone': 'casual'
            },
            'email_reply': {
                'max_delay_hours': 24,    # Email can wait longer
                'preferred_hours': (9, 18), # Business hours
                'message_max_length': 1000, # Longer emails OK
                'tone': 'professional'
            },
            'scheduling_phone_call': {
                'max_delay_hours': 4,     # Urgent issues need quick scheduling
                'preferred_hours': (9, 17), # Strict business hours
                'message_max_length': 500,  # Brief but informative
                'tone': 'empathetic'
            }
        }
    
    def optimize_recommendation(self, recommendation, customer_data):
        """Optimize the NBA recommendation with business rules and constraints - REASONING FROM GEMINI ONLY"""
        try:
            optimized_rec = recommendation.copy()
            
            # Optimize timing
            optimized_rec['send_time'] = self._optimize_send_time(
                recommendation['send_time'], 
                recommendation['channel'],
                customer_data['urgency_level']
            )
            
            # Optimize message
            optimized_rec['message'] = self._optimize_message(
                recommendation['message'],
                recommendation['channel'],
                customer_data
            )
            
            # KEEP REASONING FROM GEMINI UNCHANGED - NO ENHANCEMENT
            # optimized_rec['reasoning'] stays exactly as provided by Gemini LLM
            
            # KEEP ISSUE STATUS FROM GEMINI UNCHANGED
            # optimized_rec['issue_status'] stays exactly as provided by Gemini LLM
            
            # Validate final recommendation
            if self._validate_recommendation(optimized_rec):
                return optimized_rec
            else:
                logger.warning("Recommendation validation failed, using fallback")
                
        except Exception as e:
            logger.error(f"Error optimizing recommendation: {str(e)}")
    
    def _optimize_send_time(self, original_time, channel, urgency):
        """Optimize send time based on channel constraints and urgency"""
        try:
            # Parse original time
            if isinstance(original_time, str):
                send_time = datetime.fromisoformat(original_time.replace('Z', '+00:00'))
            else:
                send_time = datetime.now()
            
            constraints = self.channel_constraints[channel]
            
            # Adjust based on urgency
            if urgency in ['critical', 'high']:
                # For urgent issues, send ASAP within business hours
                now = datetime.now()
                if now.hour < self.business_hours['start']:
                    # If before business hours, schedule for start of business
                    send_time = now.replace(hour=self.business_hours['start'], minute=0, second=0)
                elif now.hour >= self.business_hours['end']:
                    # If after business hours, schedule for next business day
                    send_time = (now + timedelta(days=1)).replace(hour=self.business_hours['start'], minute=0, second=0)
                else:
                    # During business hours, send within 1 hour
                    send_time = now + timedelta(hours=1)
            
            else:
                # For normal urgency, respect channel preferences
                preferred_start, preferred_end = constraints['preferred_hours']
                
                if send_time.hour < preferred_start:
                    send_time = send_time.replace(hour=preferred_start, minute=0)
                elif send_time.hour >= preferred_end:
                    send_time = send_time.replace(hour=preferred_end-1, minute=0)
            
            # Ensure not on weekends for business channels
            if channel == 'email_reply' and send_time.weekday() >= 5:  # Saturday=5, Sunday=6
                # Move to next Monday
                days_ahead = 7 - send_time.weekday()
                send_time = send_time + timedelta(days=days_ahead)
                send_time = send_time.replace(hour=self.business_hours['start'], minute=0)
            
            return send_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            
        except Exception as e:
            logger.error(f"Error optimizing send time: {str(e)}")
            # Fallback: 2 hours from now
            return (datetime.now() + timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    def _optimize_message(self, original_message, channel, customer_data):
        """Optimize message based on channel constraints and customer context"""
        try:
            constraints = self.channel_constraints[channel]
            message = original_message
            
            # Truncate if too long
            max_length = constraints['message_max_length']
            if len(message) > max_length:
                message = message[:max_length-3] + "..."
            
            # Add channel-specific prefixes/suffixes
            if channel == 'twitter_dm_reply':
                # Add Twitter-friendly greeting
                if not message.startswith('Hi') and not message.startswith('Hello'):
                    message = f"Hi {customer_data['customer_id']}, " + message
                
                # Add quick response indicator
                if 'urgent' in customer_data['urgency_level']:
                    message += " We'll respond quickly!"
            
            elif channel == 'email_reply':
                # Add professional email structure
                if not message.startswith('Dear') and not message.startswith('Hello'):
                    message = f"Dear {customer_data['customer_id']},\n\n" + message
                
                message += "\n\nBest regards,\nCustomer Support Team"
            
            elif channel == 'scheduling_phone_call':
                # Add call scheduling context
                if 'call' not in message.lower():
                    message += " We'd like to schedule a quick call to resolve this personally."
                
                message += " Please let us know your preferred time."
            
            return message
            
        except Exception as e:
            logger.error(f"Error optimizing message: {str(e)}")
            return original_message
    
    
    
    def _validate_recommendation(self, recommendation):
        """Validate the final recommendation meets all constraints"""
        try:
            # Check required fields
            required_fields = ['customer_id', 'channel', 'send_time', 'message', 'reasoning', 'issue_status']
            for field in required_fields:
                if field not in recommendation or not recommendation[field]:
                    logger.warning(f"Missing or empty field: {field}")
                    return False
            
            # Validate channel
            valid_channels = ['twitter_dm_reply', 'email_reply', 'scheduling_phone_call']
            if recommendation['channel'] not in valid_channels:
                logger.warning(f"Invalid channel: {recommendation['channel']}")
                return False
            
            # Validate issue status
            valid_statuses = ['resolved', 'pending_customer_reply']
            if recommendation['issue_status'] not in valid_statuses:
                logger.warning(f"Invalid issue status: {recommendation['issue_status']}")
                return False
            
            # Validate time format
            try:
                datetime.fromisoformat(recommendation['send_time'].replace('Z', '+00:00'))
            except:
                logger.warning(f"Invalid time format: {recommendation['send_time']}")
                return False
            
            # Check message length
            channel = recommendation['channel']
            if channel in self.channel_constraints:
                max_length = self.channel_constraints[channel]['message_max_length']
                if len(recommendation['message']) > max_length:
                    logger.warning(f"Message too long for {channel}: {len(recommendation['message'])} > {max_length}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating recommendation: {str(e)}")
            return False

    
    