import logging
from datetime import datetime, timedelta

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
                'message_max_length': 280,  # Twitter character limit
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
        """Optimize the NBA recommendation with business rules and constraints"""
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
            
            # Add channel justification to reasoning
            optimized_rec['reasoning'] = self._enhance_reasoning(
                recommendation['reasoning'],
                recommendation['channel'],
                customer_data
            )
            
            # Validate final recommendation
            if self._validate_recommendation(optimized_rec):
                return optimized_rec
            else:
                logger.warning("Recommendation validation failed, using fallback")
                return self._create_safe_fallback(customer_data)
                
        except Exception as e:
            logger.error(f"Error optimizing recommendation: {str(e)}")
            return self._create_safe_fallback(customer_data)
    
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
    
    def _enhance_reasoning(self, original_reasoning, channel, customer_data):
        """Enhance reasoning with channel-specific justification"""
        try:
            enhanced_reasoning = original_reasoning
            
            # Add channel selection justification
            channel_justifications = {
                'twitter_dm_reply': f"Twitter DM selected because: fast response expected, customer is active on social media, and issue complexity is manageable via messaging. Customer sentiment ({customer_data['customer_sentiment']}) and urgency ({customer_data['urgency_level']}) support quick digital resolution.",
                
                'email_reply': f"Email selected because: issue requires detailed explanation, customer behavior ({customer_data['customer_behavior']}) suggests preference for written communication, and urgency level ({customer_data['urgency_level']}) allows for comprehensive response.",
                
                'scheduling_phone_call': f"Phone call scheduled because: high urgency ({customer_data['urgency_level']}), customer sentiment ({customer_data['customer_sentiment']}) indicates frustration requiring personal touch, and issue complexity needs real-time problem-solving."
            }
            
            if channel in channel_justifications:
                enhanced_reasoning += f" CHANNEL OPTIMIZATION: {channel_justifications[channel]}"
            
            # Add NBA principles explanation
            enhanced_reasoning += f" NBA PRINCIPLES APPLIED: Customer-centric approach based on cohort analysis ({customer_data['customer_cohorts']}), behavioral patterns, and resolution optimization."
            
            return enhanced_reasoning
            
        except Exception as e:
            logger.error(f"Error enhancing reasoning: {str(e)}")
            return original_reasoning
    
    def _validate_recommendation(self, recommendation):
        """Validate the final recommendation meets all constraints"""
        try:
            # Check required fields
            required_fields = ['customer_id', 'channel', 'send_time', 'message', 'reasoning']
            for field in required_fields:
                if field not in recommendation or not recommendation[field]:
                    logger.warning(f"Missing or empty field: {field}")
                    return False
            
            # Validate channel
            valid_channels = ['twitter_dm_reply', 'email_reply', 'scheduling_phone_call']
            if recommendation['channel'] not in valid_channels:
                logger.warning(f"Invalid channel: {recommendation['channel']}")
                return False
            
            # Validate time format
            try:
                datetime.fromisoformat(recommendation['send_time'].replace('Z', '+00:00'))
            except ValueError:
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
    
    def _create_safe_fallback(self, customer_data):
        """Create a safe fallback recommendation"""
        urgency = customer_data.get('urgency_level', 'medium')
        sentiment = customer_data.get('customer_sentiment', 'neutral')
        
        # Safe channel selection
        if urgency in ['high', 'critical']:
            channel = 'scheduling_phone_call'
        elif sentiment in ['frustrated', 'angry']:
            channel = 'email_reply'  # Professional written response
        else:
            channel = 'twitter_dm_reply'  # Default to quick response
        
        # Safe timing: 4 hours from now during business hours
        send_time = datetime.now() + timedelta(hours=4)
        if send_time.hour < 9:
            send_time = send_time.replace(hour=9, minute=0)
        elif send_time.hour >= 18:
            send_time = send_time.replace(hour=16, minute=0)  # 4 PM
        
        return {
            'customer_id': customer_data['customer_id'],
            'conversation_id': customer_data.get('conversation_id', ''),
            'channel': channel,
            'send_time': send_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'message': f"Hello! We're working to resolve your {customer_data.get('nature_of_request', 'inquiry')}. Thank you for your patience.",
            'reasoning': f"Safe fallback recommendation based on urgency ({urgency}) and sentiment ({sentiment}). Applied conservative NBA principles for customer satisfaction."
        }