"""
HTML templates and CSS styles for the ResearchAI chat interface.

Contains styling for chat messages and templates for bot/user message display.
"""

CSS = '''
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    background-color: #f0f0f0;
    color: #333;
}

.chat-message.bot {
    background-color: #475569;
    color: #f8fafc;
}

.chat-message.user {
    background-color: #1e40af;
    color: #f8fafc;
}

.chat-message .message {
    padding: 0.5rem 1rem;
    flex-grow: 1;
    line-height: 1.5;
}
</style>
'''

BOT_TEMPLATE = '''
<div class="chat-message bot">
    <div class="message">{{MSG}}</div>
</div>
'''

USER_TEMPLATE = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
</div>
'''

# Backward compatibility aliases
css = CSS
bot_template = BOT_TEMPLATE
user_template = USER_TEMPLATE
