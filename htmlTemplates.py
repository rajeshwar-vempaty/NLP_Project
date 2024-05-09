css = '''
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
    background-color: #e0e0e0;  /* Light grey for bot messages */
}
.chat-message.user {
    background-color: #d0d0d0;  /* Slightly darker grey for user messages */
}
.chat-message .message {
  padding: 0.5rem 1rem;
  flex-grow: 1;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
</div>
'''

