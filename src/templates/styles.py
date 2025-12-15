"""
CSS styles and HTML templates for the ResearchAI application.
"""


class Styles:
    """CSS and HTML template definitions."""

    # Theme colors
    LIGHT_THEME = {
        'bg_primary': '#ffffff',
        'bg_secondary': '#f8fafc',
        'bg_tertiary': '#e2e8f0',
        'text_primary': '#1e293b',
        'text_secondary': '#64748b',
        'accent': '#3b82f6',
        'accent_hover': '#2563eb',
        'success': '#22c55e',
        'warning': '#eab308',
        'error': '#ef4444',
        'border': '#e2e8f0',
        'user_msg_bg': '#3b82f6',
        'bot_msg_bg': '#f1f5f9',
        'card_bg': '#ffffff',
        'card_shadow': 'rgba(0,0,0,0.1)'
    }

    DARK_THEME = {
        'bg_primary': '#0f172a',
        'bg_secondary': '#1e293b',
        'bg_tertiary': '#334155',
        'text_primary': '#f8fafc',
        'text_secondary': '#94a3b8',
        'accent': '#60a5fa',
        'accent_hover': '#3b82f6',
        'success': '#4ade80',
        'warning': '#facc15',
        'error': '#f87171',
        'border': '#334155',
        'user_msg_bg': '#3b82f6',
        'bot_msg_bg': '#334155',
        'card_bg': '#1e293b',
        'card_shadow': 'rgba(0,0,0,0.3)'
    }

    @classmethod
    def get_main_css(cls, theme: str = 'light') -> str:
        """Get main CSS with theme support."""
        colors = cls.DARK_THEME if theme == 'dark' else cls.LIGHT_THEME

        return f'''
<style>
    /* Global Styles */
    .stApp {{
        background-color: {colors['bg_primary']};
        color: {colors['text_primary']};
    }}

    /* Ensure all text elements have proper color */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: {colors['text_primary']} !important;
    }}

    .stApp p, .stApp li, .stApp span, .stApp div {{
        color: {colors['text_primary']};
    }}

    .stMarkdown, .stMarkdown p, .stMarkdown li {{
        color: {colors['text_primary']} !important;
    }}

    /* Secondary text styling */
    .stApp .secondary-text {{
        color: {colors['text_secondary']};
    }}

    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Chat Messages */
    .chat-container {{
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }}

    .chat-message {{
        padding: 1rem 1.25rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        animation: fadeIn 0.3s ease-out;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .chat-message.user {{
        background: linear-gradient(135deg, {colors['user_msg_bg']}, {colors['accent_hover']});
        color: white;
        margin-left: 2rem;
        border-bottom-right-radius: 0.25rem;
    }}

    .chat-message.bot {{
        background-color: {colors['bot_msg_bg']};
        color: {colors['text_primary']};
        margin-right: 2rem;
        border-bottom-left-radius: 0.25rem;
        border: 1px solid {colors['border']};
    }}

    .chat-message .avatar {{
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        flex-shrink: 0;
    }}

    .chat-message.user .avatar {{
        background-color: rgba(255,255,255,0.2);
    }}

    .chat-message.bot .avatar {{
        background-color: {colors['accent']};
        color: white;
    }}

    .chat-message .message {{
        flex-grow: 1;
        line-height: 1.6;
    }}

    /* Source Citations */
    .sources-container {{
        margin-top: 1rem;
        padding-top: 0.75rem;
        border-top: 1px solid {colors['border']};
    }}

    .source-chip {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        background-color: {colors['bg_tertiary']};
        border-radius: 1rem;
        font-size: 0.75rem;
        color: {colors['text_secondary']};
    }}

    /* Dashboard Cards */
    .dashboard-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }}

    .metric-card {{
        background: {colors['card_bg']};
        border-radius: 0.75rem;
        padding: 1.25rem;
        box-shadow: 0 2px 8px {colors['card_shadow']};
        border: 1px solid {colors['border']};
        transition: transform 0.2s, box-shadow 0.2s;
    }}

    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px {colors['card_shadow']};
    }}

    .metric-card .icon {{
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }}

    .metric-card .value {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {colors['text_primary']};
    }}

    .metric-card .label {{
        font-size: 0.875rem;
        color: {colors['text_secondary']};
        margin-top: 0.25rem;
    }}

    /* Section Cards */
    .section-card {{
        background: {colors['card_bg']};
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid {colors['border']};
    }}

    .section-card h3 {{
        color: {colors['text_primary']};
        margin-bottom: 1rem;
        font-size: 1.125rem;
        font-weight: 600;
    }}

    /* Keywords Cloud */
    .keyword-cloud {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        padding: 1rem;
    }}

    .keyword-tag {{
        padding: 0.375rem 0.75rem;
        background: linear-gradient(135deg, {colors['accent']}, {colors['accent_hover']});
        color: white;
        border-radius: 1rem;
        font-weight: 500;
        transition: transform 0.2s;
    }}

    .keyword-tag:hover {{
        transform: scale(1.05);
    }}

    /* Summary Box */
    .summary-box {{
        background: linear-gradient(135deg, {colors['bg_secondary']}, {colors['bg_tertiary']});
        border-radius: 0.75rem;
        padding: 1.5rem;
        border-left: 4px solid {colors['accent']};
        margin: 1rem 0;
    }}

    .summary-box p {{
        color: {colors['text_primary']};
        line-height: 1.7;
        margin: 0;
    }}

    /* Key Findings List */
    .findings-list {{
        list-style: none;
        padding: 0;
        margin: 0;
    }}

    .findings-list li {{
        padding: 0.75rem 1rem;
        background: {colors['bg_secondary']};
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        color: {colors['text_primary']};
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }}

    .findings-list li::before {{
        content: "\\2713";
        color: {colors['success']};
        font-weight: bold;
    }}

    /* Questions List */
    .question-item {{
        padding: 0.75rem 1rem;
        background: {colors['bg_secondary']};
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        color: {colors['accent']};
        cursor: pointer;
        transition: background 0.2s;
        border: 1px solid {colors['border']};
    }}

    .question-item:hover {{
        background: {colors['bg_tertiary']};
    }}

    /* Progress Bar */
    .progress-container {{
        width: 100%;
        background: {colors['bg_tertiary']};
        border-radius: 0.5rem;
        overflow: hidden;
        height: 8px;
    }}

    .progress-bar {{
        height: 100%;
        background: linear-gradient(90deg, {colors['accent']}, {colors['success']});
        transition: width 0.3s ease;
    }}

    /* Readability Gauge */
    .gauge-container {{
        text-align: center;
        padding: 1rem;
    }}

    .gauge-value {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {colors['text_primary']};
    }}

    .gauge-label {{
        font-size: 0.875rem;
        color: {colors['text_secondary']};
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: {colors['bg_secondary']};
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        color: {colors['text_primary']};
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {colors['accent']};
        color: white;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {colors['bg_secondary']};
    }}

    [data-testid="stSidebar"] .stButton button {{
        width: 100%;
        background: linear-gradient(135deg, {colors['accent']}, {colors['accent_hover']});
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        font-weight: 600;
        transition: transform 0.2s, box-shadow 0.2s;
    }}

    [data-testid="stSidebar"] .stButton button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px {colors['card_shadow']};
    }}

    /* Export Button */
    .export-btn {{
        background: {colors['bg_tertiary']};
        border: 1px solid {colors['border']};
        color: {colors['text_primary']};
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }}

    /* Spinner */
    .loading-spinner {{
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 2px solid {colors['border']};
        border-top-color: {colors['accent']};
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }}

    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}

    /* Responsive */
    @media (max-width: 768px) {{
        .chat-message {{
            margin-left: 0;
            margin-right: 0;
        }}

        .dashboard-grid {{
            grid-template-columns: 1fr 1fr;
        }}
    }}
</style>
'''

    @staticmethod
    def get_user_message_template() -> str:
        """Get user message HTML template."""
        return '''
<div class="chat-message user">
    <div class="avatar">👤</div>
    <div class="message">{message}</div>
</div>
'''

    @staticmethod
    def get_bot_message_template(include_sources: bool = True) -> str:
        """Get bot message HTML template."""
        sources_html = '''
<div class="sources-container">
    <small><strong>Sources:</strong></small>
    {sources}
</div>
''' if include_sources else ''

        return f'''
<div class="chat-message bot">
    <div class="avatar">🤖</div>
    <div class="message">
        {{message}}
        {sources_html}
    </div>
</div>
'''

    @staticmethod
    def get_metric_card_template() -> str:
        """Get metric card HTML template."""
        return '''
<div class="metric-card">
    <div class="icon">{icon}</div>
    <div class="value">{value}</div>
    <div class="label">{label}</div>
</div>
'''

    @staticmethod
    def get_source_chip_template() -> str:
        """Get source chip HTML template."""
        return '<span class="source-chip">{section}</span>'
