import streamlit as st
import google.generativeai as genai
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Gafur Khussanbayev | Data Analyst Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme with Astana aesthetic
st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0f3460 100%);
    }

    /* Astana-inspired accent colors (blue and gold) */
    :root {
        --astana-blue: #00d4ff;
        --astana-gold: #ffd700;
        --dark-bg: #0a0a0a;
        --card-bg: rgba(26, 26, 46, 0.8);
    }

    /* Headers styling */
    h1, h2, h3 {
        background: linear-gradient(90deg, #00d4ff, #ffd700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Card styling */
    .project-card {
        background: rgba(26, 26, 46, 0.9);
        border: 1px solid #00d4ff;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .project-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.4);
    }

    /* Skill badge styling */
    .skill-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00d4ff20, #ffd70020);
        border: 1px solid #00d4ff;
        color: #00d4ff;
        padding: 8px 16px;
        margin: 5px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 500;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid #00d4ff;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }

    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        color: #ffd700;
    }

    .metric-label {
        color: #00d4ff;
        font-size: 1em;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a, #1a1a2e);
        border-right: 1px solid #00d4ff;
    }

    /* Chat container */
    .chat-container {
        background: rgba(26, 26, 46, 0.9);
        border: 1px solid #00d4ff;
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
    }

    /* Astana skyline decoration */
    .astana-header {
        text-align: center;
        padding: 20px;
        border-bottom: 2px solid #00d4ff;
        margin-bottom: 30px;
    }

    /* Location badge */
    .location-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 215, 0, 0.1);
        border: 1px solid #ffd700;
        color: #ffd700;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 14px;
        margin-top: 10px;
    }

    /* Timeline for experience */
    .timeline-item {
        border-left: 3px solid #00d4ff;
        padding-left: 20px;
        margin-left: 10px;
        margin-bottom: 20px;
    }

    .timeline-dot {
        width: 15px;
        height: 15px;
        background: #ffd700;
        border-radius: 50%;
        position: relative;
        left: -28px;
        top: 5px;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0077b6);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #ffd700, #ffaa00);
        transform: scale(1.05);
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        border-top: 1px solid #00d4ff;
        margin-top: 50px;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)


def init_gemini():
    """Initialize Gemini API"""
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if api_key:
        genai.configure(api_key=api_key)
        return True
    return False


def get_ai_response(prompt: str, context: str) -> str:
    """Get response from Gemini AI"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        full_prompt = f"""You are an AI assistant for Gafur Khussanbayev's portfolio website.
        You help visitors learn about Gafur's skills, projects, and experience.

        Context about Gafur:
        {context}

        User question: {prompt}

        Provide a helpful, professional response. Keep it concise but informative."""

        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"I apologize, but I'm unable to process your request at the moment. Error: {str(e)}"


# Portfolio context for AI
PORTFOLIO_CONTEXT = """
Gafur Khussanbayev is a Big Data Analysis student at Astana IT University, Kazakhstan.
He has professional experience as a Data & Test Analyst at 'Global Digital Innovations' where he worked with PostgreSQL, Kafka, and Kubernetes.
He has built an LLM agent with a RAG (Retrieval-Augmented Generation) system.

Key Projects:
1. Fraud Detection Model - Built using CatBoost achieving 80% recall, deployed as a Telegram bot for real-time predictions.
2. Retail Sales Forecasting - Developed using XGBoost, improved RMSE by 23% compared to baseline models.

Technical Skills:
- Programming: Python, SQL
- Data Science: Pandas, NumPy, Scikit-learn, PyTorch, TensorFlow
- ML/DL: CatBoost, XGBoost, LightGBM, Neural Networks
- Big Data: Apache Kafka, Apache Spark
- Databases: PostgreSQL, MongoDB
- DevOps: Docker, Kubernetes, Git
- Cloud: Google Cloud Platform
- Visualization: Matplotlib, Seaborn, Plotly, Streamlit

Location: Astana, Kazakhstan
"""


def render_sidebar():
    """Render sidebar navigation"""
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h2 style='margin-bottom: 5px;'>Gafur Khussanbayev</h2>
            <p style='color: #00d4ff;'>Data Analyst & ML Engineer</p>
            <div class='location-badge'>
                📍 Astana, Kazakhstan
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigation",
            ["🏠 About Me", "💻 Technical Skills", "🚀 Projects", "🤖 AI Chat"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Contact info
        st.markdown("### 📬 Contact")
        st.markdown("""
        - 📧 [Email](mailto:your.email@example.com)
        - 💼 [LinkedIn](https://linkedin.com/in/yourprofile)
        - 🐙 [GitHub](https://github.com/yourprofile)
        """)

        # Current time in Astana
        st.markdown("---")
        st.markdown(f"🕐 Astana Time: **{datetime.now().strftime('%H:%M')}**")

        return page


def render_about():
    """Render About Me section"""
    st.markdown("""
    <div class='astana-header'>
        <h1>👋 Hello, I'm Gafur Khussanbayev</h1>
        <p style='font-size: 1.3em; color: #aaa;'>Big Data Analysis Student | Data Analyst | ML Engineer</p>
        <div class='location-badge'>📍 Astana, Kazakhstan</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## 🎯 About Me")
        st.markdown("""
        I am a passionate **Big Data Analysis** student at **Astana IT University**,
        dedicated to transforming complex data into actionable insights. My journey in
        data science combines academic excellence with hands-on industry experience.

        Currently, I'm exploring the intersection of **Machine Learning** and **Large Language Models**,
        having recently built an LLM agent with a RAG system that showcases my ability to work
        with cutting-edge AI technologies.
        """)

        st.markdown("## 💼 Professional Experience")
        st.markdown("""
        <div class='timeline-item'>
            <div class='timeline-dot'></div>
            <h4 style='color: #00d4ff; margin: 0;'>Data & Test Analyst</h4>
            <p style='color: #ffd700; margin: 5px 0;'>Global Digital Innovations</p>
            <p style='color: #888;'>
                • Worked with PostgreSQL for data management and analysis<br>
                • Implemented data pipelines using Apache Kafka<br>
                • Deployed and managed applications on Kubernetes clusters<br>
                • Conducted comprehensive data quality testing
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("## 🎓 Education")
        st.markdown("""
        <div class='timeline-item'>
            <div class='timeline-dot'></div>
            <h4 style='color: #00d4ff; margin: 0;'>Big Data Analysis</h4>
            <p style='color: #ffd700; margin: 5px 0;'>Astana IT University</p>
            <p style='color: #888;'>Focusing on data science, machine learning, and big data technologies</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("## 📊 Quick Stats")

        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>2+</div>
            <div class='metric-label'>ML Projects Deployed</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>80%</div>
            <div class='metric-label'>Fraud Detection Recall</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>23%</div>
            <div class='metric-label'>RMSE Improvement</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>RAG</div>
            <div class='metric-label'>LLM Agent Built</div>
        </div>
        """, unsafe_allow_html=True)


def render_skills():
    """Render Technical Skills section"""
    st.markdown("""
    <div class='astana-header'>
        <h1>💻 Technical Skills</h1>
        <p style='color: #aaa;'>Technologies and tools I work with</p>
    </div>
    """, unsafe_allow_html=True)

    # Programming Languages
    st.markdown("## 🐍 Programming Languages")
    langs = ["Python", "SQL", "Bash"]
    st.markdown("".join([f"<span class='skill-badge'>{lang}</span>" for lang in langs]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Data Science & ML
    st.markdown("## 📊 Data Science & Machine Learning")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Core Libraries")
        core_libs = ["Pandas", "NumPy", "Scikit-learn", "SciPy", "Statsmodels"]
        st.markdown("".join([f"<span class='skill-badge'>{lib}</span>" for lib in core_libs]), unsafe_allow_html=True)

        st.markdown("### Deep Learning")
        dl_libs = ["PyTorch", "TensorFlow", "Keras", "Transformers"]
        st.markdown("".join([f"<span class='skill-badge'>{lib}</span>" for lib in dl_libs]), unsafe_allow_html=True)

    with col2:
        st.markdown("### ML Algorithms")
        ml_libs = ["CatBoost", "XGBoost", "LightGBM", "Random Forest"]
        st.markdown("".join([f"<span class='skill-badge'>{lib}</span>" for lib in ml_libs]), unsafe_allow_html=True)

        st.markdown("### NLP & LLM")
        nlp_libs = ["LangChain", "RAG Systems", "Gemini API", "Hugging Face"]
        st.markdown("".join([f"<span class='skill-badge'>{lib}</span>" for lib in nlp_libs]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Big Data & Databases
    st.markdown("## 🗄️ Big Data & Databases")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Big Data Tools")
        bigdata = ["Apache Kafka", "Apache Spark", "Airflow"]
        st.markdown("".join([f"<span class='skill-badge'>{tool}</span>" for tool in bigdata]), unsafe_allow_html=True)

    with col2:
        st.markdown("### Databases")
        databases = ["PostgreSQL", "MongoDB", "Redis", "ClickHouse"]
        st.markdown("".join([f"<span class='skill-badge'>{db}</span>" for db in databases]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # DevOps & Cloud
    st.markdown("## ☁️ DevOps & Cloud")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### DevOps")
        devops = ["Docker", "Kubernetes", "Git", "CI/CD", "Linux"]
        st.markdown("".join([f"<span class='skill-badge'>{tool}</span>" for tool in devops]), unsafe_allow_html=True)

    with col2:
        st.markdown("### Cloud Platforms")
        cloud = ["Google Cloud Platform", "AWS", "Azure"]
        st.markdown("".join([f"<span class='skill-badge'>{platform}</span>" for platform in cloud]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualization
    st.markdown("## 📈 Data Visualization")
    viz_tools = ["Matplotlib", "Seaborn", "Plotly", "Streamlit", "Tableau", "Power BI"]
    st.markdown("".join([f"<span class='skill-badge'>{tool}</span>" for tool in viz_tools]), unsafe_allow_html=True)


def render_projects():
    """Render Projects section"""
    st.markdown("""
    <div class='astana-header'>
        <h1>🚀 Featured Projects</h1>
        <p style='color: #aaa;'>Showcasing my data science and machine learning work</p>
    </div>
    """, unsafe_allow_html=True)

    # Project 1: Fraud Detection
    st.markdown("""
    <div class='project-card'>
        <h2>🔍 Fraud Detection Model</h2>
        <p style='color: #ffd700;'>Machine Learning | CatBoost | Telegram Bot</p>
        <hr style='border-color: #00d4ff33;'>
        <p>
            Developed a robust fraud detection system using CatBoost algorithm to identify
            fraudulent transactions in real-time. The model was deployed as an interactive
            Telegram bot for easy access and real-time predictions.
        </p>
        <h4 style='color: #00d4ff;'>Key Achievements:</h4>
        <ul>
            <li>Achieved <strong style='color: #ffd700;'>80% recall</strong> on fraud detection</li>
            <li>Implemented feature engineering for transaction patterns</li>
            <li>Built end-to-end pipeline from data processing to deployment</li>
            <li>Created user-friendly Telegram bot interface for predictions</li>
        </ul>
        <h4 style='color: #00d4ff;'>Technologies Used:</h4>
    </div>
    """, unsafe_allow_html=True)

    fraud_tech = ["Python", "CatBoost", "Pandas", "Telegram API", "Docker"]
    st.markdown("".join([f"<span class='skill-badge'>{tech}</span>" for tech in fraud_tech]), unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Project 2: Retail Sales Forecasting
    st.markdown("""
    <div class='project-card'>
        <h2>📈 Retail Sales Forecasting</h2>
        <p style='color: #ffd700;'>Time Series | XGBoost | Business Analytics</p>
        <hr style='border-color: #00d4ff33;'>
        <p>
            Built a sophisticated sales forecasting model for retail business to predict
            future sales and optimize inventory management. The model significantly outperformed
            baseline approaches.
        </p>
        <h4 style='color: #00d4ff;'>Key Achievements:</h4>
        <ul>
            <li>Improved RMSE by <strong style='color: #ffd700;'>23%</strong> compared to baseline</li>
            <li>Implemented advanced feature engineering with lag features and rolling statistics</li>
            <li>Analyzed seasonal patterns and trends in sales data</li>
            <li>Provided actionable insights for inventory optimization</li>
        </ul>
        <h4 style='color: #00d4ff;'>Technologies Used:</h4>
    </div>
    """, unsafe_allow_html=True)

    retail_tech = ["Python", "XGBoost", "Scikit-learn", "Pandas", "Matplotlib"]
    st.markdown("".join([f"<span class='skill-badge'>{tech}</span>" for tech in retail_tech]), unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Project 3: LLM Agent with RAG
    st.markdown("""
    <div class='project-card'>
        <h2>🤖 LLM Agent with RAG System</h2>
        <p style='color: #ffd700;'>NLP | Large Language Models | Retrieval-Augmented Generation</p>
        <hr style='border-color: #00d4ff33;'>
        <p>
            Developed an intelligent LLM agent powered by Retrieval-Augmented Generation (RAG)
            system. The agent can answer questions based on custom knowledge bases with high
            accuracy and contextual understanding.
        </p>
        <h4 style='color: #00d4ff;'>Key Features:</h4>
        <ul>
            <li>Custom knowledge base integration with vector embeddings</li>
            <li>Efficient document retrieval using semantic search</li>
            <li>Context-aware response generation</li>
            <li>Scalable architecture for enterprise use</li>
        </ul>
        <h4 style='color: #00d4ff;'>Technologies Used:</h4>
    </div>
    """, unsafe_allow_html=True)

    llm_tech = ["Python", "LangChain", "OpenAI", "ChromaDB", "FastAPI"]
    st.markdown("".join([f"<span class='skill-badge'>{tech}</span>" for tech in llm_tech]), unsafe_allow_html=True)


def render_chat():
    """Render AI Chat section"""
    st.markdown("""
    <div class='astana-header'>
        <h1>🤖 AI Assistant</h1>
        <p style='color: #aaa;'>Ask me anything about Gafur's skills, projects, or experience!</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize Gemini
    gemini_available = init_gemini()

    if not gemini_available:
        st.warning("""
        ⚠️ **Gemini API key not configured.**

        To enable the AI chat feature, add your Gemini API key to Streamlit secrets:

        1. Create a file `.streamlit/secrets.toml`
        2. Add: `GEMINI_API_KEY = "your-api-key-here"`

        Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        """)
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm the AI assistant for Gafur's portfolio. Feel free to ask me about his skills, projects, or experience! 🇰🇿"}
        ]

    # Display chat history
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.markdown("</div>", unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask about my skills, projects, or experience..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_ai_response(prompt, PORTFOLIO_CONTEXT)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    # Sample questions
    st.markdown("### 💡 Try asking:")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        - What projects has Gafur worked on?
        - What ML frameworks does he know?
        - Tell me about his fraud detection project
        """)

    with col2:
        st.markdown("""
        - What is his experience with big data?
        - What cloud platforms does he use?
        - Describe his RAG system project
        """)


def main():
    """Main application"""
    page = render_sidebar()

    if "🏠 About Me" in page:
        render_about()
    elif "💻 Technical Skills" in page:
        render_skills()
    elif "🚀 Projects" in page:
        render_projects()
    elif "🤖 AI Chat" in page:
        render_chat()

    # Footer
    st.markdown("""
    <div class='footer'>
        <p>Built with ❤️ in Astana, Kazakhstan | © 2025 Gafur Khussanbayev</p>
        <p style='font-size: 0.8em;'>Powered by Streamlit & Google Gemini</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
