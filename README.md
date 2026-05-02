# Preventive Care Recommendation Agent

This project is an Agentic AI-powered Preventive Care Recommendation system built with LangChain, ChromaDB, and Streamlit. It uses a Llama-3.1 model to retrieve guidelines from a mock vector database consisting of 12 cardiovascular and metabolic health PDFs, computes a Framingham risk score, and synthesizes a structured care plan.

## Setup Instructions

1. **Clone or Download the Repository**
   Ensure you are in the `AGENTIC` directory.

2. **Set up the Virtual Environment & Dependencies**
   If not already configured, create a virtual environment and install the requirements:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Check Environment Variables**
   Ensure the `.env` file exists in the root directory and contains:
   ```env
   HF_CHAT_URL=https://router.huggingface.co/v1/chat/completions
   HF_CHAT_MODEL=meta-llama/Llama-3.1-8B-Instruct
   ```

4. **Run the Application**
   Start the Streamlit UI:
   ```bash
   .\venv\Scripts\activate
   streamlit run app.py
   ```

5. **Usage**
   - Open the web interface in your browser (usually `http://localhost:8501`).
   - Enter a patient's vitals on the left sidebar.
   - Click "Generate Recommendation".
   - View the agent's logic in the terminal logs, and the final structured care plan in the Streamlit UI.

**Note:** The mock cardiovascular PDF data has already been generated and indexed into the `chroma_db` folder. The data generation script was removed as requested.
