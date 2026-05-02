import streamlit as st
import json
import os
from agent import create_agent

st.set_page_config(page_title="Preventive Care Recommendation Agent", layout="wide")

st.title("Cardiovascular Preventive Care Agent")
st.markdown("""
This application uses an Agentic AI-powered framework leveraging LangChain and 
a RAG knowledge base of clinical guidelines to generate personalized preventive care recommendations.
""")

# Run DB init if needed (this would ideally be done offline)
@st.cache_resource
def load_agent():
    return create_agent()

executor = load_agent()

with st.sidebar:
    st.header("Patient Data")
    with st.form("patient_form"):
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
        hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=20, max_value=100, value=50)
        sbp = st.number_input("Systolic BP (mmHg)", min_value=90, max_value=250, value=120)
        smoker = st.checkbox("Smoker")
        diabetes = st.checkbox("Diabetes")
        bp_meds = st.checkbox("On BP Medications")
        
        submit = st.form_submit_button("Generate Recommendation")

if submit:
    patient_json = {
        "age": age,
        "sex": sex,
        "total_cholesterol": chol,
        "hdl_cholesterol": hdl,
        "systolic_bp": sbp,
        "smoker": smoker,
        "diabetes": diabetes,
        "bp_meds": bp_meds
    }
    
    st.write("### Patient Profile Submitted")
    st.json(patient_json)
    
    with st.spinner("Agent is reasoning..."):
        prompt = f"""
        I have a patient with the following profile:
        {json.dumps(patient_json, indent=2)}
        
        Please use your tools to:
        1. Assess their ten-year cardiovascular risk.
        2. Retrieve relevant guidelines for their profile.
        3. Synthesize a structured recommendation.
        4. Validate it using the SelfCritiqueTool.
        Provide the final validated recommendation to me.
        """
        try:
            # We can capture the agent's intermediate steps if we want, but AgentExecutor with verbose=True outputs to console.
            # In Streamlit, we could use StreamlitCallbackHandler, but we'll keep it simple here.
            response = executor.invoke({"input": prompt})
            st.success("Recommendation Generated!")
            st.markdown("### Final Care Plan")
            st.write(response["output"])
        except Exception as e:
            st.error(f"An error occurred during agent execution: {str(e)}")
