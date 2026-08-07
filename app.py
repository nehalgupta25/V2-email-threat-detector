#==========================STEP !: LOAD MODULES=================================
import os
import base64
from getpass import getpass
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_agent
import streamlit as st

st.set_page_config(layout = "wide")
def get_base64(image_file):
    with open(image_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64("finalBg.png")

st.markdown(
    f"""
    <style>

    .stApp {{
         
        background-image: url("data:image/png;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: repeat;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

#================Step 2 LOAD ENV and API-KEYS================
st.title("🛡️ Email Threat / Phishing Detector")

st.write("AI-powered email security analysis using Machine Learning, RAG, Web Search and Gemini.")
st.sidebar.title("MUST GIVE API KEYS")

try:
  GOOGLE_API_KEY = st.sidebar.text_input("GOOGLE_API_KEY",type="password")
  TAVILY_API_KEY = st.sidebar.text_input("TAVILY_API_KEY",type="password")
  
  ALL_API = [GOOGLE_API_KEY,TAVILY_API_KEY]
  
  if not all(ALL_API):
    st.sidebar.error("MUST PASS ALL API-KETS")
    url = "https://aistudio.google.com/api-keys"
    st.markdown(f"Get Google API Key-{url}")
    
    url = "https://app.tavily.com/playground"
    st.markdown(f"Get Tavily API Key-{url}")
    
  elif all(ALL_API):
    st.sidebar.success("API KEYS LOADED")
    
    options = [
      "gemini-3.5-flash-lite",
      "gemini-3.5-flash",
      "gemini-2.5-flash-lite"
  ]
  
    selected_model = st.sidebar.selectbox("Select Model", options=options)
      
    model = ChatGoogleGenerativeAI(
          model=selected_model,
          google_api_key=GOOGLE_API_KEY
      )
    
    tavily_tool = TavilySearchResults(
          max_results=3,
          tavily_api_key=TAVILY_API_KEY
      )
  
  else:
    st.sidebar.info("Try Valid API-keys")

except Exception as err:
  pass

st.sidebar.markdown("""
  <div class="about-box">
  
  <div class="about-title">
      🔐 About This Project
  </div>
  
  <div class="about-text">
  
  An AI-powered cybersecurity assistant designed
  to detect and analyze potentially malicious or
  phishing emails.
  
  <br>
  
  <b>Core Technologies:</b><br>
  • Naive Bayes Classification<br>
  • Gemini LLM<br>
  • RAG + FAISS<br>
  • Hugging Face Embeddings<br>
  • Tavily Web Search<br>
  • LangChain Agent<br>
  • Streamlit
  
  </div>
  
  </div>
  """, unsafe_allow_html=True)
  
  
#=======================BACKEND=========================================

#Cybersecurity minitextbook containing explanation

#ML Training
try: 
  knowledge_text = """
  Phishing emails often use urgency, threats, or fear to pressure users into acting quickly.
  Common phishing indicators include suspicious links, unexpected attachments, requests for passwords,
  requests for financial information, unusual sender addresses, and messages asking users to bypass
  normal security procedures.
  
  Users should not click suspicious links or open unexpected attachments.
  They should verify requests through an official website or trusted communication channel.
  Users should report suspicious emails according to their organization's security policy.
  
  Credential phishing attempts often try to trick users into entering usernames, passwords,
  one-time codes, or other sensitive information on fraudulent websites.
  
  Email spoofing occurs when an attacker makes an email appear to come from a trusted sender.
  A suspicious sender address, mismatched domain, or unusual communication style can be an indicator.
  
  If an email appears malicious, the user should avoid interacting with it, preserve the email
  for investigation, and report it to the appropriate security team.
  """
  
  def create_chunks(text,chunk_size=300):  # chunk_size: approx how many words go into 1 chunk
    words = text.split()
    print("Total Words: ",len(words))
    chunks = []
  
    for i in range(0, len(words), chunk_size):
      chunk = " ".join(words[i:i + chunk_size])
      chunks.append(chunk)
  
    return chunks
  
  chunks = create_chunks(knowledge_text, chunk_size=100)
  
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  
  vector_store = FAISS.from_texts(
      chunks,
      embedding=embeddings
  )
  
  
  retriever = vector_store.as_retriever()
  
  #emails: the examples the model learns from.
  emails = [
      "Your account will be suspended. Click this link immediately to verify your password.",
      "Congratulations! You won a prize. Send your bank details to claim it.",
      "Urgent action required. Verify your account now or it will be closed.",
      "Please click this link and enter your login credentials immediately.",
  
      "We noticed unusual activity. Please verify your account using the link below.",
      "Your payment could not be completed. Confirm your information to continue.",
      "You have received an unexpected security alert. Review your account now.",
  
      "The project meeting is scheduled for tomorrow at 10 AM.",
      "Please find the assignment attached for today's class.",
      "Reminder: your library book is due next Monday."
  ]
  
  #labels: the correct answer for each example.
  labels = [
      "Phishing",
      "Phishing",
      "Phishing",
      "Phishing",
  
      "Suspicious",
      "Suspicious",
      "Suspicious",
  
      "Safe",
      "Safe",
      "Safe"
  ]
  vectorizer = CountVectorizer()
  email_vectors = vectorizer.fit_transform(emails)
  classifier = MultinomialNB()
  
  classifier.fit(email_vectors,labels)
  
  def classify_email(email):
    email_vector = vectorizer.transform([email])
  
    prediction = classifier.predict(email_vector)
    return prediction[0]
  
  
  query = "common phishing email indicators"
  
  if TAVILY_API_KEY:
    results = tavily_tool.invoke({
        "query": query
    })
  else:
    pass
  
 
  
  agent = create_agent(
      model=model,
      tools=[tavily_tool, search_cybersecurity_knowledge],
      system_prompt="""
      You are a cybersecurity research assistant.
  
      Use the internal cybersecurity knowledge base when the
      answer can be found there.
  
      Use Tavily when current or external web information is needed.
  
      Use both when necessary.
  
      Give clear and accurate answers.
      """
    )
  def analyze_email(email):
    #Our NB classfication model
    classification = classify_email(email)
  
    retrieved_docs = retriever.invoke(email)
  
    #Here's the email AND what the ML classifier predicted. Now investigate it.
  
    prompt = f"""
      Analyze this email as a cybersecurity assistant.
  
      Email:
      {email}
  
      Machine Learning Classification:
      {classification}
  
      Use the internal cybersecurity knowledge base and web research
      when useful.
  
      Provide:
      1. Threat assessment
      2. Risk level
      3. Explanation
      4. Suggested safe actions
      """
  
    response = agent.invoke({'messages':[{'role':'user','content':prompt}]})
  
    analysis = response['messages'][-1].content[-1]['text']
  
    return {
        "classification": classification,
        "analysis": analysis,
        "sources": retrieved_docs
    }
except Exception as err:

  pass
#===========================FRONTEND========================================

#==================Step 4 STREAMLIT NAVBARS=====================
tab1,tab2 = st.tabs(["🔍 Analyze Email",
                         "💬 Chat With Cyber Expert"])

with tab1:
  st.header("📧 Email Analysis")
  email_text = st.text_area("Paste suspicious email here",height=300)
  uploaded_file = st.file_uploader(
      "Or upload a text file",
      type=["txt"]
  )
  if uploaded_file:
      email_text = uploaded_file.read().decode("utf-8")
  
  else:
      email_text = email_text
  if st.button("🔍 Analyze Email"):
    if email_text:

      try:

        with st.spinner("Analysing........"):

          result = analyze_email(email_text)

          col1, col2 = st.columns(2)

          with col1:
              st.metric(
                  "ML Classification",
                  result["classification"]
              )

          with col2:
              st.metric(
                  "Analysis Engine",
                  "Naive Bayes + Agent"
              )

          st.subheader("Security Analysis")
          st.write(result["analysis"])

          st.subheader("📚 Supporting Knowledge")

          for doc in result["sources"]:
              st.write(doc.page_content)

      except Exception as err:

          st.error(
              "⚠️ Email analysis failed. Please check your API keys and try again."
          )

    else:
      st.warning("Please enter or upload an email.")
  


with tab2:
  st.header("💬 Cybersecurity Q&A")
  user_question = st.text_input(
      "Ask a cybersecurity question:"
  )
  if st.button("Ask Question"):
  
      if user_question:
        with st.spinner("Thinking...."):
          response = agent.invoke({"messages": [{"role": "user", "content": user_question}]})
  
          answer = response["messages"][-1].content[-1]['text']
  
          st.subheader("Answer")
          st.write(answer)
  
      else:
          st.warning("Please enter a question.")
