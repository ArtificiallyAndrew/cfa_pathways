import openai
import streamlit as st
from gtts import gTTS 
from io import BytesIO 
import base64
import os
import shutil
import tempfile
import uuid
import langchain
import time
from datetime import datetime, timedelta
from io import BytesIO
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pathlib import Path
from aidata import AIDATA
import pinecone 




image_path = "PathwaysLogo.png"
st.image(image_path,  width=200)
st.markdown("""
    <style>
    .small-font {
        font-size:18px;  /* Adjust the size as needed */
    }
    </style>
    <p class="small-font">Sharing NYU's work to create pathways helping increase opportunities for underrepresented people to pursue the college and career aspirations they set for themselves.</p>
    """, unsafe_allow_html=True)

categories = {
    "Pre-College": {"College and Career Lab (CCL)" : "https://collegecareerlab.hosting.nyu.edu/",
                    "Arthur O. Eve Higher Education Opportunity Program (HEOP)" : "https://www.nyu.edu/admissions/undergraduate-admissions/how-to-apply/all-freshmen-applicants/opportunity-programs/college-programs.html", 
                    "Collegiate Science and Technology Entry Program (CSTEP)" : "https://www.nyu.edu/admissions/undergraduate-admissions/how-to-apply/all-freshmen-applicants/opportunity-programs/college-programs.html",
                    "Science and Technology Entry Program (STEP)" : "https://www.nyu.edu/admissions/undergraduate-admissions/how-to-apply/all-freshmen-applicants/opportunity-programs/pre-college-programs.html",
                    "K-12 STEM Education" : "https://engineering.nyu.edu/academics/programs/k12-stem-education", 
                    "Opportunity Programs" : "https://www.nyu.edu/admissions/undergraduate-admissions/how-to-apply/all-freshmen-applicants/opportunity-programs.html"},
     
    "Undergraduate": {"CDS Undergraduate Research Program (CURP)" : "https://cds.nyu.edu/curp/", 
                      "Martin Luther King Jr. Scholars Program" : "https://www.nyu.edu/academics/awards-and-highlights/mlk-scholars-program.html", 
                      "Camp Changemaker" : "https://www.nyu.edu/students/getting-involved/leadership-and-service/changemaker-center/programs-resources/CampChangemaker.html", 
                      "Quality Undergraduate Education and Scholarly Training (QUEST)" : "https://sites.google.com/nyu.edu/nyuquest/",
                      "Stern: Breakthrough Scholars Leadership Program" : "https://www.stern.nyu.edu/programs-admissions/undergraduate/vibrant-community/breakthrough-scholars-leadership-program",
                      "Scholarships at NYU" : "https://www.nyu.edu/admissions/undergraduate-admissions/aid-and-costs/scholarships.html",
                      "Community College Transfer Opportunity Program (CCTOP)" : "https://www.nyu.edu/admissions/undergraduate-admissions/how-to-apply/transfer-applicants/cctop.html",
                      "NYU Financial Education" : "https://www.nyu.edu/students/student-success/financial-education-at-nyu.html",
                      "Opportunity Programs" : "https://www.nyu.edu/admissions/undergraduate-admissions/how-to-apply/all-freshmen-applicants/opportunity-programs.html"},
     
    "Graduate": {"Diversity and Access Funding Programs" : "https://gsas.nyu.edu/admissions/financial-aid/diversity-and-access-funding-programs.html", 
                 "Steinhardt: Scholarships for Underrepresented Populations" : "https://steinhardt.nyu.edu/admissions/paying-your-education/masters-and-advanced-study/scholarships-and-grants#underrepresented-populations", 
                 "Scholarships at NYU" : "https://www.nyu.edu/admissions/undergraduate-admissions/aid-and-costs/scholarships.html",
                 "Community College Transfer Opportunity Program (CCTOP)" : "https://www.nyu.edu/admissions/undergraduate-admissions/how-to-apply/transfer-applicants/cctop.html",
                 "NYU Financial Education" : "https://www.nyu.edu/students/student-success/financial-education-at-nyu.html"},                             
                                                                         
    "Faculty": {"Early Career Faculty Institute (ECFI)" : "https://cfa.hosting.nyu.edu/mentoring/early-career-faculty-institute/",                 
                "Mid-Career Faculty Institute (MCFI)" : "",                   
                "Faculty Resource Network (FRN)" : "https://frn.hosting.nyu.edu/",                        
                "Leadership Development" : "https://cfa.hosting.nyu.edu/leadership-development/",
                "Faculty Cluster Initiative" : "https://cfa.hosting.nyu.edu/recruitment/cluster-initiative-opportunities/",
                "James Weldon Johnson Professorship (JWJ)" : "https://cfa.hosting.nyu.edu/awards/james-weldon-johnson-professorship/",
                "NYU Distinguished Teaching Awards (DTA)" : "https://www.nyu.edu/academics/awards-and-highlights/university-distinguishedteachingawards.html",
                "Teaching Advancement Grant (TAG)" : "https://www.nyu.edu/faculty/funding-opportunities/teaching-advancement-grant.html",
                "Provost’s Postdoctoral Fellowship Program" : "https://www.nyu.edu/faculty/faculty-diversity-and-inclusion/mentoring-and-professional-development/provosts-postdoctoral-fellowship-program.html",
                "Faculty First-Look (FFL)" : "https://ffl.blackfacts.com/",
                "FAS: Faculty Diversity, Equity, Inclusion and Development" : "https://as.nyu.edu/departments/facultydiversity/welcome.html"}
}

if 'selected_category' not in st.session_state:
    st.session_state['selected_category'] = None

# Create buttons in a row
cols = st.columns(len(categories))
for i, category in enumerate(categories.keys()):
    with cols[i]:
        if st.button(category):
            if st.session_state['selected_category'] == category:
                st.session_state['selected_category'] = None
            else:
                st.session_state['selected_category'] = category

# Display information based on the selected category
selected_category = st.session_state['selected_category']
if selected_category:
    st.write(f"Programs for {selected_category}:")
    for program_name, program_url in categories[selected_category].items():
        st.markdown(f"[{program_name}]({program_url})")


#loader = PyPDFLoader("Z:\Pathways_LLM\content\PathwaysTraining.pdf") 
#pages = loader.load_and_split()
#
#text_splitter = RecursiveCharacterTextSplitter(
#    chunk_size = 700,
#    chunk_overlap  = 150,
#    length_function = len,
#)
#
##docs = text_splitter.split_documents(pages)

openai.api_key = st.secrets["OPENAI_API_KEY"]
pinecone.init(
        api_key="6d28a0ef-2313-4d82-8fdd-a772d654856a",  # find at app.pinecone.io
        environment="asia-southeast1-gcp-free"  # next to api key in console
)
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
index_name = "pathways"
docsearch = Pinecone.from_existing_index(index_name, embeddings)
#docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=.1, streaming=True)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
template = (
            "As an AI assistant, your primary function is to provide accurate and concise answers to user"
            "queries, leveraging the data available in a specified database. For each response, ensure you"
            "include the most relevant website link from the {categories[selected_category].items()} to support your answer. In instances "
            "where the database does not contain the necessary information to address a user's request, " 
            "respond with, 'Please contact the CFA for more information at facultyadvance@nyu.edu.' Your task now is "  
            "now to respond to the following user input: " )  
qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=docsearch.as_retriever(search_kwargs={"k": 5}), memory=memory, verbose=True)



def text_to_speech(text):
    """
    Converts text to an audio file using gTTS and returns the audio file as binary data
    """
    audio_bytes = BytesIO()
    tts = gTTS(text=text, lang="en")
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes.read()

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4-1106-preview"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about Pathway Programs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("Assistant"):
        with st.spinner("Thinking..."):
         message_placeholder = st.text("")
         full_response = ""
         #response = agent.run(template + prompt)
         response = qa({"question": template + prompt})  # QA chain
         for response in response["answer"]:
         #for response in response:        
                     full_response += response
                     time.sleep(0.02)
                     message_placeholder.markdown(full_response + "▌")
         message_placeholder.markdown(full_response)
         st.session_state.messages.append(
                 {"role": "assistant", "content": full_response}
             )

    st.audio(text_to_speech(full_response), format="audio/wav")
    

#def clear_chat_history():
 #   st.session_state.messages = []
#st.button('Clear Chat History', on_click=clear_chat_history)
