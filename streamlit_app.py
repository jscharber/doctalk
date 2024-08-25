import os
import streamlit as st
import hmac
from streamlit_option_menu import option_menu
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

noodle = {
    'Name': 'Noodle',
    'Type': 'Filesystem',
    'Files': '',
    'Chunks': '',
    'URL': 'https://www.google.com',
    'Database': 'sqlite://noodle.db',
    'AgentType': ['QA', 'Conversation'],
    'Data': [],
    'Step': 1
}

upload_dir = './uploads'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

st.set_page_config(
    page_title="Doc talk",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state['history'] = []

if "crc" not in st.session_state:
    st.session_state['crc'] = None


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.write("🔒 Please enter the password to continue.")
    st.text_input(
        "password", type="password", on_change=password_entered, key="password", label_visibility="hidden"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.


def add_sidebar():
    """Adds a sidebar to the app."""
    with st.sidebar:
        selected = option_menu("NimbleBrain", ["Data sources", 'Agents', 'Jobs', 'Activity'],
                               icons=['database', 'app', 'list-task', 'activity'], menu_icon="cast", default_index=0)
        selected2 = st.selectbox("LLM", [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            'gpt-4o'
            'gpt-4o-turbo',])

    selected2 = option_menu(None, ['', '', '', 'Settings'],
                            icons=['', '', '', 'gear'],
                            menu_icon="cast", default_index=3, orientation="horizontal")


# Create an OpenAI client.st.title('Noodle builder')
add_sidebar()

st.title('WelcomeNoodle builder')
st.write('This nodel supports a conversational agent that can be trained on a variety of data sources. You can use Noodle to build a QA system or a conversation agent. Noodle uses OpenAI GPT-3 for its conversational capabilities.')
st.write('You can use Noodle to build a QA system or a conversation agent. Noodle uses OpenAI GPT-3 for its conversational capabilities.')
st.write('Noodle uses OpenAI GPT-3 for its conversational capabilities.')
noodle['Type'] = st.selectbox('What type of data do you want to use', [
                              'Filesystem', 'URL', "Database"])

st.session_state['button_text'] = 'Load'
st.session_state['button_state'] = False


if str(noodle['Type']) == 'Filesystem':
    files = st.file_uploader('Upload a file', type=[
                             'pdf', 'txt', 'docx'], accept_multiple_files=True, help='Upload a file to use as data source')
    loadButton = st.button(st.session_state['button_text'], disabled=st.session_state['button_state'],
                           type='primary')
    if len(files) > 0 and loadButton:
        progress = 0
        pb = st.progress(progress, text='Loading data...')
        for file in files:
            pb.progress(0, text='Loading data...')
            rf = file.read()
            file_name = os.path.join("./uploads", file.name)
            with open(file_name, 'wb') as f:
                f.write(rf)

            if file.name.endswith('.pdf'):
                noodle['Files'] = PyPDFLoader(file_name).load()
            elif file.name.endswith('.txt'):
                noodle['Files'] = TextLoader(file_name).load()
            elif file.name.endswith('.docx'):
                noodle['Files'] = Docx2txtLoader(file_name).load()
            else:
                st.write('Unsupported file format')
                continue

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200)
            chunked_text = text_splitter.split_documents(noodle['Files'])
            noodle['Chunks'] = chunked_text

            progress += 1
            pb.progress(progress/len(files), text='Loading data...')
        pb.empty()
        st.success('Data loaded successfully')
        # st.session_state['button_text'] = 'Loaded'
        # st.session_state['button_state'] = True
elif str(noodle['Type']) == 'URL':
    url = st.text_input(
        'Enter a URL', help='Enter a URL to use as data source')
    st.write('You entered:', url)
else:
    conn = st.text_input('Enter a connection string',
                         help='Enter a connection string to use as data source')
    st.write('You entered:', conn)


# set env for openai correctly so you don't have to keep entering the key
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai-token"])
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.5,
                 max_tokens=100, openai_api_key=st.secrets["openai-token"])

if noodle['Chunks'] != '':
    # st.write('Chunks:', noodle['Chunks'])
    vector_store = Chroma.from_documents(noodle['Chunks'], embeddings)
    retriever = vector_store.as_retriever()
    # client = OpenAI(api_key=st.secrets["openai-token"])
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    st.session_state['crc'] = crc

question = st.text_input('Ask me a question', help='Ask me a question')

if question and st.session_state['crc']:

    crc = st.session_state['crc']

    response = crc.invoke({
        'question': question,
        'chat_history': st.session_state['history']
    })

    st.write('Answer:', response['answer'])

    st.divider()
    st.header('Chat history')
    for prompts in st.session_state['history']:
        st.write('Qustion:', prompts[0])
        st.write('Answer:', prompts[1])

    st.session_state['history'].append((question, response['answer']))