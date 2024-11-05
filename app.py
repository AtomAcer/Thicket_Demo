# Package imports
import warnings
import streamlit as st
import os
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder
warnings.filterwarnings("ignore")

# Langchain components
from langchain_openai import ChatOpenAI
from langchain.memory.summary_buffer import ConversationSummaryBufferMemory
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_community.retrievers import BM25Retriever

# Import code modules
from transcribe_voice_openai import *
from vector_store import *


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]

# Define the directory for all files
base_directory = "data"  # Modify if needed

# Ensure the base directory exists
os.makedirs(base_directory, exist_ok=True)

# create LLM var
global llm

# Voice dictionary for voice selection
voice_dict = {
    'Ocean': 'shimmer',
    'Onyx': 'onyx',
    'Nova': 'nova',
    'Adam': 'alloy'
}

def initialize_llm(OPENAI_KEY):
    """
    Initialize the language model (LLM).

    Input:
        OPENAI_KEY (str): OpenAI API key
    
    Output:
        llm (ChatOpenAI): Initialized ChatOpenAI object
    """
    return ChatOpenAI(model_name='gpt-4o', temperature=0, openai_api_key=OPENAI_KEY)

def initialize_memory():
    """
    Initialize the memory buffer for conversation summarization.
    
    Output:
        memory (ConversationSummaryBufferMemory): Initialized ConversationSummaryBufferMemory object
    """
    summarizer_llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")  # type: ignore
    return ConversationSummaryBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        llm=summarizer_llm,
        output_key='answer'
    )

def get_system_prompts():
    """
    Define the system prompts for contextualizing questions and generating QA responses.
    
    Output:
        Tuple (contextualize_q_system_prompt, qa_system_prompt): System prompts for contextualization and QA response generation
    """
    contextualize_q_system_prompt = (
        """Given a chat history and the latest user question which might reference context in the chat history, 
        formulate a standalone question which can be understood without the chat history. 
        If the chat history is limited or is not there, rephrase the question and try to add as much context
        as possible to help find the right answer. Do not go out of scope of the asked question.
        Do NOT answer the question, just reformulate and make it more clear if needed, otherwise return it as is.
        You can also rephrase the original question into multiple questions when needed, seperated by '?'"""
    )

    qa_system_prompt = (
        """You are an AI assistant to help answers questions based on a given document context.
        Response rules:
        1. You are to respond only if you find the information the user is looking for.
        2. The response should be specific to the question asked.
        3. You can ask a follow up question if you cannot find the information requested or if its too broad.
        4. Do not say "the document" too much in your responses, keep the information flow natural.
        "\n\n"
        context: {context}
        Answer:"""
    )
    return contextualize_q_system_prompt, qa_system_prompt

def create_prompts(contextualize_q_system_prompt, qa_system_prompt):
    """
    Create prompt templates using the provided system prompts.
    
    Input:
        contextualize_q_system_prompt (str): Prompt for contextualizing questions
        qa_system_prompt (str): Prompt for generating QA responses
    
    Output:
        Tuple (contextualize_q_prompt, qa_prompt): Created prompt templates
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return contextualize_q_prompt, qa_prompt

def initialize_question_answer_chain(llm, qa_prompt):
    """
    Initialize the question-answer chain.
    
    Input:
        llm (ChatOpenAI): The language model
        qa_prompt (ChatPromptTemplate): The QA prompt template
    
    Output:
        question_answer_chain (DocumentChain): Initialized DocumentChain object
    """
    return create_stuff_documents_chain(llm, qa_prompt)

def initialize_BM25Retriever():
    # Ensure the directory exists for indexes
    index_directory = base_directory

    # List files only within the base directory
    files = os.listdir(index_directory)
    collection_list = [file for file in files if file.endswith('.txt')]
    collection_list.append("Create new collection")

    collection_name_str = st.selectbox('Select a collection or create a new one:', collection_list)

    if collection_name_str == "Create new collection":
        new_collection_name = st.text_input("Enter a new collection name")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if st.button('Submit'):
            if new_collection_name and uploaded_file is not None:
                # Save the uploaded PDF file within the base directory
                pdf_path = os.path.join(base_directory, f"{new_collection_name}.pdf")
                with open(pdf_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())

                # Create a new collection and retrieve splits
                splits = create_new_collection_streamlit(collection_name_str=new_collection_name, pdf_file=new_collection_name)
                
                return BM25Retriever.from_documents(splits)                                            
    else:
        # Use the specified collection if it exists
        return load_BM25Retriever(collection_name_str)


def run_chatbot(client, llm, retriever, contextualize_q_prompt, question_answer_chain, voice_key, audio_bytes):
    """
    Run the chatbot, handling user input and generating responses.
    
    Input:
        client (OpenAI): OpenAI client
        vectordb (VectorDatabase): Vector database
        contextualize_q_prompt (ChatPromptTemplate): Prompt for contextualizing questions
        question_answer_chain (DocumentChain): Question-answer chain
        voice_key (str): Selected voice key
    
    Output:
        None
    """
    if retriever is not None:
        # retriever = vectordb.as_retriever()
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        chat_history = []

        # audio_bytes = audio_recorder()
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")

        # if st.button('Ask Question'):
            transcript = record_and_transcribe(client, audio_bytes)
            user_input = transcript

            ai_msg_dict = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
            response = ai_msg_dict["answer"]
            chat_history.extend([HumanMessage(content=user_input), response])

            create_output_speech(client, response, voice=voice_dict[voice_key])

            conversation_history = st.session_state.get('conversation_history', [])
            conversation_history.append(('You', user_input))
            conversation_history.append(('Bot', response))
            st.session_state['conversation_history'] = conversation_history

            for role, text in conversation_history:
                st.markdown(f'**{role}**: {text}')

            audio_base64 = convert_audio_to_base64('speech.wav')
            st.markdown(f'<audio controls autoplay><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>', unsafe_allow_html=True)

# Background image CSS
def add_background():
    st.markdown(
        """
        <style>
        /* Background image with overlay */
        .stApp {
            background-image: linear-gradient(rgba(255, 255, 255, 0.7), rgba(255, 255, 255, 0.7)), 
                              url('https://bombardier.com/sites/default/files/styles/retina_2700x860/public/2024-10/Inflight-Bombardier-Challenger-650---clouds-2700x1200.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }

        /* Main container styles for readability */
        .main-container {
            background-color: rgba(255, 255, 255, 0.8); /* slightly opaque white */
            padding: 2rem;
            border-radius: 8px;
        }

        /* Customize headers and text */
        h1, h2, h3, h4 {
            color: #003366; /* Bombardier-inspired dark blue */
        }
        
        .stMarkdown p {
            font-size: 18px;
            color: #333333; /* Dark gray for readability */
        }

        /* Adjust text input fields and buttons */
        .stTextInput > div > input, .stButton button {
            border-radius: 8px;
            padding: 0.5rem;
            font-size: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


## main 
def main():
    # Apply custom background and styling
    add_background()

    # Main UI content
    st.header('THICKET: Learning and Education Training')

    # Add voice key selection
    voice_key = st.selectbox('Select Voice:', list(voice_dict.keys()))

    # Explanation textbox
    explanation_text = """
    #### This application contains two sample collections:
    
    1. **Bombardier challenger Pilot Training Manual**: Pilot Training Manual for Model CL-600-2B16
    
    2. **Ontario Health Best Practices**: Best Practices for Cleaning, Disinfection and Sterilization of Medical Equipment/Devices In All Health Care Settings
    """
    
    # Wrap main content in a container for background color effect
    with st.container():
        st.markdown(explanation_text)
    
    audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=41_000)
    client = OpenAI()

    llm = initialize_llm(OPENAI_KEY=os.environ["OPENAI_API_KEY"])
    memory = initialize_memory()

    contextualize_q_system_prompt, qa_system_prompt = get_system_prompts()
    contextualize_q_prompt, qa_prompt = create_prompts(contextualize_q_system_prompt, qa_system_prompt)

    question_answer_chain = initialize_question_answer_chain(llm, qa_prompt)
    retriever = initialize_BM25Retriever()

    # Main chatbot function
    run_chatbot(client, llm, retriever, contextualize_q_prompt, question_answer_chain, voice_key, audio_bytes)

if __name__ == '__main__':
    main()
    
