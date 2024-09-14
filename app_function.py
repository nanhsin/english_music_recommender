import streamlit as st
from streamlit_js_eval import streamlit_js_eval
import choosingdata as choice
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub


def get_text_chunks(text):
    """
    Splits the given text into chunks based on specified character settings.

    Parameters:
    - text (str): The text to be split into chunks.

    Returns:
    - list: A list of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """
    Generates a vector store from a list of text chunks using specified embeddings.

    Parameters:
    - text_chunks (list of str): Text segments to convert into vector embeddings.

    Returns:
    - FAISS: A FAISS vector store containing the embeddings of the text chunks.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """
    Initializes a conversational retrieval chain that uses a large language model
    for generating responses based on the provided vector store.

    Parameters:
    - vectorstore (FAISS): A vector store to be used for retrieving relevant content.

    Returns:
    - ConversationalRetrievalChain: An initialized conversational chain object.
    """
    llm = ChatOpenAI(
        model_name="gpt-4-1106-preview",
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def set_prompt(text_block):
    """
    Callback function that sets the chosen prompt in the session state.

    Parameters:
    - text_block (str): The prompt text selected by the user.
    """
    st.session_state["messages"].append({"role": "user", "content": text_block})
    st.session_state["prompts"] = text_block


def prompts():
    """
    Renders clickable buttons for predefined prompts in the Streamlit application,
    allowing the user to select a prompt to send to the conversation chain.
    """
    potential_prompts = [
        f"What is the meaning of the song {st.session_state['title']} by {st.session_state['artist']}?",
        f"What is the most difficult English grammar point in the song {st.session_state['title']} by {st.session_state['artist']}? Can you explain it?",
        f"What is the most common English word in the song {st.session_state['title']} by {st.session_state['artist']} (excluding stopwords)? Can you give some example sentences using that word?",
        f"What is the most worth learning English phrase in the song {st.session_state['title']} by {st.session_state['artist']}? Can you explain it and provide practical example using the phrase?",
    ]
    chosen_prompt = None
    for index, text_block in enumerate(potential_prompts):
        st.button(
            f"Prompt {index + 1}: {text_block}", on_click=set_prompt, args=(text_block,)
        )


def get_lyrics():
    """
    Retrieves the lyrics stored in the session state.

    Returns:
    - str: The lyrics of the currently selected song.
    """
    lyrics = st.session_state["lyrics"]
    return lyrics


def page_title():
    """
    Sets the title of the Streamlit page based on the selected song and artist.
    """
    if st.session_state["title"] and st.session_state["artist"]:
        st.title(
            f'🎵 English Music Recommender 💬  ({st.session_state["title"]} by {st.session_state["artist"]})'
        )
    else:
        st.title("🎵 English Music Recommender 💬")


def chat_sidebar():
    """
    Renders the sidebar in the Streamlit application for selecting music preferences
    and handling song recommendations.
    """
    with st.sidebar:
        st.title("💚 Music Preferences")

        user_difficulty = st.sidebar.radio(
            "Choose a difficulty level:", ("Easy", "Medium", "Hard")
        )

        user_danceability = st.sidebar.radio(
            "How much do you want to dance?", ("Low", "Medium", "High")
        )

        user_valence = st.sidebar.radio(
            "What energy are you feeling?", ("Negative", "Neutral", "Positive")
        )

        if not st.session_state["song_bool"]:

            if st.sidebar.button("Submit"):
                df = choice.process_data("data.json")
                recommendations = choice.recommendation(
                    df,
                    dance_choice=user_danceability,
                    valence_choice=user_valence,
                    difficulty_choice=user_difficulty,
                )

                st.session_state["title"] = recommendations["title"].values[0]
                st.session_state["artist"] = recommendations["artist"].values[0]
                st.session_state["lyrics"] = recommendations["lyrics"].values[0]
                st.session_state["id"] = (
                    f'https://open.spotify.com/track/{recommendations["id"].values[0]}'
                )
                st.session_state["song_bool"] = True

                st.rerun()

        else:
            if st.session_state["song_bool"]:

                st.write("### We would recommend you...")
                st.write(f"## {st.session_state['title']}")
                st.write(f" by {st.session_state['artist']}")
                st.markdown(
                    f'<a href="{st.session_state["id"]}"><img src="{st.session_state["icon"]}" alt="Clickable image" style="height:60px;"></a>',
                    unsafe_allow_html=True,
                )
                st.write("Please refresh the page for a new recommendation.")
                if st.button("Reload page"):
                    streamlit_js_eval(js_expressions="parent.window.location.reload()")


def chat():
    """
    Manages the chat interface in the Streamlit application, handling the conversation
    flow and displaying the chat history.
    """
    if st.session_state["lyrics"]:

        text_chunks = get_text_chunks(get_lyrics())
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation_chain(vectorstore)

        if len(st.session_state.messages) == 1:
            message = st.session_state.messages[0]
            with st.chat_message(message["role"]):
                st.write(message["content"])
                prompts()

        else:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        # User-provided prompt
        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.prompts = prompt
            with st.chat_message("user"):
                st.write(prompt)

        if st.session_state.messages[-1]["role"] != "system":

            with st.chat_message("system"):
                with st.spinner("Generating response..."):
                    response = st.session_state.conversation.invoke(
                        {"question": st.session_state.prompts}
                    )
                    st.session_state.chat_history = response["chat_history"]
                    message = st.session_state.chat_history[-1]
                    st.write(message.content)
                    message = {"role": "system", "content": message.content}
                    st.session_state.messages.append(message)

    else:
        st.write("You can chat with GPT once a song has been recommended to you!")


def init():
    """
    Initializes the session state variables used in the Streamlit application and
    loads environment variables.
    """
    load_dotenv()

    if "title" not in st.session_state:
        st.session_state["title"] = ""
    if "artist" not in st.session_state:
        st.session_state["artist"] = ""
    if "icon" not in st.session_state:
        st.session_state["icon"] = (
            "https://thereceptionist.com/wp-content/uploads/2021/02/Podcast-Listen-On-Spotify-1.png"
        )
    if "id" not in st.session_state:
        st.session_state["id"] = ""
    if "song_bool" not in st.session_state:
        st.session_state["song_bool"] = False
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {
                "role": "system",
                "content": "What do you want to learn about? Here are some suggested prompts: ",
            }
        ]
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "lyrics" not in st.session_state:
        st.session_state["lyrics"] = ""
    if "prompts" not in st.session_state:
        st.session_state["prompts"] = ""
