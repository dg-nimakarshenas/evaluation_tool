import streamlit as st
import os
from io import BytesIO
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import BaseCallbackHandler

# --- Page configuration ---
st.set_page_config(page_title="Feedback portal", page_icon="⭐")

PROMPT_TEMPLATES = {
    "resident": """
    You are a multi-lingual support officer for the SHDF retrofit program in the Royal Borough of Greenwich. Your task is to listen to the feedback of residents, and aim to ask follow up questions 
    that will help draw out as much of the residents' feedback regarding the SHDF program as possible, as long as the resident seems willing to answer them. You will be provided with a list of 
    'target' questions that you should aim to ask the resident, but you should also be able to ask other questions that are not on the list if they are relevant to the conversation, but
    make sure that you do not veer away from the topic of the SHDF program. You should also be able to ask clarifying questions if the resident's feedback is not clear.
    Initially, ask an open-ended question to get the resident talking about their experience with the SHDF program.
    
    
    # Instructions:
    - Do not discuss prohibited topics (politics, religion, controversial current events, medical, legal, or financial advice, personal conversations, internal company operations, or criticism of any people or company).
    - Rely on sample phrases whenever appropriate, but never repeat a sample phrase in the same conversation. Feel free to vary the sample phrases to avoid sounding repetitive and make it more appropriate for the user.
    - Maintain a professional and concise tone in all responses, and use emojis between sentences.
    - Your role is to listen and ask questions, you are not able to answer questions or provide information about the SHDF program.
    - You are not able to provide any information about the SHDF program, including the eligibility criteria, the application process, or the timeline for the program.
    - Do not ask the same question twice in the same conversation.
    - If you feel an answer sufficiently answers other questions, you can skip those questions.
    - If the user asks a question that is not related to the SHDF program, you should respond with "I'm sorry but I cannot answer that question. 
      My role is to listen to your feedback regarding the SHDF program and ask follow up questions to help draw out as much of your feedback as possible."

    # Target Questions:
    - How have you felt regrarding the communication you have received from the council, as well as the contractors?
    - How do you feel about the contractors that have been working on your home? Have they been respectful and professional?
    - How do you feel about the work that has been done on your home? Are you happy with the results?
    - How do you feel about the impact that the work has had on your home? Has it been disruptive or inconvenient?
    - Are you happy with the way that the work has been carried out? Have there been any issues or problems that you have encountered?
    - How do you feel about the way that the work has been communicated to you? Have you been kept informed about what is happening and when?
    - How do you feel about the way that the work has been managed? Have there been any delays or issues that you have encountered?
    - Are you confident that the work will resolve the issues that you have been experiencing in your home?
    - Do you think that the work will lead to a reduction in your energy bills?
    """,
    "contractor": """""",
    "staff":"""""", 
    "translator": """You are a simple translator. Your task is to translate the text that you are given into the language that is specified in the input. Note that 
    the context of what you are translating is that you are a feedback officer for the SHDF retrofit program in the Royal Borough of Greenwich."""}

# Initialize page state
if "page" not in st.session_state:
    st.session_state.page = "form"

# --- Setup form ---
if st.session_state.page == "form":
    st.title('Welcome to the SHDF Feedback Chatbot!')
    st.write("""This chatbot is designed to assist you with your feedback and inquiries.
    Your responses will help us tailor the experience to your needs.
    Please note that this is a demo version and may not reflect the final product
    We appreciate your feedback!         
    Please fill out the form below to get started.""")    
    st.header('Please select your preferences')
    language = st.selectbox(
        'Which language would you like to communicate in?',
        options=["English", "French", "Spanish", "Hindi"] + sorted(["Mandarin Chinese",  "German",
                "Russian", "Arabic", "Italian", "Korean", "Punjabi", "Bengali",
                "Portuguese", "Indonesian", "Urdu", "Persian (Farsi)", "Vietnamese",
                "Polish", "Samoan", "Thai", "Ukrainian", "Turkish", "Norwegian", "Dutch",
                "Greek", "Romanian", "Swahili", "Hungarian", "Hebrew", "Swedish", "Czech",
                "Finnish", "Tagalog", "Burmese", "Tamil", "Kannada", "Pashto", "Yoruba",
                "Malay", "Haitian Creole", "Nepali", "Sinhala", "Catalan", "Malagasy",
                "Latvian", "Lithuanian", "Estonian", "Somali", "Maltese", "Corsican",
                "Luxembourgish", "Occitan", "Welsh", "Albanian", "Macedonian", "Icelandic",
                "Slovenian", "Galician", "Basque", "Azerbaijani", "Uzbek", "Kazakh",
                "Mongolian", "Lao", "Telugu", "Marathi", "Chichewa", "Esperanto",
                "Tajik", "Yiddish", "Zulu", "Sundanese", "Tatar", "Tswana"
            ])
    )
    role = st.radio(
        'Are you a resident or a contractor?',
        options=['Resident', 'Contractor']
    )
    if st.button('Submit'):
        st.session_state.language = language
        st.session_state.role = role
        # 1) Clear out any old chat messages
        if "messages" in st.session_state:
            st.session_state.pop("messages", None)
        # 2) Reset the page‐marker so your decorator will know
        #    it’s a “new” run of ContextChatbot.main()
        st.session_state.page = "chat"
        st.rerun()

# --- Chat interface ---
elif st.session_state.page == "chat":
    st.title('SHDF Feedback Chatbot')
    st.write(f"**Role:** {st.session_state.role}")


    # Initialize chat history
    logger = st.get_logger('Langchain-Chatbot') if hasattr(st, 'get_logger') else None
    def enable_chat_history(func):
        def wrapper(*args, **kwargs):
            if os.environ.get("OPENAI_API_KEY"):
                page = func.__qualname__
                if st.session_state.get("current_page") != page:
                    st.session_state["current_page"] = page
                    st.session_state.pop("messages", None)
            return func(*args, **kwargs)
        return wrapper
    
    def display_msg(msg, author):
        st.session_state["messages"].append({"role": author, "content": msg})
        st.chat_message(author).write(msg)

    def configure_llm():
        return ChatOpenAI(
            model_name="gpt-4.1-mini-2025-04-14",
            temperature=0,
            streaming=True,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text = initial_text
        def on_llm_new_token(self, token: str, **kwargs):
            self.text += token
            self.container.markdown(self.text)

    class ContextChatbot:
        def __init__(self):
            self.llm = configure_llm()


        def setup_chain(_self):
            # Return messages instead of a concatenated string for the history variable
            print("in setup_chain")
            memory = ConversationBufferMemory(return_messages=True)
            role_key = st.session_state.role.lower()
            system_template = PROMPT_TEMPLATES.get(role_key, "")
            system_template+= f"You should be communicating in {st.session_state.language}, you are trying to ask the same questions, but in the selected language."
            system_message = SystemMessagePromptTemplate.from_template(system_template)
            messages = [
                system_message,
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
            prompt = ChatPromptTemplate.from_messages(messages)
            return ConversationChain(llm=_self.llm, memory=memory, prompt=prompt, verbose=False)

        def get_translation(self, text, language):
            prompt = ChatPromptTemplate.from_messages([
                        ("system",
                        PROMPT_TEMPLATES.get("translator", "")),
                        ("human", f"Translate the following text into {language}:\n{{input}}"),
                ])
            translation_chain = prompt | self.llm
            with st.chat_message("assistant"):
                translation_chain.invoke({"input": text}, {"callbacks": [StreamHandler(st.empty())]})
            return text
        
        def change_language(self, chain):
            chain.prompt.messages[0] = SystemMessagePromptTemplate.from_template(PROMPT_TEMPLATES.get(st.session_state.role.lower(), "") + 
                                                                                     f"You should be communicating in {st.session_state.language}, you are trying to ask the same questions, but in the selected language.")
            if "messages" in st.session_state:
                print(st.session_state.language)
                assisstant_msgs = [message for message in st.session_state["messages"] if message["role"] == "assistant"]
                translation = self.get_translation(assisstant_msgs[-1]["content"], st.session_state.language)
                st.session_state["messages"].append({"role": 'system', "content": f"You should continue where you left off, but in the selected language, {st.session_state.language}."})
                print("hello")

        @enable_chat_history
        def main(self):
            chain = self.setup_chain()
            st.selectbox(key='language',
                label='You can change the language here:',
                options=list(dict.fromkeys([st.session_state.language] + ["English", "French", "Spanish", "Hindi"] + sorted(["Mandarin Chinese",  "German",
                        "Russian", "Arabic", "Italian", "Korean", "Punjabi", "Bengali",
                        "Portuguese", "Indonesian", "Urdu", "Persian (Farsi)", "Vietnamese",
                        "Polish", "Samoan", "Thai", "Ukrainian", "Turkish", "Norwegian", "Dutch",
                        "Greek", "Romanian", "Swahili", "Hungarian", "Hebrew", "Swedish", "Czech",
                        "Finnish", "Tagalog", "Burmese", "Tamil", "Kannada", "Pashto", "Yoruba",
                        "Malay", "Haitian Creole", "Nepali", "Sinhala", "Catalan", "Malagasy",
                        "Latvian", "Lithuanian", "Estonian", "Somali", "Maltese", "Corsican",
                        "Luxembourgish", "Occitan", "Welsh", "Albanian", "Macedonian", "Icelandic",
                        "Slovenian", "Galician", "Basque", "Azerbaijani", "Uzbek", "Kazakh",
                        "Mongolian", "Lao", "Telugu", "Marathi", "Chichewa", "Esperanto",
                        "Tajik", "Yiddish", "Zulu", "Sundanese", "Tatar", "Tswana"
                    ]))), 
                on_change = self.change_language(chain),
            )
            if "messages" not in st.session_state:
                chain.memory.clear()
                self.change_language(chain)
                with st.chat_message("assistant"):
                    handler = StreamHandler(st.empty())
                    resp = chain.invoke({"input": ""}, {"callbacks": [handler]})
                    answer = resp.get("response") or resp
                st.session_state["messages"] = []
                st.session_state["messages"].append({"role": 'assistant', "content": answer})
            
            client = OpenAI()
            user_query = st.chat_input(placeholder="Write your message here...")
            audio_file = st.audio_input("You can also record a voice message in your preferred language instead of typing! If you'd like to " \
            "record a voice message, press the 'bin' button to clear the chat history and then press the 'record' button to start recording.")

            if user_query:
                display_msg(user_query, 'user')
                with st.chat_message("assistant"):
                    handler = StreamHandler(st.empty())
                    resp = chain.invoke({"input": user_query}, {"callbacks": [handler]})
                    answer = resp.get("response") or resp
                st.session_state["messages"].append({"role": 'assistant', "content": answer})


            if audio_file and st.button("Transcribe Audio"):
                # Read bytes and wrap in Blob
                st.write("Transcribing...")
                # Treat transcription as user input
                transcript = client.audio.transcriptions.create(
                                        model="gpt-4o-mini-transcribe",
                                        file = audio_file,
                )
                transcript_text = transcript.text
                with st.chat_message("user"):
                    st.write(transcript_text)
                st.session_state.messages.append({"role": "user", "content": transcript_text})
                with st.chat_message("assistant"):
                    handler = StreamHandler(st.empty())
                    resp = chain.invoke({"input": transcript_text}, {"callbacks": [handler]})
                    answer = resp.get("response") or resp
                st.session_state["messages"].append({"role": 'assistant', "content": answer})
                audio_file.close()
                
    chatbot = ContextChatbot()
    chatbot.main()
