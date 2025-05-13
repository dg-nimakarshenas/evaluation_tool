import streamlit as st
import os
from io import BytesIO
from langchain_openai import ChatOpenAI
from openai import OpenAI # Import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage # Import message types

# --- Assume PROMPT_TEMPLATES and page setup code from above exists ---

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
    "contractor": """(Placeholder for Contractor Prompt)""", # Added placeholder
    "staff":"""(Placeholder for Staff Prompt)""", # Added placeholder
    "translator": """You are a simple translator. Your task is to translate the text that you are given into the language that is specified in the input. Note that
    the context of what you are translating is that you are a feedback officer for the SHDF retrofit program in the Royal Borough of Greenwich. Respond only with the translation, nothing else."""} # Added instruction for translator

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
    # Define language options (ensure unique keys if needed later)
    language_options = ["English", "French", "Spanish", "Hindi"] + sorted([
        "Mandarin Chinese", "German", "Russian", "Arabic", "Italian", "Korean", "Punjabi", "Bengali",
        "Portuguese", "Indonesian", "Urdu", "Persian (Farsi)", "Vietnamese", "Polish", "Samoan",
        "Thai", "Ukrainian", "Turkish", "Norwegian", "Dutch", "Greek", "Romanian", "Swahili",
        "Hungarian", "Hebrew", "Swedish", "Czech", "Finnish", "Tagalog", "Burmese", "Tamil",
        "Kannada", "Pashto", "Yoruba", "Malay", "Haitian Creole", "Nepali", "Sinhala", "Catalan",
        "Malagasy", "Latvian", "Lithuanian", "Estonian", "Somali", "Maltese", "Corsican",
        "Luxembourgish", "Occitan", "Welsh", "Albanian", "Macedonian", "Icelandic", "Slovenian",
        "Galician", "Basque", "Azerbaijani", "Uzbek", "Kazakh", "Mongolian", "Lao", "Telugu",
        "Marathi", "Chichewa", "Esperanto", "Tajik", "Yiddish", "Zulu", "Sundanese", "Tatar", "Tswana"
    ])
    language = st.selectbox(
        'Which language would you like to communicate in?',
        options=language_options,
        key='selected_language' # Use a distinct key for the selection widget
    )
    role = st.radio(
        'Are you a resident or a contractor?',
        options=['Resident', 'Contractor'],
        key='selected_role' # Use a distinct key
    )
    if st.button('Submit'):
        st.session_state.language = st.session_state.selected_language # Store in standard keys
        st.session_state.role = st.session_state.selected_role
        # Clear ALL chat-related state when submitting the form to start fresh
        st.session_state.pop("messages", None)
        st.session_state.pop("chain", None)
        st.session_state.pop("initial_message_sent", None)
        st.session_state.pop("current_page", None) # Reset page marker
        st.session_state.pop("display_translated_message", None) # Clear any pending translated message

        st.session_state.page = "chat"
        st.rerun()


# --- Chat interface ---
elif st.session_state.page == "chat":
    st.title('SHDF Feedback Chatbot')
    # Display selected role and language if they exist
    if "role" in st.session_state:
        st.write(f"**Role:** {st.session_state.role}")
    if "language" in st.session_state:
        st.write(f"**Language:** {st.session_state.language}")


    # --- Helper Functions and Classes ---

    # Initialize chat history decorator
    def enable_chat_history(func):
        def wrapper(*args, **kwargs):
            if os.environ.get("OPENAI_API_KEY"):
                page = func.__qualname__
                if st.session_state.get("current_page") != page:
                    print(f"Setting current page context to: {page}")
                    st.session_state["current_page"] = page
            else:
                 st.error("OpenAI API Key not found. Please set the OPENAI_API_KEY environment variable.")
                 st.stop()
            return func(*args, **kwargs)
        return wrapper

    # Function to display messages
    def display_msg(msg_content, author_role):
        if "messages" not in st.session_state:
             st.session_state.messages = []
        st.session_state.messages.append({"role": author_role, "content": msg_content})
        with st.chat_message(author_role):
            st.write(msg_content)

    # Configure LLM
    def configure_llm():
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OPENAI_API_KEY environment variable not set.")
            return None
        try:
            return OpenAI(), ChatOpenAI(
                model_name="gpt-4.1-mini-2025-04-14",
                temperature=0,
                streaming=True,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        except Exception as e:
            st.error(f"Failed to initialize LLM: {e}")
            return None


    # Handler for streaming output
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text = initial_text
        def on_llm_new_token(self, token: str, **kwargs):
            self.text += token
            self.container.markdown(self.text + "▌")
        def on_llm_end(self, response, **kwargs):
             self.container.markdown(self.text)


    # Main Chatbot Class
    class ContextChatbot:
        def __init__(self):
            self.audio_llm, self.llm = configure_llm()
            api_key = os.getenv("OPENAI_API_KEY")
            self.chat_input_placeholder = "Write your message here..." if st.session_state.language == "English" else self.translate_text("Write your message here...", st.session_state.language)
            self.upload_button_text = "Send Audio" if st.session_state.language == "English" else self.translate_text("Send Audio", st.session_state.language)
            if not api_key:
                 self.client = None
                 st.stop()
            else:
                 try:
                     self.client = OpenAI(api_key=api_key)
                 except Exception as e:
                     st.error(f"Failed to initialize OpenAI client: {e}")
                     self.client = None
                     st.stop()

        # Helper function for simple translation
        def translate_text(self, text_to_translate, target_language):
            if not self.llm or not text_to_translate:
                return None

            print(f"Attempting to translate to {target_language}: '{text_to_translate[:50]}...'")
            try:
                translate_prompt = ChatPromptTemplate.from_messages([
                    ("system", PROMPT_TEMPLATES["translator"]),
                    ("human", f"Translate the following text into {target_language}:\n\n{text_to_translate}")
                ])
                # Use invoke for a single translation call
                response = self.llm.invoke(translate_prompt.format_prompt(text=text_to_translate).to_messages())
                translated_text = response.content
                print(f"Translation result: '{translated_text[:50]}...'")
                return translated_text
            except Exception as e:
                print(f"Error during translation: {e}")
                st.warning(f"Could not translate the previous message due to an error: {e}")
                return None


        def setup_chain(self):
            if "chain" in st.session_state and st.session_state.chain:
                 print("Retrieving existing ConversationChain from session state.")
                 return st.session_state.chain

            if not self.llm: return None

            print("Setting up new ConversationChain...")
            messages_history = st.session_state.get("messages", [])
            memory = ConversationBufferMemory(memory_key="history", return_messages=True)

            for msg in messages_history:
                 if msg["role"] == "user":
                     memory.chat_memory.add_user_message(msg["content"])
                 elif msg["role"] == "assistant":
                     memory.chat_memory.add_ai_message(msg["content"])
                 elif msg["role"] == "system":
                     memory.chat_memory.add_message(SystemMessage(content=msg["content"]))

            role_key = st.session_state.role.lower()
            language = st.session_state.language
            system_template = PROMPT_TEMPLATES.get(role_key, "You are a helpful assistant.")
            system_template += f"\n\nYou must communicate ONLY in {language}. Ask questions relevant to the SHDF program feedback based on the user's role ({st.session_state.role})."
            system_message = SystemMessagePromptTemplate.from_template(system_template)

            messages_prompt = [
                system_message,
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
            prompt = ChatPromptTemplate.from_messages(messages_prompt)

            try:
                chain = ConversationChain(llm=self.llm, memory=memory, prompt=prompt, verbose=True)
                print("ConversationChain setup complete.")
                st.session_state.chain = chain
                return chain
            except Exception as e:
                 st.error(f"Failed to create ConversationChain: {e}")
                 return None

        def change_language_callback(self):
            # This callback runs ONLY when the selectbox value changes
            if "chain" not in st.session_state or not st.session_state.chain:
                 print("Warning: Chain not found in session state during language change.")
                 st.error("An error occurred. Please refresh the page or restart the chat.")
                 return

            chain = st.session_state.chain
            new_language = st.session_state.language # Streamlit updates this key

            print(f"Language change callback triggered. New language: {new_language}")

            # 1. Find the last assistant message in the official history
            last_assistant_message_content = None
            if "messages" in st.session_state:
                for msg in reversed(st.session_state.messages):
                    if msg.get("role") == "assistant":
                        last_assistant_message_content = msg.get("content")
                        break

            # 2. Update the system prompt in the existing chain object
            role_key = st.session_state.role.lower()
            system_template = PROMPT_TEMPLATES.get(role_key, "You are a helpful assistant.")
            system_template += f"\n\nYou must communicate ONLY in {new_language}. Ask questions relevant to the SHDF program feedback based on the user's role ({st.session_state.role})."

            try:
                chain.prompt.messages[0] = SystemMessagePromptTemplate.from_template(system_template)
                print("System prompt updated in chain.")

                # 3. Add system guidance message to history and memory (for context)
                system_guidance = f"System Notification: The conversation language has now changed to {new_language}. Please continue the conversation ONLY in {new_language}."
                st.session_state.setdefault("messages", []).append({"role": 'system', "content": system_guidance})
                if hasattr(chain.memory, 'chat_memory'):
                     chain.memory.chat_memory.add_message(SystemMessage(content=system_guidance))
                     print("System guidance message added to history and memory.")
                else:
                     print("Warning: chat_memory not found on chain.memory.")
                self.chat_input_placeholder = self.translate_text(self.chat_input_placeholder, new_language)
                self.upload_button_text = self.translate_text(self.upload_button_text, new_language)    
                # 4. Translate the last assistant message (if found) and store for later display
                st.session_state.pop("display_translated_message", None) # Clear any previous pending message
                if last_assistant_message_content:
                    print("Attempting to translate the last assistant message.")
                    with st.spinner(f"Translating last message to {new_language}..."):
                         translated_content = self.translate_text(last_assistant_message_content, new_language)

                    if translated_content:
                         print("Storing translated message for display.")
                         # Store the translated message to be displayed in the main function flow
                         st.session_state.display_translated_message = translated_content
                    else:
                         print("Translation failed or returned empty.")
                st.rerun() # Rerun to refresh the page with the new language
                         # Optionally store a fallback message if translation fails
                         # st.session_state.display_translated_message = f"(Could not translate previous message to {new_language})"

                # --- REMOVED direct display from here ---
                # with st.chat_message("assistant"):
                #      st.write(translated_content) # REMOVED

            except Exception as e:
                print(f"Error during language change processing: {e}")
                st.error(f"Error applying language change: {e}")

            # A rerun might still be needed implicitly by Streamlit due to state change

        @enable_chat_history
        def main(self):
            if not self.llm or not self.client:
                 st.error("Chatbot initialization failed. Cannot proceed.")
                 st.stop()

            chain = self.setup_chain()
            if not chain:
                 st.error("Failed to initialize or retrieve conversation chain.")
                 st.stop()

            # --- Language Selection ---
            language_options = list(dict.fromkeys([st.session_state.language] + ["English", "French", "Spanish", "Hindi"] + sorted([
                "Mandarin Chinese", "German", "Russian", "Arabic", "Italian", "Korean", "Punjabi", "Bengali",
                "Portuguese", "Indonesian", "Urdu", "Persian (Farsi)", "Vietnamese", "Polish", "Samoan",
                "Thai", "Ukrainian", "Turkish", "Norwegian", "Dutch", "Greek", "Romanian", "Swahili",
                "Hungarian", "Hebrew", "Swedish", "Czech", "Finnish", "Tagalog", "Burmese", "Tamil",
                "Kannada", "Pashto", "Yoruba", "Malay", "Haitian Creole", "Nepali", "Sinhala", "Catalan",
                "Malagasy", "Latvian", "Lithuanian", "Estonian", "Somali", "Maltese", "Corsican",
                "Luxembourgish", "Occitan", "Welsh", "Albanian", "Macedonian", "Icelandic", "Slovenian",
                "Galician", "Basque", "Azerbaijani", "Uzbek", "Kazakh", "Mongolian", "Lao", "Telugu",
                "Marathi", "Chichewa", "Esperanto", "Tajik", "Yiddish", "Zulu", "Sundanese", "Tatar", "Tswana"
            ])))

            current_language = st.session_state.language
            if current_language not in language_options:
                print(f"Warning: Current language '{current_language}' not in options. Defaulting to English.")
                st.session_state.language = "English"
                current_language = "English"
                st.rerun()

            st.selectbox(
                key='language',
                label='You can change the language here:',
                options=language_options,
                index=language_options.index(current_language),
                on_change=self.change_language_callback
            )

            # --- Initialize Chat History and First Message ---
            if "messages" not in st.session_state:
                st.session_state.messages = []
                print("Messages list initialized.")

            if "initial_message_sent" not in st.session_state and not st.session_state.messages:
                print("Generating initial assistant message (first time only)...")
                with st.chat_message("assistant"):
                    msg_placeholder = st.empty()
                    handler = StreamHandler(msg_placeholder)
                    try:
                        resp = chain.invoke({"input": ""}, {"callbacks": [handler]})
                        answer = resp.get("response") if isinstance(resp, dict) else resp
                        if answer:
                            st.session_state.messages.append({"role": 'assistant', "content": answer})
                            print("Initial assistant message generated and added.")
                        else:
                            print("Warning: Initial assistant response was empty.")
                            fallback_msg = f"Hello! How can I help you with your feedback today in {st.session_state.language}?"
                            st.session_state.messages.append({"role": 'assistant', "content": fallback_msg})
                            msg_placeholder.markdown(fallback_msg)

                        st.session_state.initial_message_sent = True
                        print("Initial message flag set.")
                        st.rerun()

                    except Exception as e:
                        print(f"Error invoking chain for initial message: {e}")
                        st.error("Sorry, I couldn't start the conversation.")
                        error_msg = "Error starting conversation."
                        st.session_state.messages.append({"role": 'assistant', "content": error_msg})
                        msg_placeholder.markdown(error_msg)


            # --- Display Chat Messages ---
            # Display all non-system messages from the official history
            for msg in st.session_state.get("messages", []):
                 if msg.get("role") != "system":
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

            # --- Display Pending Translated Message (if any) --- ADDED THIS BLOCK
            if "display_translated_message" in st.session_state and st.session_state.display_translated_message:
                print("Displaying pending translated message.")
                with st.chat_message("assistant"):
                    st.write(st.session_state.display_translated_message)
                # Clear the message after displaying it
                st.session_state.pop("display_translated_message", None)

            # --- User Input Handling (Text and Audio) ---

            # Text Input
            user_query = st.chat_input(placeholder=self.chat_input_placeholder)

            # Audio Input (Upload)
            audio_file = st.audio_input("You can also record a voice message in your preferred language instead of typing! If you'd like to " \
            "record a voice message, press the 'bin' button to clear the chat history and then press the 'record' button to start recording.")

            # --- Process Inputs ---
            processed_input = None # Variable to hold the input to send to the LLM
            send_audio_button = st.button(f"✅ {self.upload_button_text}")

            if user_query:
                print(f"Processing text input: {user_query}")
                processed_input = user_query
                # Display user message immediately
                display_msg(user_query, 'user')
            
            elif audio_file and send_audio_button:
                # Read bytes and wrap in Blob
                st.write("Transcribing...")
                # Treat transcription as user input
                transcript = self.audio_llm.audio.transcriptions.create(
                                        model="gpt-4o-mini-transcribe",
                                        file = audio_file,
                )
                transcript_text = transcript.text
                processed_input = transcript_text
                # Display transcript as user message
                display_msg(transcript_text, 'user')


            # --- LLM Invocation (if input was processed) ---
            if processed_input:
                with st.chat_message("assistant"):
                    msg_placeholder = st.empty()
                    handler = StreamHandler(msg_placeholder)
                    try:
                        # Use the chain from session state
                        resp = chain.invoke({"input": processed_input}, {"callbacks": [handler]})
                        answer = resp.get("response") if isinstance(resp, dict) else resp
                        if answer:
                             # Add assistant response to state *after* generation
                             st.session_state.messages.append({"role": 'assistant', "content": answer})
                        else:
                             print("Warning: Assistant response was empty.")
                             fallback_ans = "..."
                             st.session_state.messages.append({"role": 'assistant', "content": fallback_ans})
                             msg_placeholder.markdown(fallback_ans) # Display placeholder

                    except Exception as e:
                        print(f"Error invoking chain for user input: {e}")
                        st.error("Sorry, I encountered an error processing your message.")
                        error_msg = "Error processing message."
                        st.session_state.messages.append({"role": 'assistant', "content": error_msg})
                        msg_placeholder.markdown(error_msg)

                # Rerun to clear text input and potentially reset file uploader
                st.rerun()


    # --- Run the Chatbot ---
    if "role" in st.session_state and "language" in st.session_state:
        chatbot = ContextChatbot()
        chatbot.main()
    else:
        st.warning("Role or language not selected. Please go back to the form.")
        if st.button("Back to Form"):
             st.session_state.page = "form"
             # Explicitly clear state when navigating back
             st.session_state.pop("messages", None)
             st.session_state.pop("chain", None)
             st.session_state.pop("initial_message_sent", None)
             st.session_state.pop("current_page", None)
             st.rerun()
