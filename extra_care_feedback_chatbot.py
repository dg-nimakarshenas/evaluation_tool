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
import json # Added for saving conversation data
import time
import datetime # CHANGED: No longer importing timedelta
import psycopg2 # Added for PostgreSQL integration
from psycopg2 import sql # For safe SQL query construction if needed
from datetime import datetime as dt # REFINED: Using an alias for clarity

today = dt.now().strftime("%d %B %Y")
# --- Database Configuration ---
# IMPORTANT: Replace these with your actual PostgreSQL connection details
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres.oedyrmoxycaidzifeucl")
DB_PASSWORD = os.getenv("DB_PASSWORD", "4Abetterfuture!")
DB_HOST = os.getenv("DB_HOST", "aws-0-eu-west-2.pooler.supabase.com")
DB_PORT = os.getenv("DB_PORT", "6543")

# --- Database Connection Function with Streamlit Caching ---
@st.cache_resource  # This decorator does the magic!
def get_db_connection():
    """
    Establishes and returns a database connection.
    Streamlit's @st.cache_resource ensures this function's core logic
    (connecting to the DB) runs only once per session unless the cache is cleared.
    """
    print("Attempting to establish database connection (cached resource)...")
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
        )
        # Test the connection
        with conn.cursor() as cur_test:
            cur_test.execute("SELECT 1")
        print(f"Database connection successfully established to {DB_HOST}:{DB_PORT} via cache_resource.")
        return conn
    except psycopg2.Error as e:
        print(f"Failed to connect to database (cache_resource): {e}")
        st.error(f"Database Connection Error: {e}. Please check settings or server status.")
        return None # Return None if connection fails

# --- Attempt to establish DB connection when app loads/script runs ---
initial_conn_on_load = get_db_connection()

@st.cache_resource
def get_llm_for_translation():
    """
    Initializes and caches a ChatOpenAI model instance specifically for UI translation.
    This prevents re-initializing the model on every text translation.
    """
    print("Initializing LLM for UI translation (cached)...")
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OPENAI_API_KEY is not set. Translation services will not be available.")
            return None
        return ChatOpenAI(
            # REFINED: Using a more standard and robust model for this task.
            model_name="gpt-4o-mini",
            temperature=0,
            api_key=api_key
        )
    except Exception as e:
        print(f"Failed to initialize translation LLM: {e}")
        st.error(f"Could not initialize translation service: {e}")
        return None

def translate_text(text_to_translate, target_language, llm):
    """
    Global function to translate a given piece of text to a target language using the provided LLM.
    """
    # FIXED: Added a check to prevent unnecessary API calls for English.
    if target_language == "English" or not llm or not text_to_translate:
        return text_to_translate
    
    # Use a session state cache to avoid re-translating the same text in the same session
    if "ui_translation_cache" not in st.session_state:
        st.session_state.ui_translation_cache = {}
    
    cache_key = f"{target_language}|{text_to_translate}"
    if cache_key in st.session_state.ui_translation_cache:
        return st.session_state.ui_translation_cache[cache_key]

    print(f"Translating to {target_language}: '{text_to_translate[:50]}...'")
    try:
        # REFINED: Simplified the prompt creation.
        messages = [
            ("system", PROMPT_TEMPLATES["translator"]),
            ("human", f"Translate the following text into {target_language}:\n\n---\n{text_to_translate}\n---")
        ]
        response = llm.invoke(messages)
        translated_text = response.content
        st.session_state.ui_translation_cache[cache_key] = translated_text # Cache the result
        return translated_text
    except Exception as e:
        print(f"Error during translation: {e}")
        # Do not cache failures
        return text_to_translate

def save_conversation_data():
    """
    Saves the current conversation data from st.session_state to a PostgreSQL database
    using a cached connection.
    """
    if "messages" not in st.session_state or not st.session_state.messages:
        print("No messages to save.")
        st.toast("No messages to save.", icon="🤷")
        return

    conn = get_db_connection()

    if conn is None or conn.closed:
        st.error("Cannot save data: Database connection is not available or closed.")
        print("Save operation failed: Database connection is None or closed.")
        return

    cur = None
    try:
        cur = conn.cursor()
        print(f"Saving conversation data. DB Connection active: {not conn.closed}")

        # REFINED: Simplified timestamp creation.
        now_utc = dt.now(datetime.timezone.utc)
        
        language = st.session_state.get("language", "N/A")
        user_role = st.session_state.get("role", "N/A")
        address = st.session_state.get("address", "N/A")
        contact_details = st.session_state.get("contact_details", "N/A")

        insert_conversation_query = """
        INSERT INTO conversations (timestamp, language, role, address, contact_details)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
        """
        cur.execute(insert_conversation_query, (
            now_utc, language, user_role, address, contact_details
        ))
        conversation_id = cur.fetchone()[0]

        messages_to_save = st.session_state.get("messages", [])
        insert_message_query = """
        INSERT INTO messages (conversation_id, role, content, message_timestamp)
        VALUES (%s, %s, %s, %s);
        """
        
        # REFINED: Use a single timestamp for all messages in this batch save for consistency.
        message_save_timestamp = dt.now(datetime.timezone.utc)
        
        for message in messages_to_save:
            message_role = message.get("role")
            message_content = message.get("content")
            
            cur.execute(insert_message_query, (
                conversation_id,
                message_role,
                message_content,
                message_save_timestamp
            ))

        conn.commit()
        print(f"Conversation data (ID: {conversation_id}) saved to PostgreSQL database.")
        st.toast(f"Conversation data saved to database.", icon="💾")

    except psycopg2.Error as e:
        print(f"Database error during save operation: {e}")
        st.error(f"Could not save conversation data to database: {e}")
        if conn and not conn.closed:
            try:
                print("Attempting to rollback transaction...")
                conn.rollback()
            except psycopg2.Error as rb_e:
                print(f"Rollback failed: {rb_e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving conversation data: {e}")
        st.error(f"An unexpected error occurred: {e}")
    finally:
        if cur:
            cur.close()

# --- PROMPT_TEMPLATES and other globals ---
PROMPT_TEMPLATES = {
    "resident": """
You are a multi‑lingual Engagement Officer for the Royal Borough of Greenwich,
tasked with speaking to older and potentially vulnerable residents about their
current housing and their thoughts on **Extra Care Housing** (independent flats
with on‑site care, social rent, and shared communal spaces).

Your single goal is to **listen and gently explore** the resident’s views,
needs, hopes and concerns—never to give advice or make promises.  
Start with a warm, open‑ended question that invites them to talk about their
current living situation, then try to follow the structure below, adapting naturally
to what the resident has already shared. Try and explore what the resident is saying, not only jumping 
to the next question in the provided structure. Always put the resident at ease, using clear, 
jargon‑free language.


##CONVERSATION STRUCTURE & TARGET QUESTIONS
(Ask only if relevant and not already answered; feel free to re‑phrase.)

1. **Introduction & Purpose**
Example introduction:    
   • “We’re speaking with older residents in Greenwich to understand what matters most to you when it comes to housing as you get older. 
      We’re especially looking at Extra Care Housing—homes that support independent living with care available if needed, mainly for social rent.”  
   • Ask: “To get started, could you tell me a little about your current
     home and how long you’ve lived there?”

2. **Current Living Situation**
Example questions:  
   – Where do you live now? Are you an owner‑occupier, leaseholder, or tenant?  
   – What do you like most about your home?  
   – Are there any challenges or things you’d change?

3. **Looking Ahead: Future Housing Needs**
Example questions:  
   – Have you thought about how your housing needs might change as you get older?  
   – What would be important to you in a future home (location, accessibility,
     support, community, etc.)?  
   – How important is staying independent?

4. **Introducing Extra Care (if not already covered)**
Example questions:  
   – Have you heard of Extra Care Housing before?  
   – What are your first thoughts?

5. **Exploring Preferences Around Extra Care**
Example questions:  
   – What would make a place like that appealing to you?  
   – What concerns might you have?  
   – Which services or features would matter most?  
   – How important is affordability?

6. **Barriers, Motivators & Communication** Example questions: 
   – What might stop you from considering a move like this?  
   – What might encourage you?  
   – How would you prefer to hear about options like this?

7. **Personal Preferences & Inclusion**
Example questions:  
   – Are there any particular needs linked to your background, identity,
     culture, language, religion, gender, disability, or anything else that
     would be important in a new home?

8. **Opportunities for Further Involvement**  
   – Would you like to join the *Extra Care Residents Design Group*?  
   – Would you like to be added to the consultation list for the *Housing
     Strategy 2021–26*?  
   (Record preferences only if they say yes.)

9. **Wrap‑Up**  
   – Is there anything else you’d like to share?

   
##TONE & STYLE GUIDELINES
• Warm, patient, and respectful; speak slowly and clearly.  
• Use plain English; avoid jargon and acronyms.  
• Empathise without over‑promising: “I understand that can be challenging.”  
• Summarise complex points to check understanding: “So, if I’ve understood
  correctly…”  
• Use emojis sparingly (e.g. 🙂) only when they enhance warmth; never more than
  one per message.  
• Allow silence: if the resident pauses, wait a moment before prompting again.  
• Do not rush; adapt to the resident’s pace.

##GUARDRAILS & LIMITATIONS
• **Do NOT** provide medical, legal, financial, or professional housing advice.  
• **Do NOT** discuss or debate politics, religion, controversial current
  events, internal council matters, or criticise individuals or organisations.  
• If asked for information outside your remit, reply:  
  “I’m sorry—I’m here just to listen to your thoughts about housing and ask
   follow‑up questions.”  
• Never ask the same question twice.  
• Skip any target question that has already been clearly answered.  
• Maintain confidentiality: never request or record sensitive personal data
  such as National Insurance numbers, bank details, or medical records.  
• If the resident becomes distressed, respond gently (“It sounds like this is
  upsetting—would you like a moment?”). If they mention immediate risk to
  themselves or others, advise contacting emergency services and offer to end
  the conversation.

""",
    "contractor": """(Placeholder for Contractor Prompt)""",
    "staff": """(Placeholder for Staff Prompt)""",
    "translator": """You are a simple and direct translator. Your only task is to translate the text you are given into the specified language. The context is a local government housing survey. Respond ONLY with the raw translated text and nothing else. Do not add explanations or pleasantries."""
}

LANGUAGE_OPTIONS = sorted(list(set(["English", "French (Français)", "Spanish (Español)", "Hindi (हिन्दी)"] + [
    "Albanian (Shqip)", "Amharic (አማርኛ)", "Arabic (العربية)", "Armenian (Հայերեն)",
    "Bengali (বাংলা)", "Bosnian (Bosanski)", "Bulgarian (Български)", "Burmese (မြန်မာဘာသာ)",
    "Croatian (Hrvatski)", "Czech (Čeština)", "Danish (Dansk)", "Dutch (Nederlands)",
    "Estonian (Eesti)", "Finnish (Suomi)", "Georgian (ქართული)",
    "German (Deutsch)", "Greek (Ελληνικά)", "Gujarati (ગુજરાતી)", "Hungarian (Magyar)",
    "Icelandic (Íslenska)", "Indonesian (Bahasa Indonesia)", "Italian (Italiano)", "Japanese (日本語)",
    "Kannada (ಕನ್ನಡ)", "Kazakh (Қазақ тілі)", "Korean (한국어)", "Latvian (Latviešu)", "Lithuanian (Lietuvių)",
    "Macedonian (Македонски)", "Malay (Bahasa Melayu)", "Malayalam (മലയാളം)", "Maltese (Malti)",
    "Mandarin Chinese (普通话)", "Marathi (मराठी)", "Nepali (नेपाली)", "Norwegian (Norsk)", "Pashto (پښتو)",
    "Persian (Farsi) (فارسی)", "Polish (Polski)", "Portuguese (Português)", "Punjabi (ਪੰਜਾਬੀ)", "Romanian (Română)",
    "Russian (Русский)", "Serbian (Српски)", "Sinhala (සිංහල)", "Slovak (Slovenčina)", "Slovenian (Slovenščina)",
    "Somali (Soomaali)", "Swahili (Kiswahili)","Swedish (Svenska)", "Tagalog (Tagalog)",
    "Tamil (தமிழ்)", "Telugu (తెలుగు)", "Thai (ภาษาไทย)", "Turkish (Türkçe)", "Ukrainian (Українська)",
    "Urdu (اردو)", "Uzbek (Oʻzbekcha)", "Vietnamese (Tiếng Việt)"
])))


if "page" not in st.session_state:
    st.session_state.page = "language_selection"

# --- PAGE 1: Language Selection ---
if st.session_state.page == "language_selection":
    st.title("Welcome / Bienvenue / Bienvenido / स्वागत है")
    st.header("Please Select Your Language")

    keys_to_clear = ["messages", "chain", "language", "role", "address", "contact_details"]
    for key in keys_to_clear:
        st.session_state.pop(key, None)

    selected_language = st.selectbox(
        label="Choose your preferred language to continue:",
        options=LANGUAGE_OPTIONS,
        index=LANGUAGE_OPTIONS.index("English")
    )

    if st.button("Confirm and Continue"):
        st.session_state.language = selected_language
        st.session_state.page = "form"
        st.rerun()

# --- PAGE 2: Form ---
elif st.session_state.page == "form":
    translation_llm = get_llm_for_translation()
    lang = st.session_state.get("language", "English")

    @st.cache_data(show_spinner=f"Preparing form in {lang}...")
    def get_translated_ui_texts(_lang):
        ui_elements = {
            "form_title": "Welcome to the Extra Care Feedback Chatbot!",
            "welcome_message": "Thank you for sharing your thoughts with us. We’re exploring Extra Care Housing – self‑contained flats for older residents with on‑site care to help you stay independent. This chatbot is here to listen to your experiences. Your feedback will guide the Royal Borough of Greenwich. Please complete the short form below to get started.",
            "form_header": "Please fill out this form to begin.",
            "address_label": "Your Address (Required)",
            "contact_label": "Your Contact Details (e.g., email or phone - Optional)",
            "submit_button": "Submit and Start Chat",
            "address_error": "Address is required. Please enter your address."
        }
        
        if _lang == "English":
            return ui_elements
            
        return {key: translate_text(text, _lang, translation_llm) for key, text in ui_elements.items()}

    T = get_translated_ui_texts(lang)

    st.title(T["form_title"])
    st.markdown(T["welcome_message"])
    
    with st.form(key="user_details_form"):
        st.header(T["form_header"])
        form_address = st.text_input(T["address_label"], value=st.session_state.get("address", ""))
        form_contact_details = st.text_input(T["contact_label"], value=st.session_state.get("contact_details", ""))
        
        submitted = st.form_submit_button(T["submit_button"])

    if submitted:
        if not form_address:
            st.error(T["address_error"])
        else:
            st.session_state.address = form_address
            st.session_state.contact_details = form_contact_details
            st.session_state.role = "resident"
            
            # Clear previous chat state before starting a new one
            keys_to_pop = ["messages", "chain", "initial_message_sent", "last_interaction_time"]
            for key in keys_to_pop:
                st.session_state.pop(key, None)

            st.session_state.page = "chat"
            st.rerun()

# --- PAGE 3: Chat Interface ---
elif st.session_state.page == "chat":
    # --- Timeout Logic ---
    CHAT_TIMEOUT_SECONDS = 30 * 60 # 30 minutes
    if "last_interaction_time" in st.session_state:
        time_since_last_interaction = time.time() - st.session_state.last_interaction_time
        if time_since_last_interaction > CHAT_TIMEOUT_SECONDS:
            st.warning(f"Session timed out due to inactivity. Saving conversation...")
            save_conversation_data()
            st.session_state.page = "form"
            st.toast("Session ended. Returning to form.", icon="⏱️")
            st.rerun()
    
    translation_llm = get_llm_for_translation()
    lang = st.session_state.get("language", "English")

    @st.cache_data(show_spinner=f"Preparing chat in {lang}...")
    def get_chat_ui_translations(_lang):
        ui = {
            "chat_title": "Extra Care Feedback Chat",
            "language_label": "Language",
            "change_lang_label": "Change chat language:",
            "end_button": "End & Save Conversation",
            "ending_toast": "Ending conversation and saving data...",
            "ended_toast": "Conversation ended and saved.",
            "audio_prompt": "Or, record your message:",
            "chat_placeholder": "Write your message here...",
            "transcribing_audio": "Transcribing audio, please wait...",
        }
        if _lang == "English": return ui
        return {k: translate_text(v, _lang, translation_llm) for k, v in ui.items()}

    T_CHAT = get_chat_ui_translations(lang)
    
    st.title(T_CHAT["chat_title"])

    # --- Helper Classes ---
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text = initial_text
        def on_llm_new_token(self, token: str, **kwargs):
            self.text += token
            self.container.markdown(self.text + "▌")
        def on_llm_end(self, response, **kwargs):
            self.container.markdown(self.text)

    class ContextChatbot:
        def __init__(self):
            # REFINED: Initialize clients in one place
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OPENAI_API_KEY environment variable not set.")
                self.client = None
                self.llm = None
                return
            
            self.client = OpenAI(api_key=api_key)
            self.llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.7, # Slightly more creative for conversation
                streaming=True,
                api_key=api_key
            )
        
        def setup_chain(self):
            if "chain" in st.session_state and st.session_state.chain:
                return st.session_state.chain

            if not self.llm: return None

            print("Setting up new ConversationChain...")
            memory = ConversationBufferMemory(memory_key="history", return_messages=True)
            
            # Rehydrate memory from session state
            for msg in st.session_state.get("messages", []):
                if msg["role"] == "user":
                    memory.chat_memory.add_user_message(msg["content"])
                elif msg["role"] == "assistant":
                    memory.chat_memory.add_ai_message(msg["content"])

            role_key = st.session_state.role.lower()
            language = st.session_state.language
            system_template = PROMPT_TEMPLATES.get(role_key, "You are a helpful assistant.")
            system_template += f"\n\nIMPORTANT: You must conduct the entire conversation ONLY in {language}. Adhere strictly to this language requirement."
            
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ])

            chain = ConversationChain(llm=self.llm, memory=memory, prompt=prompt, verbose=True)
            st.session_state.chain = chain
            return chain

        def change_language_callback(self):
            # This callback runs when the selectbox value changes.
            # Its ONLY job is to update the backend conversation state. The UI will be updated on the subsequent rerun.
            
            new_language = st.session_state.language # This key is automatically updated by Streamlit
            print(f"Language change callback triggered. New language: {new_language}")

            if "chain" not in st.session_state:
                print("Error: Chain not found during language change.")
                st.error("An error occurred. Please restart the chat.")
                return

            # 1. Update the system prompt in the existing chain
            role_key = st.session_state.role.lower()
            system_template = PROMPT_TEMPLATES.get(role_key, "You are a helpful assistant.")
            system_template += f"\n\nIMPORTANT: You must conduct the entire conversation ONLY in {new_language}. Adhere strictly to this language requirement."
            st.session_state.chain.prompt.messages[0] = SystemMessagePromptTemplate.from_template(system_template)
            print("System prompt updated in chain.")

            # 2. Add a system message to history for context and translate the last AI message
            last_ai_message = next((m['content'] for m in reversed(st.session_state.get("messages",[])) if m['role'] == 'assistant'), None)
            
            if last_ai_message:
                with st.spinner(f"Translating to {new_language}..."):
                    translated_content = translate_text(last_ai_message, new_language, translation_llm)
                # Store the translated message to be displayed after the rerun
                st.session_state.display_translated_message = translated_content
            
            # The script will now rerun automatically, redrawing the UI with the new language.

        def main(self):
            if not self.llm or not self.client:
                st.stop()

            chain = self.setup_chain()
            if not chain:
                st.error("Failed to set up conversation. Please try again.")
                st.stop()
            
            # --- UI Controls (Sidebar) ---
            with st.sidebar:
                st.write(f"**Role:** {st.session_state.role.capitalize()}")
                st.selectbox(
                    key='language',
                    label=T_CHAT['change_lang_label'],
                    options=LANGUAGE_OPTIONS,
                    index=LANGUAGE_OPTIONS.index(st.session_state.language),
                    on_change=self.change_language_callback
                )

                if st.button(T_CHAT["end_button"], use_container_width=True):
                    st.toast(T_CHAT["ending_toast"])
                    save_conversation_data()
                    st.session_state.page = "form"
                    st.toast(T_CHAT["ended_toast"], icon="👋")
                    st.rerun()

            # --- Initial Assistant Message ---
            if "messages" not in st.session_state:
                st.session_state.messages = []

            if not st.session_state.messages:
                print("Generating initial assistant message...")
                with st.chat_message("assistant"):
                    msg_placeholder = st.empty()
                    handler = StreamHandler(msg_placeholder)
                    initial_input = "Please start the conversation by introducing yourself and asking me about my current home."
                    resp = chain.invoke({"input": initial_input}, {"callbacks": [handler]})
                    answer = resp.get("response")
                    st.session_state.messages.append({"role": 'assistant', "content": answer})
                    st.session_state.last_interaction_time = time.time()
                st.rerun()

            # --- Display Chat History ---
            for msg in st.session_state.get("messages", []):
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            # --- Display Pending Translated Message (from language change) ---
            if "display_translated_message" in st.session_state:
                with st.chat_message("assistant"):
                    st.info("Continuing in the new language:")
                    st.write(st.session_state.display_translated_message)
                st.session_state.pop("display_translated_message", None)

            # --- User Input Handling ---
            processed_input = None

            # REFINED: Unified text and audio input handling
            user_text = st.chat_input(placeholder=T_CHAT["chat_placeholder"])
            audio_bytes = st.audio_input(T_CHAT['audio_prompt'])

            if user_text:
                processed_input = user_text
                st.session_state.messages.append({"role": "user", "content": processed_input})
            
            elif audio_bytes:
                # FIXED: Correctly handle audio bytes and prevent reprocessing
                if "last_audio" not in st.session_state or st.session_state.last_audio != audio_bytes:
                    st.session_state.last_audio = audio_bytes
                    with st.spinner(T_CHAT['transcribing_audio']):
                        # FIXED: Pass audio as a file-like tuple to the API
                        transcript = self.client.audio.transcriptions.create(
                            model="whisper-1",
                            file=("audio.wav", audio_bytes)
                        )
                    processed_input = transcript.text
                    st.session_state.messages.append({"role": "user", "content": f"🎤: \"{processed_input}\""})
            
            # --- LLM Invocation ---
            if processed_input:
                st.session_state.last_interaction_time = time.time()
                with st.chat_message("assistant"):
                    msg_placeholder = st.empty()
                    handler = StreamHandler(msg_placeholder)
                    resp = chain.invoke({"input": processed_input}, {"callbacks": [handler]})
                    answer = resp.get("response")
                    st.session_state.messages.append({"role": 'assistant', "content": answer})
                st.rerun()

    # --- Run the Chatbot ---
    if "role" in st.session_state and "language" in st.session_state:
        chatbot = ContextChatbot()
        chatbot.main()
    else:
        st.warning("Role or language not set. Returning to form.")
        if st.button("Go to Form"):
            st.session_state.page = "form"
            st.rerun()