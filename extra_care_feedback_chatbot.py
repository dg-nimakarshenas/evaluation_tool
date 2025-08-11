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
import datetime
import psycopg2 # Added for PostgreSQL integration
from psycopg2 import sql # For safe SQL query construction if needed
from datetime import datetime, timedelta
    
    
    
today = datetime.now().strftime("%d %B %Y")
# --- Database Configuration ---
# IMPORTANT: Replace these with your actual PostgreSQL connection details
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres.oedyrmoxycaidzifeucl")
DB_PASSWORD = os.getenv("DB_PASSWORD", "4Abetterfuture!")
DB_HOST = os.getenv("DB_HOST", "aws-0-eu-west-2.pooler.supabase.com")
DB_PORT = os.getenv("DB_PORT", "6543")

# --- Database Connection Function with Streamlit Caching ---
@st.cache_resource  
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
        return None # Return None if connection fails

# --- Attempt to establish DB connection when app loads/script runs ---
# This will now use the cached function.
# The connection logic inside get_db_connection() will only run once
# per session unless the cache is invalidated.
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
            model_name="gpt-4.1-nano-2025-04-14",
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
    if not llm or not text_to_translate:
        return text_to_translate
    
    # Use a session state cache to avoid re-translating the same text in the same session
    if "ui_translation_cache" not in st.session_state:
        st.session_state.ui_translation_cache = {}
    
    cache_key = f"{target_language}|{text_to_translate}"
    if cache_key in st.session_state.ui_translation_cache:
        return st.session_state.ui_translation_cache[cache_key]

    print(f"Translating to {target_language}: '{text_to_translate[:50]}...'")
    try:
        translate_prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_TEMPLATES["translator"]),
            ("human", f"Translate the following text into {target_language}:\n\n{text_to_translate}")
        ])
        response = llm.invoke(translate_prompt.format_prompt(text=text_to_translate).to_messages())
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

    # Get the cached database connection
    conn = get_db_connection() # This will return the cached connection object

    if conn is None or conn.closed: # Check if connection is valid
        st.error("Cannot save data: Database connection is not available or closed.")
        print("Save operation failed: Database connection is None or closed.")
        if conn is None and initial_conn_on_load is None: # if it never connected
             print("The initial connection attempt also failed.")
        elif conn and conn.closed: # if it was connected but now closed
             print("The previously cached connection is now closed. Consider refreshing the page to re-initialize.")
             # For more advanced handling, you might try to clear the cache and retry:
             # st.cache_resource.clear()
             # conn = get_db_connection()
             # if conn is None or conn.closed:
             #    st.error("Still no valid DB connection after cache clear attempt.")
             #    return
        return

    cur = None
    try:
        cur = conn.cursor()
        print(f"Saving conversation data. DB Connection active: {not conn.closed}")

        now_local_aware = datetime.datetime.now(datetime.timezone.utc).astimezone()
        conversation_timestamp_str = now_local_aware.isoformat(sep=' ', timespec='seconds')
        
        language = st.session_state.get("language", "N/A")
        user_role = st.session_state.get("role", "N/A")
        address = st.session_state.get("address", "N/A")
        contact_details = st.session_state.get("contact_details", "N/A")

        insert_conversation_query = """
        INSERT INTO conversations (timestamp, language, role, address, contact_details, feedback_type)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        cur.execute(insert_conversation_query, (
            conversation_timestamp_str, language, user_role, address, contact_details, "Extra Care Housing Feedback"
        ))
        conversation_id = cur.fetchone()[0]

        messages_to_save = st.session_state.get("messages", [])
        insert_message_query = """
        INSERT INTO messages (conversation_id, role, content, message_timestamp)
        VALUES (%s, %s, %s, %s);
        """
        for message in messages_to_save:
            message_role = message.get("role")
            message_content = message.get("content")
            msg_now_local_aware = datetime.datetime.now(datetime.timezone.utc).astimezone()
            message_instance_timestamp_str = msg_now_local_aware.isoformat(sep=' ', timespec='seconds')
            
            cur.execute(insert_message_query, (
                conversation_id,
                message_role,
                message_content,
                message_instance_timestamp_str
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
        # We DO NOT close the connection `conn` here.
        # @st.cache_resource manages its lifecycle (implicitly, no explicit close needed here
        # unless you define a cleanup function for @st.cach

# --- Assume PROMPT_TEMPLATES and page setup code from above exists ---

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

6. **Barriers, Motivators & Communication** 
Example questions: 
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
    "contractor": """(Placeholder for Contractor Prompt)""", # Added placeholder
    "staff":"""(Placeholder for Staff Prompt)""", # Added placeholder
    "translator": """You are a simple translator. Your task is to translate the text that you are given into the language that is specified in the input. Note that
    the context of what you are translating is that you are a feedback officer for the SHDF retrofit program in the Royal Borough of Greenwich. Respond only with the translation, nothing else."""} # Added instruction for translator

# Initialize page state
if "page" not in st.session_state:
    st.session_state.page = "language_selection"

# --- Language and Role Options ---
# Defined globally to be accessible on multiple pages
LANGUAGE_OPTIONS = ["English", "French (Français)", "Spanish (Español)", "Hindi (हिन्दी)"] + sorted([
    "Albanian (Shqip)", "Amharic (አማርኛ)", "Arabic (العربية)", "Armenian (Հայերեն)",
    "Bengali (বাংলা)", "Bosnian (Bosanski)", "Bulgarian (Български)", "Burmese (မြန်မာဘာသာ)",
    "Croatian (Hrvatski)", "Czech (Čeština)", "Danish (Dansk)", "Dutch (Nederlands)",
    "English", "Estonian (Eesti)", "Finnish (Suomi)", "French (Français)", "Georgian (ქართული)",
    "German (Deutsch)", "Greek (Ελληνικά)", "Gujarati (ગુજરાતી)", "Hindi (हिन्दी)",  "Hungarian (Magyar)",
    "Icelandic (Íslenska)", "Indonesian (Bahasa Indonesia)", "Italian (Italiano)", "Japanese (日本語)",
    "Kannada (ಕನ್ನಡ)", "Kazakh (Қазақ тілі)", "Korean (한국어)", "Latvian (Latviešu)", "Lithuanian (Lietuvių)",
    "Macedonian (Македонски)", "Malay (Bahasa Melayu)", "Malayalam (മലയാളം)", "Maltese (Malti)",
    "Mandarin Chinese (普通话)", "Marathi (मराठी)", "Nepali (नेपाली)", "Norwegian (Norsk)", "Pashto (پښتو)",
    "Persian (Farsi) (فارسی)", "Polish (Polski)", "Portuguese (Português)", "Punjabi (ਪੰਜਾਬੀ)", "Romanian (Română)",
    "Russian (Русский)", "Serbian (Српски)", "Sinhala (සිංහල)", "Slovak (Slovenčina)", "Slovenian (Slovenščina)",
    "Somali (Soomaali)",  "Spanish (Español)", "Swahili (Kiswahili)","Swedish (Svenska)", "Tagalog (Tagalog)",
    "Tamil (தமிழ்)", "Telugu (తెలుగు)", "Thai (ภาษาไทย)", "Turkish (Türkçe)", "Ukrainian (Українська)",
    "Urdu (اردو)",  "Uzbek (Oʻzbekcha)", "Vietnamese (Tiếng Việt)"])

if st.session_state.page == "language_selection":
    st.title("Welcome / Bienvenue / Bienvenido / स्वागत है")
    st.header("Please Select Your Language")

    # Clean up any previous session data if user lands here
    keys_to_clear = ["messages", "chain", "language", "role", "address", "contact_details"]
    for key in keys_to_clear:
        st.session_state.pop(key, None)

    selected_language = st.selectbox(
        label="Choose your preferred language to continue:",
        options=LANGUAGE_OPTIONS,
        index=0
    )

    if st.button("Confirm and Continue"):
        st.session_state.language = selected_language
        st.session_state.current_language = selected_language
        st.session_state.page = "form"
        st.rerun()

# --- Setup form ---
elif st.session_state.page == "form":
    # Get the LLM for translation. This will hit the cache after the first run.
    translation_llm = get_llm_for_translation()
    lang = st.session_state.get("language", "English")

    # Helper to translate text, showing a spinner for the first translation batch
    @st.cache_data(show_spinner=f"Preparing form in {lang}...")
    def get_translated_ui_texts(_lang):
        # This function runs once per language and caches the results
        ui_elements = {
            "form_title": "Welcome to the Extra Care Feedback Chatbot!",
            "welcome_message": """Welcome 🙂
Thank you for taking a moment to share your thoughts with us.
We’re exploring Extra Care Housing – self‑contained flats for older residents that include on‑site care and friendly communal spaces to help you stay independent.

This demo chatbot is here to listen to your experiences, ideas, and questions.
Your feedback will guide the Royal Borough of Greenwich as we shape future housing options.
There are no right or wrong answers – every comment counts.

Please complete the short form below to get started, and then we’ll chat at your own pace.
We appreciate your time and insights!""",
            "form_header": "Please fill out the form below to get started.",
            "details_subheader": "Your Details",
            "address_label": "Your Address (Required)",
            "contact_label": "Your Contact Details (e.g., email or phone - Optional)",
            "chat_prefs_subheader": "Chat Preferences",
            "language_label": "Which language would you like to communicate in?",
            "role_label": "Are you a resident or a contractor?",
            "submit_button": "Submit and Start Chat",
            "address_error": "Address is required. Please enter your address."
        }
        
        if _lang == "English":
            return ui_elements
            
        translated_texts = {}
        for key, text in ui_elements.items():
            translated_texts[key] = translate_text(text, _lang, translation_llm)
        return translated_texts

    T = get_translated_ui_texts(lang)

    st.title(T["form_title"])
    st.write(T["welcome_message"])
    
    # Safeguard logic remains the same...
    if "messages" in st.session_state and st.session_state.messages:
        if not st.session_state.get("conversation_saved_on_form_load_safeguard", False):
            print("Form page loaded with existing messages. Safeguard: Saving conversation...")
            save_conversation_data()
            st.session_state.conversation_saved_on_form_load_safeguard = True
    else:
        st.session_state.pop("conversation_saved_on_form_load_safeguard", None)

    st.header(T["form_header"])
    with st.form(key="user_details_form"):
        st.subheader(T["details_subheader"])
        address_val = st.session_state.get("address", "")
        contact_val = st.session_state.get("contact_details", "")
        
        form_address = st.text_input(T["address_label"], value=address_val)
        form_contact_details = st.text_input(T["contact_label"], value=contact_val)
        
        role_idx = 0
        submit_button = st.form_submit_button(T["submit_button"])

    if submit_button:
        if not form_address:
            st.error(T["address_error"])
        else:
            st.session_state.address = form_address
            st.session_state.contact_details = form_contact_details
            st.session_state.role = "resident"
            
            # Clear chat state and proceed
            keys_to_pop = ["messages", "chain", "initial_message_sent", "last_interaction_time"]
            for key in keys_to_pop:
                st.session_state.pop(key, None)

            st.session_state.page = "chat"
            st.rerun()


# --- Chat interface ---
elif st.session_state.page == "chat":
    # --- Timeout Logic ---
    CHAT_TIMEOUT_SECONDS = 30 * 60 # 30 minutes
    if "last_interaction_time" in st.session_state:
        # Only apply timeout if a conversation is considered active
        is_active_conversation = ("messages" in st.session_state and st.session_state.messages) or \
                                 st.session_state.get("initial_message_sent", False)

        if is_active_conversation:
            time_since_last_interaction = time.time() - st.session_state.last_interaction_time
            if time_since_last_interaction > CHAT_TIMEOUT_SECONDS:
                st.warning(f"Session timed out due to inactivity for over {int(CHAT_TIMEOUT_SECONDS/60)} minutes. Saving conversation...")
                save_conversation_data()

                # Clear chat-specific state and redirect to form
                keys_to_pop_on_timeout = ["messages", "chain", "initial_message_sent", "current_page",
                               "display_translated_message", "last_interaction_time"]
                for key in keys_to_pop_on_timeout:
                    st.session_state.pop(key, None)

                # Keep user details (address, contact, language, role) for convenience.
                st.session_state.page = "form"
                st.toast("Session ended due to inactivity. Data saved. Returning to form.", icon="⏱️")
                st.rerun()
    
    translation_llm = get_llm_for_translation()
    current_lang = st.session_state.get("current_language", st.session_state.get("language", "English"))

    def get_chat_ui_translations(_lang):
        ui = {
            "chat_title": "Extra Care Feedback Chatbot",
            "role_label": "Role",
            "language_label": "Language",
            "change_lang_label": "You can change the language here:",
            "end_button": "End Conversation and Save",
            "ending_toast": "Ending conversation and saving data...",
            "ended_toast": "Conversation ended and saved. Returning to form.",
            "back_to_form_button": "Back to Form",
            "timeout_warning": f"Session timed out due to inactivity for over {int(CHAT_TIMEOUT_SECONDS/60)} minutes. Saving conversation...",
            "timeout_toast": "Session ended due to inactivity. Data saved. Returning to form.",
            "audio_prompt": "You can also record a voice message. Press the microphone button to recored your new message.",
            "send_audio_button": "Send Audio",
            "chat_placeholder": "Write your message here...",
            "transcribing_audio": "Transcribing audio...",
        }
        if _lang == "English": 
            return ui
        
        translated_texts = {}
        for key, text in ui.items():
            translated_texts[key] = translate_text(text, _lang, translation_llm)
        return translated_texts

    T_CHAT = get_chat_ui_translations(current_lang)

    # Update timeout messages with translated text
    if "last_interaction_time" in st.session_state:
        # ... (timeout logic)
        if time_since_last_interaction > CHAT_TIMEOUT_SECONDS:
            st.warning(T_CHAT["timeout_warning"])
            # ... (rest of timeout logic)
            st.toast(T_CHAT["timeout_toast"], icon="⏱️")
            st.rerun()
    
    st.title(T_CHAT["chat_title"])
    if "language" in st.session_state:
        st.write(f"**{T_CHAT['language_label']}:** {st.session_state.language}")

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
                model_name="gpt-4.1-mini",
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

    class ContextChatbot:
        def __init__(self, ui_text=T_CHAT):
            self.audio_llm, self.llm = configure_llm()
            self.ui_text = ui_text
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

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
            """
            Callback function to handle language change.
            This updates the system prompt, chat history, and memory to reflect the new language.
            """
            print(f"Language callback triggered. Selected: {st.session_state.language}")
            
            # Get previous and new language
            new_language = st.session_state.language
            print(f"Language change callback triggered for new language: {new_language}")
            
            if "chain" not in st.session_state or not st.session_state.chain:
                print("Warning: Chain not found in session state during language change.")
                return
            
            chain = st.session_state.chain

            # 1. Find the last assistant message in the official history
            last_assistant_message_content = None
            if "messages" in st.session_state:
                for msg in reversed(st.session_state.messages):
                    if msg.get("role") == "assistant":
                        last_assistant_message_content = msg.get("content")
                        break

            # 2. Update the system prompt in the existing chain object
            system_template = PROMPT_TEMPLATES.get("resident", "You are a helpful assistant.")
            system_template += f"\n\nYou must communicate ONLY in {new_language}. Ask questions relevant to the SHDF program feedback based on the user's role ({st.session_state.role})."
            
            try:
                chain.prompt.messages[0] = SystemMessagePromptTemplate.from_template(system_template)
                print("System prompt updated in chain.")

                # 3. Add system guidance message to history and memory
                system_guidance = f"System Notification: The conversation language has now changed to {new_language}. Please continue the conversation ONLY in {new_language}."
                st.session_state.setdefault("messages", []).append({"role": 'system', "content": system_guidance})
                if hasattr(chain.memory, 'chat_memory'):
                    chain.memory.chat_memory.add_message(SystemMessage(content=system_guidance))
                    print("System guidance message added to history and memory.")
                else:
                    print("Warning: chat_memory not found on chain.memory.")
                    
                # 4. Translate the last assistant message (if found)
                st.session_state.pop("display_translated_message", None)
                if last_assistant_message_content:
                    print("Attempting to translate the last assistant message.")
                    with st.spinner(f"Translating last message to {new_language}..."):
                        translated_content = translate_text(last_assistant_message_content, new_language, self.llm)
                    if translated_content:
                        print("Storing translated message for display.")
                        st.session_state.display_translated_message = translated_content
                    else:
                        print("Translation failed or returned empty.")

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

            current_language = st.session_state.language
            if current_language not in LANGUAGE_OPTIONS:
                print(f"Warning: Current language '{current_language}' not in options. Defaulting to English.")
                st.session_state.language = "English"
                current_language = "English"
                st.rerun()
            
            T_CHAT_CURRENT = get_chat_ui_translations(st.session_state.language)

            # --- Language Selection ---
            st.selectbox(
                key='language',
                label=T_CHAT_CURRENT['change_lang_label'],
                options=LANGUAGE_OPTIONS,
                index=LANGUAGE_OPTIONS.index(st.session_state.language),
                on_change=self.change_language_callback
            )

            # --- Save Conversation Button ---
            if st.button(T_CHAT_CURRENT["end_button"], key="end_conversation_button"): # Use translated text
                # This message will be in the current language
                st.info(T_CHAT_CURRENT["ending_toast"])
                save_conversation_data()

                # Clear chat-specific state, keep user details for form prefill
                keys_to_pop_on_end = ["messages", "chain", "initial_message_sent", "current_page",
                                      "display_translated_message", "last_interaction_time"]
                for key in keys_to_pop_on_end:
                    st.session_state.pop(key, None)

                st.session_state.page = "form"
                st.toast(T_CHAT_CURRENT["ended_toast"], icon="👋")
                st.rerun()

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
            
            processed_input = None
            user_text = st.chat_input(placeholder=T_CHAT["chat_placeholder"])
            audio_bytes = st.audio_input(T_CHAT['audio_prompt'])

            if user_text:
                processed_input = user_text
                st.session_state.messages.append({"role": "user", "content": processed_input})
            
            elif audio_bytes:
                if "last_audio" not in st.session_state or st.session_state.last_audio != audio_bytes:
                    st.session_state.last_audio = audio_bytes
                    with st.spinner(T_CHAT['transcribing_audio']):
                        # FIXED: Pass audio as a file-like tuple to the API
                        transcript = self.audio_llm.audio.transcriptions.create(
                                        model="gpt-4o-mini-transcribe",
                                        file = audio_bytes,
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
        st.warning("Role or language not selected. Please go back to the form.")
        if st.button("Back to Form"):
            if "messages" in st.session_state and st.session_state.messages:
                print("Back to Form button clicked. Saving conversation...")
                save_conversation_data()
            # --- END Save conversation ---

            st.session_state.page = "form"
            # Explicitly clear chat-specific state when navigating back
            # Keep address, contact, language, role in session state for form prefill
            keys_to_pop_on_back_to_form = ["messages", "chain", "initial_message_sent", "current_page",
                                           "display_translated_message", "last_interaction_time"] # Added last_interaction_time
            for key in keys_to_pop_on_back_to_form:
                st.session_state.pop(key, None)
            st.rerun()
