from flask import Flask, render_template, session, request
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from flask_socketio import SocketIO, emit
from llama_cpp import Llama
import llama_cpp
import os
from dotenv import load_dotenv
import logging
import json
import psutil
import gc
from difflib import get_close_matches
import random

print("Version:", llama_cpp.__version__)
print("System Info:\n")
print(llama_cpp.llama_cpp._lib.llama_print_system_info().decode())


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "your-secret-key")
app.config["SESSION_TYPE"] = "filesystem"
socketio = SocketIO(app, manage_session=True)

# Updated Porsche Models and competitors
porsche_models = [
    "911",
    "911 carrera",
    "911 carrera s",
    "911 turbo",
    "911 turbo s",
    "911 gt3",
    "911 gt3 rs",
    "911 targa",
    "911 speedster",
    "911 r",
    "911 turbo cabriolet",
    "911 gt2 rs",
    "911 dakar",
    "718 cayman",
    "718 cayman s",
    "718 cayman gts 4.0",
    "718 boxster",
    "718 boxster s",
    "718 boxster gts 4.0",
    "taycan",
    "taycan 4s",
    "taycan turbo",
    "taycan turbo s",
    "taycan cross turismo",
    "taycan cross turismo 4s",
    "taycan cross turismo turbo",
    "taycan cross turismo turbo s",
    "panamera",
    "panamera 4",
    "panamera 4s",
    "panamera turbo",
    "panamera turbo s",
    "panamera gts",
    "panamera 4 e-hybrid",
    "panamera turbo s e-hybrid",
    "panamera sport turismo",
    "macan",
    "macan s",
    "macan gts",
    "macan turbo",
    "cayenne",
    "cayenne s",
    "cayenne gts",
    "cayenne turbo",
    "cayenne turbo s e-hybrid",
    "cayenne coupe",
    "cayenne turbo coupe",
    "cayenne e-hybrid",
    "cayenne coupe e-hybrid",
    "918 spyder",
    "cayman gt4",
    "cayman gt4 rs",
    "boxster spyder",
    "macan ev",
    "911 sport classic",
]

other_models = [
    "urus",
    "aventador",
    "huracan",
    "model x",
    "model y",
    "ferrari f8",
    "ferrari 812",
    "ferrari portofino",
    "m5",
    "m8",
    "x5 m",
    "x6 m",
    "rs7",
    "rs5",
    "q7",
    "q8 rs",
    "mercedes amg gt",
    "sls",
    "g-class",
    "e-class coupe",
    "vantage",
    "db11",
    "dbs superleggera",
    "720s",
    "570s",
    "gt",
    "continental gt",
    "bentley bentayga",
    "f-type",
    "i-pace",
    "tesla model s",
    "tesla model x",
    "tesla model 3",
    "tesla model y",
    "range rover sport",
    "range rover velar",
]


def run_rag(query, k=3):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        db = FAISS.load_local("faiss_index", embeddings)
        results = db.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in results])
        return context
    except Exception as e:
        logger.error(f"RAG error: {str(e)}")
        return ""


# Spell corrector
def correct_spelling(user_input):
    words = user_input.lower().split()
    corrected = []
    for word in words:
        match = get_close_matches(word, porsche_models + other_models, n=1, cutoff=0.8)
        corrected.append(match[0] if match else word)
    return " ".join(corrected)


# Extract model mentions
def extract_models(user_input):
    return [kw for kw in porsche_models if kw in user_input.lower()]


# Resource optimization
def get_optimal_thread_count():
    cpu_count = psutil.cpu_count(logical=False)
    return max(1, min(cpu_count - 1, 4))


def get_optimal_context_size():
    """
    Determine the optimal context window size for the LLM based on system memory.
    Allows override via environment variable 'LLM_CONTEXT_SIZE'.
    """
    try:
        # Allow manual override from environment
        ctx_override = os.getenv("LLM_CONTEXT_SIZE")
        if ctx_override:
            ctx = int(ctx_override)
            logger.info(f"Context size overridden via environment: {ctx}")
            return ctx

        memory = psutil.virtual_memory()
        gb = memory.total / (1024 * 1024 * 1024)

        if gb > 16:
            ctx = 2048
        elif gb > 12:
            ctx = 1536
        elif gb > 8:
            ctx = 1024
        else:
            ctx = 512

        logger.info(
            f"Total system memory: {gb:.2f} GB — Optimal context size selected: {ctx}"
        )
        return ctx
    except Exception as e:
        logger.warning(
            f"Error determining context size: {str(e)} — Falling back to default 512"
        )
        return 512


# Load model
try:
    n_threads = get_optimal_thread_count()
    n_ctx = get_optimal_context_size()
    logger.info(f"Loading model with {n_threads} threads and {n_ctx} context size")
    llm = Llama(
        model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_ctx=n_ctx,
        n_threads=n_threads,
        use_mlock=False,
        use_mmap=True,
        n_batch=512,
        n_gpu_layers=40,
        n_predict=50,
        repeat_penalty=1.1,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        seed=42,
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Menu prompt context (if applicable)
MENU_PROMPTS = {
    "models": "You are a Porsche model expert...",
    "dealerships": "You are a Porsche dealership locator...",
    "test-drive": "You are a Porsche test drive coordinator...",
    "build": "You are a Porsche configuration expert...",
}

MODEL_INFO = {
    "911": "The iconic Porsche 911 is our flagship sports car, known for its rear-engine layout and exceptional handling.",
    "taycan": "The Porsche Taycan is our first all-electric sports car, combining performance with sustainability.",
    "macan": "The Porsche Macan is our compact SUV, offering sports car performance in a practical package.",
    "cayenne": "The Porsche Cayenne is our luxury SUV, combining comfort with impressive performance.",
    "panamera": "The Porsche Panamera is our luxury sedan, offering sports car performance with executive comfort.",
    "718": "The Porsche 718 Boxster and Cayman are our mid-engine sports cars, offering pure driving pleasure.",
}

porsche_competitor_brands = [
    "Ferrari",
    "Lamborghini",
    "McLaren",
    "Aston Martin",
    "Chevrolet",
    "Nissan",
    "BMW",
    "Mercedes-Benz",
    "Audi",
    "Tesla",
    "Lucid Motors",
    "Rivian",
    "Polestar",
    "Bentley",
    "Jaguar",
    "Lotus",
    "Range Rover",
    "Land Rover",
    "Maserati",
    "Genesis",
    "Cadillac",
    "Lexus",
    "Infiniti",
]

SUGGESTED_QUESTIONS = {
    "models": {
        "911": [
            "What are the different 911 variants?",
            "What is the 0-60 time of the 911?",
            "How much does the 911 cost?",
            "What are the key features of the 911?",
            "Which 911 is best for daily driving?",
            "What's the difference between the 911 Turbo and Turbo S?",
            "Is the 911 GT3 track-ready?",
            "What’s new in the latest 911 model year?",
        ],
        "taycan": [
            "What is the range of the Taycan?",
            "How fast can the Taycan charge?",
            "What are the different Taycan models?",
            "What is the price of the Taycan?",
            "How does Taycan compare to Tesla?",
            "What tech features come with the Taycan?",
            "Is the Taycan suitable for long-distance travel?",
            "What’s the top speed of the Taycan Turbo S?",
        ],
        "macan": [
            "What are the Macan's performance specs?",
            "How much cargo space does the Macan have?",
            "What are the available Macan trims?",
            "What is the starting price of the Macan?",
            "Is the Macan available as an EV?",
            "How does Macan compare to other luxury SUVs?",
            "What driver assistance features are available in the Macan?",
            "What’s the difference between Macan and Macan S?",
        ],
        "cayenne": [
            "What are the Cayenne's engine options?",
            "How much can the Cayenne tow?",
            "What are the Cayenne's luxury features?",
            "What is the price range of the Cayenne?",
            "Is the Cayenne available as a coupe?",
            "Does the Cayenne offer hybrid options?",
            "What is the Cayenne Turbo GT?",
            "How does the Cayenne handle off-road driving?",
        ],
        "panamera": [
            "What are the Panamera's performance specs?",
            "How many passengers can the Panamera seat?",
            "What are the Panamera's luxury features?",
            "What is the starting price of the Panamera?",
            "What’s the difference between Panamera and Panamera Sport Turismo?",
            "Is the Panamera available as a hybrid?",
            "Does the Panamera have a performance variant?",
            "How does the Panamera compare to other executive sedans?",
        ],
        "718": [
            "What's the difference between Boxster and Cayman?",
            "What are the 718's performance specs?",
            "How much does the 718 cost?",
            "What are the available 718 variants?",
            "Is the 718 suitable for track use?",
            "How does the 718 handle compared to the 911?",
            "What engine options are available in the 718?",
            "Is the 718 a good option for a weekend sports car?",
        ],
    },
    "dealerships": [
        "What are the dealership hours?",
        "Do you offer financing options?",
        "Can I schedule a test drive?",
        "What services do you offer?",
    ],
    "test-drive": [
        "What models are available for test drive?",
        "How long is a test drive?",
        "Do I need to make an appointment?",
        "What documents do I need to bring?",
    ],
    "build": [
        "What are the available colors?",
        "What interior options are available?",
        "What performance packages are offered?",
        "How long does delivery take?",
    ],
}


def get_suggested_questions(menu_context, user_input=None):
    from difflib import get_close_matches

    """ if selected_model:
        model = selected_model.lower() """
    if user_input:
        models_mentioned = extract_models(user_input)
        model = models_mentioned[0] if models_mentioned else None
    else:
        model = None

    if menu_context == "models" and model:
        questions = SUGGESTED_QUESTIONS["models"].get(model, [])
        return random.sample(questions, min(2, len(questions))) if questions else []

    elif menu_context in SUGGESTED_QUESTIONS:
        questions = SUGGESTED_QUESTIONS[menu_context]
        if isinstance(questions, list):
            return random.sample(questions, min(3, len(questions)))

    return []


# Format prompt for LLM
def format_prompt(user_input, search_result="", menu_context=None, selected_model=None):
    full_user_input = (
        f"User question: {user_input}\nReference: {search_result}\nAnswer:"
    )
    user_input_clean = user_input.strip().lower()

    if menu_context and menu_context in MENU_PROMPTS:
        base_instruction = MENU_PROMPTS[menu_context]
        if selected_model and selected_model in MODEL_INFO:
            base_instruction += f"\nThe user is interested in the {selected_model.upper()}. {MODEL_INFO[selected_model]}"
    else:
        base_instruction = (
            "You are a professional Porsche product expert. Always speak positively about Porsche. "
            "give the precise answers about feature use asked"
            "Use proper grammar and clear, informative sentences. Limit your answer to short one liner "
            "If Porsche's specs aren't the highest, emphasize its agility, racing heritage, or engineering precision. "
            "Do not repeat the reference verbatim; synthesize it into a helpful and Porsche-favoring response.\n\n"
            "Do not return User question and reference\n"
        )

    porsche_models = ["911", "taycan", "macan", "cayenne", "panamera", "boxster", "718"]

    if user_input_clean in ["hi", "hello", "hey", "how are you"]:
        return (
            "You are a casual and friendly Porsche chatbot. Respond warmly to the greeting.\n"
            "Limit your answer to short one liner and complete\n"
            + "Do not return User question and reference\n"
            + full_user_input
        )

    if any(brand in user_input_clean for brand in porsche_competitor_brands) and any(
        p_model in user_input_clean for p_model in porsche_models
    ):
        return (
            base_instruction
            + "Focus on what user asked and give answers in a clear and concise manner and keep the answer short one liner and complete\n"
            + "like if user asked speed comparison mention speed in the response.\n"
            + "make is positively biased towards Porsche, always emphasize Porsche's superior handling, racing heritage, "
            "uncompromising engineering, and precision...\n"
            + "Do not return User question and reference\n"
            + full_user_input
        )

    if any(model in user_input_clean for model in porsche_models) and (
        "difference" in user_input_clean or "compare" in user_input_clean
    ):
        return (
            base_instruction
            + "You are a Porsche performance expert. Focus on what user asked and give answers in a clear and concise manner.\n"
            + "Limit your answer to short one liner and complete\n"
            + " like if user asked speed comparison mention speed in the response.\n"
            + "Do not return User question and reference\n"
            + full_user_input
        )

    if any(
        kw in user_input_clean
        for kw in [
            "electric",
            "ev",
            "taycan",
            "charge",
            "sustainability",
            "electrification",
        ]
    ):
        return (
            base_instruction
            + "You are a Porsche EV expert. Mention Porsche's electric models like the Taycan and Macan Electric, "
            "Limit your answer to short one liner and complete\n"
            + "and highlight their performance and engineering.\n"
            + "Do not return User question and reference\n"
            + full_user_input
        )

    if any(
        kw in user_input_clean
        for kw in ["price", "leasing", "quote", "offers", "financing"]
    ):
        return (
            base_instruction
            + "You are a Porsche sales advisor. Share concise pricing or financing points if available.\n"
            + "Limit your answer to short one liner and complete\n"
            + "Do not return User question and reference\n"
            + full_user_input
        )

    if any(
        kw in user_input_clean
        for kw in ["acceleration", "0-60", "top speed", "horsepower", "torque"]
    ):
        logger.info("Prompt condition: Performance-related keyword detected")
        return (
            base_instruction
            + "You are a Porsche performance expert. Focus on what user asked and give answers in a clear and concise manner.\n"
            + "Limit your answer to short one liner and complete\n"
            + " like if user asked speed comparison mention speed in the response.\n"
            + "Do not return User question and reference\n"
            + full_user_input
        )

    if any(
        kw in user_input_clean
        for kw in ["service", "maintenance", "schedule", "center", "plans"]
    ):
        return (
            base_instruction
            + "You are a Porsche service advisor. Provide reliable service info with a friendly tone.\n"
            + "Limit your answer to short one liner and complete\n"
            + "Do not return User question and reference\n"
            + full_user_input
        )

    if any(
        kw in user_input_clean
        for kw in [
            "design",
            "customize",
            "paint to sample",
            "interior color",
            "manufaktur",
        ]
    ):
        return (
            base_instruction
            + "You are a Porsche customization expert. Briefly describe personalization and premium options.\n"
            + "Limit your answer to short one liner and complete\n"
            + "Do not return User question and reference\n"
            + full_user_input
        )

    if any(model in user_input_clean for model in porsche_models):
        return (
            base_instruction
            + "You are a Porsche model specialist. Give a short, passionate overview of the mentioned model.\n"
            + "Limit your answer to short one liner and complete\n"
            + "Do not return User question and reference\n"
            + full_user_input
        )

    return (
        base_instruction
        + "You are a Porsche brand expert. Answer directly and favorably.\n"
        + "Limit your answer to short one liner and complete\n"
        + "Do not return User question and reference\n"
        + full_user_input
    )


# Generate response with memory management and caching
response_cache = {}
CACHE_SIZE = 100  # Maximum number of cached responses


def get_cache_key(prompt, menu_context, selected_model):
    return f"{prompt}_{menu_context}_{selected_model}"


def ask_mistral(prompt, search_result="", menu_context=None, selected_model=None):
    try:
        # Check cache first
        cache_key = get_cache_key(prompt, menu_context, selected_model)
        if cache_key in response_cache:
            return response_cache[cache_key]

        # Force garbage collection before generating response
        gc.collect()

        full_prompt = format_prompt(prompt, search_result, menu_context, selected_model)

        # Optimize response generation
        response = llm(
            full_prompt,
            max_tokens=200,  # Reduced from 150
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.1,
            stop=["User:", "\n\n"],  # Add stop sequences
            echo=False,  # Don't echo the prompt
        )

        # Cache the response
        if len(response_cache) >= CACHE_SIZE:
            # Remove oldest entry if cache is full
            response_cache.pop(next(iter(response_cache)))
        response_cache[cache_key] = response["choices"][0]["text"].strip()

        # Force garbage collection after generating response
        gc.collect()

        return response_cache[cache_key]
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again."


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("message")
def handle_message(message):
    try:
        # Handle both string and dictionary message formats
        if isinstance(message, dict):
            menu_context = message.get("menu_context", None)
            selected_model = message.get("selected_model", None)
            user_message = message.get("message", "")
        else:
            menu_context = None
            selected_model = None
            user_message = str(message)

        if not user_message:
            emit("error", "Empty message received")
            return

        search_result = run_rag(user_message)
        response = ask_mistral(
            user_message,
            search_result=search_result,
            menu_context=menu_context,
            selected_model=selected_model,
        )

        # Get suggested questions based on context
        suggested_questions = get_suggested_questions(menu_context, user_message)

        emit(
            "response",
            {"message": response, "suggested_questions": suggested_questions},
        )
    except Exception as e:
        logger.error(f"Error handling message: {str(e)}")
        emit("error", "Sorry, I encountered an error. Please try again.")


if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
