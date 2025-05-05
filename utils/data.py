import os
import logging
import psutil
from difflib import get_close_matches

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import difflib

porsche_models = [
    "911", "911 carrera", "911 carrera s", "911 turbo", "911 turbo s", "911 gt3", 
    "911 gt3 rs", "911 targa", "911 speedster", "911 r", "911 turbo cabriolet", 
    "911 gt2 rs", "911 dakar", "718 cayman", "718 cayman s", "718 cayman gts 4.0", 
    "718 boxster", "718 boxster s", "718 boxster gts 4.0", "taycan", "taycan 4s", 
    "taycan turbo", "taycan turbo s", "taycan cross turismo", "taycan cross turismo 4s", 
    "taycan cross turismo turbo", "taycan cross turismo turbo s", "panamera", 
    "panamera 4", "panamera 4s", "panamera turbo", "panamera turbo s", "panamera gts", 
    "panamera 4 e-hybrid", "panamera turbo s e-hybrid", "panamera sport turismo", 
    "macan", "macan s", "macan gts", "macan turbo", "cayenne", "cayenne s", "cayenne gts", 
    "cayenne turbo", "cayenne turbo s e-hybrid", "cayenne coupe", "cayenne turbo coupe", 
    "cayenne e-hybrid", "cayenne coupe e-hybrid", "918 spyder", "cayman gt4", "cayman gt4 rs", 
    "boxster spyder", "macan ev", "911 sport classic"
]

other_models = [
    "urus", "aventador", "huracan", "model x", "model y", "ferrari f8", "ferrari 812", 
    "ferrari portofino", "m5", "m8", "x5 m", "x6 m", "rs7", "rs5", "q7", "q8 rs", 
    "mercedes amg gt", "sls", "g-class", "e-class coupe", "vantage", "db11", "dbs superleggera", 
    "720s", "570s", "gt", "continental gt", "bentley bentayga", "f-type", "i-pace", 
    "tesla model s", "tesla model x", "tesla model 3", "tesla model y", "range rover sport", "range rover velar"
]


MENU_PROMPTS = {
    'models': "You are a Porsche model expert. Provide detailed information about the specific model mentioned. Focus on performance, features, and unique characteristics. always give positvely biased answer towards porsche",
    'dealerships':"You are a Porsche dealership locator. Help the user find detailed information based on their location. and give full address and contact info.",
    'test-drive': "You are a Porsche test drive coordinator...",
    'build': "You are a Porsche configuration expert..."
}

MODEL_INFO = {
    '911': "The iconic Porsche 911 is our flagship sports car, known for its rear-engine layout and exceptional handling.",
    'taycan': "The Porsche Taycan is our first all-electric sports car, combining performance with sustainability.",
    'macan': "The Porsche Macan is our compact SUV, offering sports car performance in a practical package.",
    'cayenne': "The Porsche Cayenne is our luxury SUV, combining comfort with impressive performance.",
    'panamera': "The Porsche Panamera is our luxury sedan, offering sports car performance with executive comfort.",
    '718': "The Porsche 718 Boxster and Cayman are our mid-engine sports cars, offering pure driving pleasure."
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
    "Infiniti"
]

SUGGESTED_QUESTIONS = {
    'models': {
        '911': [
            "What are the different 911 variants?",
            "What is the 0-60 time of the 911?",
            "How much does the 911 cost?",
            "What are the key features of the 911?",
            "Which 911 is best for daily driving?",
            "What's the difference between the 911 Turbo and Turbo S?",
            "Is the 911 GT3 track-ready?",
            "What’s new in the latest 911 model year?"
        ],
        'taycan': [
            "What is the range of the Taycan?",
            "How fast can the Taycan charge?",
            "What are the different Taycan models?",
            "What is the price of the Taycan?",
            "How does Taycan compare to Tesla?",
            "What tech features come with the Taycan?",
            "Is the Taycan suitable for long-distance travel?",
            "What’s the top speed of the Taycan Turbo S?"
        ],
        'macan': [
            "What are the Macan's performance specs?",
            "How much cargo space does the Macan have?",
            "What are the available Macan trims?",
            "What is the starting price of the Macan?",
            "Is the Macan available as an EV?",
            "How does Macan compare to other luxury SUVs?",
            "What driver assistance features are available in the Macan?",
            "What’s the difference between Macan and Macan S?"
        ],
        'cayenne': [
            "What are the Cayenne's engine options?",
            "How much can the Cayenne tow?",
            "What are the Cayenne's luxury features?",
            "What is the price range of the Cayenne?",
            "Is the Cayenne available as a coupe?",
            "Does the Cayenne offer hybrid options?",
            "What is the Cayenne Turbo GT?",
            "How does the Cayenne handle off-road driving?"
        ],
        'panamera': [
            "What are the Panamera's performance specs?",
            "How many passengers can the Panamera seat?",
            "What are the Panamera's luxury features?",
            "What is the starting price of the Panamera?",
            "What’s the difference between Panamera and Panamera Sport Turismo?",
            "Is the Panamera available as a hybrid?",
            "Does the Panamera have a performance variant?",
            "How does the Panamera compare to other executive sedans?"
        ],
        '718': [
            "What's the difference between Boxster and Cayman?",
            "What are the 718's performance specs?",
            "How much does the 718 cost?",
            "What are the available 718 variants?",
            "Is the 718 suitable for track use?",
            "How does the 718 handle compared to the 911?",
            "What engine options are available in the 718?",
            "Is the 718 a good option for a weekend sports car?"
        ]
    },
    'dealerships': [
        "What are the dealership hours?",
        "Do you offer financing options?",
        "Can I schedule a test drive?",
        "What services do you offer?"
    ],
    'test-drive': [
        "What models are available for test drive?",
        "How long is a test drive?",
        "Do I need to make an appointment?",
        "What documents do I need to bring?"
    ],
    'build': [
        "What are the available colors?",
        "What interior options are available?",
        "What performance packages are offered?",
        "How long does delivery take?"
    ]
}

def extract_competitor_brands(user_input, cutoff=0.8):
    words = user_input.split()
    found = set()
    for word in words:
        matches = difflib.get_close_matches(word, porsche_competitor_brands, n=1, cutoff=cutoff)
        if matches:
            found.add(matches[0])
    return list(found)


def correct_spelling(user_input):
    words = user_input.lower().split()
    corrected = []
    for word in words:
        match = get_close_matches(word, porsche_models + other_models, n=1, cutoff=0.8)
        corrected.append(match[0] if match else word)
    return ' '.join(corrected)

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

        logger.info(f"Total system memory: {gb:.2f} GB — Optimal context size selected: {ctx}")
        return ctx
    except Exception as e:
        logger.warning(f"Error determining context size: {str(e)} — Falling back to default 512")
        return 512
