# ============================================================
# GEOCHAT API v12.0
# ============================================================
#
# Local GeoChat-7B Satellite Vision-Language API
#
# SINGLE IMAGE:
#   POST /chat
#
# TWO IMAGE:
#   POST /compare
#   POST /compare/visual
#
# HEALTH:
#   GET /
#   GET /health
#
# JSON HANDOFF:
#   /chat -> answer
#   /compare -> answer, before.description, after.description,
#               comparison.answer, comparison.changed_area_percent,
#               comparison.regions
#
# IMPORTANT:
#   This service does NOT use external LLMs.
#   GeoChat-7B is the primary and only VLM.
# ============================================================

# ============================================================
# GPU MEMORY CONFIG
# ============================================================

import os

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

# ============================================================
# IMPORTS
# ============================================================

import gc
import io
import re
import time
import uuid
import secrets
import traceback
import threading

from typing import (
    Any,
    Iterable,
    Optional,
)

import numpy as np
import torch

from PIL import (
    Image,
    ImageDraw,
    UnidentifiedImageError,
)

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import (
    JSONResponse,
    Response,
)

from starlette.concurrency import run_in_threadpool

from geochat.conversation import (
    Chat,
    conv_templates,
)

from geochat.mm_utils import (
    get_model_name_from_path,
)

from geochat.model.builder import (
    load_pretrained_model,
)


# ============================================================
# CONFIGURATION
# ============================================================

API_VERSION = "12.0.0"

MODEL_PATH = r"D:\AI\models\geochat-7B"

GPU_ID = 0

DEVICE = f"cuda:{GPU_ID}"

HOST = "0.0.0.0"

PORT = 8000


# ============================================================
# API KEY
# ============================================================

API_KEY = os.getenv(
    "GEOCHAT_API_KEY"
)

if not API_KEY:

    raise RuntimeError(
        "\n"
        "============================================================\n"
        "GEOCHAT_API_KEY IS NOT CONFIGURED\n"
        "============================================================\n\n"
        "PowerShell:\n\n"
        '$env:GEOCHAT_API_KEY="your-secret-key"\n\n'
        "Then restart the server.\n"
        "============================================================\n"
    )


# ============================================================
# IMAGE / REQUEST LIMITS
# ============================================================

MAX_IMAGE_SIZE = 15 * 1024 * 1024

MAX_IMAGE_DIMENSION = 12000

MAX_QUESTION_LENGTH = 4000


# ============================================================
# GENERATION SETTINGS
# ============================================================

TEMPERATURE = 0.08

TOP_P = 0.75

# Increased from 300 for more detailed reports.
MAX_NEW_TOKENS = 450

MAX_LENGTH = 2000


# ============================================================
# COMPARISON SETTINGS
# ============================================================

DIFF_THRESHOLD = 28

MIN_CHANGED_AREA_PERCENT = 0.15

MAX_REPORTED_REGIONS = 20

MAX_COMPARE_PIXELS = 3000 * 3000


# ============================================================
# DESCRIPTION SETTINGS
# ============================================================

MAX_COMPARISON_DESCRIPTION_CHARS = 1400

MAX_COMPARISON_QUESTION_CHARS = 1200


# ============================================================
# GPU CONCURRENCY
# ============================================================

# Only one GeoChat generation at a time.
inference_semaphore = threading.Semaphore(1)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="GeoChat Satellite Intelligence API",
    description=(
        "Local GeoChat-7B satellite "
        "vision-language microservice."
    ),
    version=API_VERSION,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT / HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "GeoChat-7B",
        "version": API_VERSION,
        "message": "GeoChat API is running",
        "endpoints": {
            "chat": "/chat",
            "compare": "/compare",
            "compare_visual": "/compare/visual",
            "health": "/health",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "GeoChat-7B",
        "version": API_VERSION,
        "device": DEVICE,
        "gpu": GPU_NAME if "GPU_NAME" in globals() else None,
    }


# ============================================================
# REQUEST ID
# ============================================================

def generate_request_id() -> str:
    return str(uuid.uuid4())


# ============================================================
# GEOCHAT OUTPUT CLEANER
# ============================================================

def clean_geochat_output(text: Any) -> str:
    """Cleans GeoChat's raw output.

    Removes grounding markup, HTML tags, stop markers, etc.
    Does not generate new information.
    """
    if text is None:
        return ""
    text = str(text)
    if not text:
        return ""

    text = text.replace("\x00", "")

    stop_markers = (
        "</s>",
        "<|endoftext|>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|end|>",
    )
    for marker in stop_markers:
        if marker in text:
            text = text.split(marker, 1)[0]

    text = re.sub(r"^\s*(?:assistant)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*final\s+answer\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?\s*p(?:\s+[^>]*)?\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"</?\s*(?:div|span|br|b|strong|i|em|li|ul|ol|h[1-6])(?:\s+[^>]*)?>",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<\s*delim\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\{[^{}]*<\s*\d+\s*>[^{}]*\}", " ", text)
    text = re.sub(r"(?:<\s*\d+\s*>){2,}", " ", text)
    text = re.sub(r"<\s*\d+\s*>", " ", text)
    text = re.sub(r"<[^>\n]{1,120}>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"\{(?:\s*[\d<>{}|,\-]+\s*)+\}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# BAD ANSWER DETECTION
# ============================================================

def is_empty_or_useless_answer(answer: str) -> bool:
    if not answer:
        return True
    normalized = answer.strip().lower()
    useless_answers = {
        "image", "image.", "picture", "picture.",
        "photo", "photo.", "frame", "frame.",
    }
    if normalized in useless_answers:
        return True
    return len(normalized) < 5


# ============================================================
# QUESTION NORMALIZATION
# ============================================================

def normalize_question(question: str) -> str:
    question = re.sub(r"\s+", " ", question.strip())
    if not question:
        return question
    replacements = {
        r"\bannalise\b": "analyze",
        r"\banalise\b": "analyze",
        r"\banlyse\b": "analyze",
        r"\bpls\b": "please",
        r"\bplz\b": "please",
        r"\bpics\b": "images",
        r"\bpic\b": "image",
    }
    for pattern, replacement in replacements.items():
        question = re.sub(pattern, replacement, question, flags=re.IGNORECASE)
    return question


# ============================================================
# CASUAL / VAGUE QUESTION HELPERS
# ============================================================

def is_casual_question(question: str) -> bool:
    q = re.sub(r"\s+", " ", question.strip().lower()).rstrip("!?., ")
    casual_questions = {
        "hi", "hii", "hiii", "hello", "helo", "heloo",
        "hey", "heyy", "hlo", "hola", "yo", "sup",
        "good morning", "good afternoon", "good evening",
        "thanks", "thank you", "thx",
        "ok", "okay",
    }
    return q in casual_questions


def is_vague_visual_question(question: str) -> bool:
    q = re.sub(r"\s+", " ", question.strip().lower()).rstrip(".!?")
    vague_questions = {
        "analyze the image", "analyse the image", "analyze image", "analyse image",
        "look at the image", "look at this image", "describe the image",
        "describe this image", "what is this", "what is this image",
        "what is this satellite image", "what is in this image",
        "what is in the image", "what do you see", "tell me about the image",
        "image analysis", "what is visible in this image",
        "what is visible in the image", "what is visible in this satellite image",
    }
    return q in vague_questions


# ============================================================
# QUESTION CLASSIFICATION
# ============================================================

def detect_question_type(question: str) -> str:
    q = normalize_question(question).lower().strip()
    if is_casual_question(q):
        return "casual"
    if is_vague_visual_question(q):
        return "general"
    if any(x in q for x in (
        "compare", "difference", "differences", "changed",
        "change", "before and after", "before vs after",
    )):
        return "comparison"
    if any(x in q for x in ("how many", "count", "number of", "quantity")):
        return "counting"
    if any(q.startswith(x) for x in (
        "is there", "are there", "do you see", "can you see",
        "does the image", "is the ", "are the ",
    )):
        return "binary"
    if any(x in q for x in (
        "where", "located", "location", "position",
        "which part", "which area",
    )):
        return "location"
    if any(x in q for x in (
        "vegetation", "tree", "trees", "forest", "greenery",
        "plants", "crop", "crops", "agriculture", "farmland",
    )):
        return "vegetation"
    if any(x in q for x in (
        "building", "buildings", "structure", "structures",
        "house", "houses", "roof", "rooftop",
    )):
        return "buildings"
    if any(x in q for x in ("bridge", "bridges")):
        return "bridges"
    if any(x in q for x in (
        "road", "roads", "street", "streets", "highway",
        "path", "intersection", "lane",
    )):
        return "roads"
    if any(x in q for x in (
        "water", "river", "lake", "pond", "sea",
        "ocean", "reservoir", "canal",
    )):
        return "water"
    if any(x in q for x in (
        "land use", "land cover", "terrain", "landscape",
        "urban", "rural", "industrial", "residential", "agricultural",
    )):
        return "land_use"
    return "general"


# ============================================================
# SINGLE IMAGE PROMPT
# ============================================================

GENERAL_PROMPT = (
    "Analyze this satellite image carefully and provide a "
    "detailed, evidence-based visual description. "
    "First directly answer the user's question. "
    "Then provide the important supporting observations "
    "visible in the image. "
    "Consider, when visible and relevant: buildings and "
    "structures, roads and paths, bridges, rivers and other "
    "water bodies, vegetation, agricultural land, vehicles, "
    "parking areas, industrial areas, residential areas, "
    "open land, terrain, and infrastructure. "
    "Describe the approximate position of important objects "
    "using image-relative terms such as upper-left, "
    "upper-right, center, lower-left, or lower-right. "
    "Mention spatial relationships between visible objects "
    "when useful, such as a bridge crossing a river, a road "
    "connecting buildings, or structures located near one "
    "another. "
    "Only mention information supported by the image. "
    "Do not invent locations, coordinates, exact dimensions, "
    "dates, causes, or unsupported facts. "
    "When something is uncertain, clearly state the uncertainty. "
    "Give a detailed natural-language answer rather than "
    "a one-line image caption."
)


def build_chat_prompt(question: str, question_type: str) -> str:
    question = normalize_question(question)

    if question_type == "general":
        return (
            f"{GENERAL_PROMPT}\n\n"
            f"User question: {question}\n\n"
            "Provide approximately 100-180 words when the "
            "image contains enough visible information. "
            "Focus on useful observations rather than "
            "repeating the question.\n\n"
            "Answer:"
        )

    if question_type == "counting":
        return (
            "Analyze the satellite image carefully.\n\n"
            f"User question: {question}\n\n"
            "Count only clearly distinguishable visible "
            "objects. Give the count first, then explain "
            "where the objects are located and any relevant "
            "visual context. If an exact count is unreliable, "
            "say so instead of fabricating a number.\n\n"
            "Answer:"
        )

    if question_type == "binary":
        return (
            "Analyze the satellite image carefully.\n\n"
            f"User question: {question}\n\n"
            "Answer Yes, No, or Unclear based only on "
            "visible evidence. Then provide a short but "
            "useful explanation describing the visible "
            "evidence and approximate location.\n\n"
            "Answer:"
        )

    if question_type == "location":
        return (
            "Analyze the satellite image carefully.\n\n"
            f"User question: {question}\n\n"
            "Describe where the requested object or feature "
            "appears using image-relative terms such as "
            "upper-left, upper-right, center, lower-left, "
            "or lower-right. Include nearby visible context. "
            "Do not invent geographic coordinates.\n\n"
            "Answer:"
        )

    if question_type == "vegetation":
        return (
            "Analyze the satellite image carefully.\n\n"
            f"User question: {question}\n\n"
            "Describe visible vegetation patterns in detail. "
            "Mention approximate location, density, shape, "
            "and surrounding land cover when visible. "
            "Do not identify plant species unless clearly "
            "supported by the image.\n\n"
            "Answer:"
        )

    if question_type == "buildings":
        return (
            "Analyze the satellite image carefully.\n\n"
            f"User question: {question}\n\n"
            "Identify visible building-like structures using "
            "footprints, rooftops, geometry, density, and "
            "surrounding context. Describe approximate "
            "locations and relevant relationships between "
            "structures and roads or open land.\n\n"
            "Answer:"
        )

    if question_type == "bridges":
        return (
            "Analyze the satellite image carefully.\n\n"
            f"User question: {question}\n\n"
            "Identify visible bridges and describe their "
            "approximate locations, apparent orientation, "
            "what they cross, and their relationship with "
            "nearby roads or infrastructure. Only state "
            "what is visually supported.\n\n"
            "Answer:"
        )

    if question_type == "roads":
        return (
            "Analyze the satellite image carefully.\n\n"
            f"User question: {question}\n\n"
            "Identify visible roads or paths. Describe their "
            "approximate locations, directions, intersections, "
            "and relationship to nearby buildings, bridges, "
            "water, or open land. Distinguish roads from "
            "rivers, drainage channels, shadows, and field "
            "boundaries where possible.\n\n"
            "Answer:"
        )

    if question_type == "water":
        return (
            "Analyze the satellite image carefully.\n\n"
            f"User question: {question}\n\n"
            "Identify visible water bodies or water-like "
            "regions using shape, texture, continuity, and "
            "context. Describe their approximate location "
            "and relationship with nearby roads, bridges, "
            "buildings, or vegetation when visible.\n\n"
            "Answer:"
        )

    if question_type == "land_use":
        return (
            "Analyze the satellite image carefully.\n\n"
            f"User question: {question}\n\n"
            "Describe visible land-use and land-cover patterns "
            "in detail. Discuss urban, residential, industrial, "
            "agricultural, vegetated, open, and water-covered "
            "areas when clearly visible. Describe approximate "
            "spatial distribution and relationships.\n\n"
            "Answer:"
        )

    return (
        f"{GENERAL_PROMPT}\n\n"
        f"User question: {question}\n\n"
        "Provide approximately 100-180 words when there is "
        "enough visible information. Answer the question "
        "directly first, then provide supporting observations.\n\n"
        "Answer:"
    )


# ============================================================
# COMPARISON PROMPT
# ============================================================

def truncate_text(text: str, maximum: int) -> str:
    text = text or ""
    if len(text) <= maximum:
        return text
    return text[:maximum].rstrip() + "..."


def build_comparison_prompt(
    question: str,
    before_description: str,
    after_description: str,
    changed_area_percent: float,
) -> str:
    question = normalize_question(question)
    question = truncate_text(question, MAX_COMPARISON_QUESTION_CHARS)
    before_description = truncate_text(before_description, MAX_COMPARISON_DESCRIPTION_CHARS)
    after_description = truncate_text(after_description, MAX_COMPARISON_DESCRIPTION_CHARS)

    return (
        "You are comparing two satellite images of the same "
        "or similar area.\n\n"
        "BEFORE IMAGE OBSERVATION:\n"
        f"{before_description}\n\n"
        "AFTER IMAGE OBSERVATION:\n"
        f"{after_description}\n\n"
        "TECHNICAL PIXEL DIFFERENCE:\n"
        f"{changed_area_percent:.2f}% of compared pixels differ.\n\n"
        "USER QUESTION:\n"
        f"{question}\n\n"
        "Provide a detailed comparison. "
        "Start with the main changes. Then discuss the "
        "most important visible differences in buildings, "
        "roads, bridges, vegetation, water, vehicles, "
        "structures, infrastructure, or land use when relevant. "
        "Describe where the changes appear in the image using "
        "relative positions such as upper-left, upper-right, "
        "center, lower-left, or lower-right. "
        "Also mention important features that appear similar "
        "or unchanged when useful. "
        "Do not invent causes, dates, exact measurements, or "
        "geographic locations. "
        "The pixel-difference percentage is a technical image "
        "difference and must not be treated as a percentage "
        "of real-world semantic change. "
        "Give a detailed natural-language report.\n\n"
        "Answer:"
    )


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="Image exceeds the 15 MB limit.")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        if image.width > MAX_IMAGE_DIMENSION or image.height > MAX_IMAGE_DIMENSION:
            raise HTTPException(
                status_code=413,
                detail=f"Image dimensions are too large. Maximum dimension is {MAX_IMAGE_DIMENSION}px.",
            )
        if image.width < 2 or image.height < 2:
            raise HTTPException(status_code=400, detail="Image dimensions are invalid.")
        return image.convert("RGB")
    except HTTPException:
        raise
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {type(exc).__name__}: {exc}")


# ============================================================
# MODEL LOADING
# ============================================================

print()
print("=" * 70)
print("LOADING GEOCHAT API v12.0")
print("=" * 70)
print("Python file:", os.path.abspath(__file__))
print("Model path:", MODEL_PATH)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. GeoChat requires an NVIDIA GPU.")

if GPU_ID >= torch.cuda.device_count():
    raise RuntimeError(
        f"GPU_ID={GPU_ID} is invalid. Available GPUs: {torch.cuda.device_count()}"
    )

GPU_NAME = torch.cuda.get_device_name(GPU_ID)
model_name = get_model_name_from_path(MODEL_PATH)

print("Model name:", model_name)
print("GPU:", GPU_NAME)
print("Device:", DEVICE)
print("Loading model...")

tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH, None, model_name, False, True, device=DEVICE,
)
model = model.eval()

chat = Chat(model, image_processor, tokenizer, device=DEVICE)

print()
print("=" * 70)
print("GEOCHAT LOADED SUCCESSFULLY")
print("=" * 70)
print("Model:", MODEL_PATH)
print("GPU:", GPU_NAME)
print("Device:", DEVICE)
print("Context length:", context_len)
print("Max new tokens:", MAX_NEW_TOKENS)
print("External LLM: DISABLED")
print("History: DISABLED")
print("Sessions: DISABLED")
print("Automatic retry: DISABLED")
print("=" * 70)


# ============================================================
# AUTHENTICATION
# ============================================================

def verify_api_key(api_key):
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key.")
    if not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key.")


# ============================================================
# STREAM TEXT EXTRACTION
# ============================================================

def extract_stream_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, dict):
        for key in ("text", "content", "delta", "answer", "response", "generated_text"):
            if key in value:
                text = extract_stream_text(value[key])
                if text:
                    return text
        choices = value.get("choices")
        if choices:
            for item in choices:
                text = extract_stream_text(item)
                if text:
                    return text
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            text = extract_stream_text(item)
            if text:
                return text
        return ""
    for attr in ("text", "content", "delta", "answer", "response", "generated_text"):
        try:
            item = getattr(value, attr, None)
            if item is not None:
                text = extract_stream_text(item)
                if text:
                    return text
        except Exception:
            pass
    try:
        return str(value)
    except Exception:
        return ""


def merge_stream_chunk(current: str, chunk: str) -> str:
    if not chunk:
        return current
    if not current:
        return chunk
    if chunk == current:
        return current
    if chunk.startswith(current):
        return chunk
    if current.startswith(chunk):
        return current
    if current in chunk:
        return chunk
    if chunk in current:
        return current
    maximum = min(len(current), len(chunk))
    for overlap in range(maximum, 0, -1):
        if current[-overlap:] == chunk[:overlap]:
            return current + chunk[overlap:]
    return current + chunk


def collect_stream_output(streamer):
    accumulated = ""
    chunk_count = 0
    for item in streamer:
        chunk = extract_stream_text(item)
        if not chunk:
            continue
        chunk_count += 1
        accumulated = merge_stream_chunk(accumulated, chunk)
    print()
    print("[GeoChat] Stream chunks:", chunk_count)
    print("[GeoChat] Raw complete output:", repr(accumulated))
    final_text = clean_geochat_output(accumulated)
    print("[GeoChat] Clean output:", repr(final_text))
    return final_text


# ============================================================
# GEOCHAT GENERATION
# ============================================================

def generate_once(image, prompt):
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    conv = conv_templates["llava_v1"].copy()
    img_list = []

    chat.upload_img(image, conv, img_list)
    if not img_list:
        raise RuntimeError("GeoChat rejected the image.")
    chat.ask(prompt, conv)
    chat.encode_img(img_list)
    if not img_list:
        raise RuntimeError("GeoChat image encoding failed.")

    print()
    print("-" * 70)
    print("[GeoChat] GENERATION START")
    print("-" * 70)
    print("[GeoChat] Prompt length:", len(prompt))

    generation_start = time.perf_counter()

    streamer = chat.stream_answer(
        conv=conv, img_list=img_list,
        temperature=TEMPERATURE, top_p=TOP_P,
        max_new_tokens=MAX_NEW_TOKENS, max_length=MAX_LENGTH,
    )

    answer = collect_stream_output(streamer)

    generation_time = time.perf_counter() - generation_start
    print("[GeoChat] Generation:", f"{generation_time:.2f}s")
    print("[GeoChat] Final cleaned answer:", repr(answer))
    print("[GeoChat] Answer length:", len(answer))
    print("-" * 70)
    return answer


def safe_generate(image, prompt):
    acquired = inference_semaphore.acquire(timeout=300)
    if not acquired:
        raise RuntimeError("GeoChat inference queue timeout.")
    try:
        return generate_once(image, prompt)
    finally:
        inference_semaphore.release()


def run_geochat(image, question, question_type):
    if question_type == "casual":
        return (
            "Hello! Upload a satellite image and "
            "ask me what you want to analyze."
        )
    prompt = build_chat_prompt(question, question_type)
    return safe_generate(image, prompt)


# ============================================================
# SINGLE IMAGE ENDPOINT
# ============================================================

@app.post("/chat")
async def chat_endpoint(
    image: UploadFile = File(...),
    question: str = Form(...),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    verify_api_key(x_api_key)
    request_id = generate_request_id()
    start_time = time.perf_counter()
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail="Question is too long. Maximum is 4000 characters.",
        )
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    try:
        image_bytes = await image.read()
        pil_image = validate_image(image_bytes)
        normalized_question = normalize_question(question)
        question_type = detect_question_type(normalized_question)
        if normalized_question.lower() == "what is this":
            question_type = "general"

        print()
        print("=" * 70)
        print("NEW GEOCHAT REQUEST")
        print("=" * 70)
        print("Request ID:", request_id)
        print("Image:", image.filename)
        print("Image size:", pil_image.size)
        print("Question:", question)
        print("Normalized:", normalized_question)
        print("Question type:", question_type)
        print("=" * 70)

        generation_start = time.perf_counter()
        answer = await run_in_threadpool(run_geochat, pil_image, normalized_question, question_type)
        generation_time = time.perf_counter() - generation_start
        answer = clean_geochat_output(answer)

        if is_empty_or_useless_answer(answer):
            total_time = time.perf_counter() - start_time
            return JSONResponse(
                status_code=502,
                content={
                    "success": False,
                    "request_id": request_id,
                    "provider": "SatQueryLocalEngine",
                    "model": "GeoChat-7B",
                    "status": "model_output_unusable",
                    "question": question,
                    "normalized_question": normalized_question,
                    "question_type": question_type,
                    "answer": None,
                    "text": None,
                    "error": "GeoChat produced an empty or unusable answer.",
                    "fallback_required": True,
                    "analysis": {
                        "visual_model_used": question_type != "casual",
                        "image_provided": True,
                        "generation_time_seconds": round(generation_time, 2),
                        "total_time_seconds": round(total_time, 2),
                    },
                },
            )

        total_time = time.perf_counter() - start_time
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = round(torch.cuda.memory_allocated(GPU_ID) / (1024 ** 3), 2)

        result = {
            "success": True,
            "request_id": request_id,
            "provider": "SatQueryLocalEngine",
            "model": "GeoChat-7B",
            "status": "completed",
            "image": {
                "filename": image.filename,
                "width": pil_image.width,
                "height": pil_image.height,
                "mode": pil_image.mode,
            },
            "question": question,
            "normalized_question": normalized_question,
            "question_type": question_type,
            "answer": answer,
            "text": answer,
            "result": {"answer": answer, "text": answer},
            "answer_length": len(answer),
            "analysis": {
                "type": "satellite_image",
                "visual_model_used": question_type != "casual",
                "image_provided": True,
                "history_used": False,
                "sessions_used": False,
                "external_synthesis_used": False,
                "automatic_retry": False,
                "generation_time_seconds": round(generation_time, 2),
                "total_time_seconds": round(total_time, 2),
                "gpu_memory_allocated_gb": gpu_memory,
            },
            "grounding": {
                "image_based": question_type != "casual",
                "evidence_artifacts": [],
            },
        }

        print()
        print("=" * 70)
        print("GEOCHAT JSON RESPONSE")
        print("=" * 70)
        print("Request ID:", request_id)
        print("Question type:", question_type)
        print()
        print("ANSWER SENT TO BACKEND:")
        print(answer)
        print()
        print("Answer length:", len(answer))
        print("Generation time:", f"{generation_time:.2f}s")
        print("Total time:", f"{total_time:.2f}s")
        print("HTTP status:", 200)
        print("=" * 70)

        return JSONResponse(
            status_code=200,
            content=result,
            headers={
                "X-GeoChat-Model": "GeoChat-7B",
                "X-GeoChat-Answer-Length": str(len(answer)),
                "X-GeoChat-Request-ID": request_id,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        print()
        print("=" * 70)
        print("GEOCHAT ERROR")
        print("=" * 70)
        print("Request ID:", request_id)
        print("Type:", type(exc).__name__)
        print("Message:", str(exc))
        traceback.print_exc()
        print("=" * 70)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


# ============================================================
# IMAGE NORMALIZATION FOR COMPARISON
# ============================================================

def _normalize_for_comparison(image, width, height):
    image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(image).astype(np.float32)


def calculate_difference(image1, image2):
    width = min(image1.width, image2.width)
    height = min(image1.height, image2.height)

    if width * height > MAX_COMPARE_PIXELS:
        scale = (MAX_COMPARE_PIXELS / (width * height)) ** 0.5
        width = max(256, int(width * scale))
        height = max(256, int(height * scale))

    arr1 = _normalize_for_comparison(image1, width, height)
    arr2 = _normalize_for_comparison(image2, width, height)

    gray1 = 0.299 * arr1[:, :, 0] + 0.587 * arr1[:, :, 1] + 0.114 * arr1[:, :, 2]
    gray2 = 0.299 * arr2[:, :, 0] + 0.587 * arr2[:, :, 1] + 0.114 * arr2[:, :, 2]
    diff = np.abs(gray1 - gray2)
    regions = []

    try:
        import cv2
        blur1 = cv2.GaussianBlur(gray1, (5, 5), 0)
        blur2 = cv2.GaussianBlur(gray2, (5, 5), 0)
        diff = np.abs(blur1 - blur2)
        mask = (diff > DIFF_THRESHOLD).astype(np.uint8) * 255

        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        total_pixels = mask.size
        min_region_area = max(150, int(total_pixels * 0.00005))
        cleaned = np.zeros_like(mask)

        for index in range(1, num_labels):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < min_region_area:
                continue
            x = int(stats[index, cv2.CC_STAT_LEFT])
            y = int(stats[index, cv2.CC_STAT_TOP])
            w = int(stats[index, cv2.CC_STAT_WIDTH])
            h = int(stats[index, cv2.CC_STAT_HEIGHT])
            cleaned[labels == index] = 255
            cx = x + w / 2
            cy = y + h / 2
            horizontal = "western" if cx < width / 3 else "eastern" if cx > width * 2 / 3 else "central"
            vertical = "northern" if cy < height / 3 else "southern" if cy > height * 2 / 3 else "central"
            if horizontal == "central" and vertical == "central":
                location = "central region"
            elif horizontal == "central":
                location = f"{vertical} region"
            elif vertical == "central":
                location = f"{horizontal} region"
            else:
                location = f"{vertical}-{horizontal} region"
            regions.append({
                "type": "visual_change",
                "location": location,
                "area_percent": round(area / total_pixels * 100, 2),
                "bounding_box": {"x": x, "y": y, "width": w, "height": h},
            })

        mask = cleaned > 0
    except ImportError:
        print("[Compare] OpenCV not installed; using simple threshold.")
        mask = diff > DIFF_THRESHOLD

    total_pixels = mask.size
    changed_pixels = int(mask.sum())
    changed_area_percent = changed_pixels / total_pixels * 100 if total_pixels else 0.0

    before = image1.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    after = image2.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)

    regions.sort(key=lambda item: item["area_percent"], reverse=True)
    return before, after, mask, changed_area_percent, regions[:MAX_REPORTED_REGIONS]


def create_highlighted_image(before, after, mask):
    before = before.copy().convert("RGB")
    after = after.copy().convert("RGB")
    width, height = before.size
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[mask] = [255, 0, 0, 165]
    highlighted = Image.alpha_composite(
        before.convert("RGBA"),
        Image.fromarray(overlay, "RGBA"),
    ).convert("RGB")
    output = Image.new("RGB", (width * 3, height), "white")
    output.paste(before, (0, 0))
    output.paste(after, (width, 0))
    output.paste(highlighted, (width * 2, 0))
    draw = ImageDraw.Draw(output)
    label_height = min(40, max(24, height // 12))
    labels = ((0, "BEFORE"), (width, "AFTER"), (width * 2, "CHANGES"))
    for x, label in labels:
        draw.rectangle((x, 0, x + width, label_height), fill="black")
        draw.text((x + 10, max(4, label_height // 4)), label, fill="white")
    return output


def analyze_comparison_image(image, label):
    prompt = (
        "Analyze this satellite image carefully and provide "
        "a detailed visual description. "
        "Describe the overall scene first. Then describe "
        "the important visible objects and structures. "
        "Pay particular attention to buildings, roads, "
        "bridges, water, vegetation, agricultural land, "
        "vehicles, infrastructure, open areas, and other "
        "distinctive features when visible. "
        "Describe approximate image-relative positions such "
        "as upper-left, upper-right, center, lower-left, "
        "or lower-right. "
        "Describe relationships between objects when useful. "
        "Only describe what is supported by the image. "
        "Do not invent exact locations, coordinates, exact "
        "measurements, dates, or causes. "
        "When something is uncertain, state the uncertainty. "
        "Return approximately 100-170 words when sufficient "
        "visual information is available."
    )
    print()
    print("=" * 70)
    print(f"GEOCHAT {label} IMAGE ANALYSIS")
    print("=" * 70)
    print("Image size:", image.size)
    print("Prompt length:", len(prompt))
    print("=" * 70)
    answer = safe_generate(image, prompt)
    answer = clean_geochat_output(answer)
    if is_empty_or_useless_answer(answer):
        answer = f"No reliable visual description was generated for the {label.lower()} image."
    print()
    print(f"[GeoChat {label}] ANSWER:")
    print(answer)
    print(f"[GeoChat {label}] LENGTH:", len(answer))
    return answer


# ============================================================
# TWO-IMAGE COMPARISON ENDPOINT
# ============================================================

@app.post("/compare")
async def compare_endpoint(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    question: str = Form(
        default=(
            "Compare these two satellite images. "
            "Describe what is visible in each image "
            "and identify the important differences "
            "between them."
        )
    ),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    verify_api_key(x_api_key)
    request_id = generate_request_id()
    start_time = time.perf_counter()
    try:
        if not image1.content_type or not image1.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="image1 must be an image.")
        if not image2.content_type or not image2.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="image2 must be an image.")

        before = validate_image(await image1.read())
        after = validate_image(await image2.read())

        print()
        print("=" * 70)
        print("NEW TWO-IMAGE GEOCHAT COMPARISON")
        print("=" * 70)
        print("Request ID:", request_id)
        print("BEFORE:", image1.filename)
        print("AFTER:", image2.filename)
        print("BEFORE size:", before.size)
        print("AFTER size:", after.size)
        print("QUESTION:", question)
        print("=" * 70)

        before_start = time.perf_counter()
        before_description = await run_in_threadpool(analyze_comparison_image, before, "BEFORE")
        before_time = time.perf_counter() - before_start

        after_start = time.perf_counter()
        after_description = await run_in_threadpool(analyze_comparison_image, after, "AFTER")
        after_time = time.perf_counter() - after_start

        difference_start = time.perf_counter()
        before_resized, after_resized, mask, changed_area_percent, regions = calculate_difference(before, after)
        difference_time = time.perf_counter() - difference_start

        combined = Image.new(
            "RGB",
            (before_resized.width * 2, before_resized.height),
        )
        combined.paste(before_resized, (0, 0))
        combined.paste(after_resized, (before_resized.width, 0))

        comparison_prompt = build_comparison_prompt(
            question, before_description, after_description, changed_area_percent,
        )
        semantic_start = time.perf_counter()
        comparison_answer = await run_in_threadpool(safe_generate, combined, comparison_prompt)
        comparison_answer = clean_geochat_output(comparison_answer)
        semantic_time = time.perf_counter() - semantic_start
        if is_empty_or_useless_answer(comparison_answer):
            comparison_answer = "The semantic comparison did not produce a reliable result."

        changes_detected = changed_area_percent >= MIN_CHANGED_AREA_PERCENT
        total_time = time.perf_counter() - start_time

        result = {
            "success": True,
            "request_id": request_id,
            "provider": "SatQueryLocalEngine",
            "model": "GeoChat-7B",
            "status": "completed",
            "before": {
                "filename": image1.filename,
                "width": before.width,
                "height": before.height,
                "description": before_description,
                "analysis_time_seconds": round(before_time, 2),
            },
            "after": {
                "filename": image2.filename,
                "width": after.width,
                "height": after.height,
                "description": after_description,
                "analysis_time_seconds": round(after_time, 2),
            },
            "question": question,
            "comparison": {
                "answer": comparison_answer,
                "text": comparison_answer,
                "changes_detected": changes_detected,
                "changed_area_percent": round(changed_area_percent, 2),
                "regions": regions,
            },
            "answer": comparison_answer,
            "text": comparison_answer,
            "answer_length": len(comparison_answer),
            "result": {
                "answer": comparison_answer,
                "text": comparison_answer,
                "before_description": before_description,
                "after_description": after_description,
                "changes_detected": changes_detected,
                "changed_area_percent": round(changed_area_percent, 2),
            },
            "analysis": {
                "type": "satellite_image_comparison",
                "two_images_analyzed_independently": True,
                "pixel_difference": True,
                "semantic_analysis": True,
                "external_synthesis_used": False,
                "automatic_retry": False,
                "history_used": False,
                "sessions_used": False,
                "before_analysis_time_seconds": round(before_time, 2),
                "after_analysis_time_seconds": round(after_time, 2),
                "difference_time_seconds": round(difference_time, 2),
                "semantic_comparison_time_seconds": round(semantic_time, 2),
                "total_time_seconds": round(total_time, 2),
            },
            "grounding": {
                "before_image_used": True,
                "after_image_used": True,
                "image_pair_used": True,
                "deterministic_pixel_analysis": True,
                "visual_model_used": True,
                "evidence_artifacts": [
                    "before_description",
                    "after_description",
                    "changed_area_percent",
                    "changed_regions",
                ],
            },
        }

        print()
        print("=" * 70)
        print("TWO-IMAGE COMPARISON COMPLETE")
        print("=" * 70)
        print("BEFORE DESCRIPTION:", before_description)
        print("AFTER DESCRIPTION:", after_description)
        print("COMPARISON ANSWER:", comparison_answer)
        print("CHANGED AREA:", f"{changed_area_percent:.2f}%")
        print("CHANGES DETECTED:", changes_detected)
        print("REGIONS:", len(regions))
        print("TOTAL:", f"{total_time:.2f}s")
        print("=" * 70)

        return JSONResponse(
            status_code=200,
            content=result,
            headers={
                "X-GeoChat-Model": "GeoChat-7B",
                "X-GeoChat-Request-ID": request_id,
                "X-GeoChat-Changed-Area": str(round(changed_area_percent, 2)),
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        print()
        print("=" * 70)
        print("TWO-IMAGE COMPARISON ERROR")
        print("=" * 70)
        print("Request ID:", request_id)
        print("Type:", type(exc).__name__)
        print("Message:", str(exc))
        traceback.print_exc()
        print("=" * 70)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


# ============================================================
# VISUAL COMPARISON ENDPOINT
# ============================================================

@app.post("/compare/visual")
async def compare_visual_endpoint(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    verify_api_key(x_api_key)
    try:
        if not image1.content_type or not image1.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="image1 must be an image.")
        if not image2.content_type or not image2.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="image2 must be an image.")

        before = validate_image(await image1.read())
        after = validate_image(await image2.read())

        before_resized, after_resized, mask, changed_area_percent, regions = calculate_difference(
            before, after,
        )
        output = create_highlighted_image(before_resized, after_resized, mask)

        buffer = io.BytesIO()
        output.save(buffer, format="PNG", optimize=True)
        buffer.seek(0)

        return Response(
            content=buffer.getvalue(),
            media_type="image/png",
            headers={
                "X-Changed-Area-Percent": str(round(changed_area_percent, 2)),
                "X-Detected-Regions": str(len(regions)),
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print()
    print("=" * 70)
    print("UNHANDLED GEOCHAT ERROR")
    print("=" * 70)
    print("Path:", request.url.path)
    print("Method:", request.method)
    print("Type:", type(exc).__name__)
    print("Message:", str(exc))
    traceback.print_exc()
    print("=" * 70)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error.",
            "type": type(exc).__name__,
        },
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print()
    print("=" * 70)
    print("STARTING GEOCHAT API v12.0")
    print("=" * 70)
    print("Python file:", os.path.abspath(__file__))
    print("Host:", HOST)
    print("Port:", PORT)
    print("Model:", MODEL_PATH)
    print("GPU:", GPU_NAME)
    print("Primary /chat output:", "answer")
    print("Primary /compare output:", "answer")
    print("Health: /health")
    print("Swagger: /docs")
    print("External LLM: DISABLED")
    print("History: DISABLED")
    print("Sessions: DISABLED")
    print("Automatic retry: DISABLED")
    print("=" * 70)

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
