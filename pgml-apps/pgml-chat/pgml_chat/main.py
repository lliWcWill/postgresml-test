import asyncio
from pgml import (
    Collection,
    Model,
    Splitter,
    Pipeline,
    migrate,
    init_logger,
    Builtins,
    OpenSourceAI,
)
import logging
import threading
from rich.logging import RichHandler
from rich.progress import track
from rich import print
import os
from dotenv import load_dotenv
import glob
import argparse
from time import time
from openai import OpenAI
import signal
from uuid import uuid4
import pendulum
from colorama import Fore, Style, init
from datetime import datetime
import readchar
import requests
import sounddevice as sd
import speech_recognition as sr
import wavio

import ast
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
import requests

import discord


def handler(signum, frame):
    print("Exiting...")
    exit(0)


init(autoreset=True)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Global variable to control the recording state
is_recording = False

# Constants for recording
WAVE_OUTPUT_FILENAME = "temp_audio.wav"
CHANNELS = 1
SAMPLE_WIDTH = 2
RATE = 16000
FORMAT = "int16"

signal.signal(signal.SIGINT, handler)

parser = argparse.ArgumentParser(
    description="PostgresML Chatbot Builder",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--collection_name",
    dest="collection_name",
    type=str,
    help="Name of the collection (schema) to store the data in PostgresML database",
    required=True,
)
parser.add_argument(
    "--root_dir",
    dest="root_dir",
    type=str,
    help="Input folder to scan for markdown files. Required for ingest stage. Not required for chat stage",
)
parser.add_argument(
    "--stage",
    dest="stage",
    choices=["ingest", "chat"],
    type=str,
    default="chat",
    help="Stage to run",
)
parser.add_argument(
    "--chat_interface",
    dest="chat_interface",
    choices=["cli", "slack", "discord"],
    type=str,
    default="cli",
    help="Chat interface to use",
)

parser.add_argument(
    "--chat_history",
    dest="chat_history",
    type=int,
    default=0,
    help="Number of messages from history used for generating response",
)

parser.add_argument(
    "--bot_name",
    dest="bot_name",
    type=str,
    default="FDBot",
    help="Name of the bot",
)

parser.add_argument(
    "--bot_language",
    dest="bot_language",
    type=str,
    default="English",
    help="Language of the bot",
)
parser.add_argument(
    "-v", "--voice", action="store_true", help="Enable voice input mode"
)
parser.add_argument(
    "--bot_topic",
    dest="bot_topic",
    type=str,
    default="Technology",
    help="Topic of the bot",
)
parser.add_argument(
    "--bot_topic_primary_language",
    dest="bot_topic_primary_language",
    type=str,
    default="SQL",
    help="Primary programming language of the topic",
)

parser.add_argument(
    "--bot_persona",
    dest="bot_persona",
    type=str,
    default="Research Assistant",
    help="Persona of the bot",
)

parser.add_argument(
    "--chat_completion_model",
    dest="chat_completion_model",
    type=str,
    default="gpt-4-1106-preview",
)

parser.add_argument(
    "--max_tokens",
    dest="max_tokens",
    type=int,
    default=2789,
    help="Maximum number of tokens to generate",
)

parser.add_argument(
    "--temperature",
    dest="temperature",
    type=float,
    default=0.7,
    help="Temperature for generating response",
)

parser.add_argument(
    "--top_p",
    dest="top_p",
    type=float,
    default=0.9,
    help="Top p for generating response",
)
parser.add_argument(
    "--vector_recall_limit",
    dest="vector_recall_limit",
    type=int,
    default=1,
    help="Maximum number of documents to retrieve from vector recall",
)

args = parser.parse_args()

FORMAT = "%(message)s"
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "ERROR"),
    format="%(asctime)s - %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
log = logging.getLogger("rich")

# Load .env file
load_dotenv(".env")


# The code is using the `argparse` module to parse command line arguments.
chat_history_collection_name = args.collection_name + "_chat_history"
collection = Collection(args.collection_name)
chat_collection = Collection(chat_history_collection_name)
stage = args.stage
chat_interface = args.chat_interface
chat_history = args.chat_history

# Get all bot related environment variables
bot_name = args.bot_name
bot_language = args.bot_language
bot_persona = args.bot_persona
bot_topic = args.bot_topic
bot_topic_primary_language = args.bot_topic_primary_language

# The above code is retrieving environment variables and assigning their values to various variables.
database_url = os.environ.get("DATABASE_URL")
splitter_name = os.environ.get("SPLITTER", "recursive_character")
splitter_params = os.environ.get(
    "SPLITTER_PARAMS", {"chunk_size": 1500, "chunk_overlap": 40}
)

splitter = Splitter(splitter_name, splitter_params)
model_name = "hkunlp/instructor-xl"
model_embedding_instruction = "Represent the %s document for retrieval: " % (bot_topic)
model_params = {"instruction": model_embedding_instruction}

model = Model(model_name, "pgml", model_params)
pipeline = Pipeline(args.collection_name + "_pipeline", model, splitter)
chat_history_pipeline = Pipeline(
    chat_history_collection_name + "_pipeline", model, splitter
)

chat_completion_model = args.chat_completion_model
max_tokens = args.max_tokens
temperature = args.temperature
vector_recall_limit = args.vector_recall_limit

query_params_instruction = (
    "Represent the %s question for retrieving supporting documents: " % (bot_topic)
)
query_params = {"instruction": query_params_instruction}

default_system_prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an anything"""

system_prompt_template = os.environ.get(
    "SYSTEM_PROMPT_TEMPLATE", default_system_prompt_template
)

system_prompt = system_prompt_template.format(
    topic=bot_topic,
    name=bot_name,
    persona=bot_persona,
    language=bot_language,
    response_programming_language=bot_topic_primary_language,
)

base_prompt = """
{conversation_history}
####
Documents
####
{context}
###
User: {question}
###

Helpful Answer:"""


openai_api_key = os.environ.get("OPENAI_API_KEY", "")

system_prompt_document = [
    {
        "text": system_prompt,
        "id": str(uuid4())[:8],
        "interface": chat_interface,
        "role": "system",
        "timestamp": pendulum.now().timestamp(),
    }
]


def voice_to_text():
    """
    Transcribes voice from recorded audio data.
    """
    global audio_frames
    # Save the recorded audio data to a WAV file
    wavio.write(WAVE_OUTPUT_FILENAME, audio_frames, RATE, sampwidth=SAMPLE_WIDTH)

    # Transcribe the saved audio file
    with open(WAVE_OUTPUT_FILENAME, "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
        return transcript.text


# Global variable to store audio frames
audio_frames = []

import subprocess


def create_session_folder():
    session_folder = f"Collections/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(session_folder, exist_ok=True)
    return session_folder


def save_transcription(folder, original, translated):
    with open(os.path.join(folder, "transcriptions.txt"), "a") as file:
        file.write(f"Original: {original}\nTranslated: {translated}\n\n")


def play_audio(audio_content):
    """
    Play the given audio content using ffplay.

    Args:
        audio_content (bytes): The audio content to be played.

    Raises:
        Exception: If there is an error playing the audio with ffplay.

    Returns:
        None
    """
    try:
        # Start a subprocess that runs ffplay
        ffplay_proc = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        # Write the audio content to ffplay's stdin
        ffplay_proc.stdin.write(audio_content)
        ffplay_proc.stdin.flush()

        # Close the stdin and wait for ffplay to finish playing the audio
        ffplay_proc.stdin.close()
        ffplay_proc.wait()
    except Exception as e:
        logger.error(Fore.RED + f"Error playing audio with ffplay: {e}\n")


def voice_stream(input_text, chosen_voice):
    """
    Generate the voice stream for the given input text and chosen voice.

    Parameters:
        input_text (str): The text to be converted into speech.
        chosen_voice (str): The voice to be used for the speech conversion.

    Returns:
        None
    """
    try:
        response = client.audio.speech.create(
            model="tts-1", voice=chosen_voice, input=input_text
        )

        # Play the audio
        play_audio(response.content)  # Implement play_audio to play the actual audio
    except Exception as e:
        logger.error(Fore.RED + f"Failed to speak text: {e}\n")


# Record Audio Function with Duration Parameter
def record_audio(duration=20):  # Default duration set to 20 seconds
    """
    Records audio for a specified duration and saves it as a WAV file.
    ...
    """
    filename = f"audio_record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    print(Fore.GREEN + f"\nRecording for {duration} seconds...\n" + Style.RESET_ALL)
    audio_data = sd.rec(
        int(duration * RATE), samplerate=RATE, channels=CHANNELS, dtype=FORMAT
    )
    sd.wait()  # Wait until the recording is finished
    wavio.write(filename, audio_data, RATE, sampwidth=SAMPLE_WIDTH)
    logger.info(Fore.GREEN + f"Confirmed Audio Saved!\n")
    return filename


# Transcribe Audio Function with Corrected Handling
def transcribe_audio(audio_file_path):
    """
    Transcribes an audio file using the specified audio file path.

    Parameters:
        audio_file_path (str): The path to the audio file.

    Returns:
        str: The transcription text if successful, None otherwise.
    """
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                prompt="Please focus solely on transcribing the content of this audio. Do not translate. Maintain the original language and context as accurately as possible.",
            )

            logger.info(f"Full API Response: {response}\n")

            # Check for transcription text
            if hasattr(response, "text") and response.text:
                return response.text
            else:
                logger.error(Fore.RED + "No transcription data found in the response\n")
                return None
    except Exception as e:
        logger.error(Fore.RED + f"Transcription failed due to an error: {e}\n")
        return None


def get_model_type(chat_completion_model: str):
    # Default model type to 'openai'
    model_type = "openai"

    try:
        client = OpenAI(api_key=openai_api_key)
        models = client.models.list()
        # Check if the specified model is in the list of OpenAI models
        if not any(model.id == chat_completion_model for model in models):
            model_type = "opensourceai"
    except Exception as e:
        log.debug(e)
        # If there is an exception, assume 'opensourceai'
        model_type = "opensourceai"

    log.info("Setting model type to " + model_type)
    return model_type


async def upsert_documents(folder: str) -> int:
    log.info("Scanning " + folder + " for markdown files")
    md_files = []
    # root_dir needs a trailing slash (i.e. /root/dir/)
    for filename in glob.iglob(folder + "/**/*.md", recursive=True):
        md_files.append(filename)

    log.info("Found " + str(len(md_files)) + " markdown files")
    documents = []
    for md_file in track(md_files, description="Extracting text from markdown"):
        with open(md_file, "r", encoding="utf-8", errors="replace") as f:
            documents.append({"text": f.read(), "id": md_file})

    log.info("Upserting documents into database")
    await collection.upsert_documents(documents)

    return len(md_files)


async def generate_chat_response(
    user_input,
    system_prompt,
    openai_api_key,
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=0.4,
    user_name="",
):
    messages = []
    messages.append({"role": "system", "content": system_prompt})

    chat_history_messages = await chat_collection.get_documents(
        {
            "limit": chat_history * 2,
            "order_by": {"timestamp": "desc"},
            "filter": {
                "metadata": {
                    "$and": [
                        {
                            "$or": [
                                {"role": {"$eq": "assistant"}},
                                {"role": {"$eq": "user"}},
                            ]
                        },
                        {"interface": {"$eq": chat_interface}},
                        {"user_name": {"$eq": user_name}},
                    ]
                }
            },
        }
    )

    # Reverse the order so that user messages are first

    chat_history_messages.reverse()

    conversation_history = ""
    for entry in chat_history_messages:
        document = entry["document"]
        if document["role"] == "user":
            conversation_history += "User: " + document["text"] + "\n"
        if document["role"] == "assistant":
            conversation_history += "Assistant: " + document["text"] + "\n"

    log.info(conversation_history)

    history_documents = []
    user_message_id = str(uuid4())[:8]
    _document = {
        "text": user_input,
        "id": user_message_id,
        "interface": chat_interface,
        "role": "user",
        "timestamp": pendulum.now().timestamp(),
        "user_name": user_name,
    }
    history_documents.append(_document)

    query = await get_prompt(user_input, conversation_history)

    messages.append({"role": "user", "content": query})

    log.info(messages)

    response = await generate_response(
        messages,
        openai_api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    _document = {
        "text": response,
        "id": str(uuid4())[:8],
        "parent_message_id": user_message_id,
        "interface": chat_interface,
        "role": "assistant",
        "timestamp": pendulum.now().timestamp(),
        "user_name": user_name,
    }
    history_documents.append(_document)

    await chat_collection.upsert_documents(history_documents)

    return response


async def generate_response(
    messages, openai_api_key, temperature=temperature, max_tokens=max_tokens, top_p=0.9
):
    model_type = get_model_type(chat_completion_model)
    if model_type == "openai":
        client = OpenAI(api_key=openai_api_key)
        log.debug("Generating response from OpenAI API: " + str(messages))
        response = client.chat.completions.create(
            model=chat_completion_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0,
        )
        output = response.choices[0].message.content
    else:
        client = OpenSourceAI(database_url=database_url)
        log.debug("Generating response from OpenSourceAI API: " + str(messages))
        response = client.chat_completions_create(
            model=chat_completion_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        output = response["choices"][0]["message"]["content"]

    return output


async def ingest_documents(folder: str):
    # Add the pipeline to the collection, does nothing if we have already added it
    await collection.add_pipeline(pipeline)
    await chat_collection.add_pipeline(chat_history_pipeline)
    # This will upsert, chunk, and embed the contents in the folder
    total_docs = await upsert_documents(folder)
    log.info("Total documents: " + str(total_docs))


async def get_prompt(user_input: str = "", conversation_history: str = "") -> str:
    query_input = "In the context of " + bot_topic + ", " + user_input
    vector_results = (
        await collection.query()
        .vector_recall(query_input, pipeline, query_params)
        .limit(vector_recall_limit)
        .fetch_all()
    )
    log.info(vector_results)

    context = ""
    for id, result in enumerate(vector_results):
        if result[0] > 0.6:
            context += "#### \n Document %d: " % (id) + result[1] + "\n"

    if conversation_history:
        conversation_history = "#### \n Conversation History: \n" + conversation_history

    query = base_prompt.format(
        conversation_history=conversation_history,
        context=context,
        question=user_input,
        topic=bot_topic,
        persona=bot_persona,
        language=bot_language,
        response_programming_language=bot_topic_primary_language,
    )

    return query


async def chat_cli():
    user_name = os.environ.get("USER")
    while True:
        try:
            if args.voice:
                print("Voice input mode activated. Press 'r' to record, 'q' to quit.")
                user_input_key = input("Your choice: ")
                if user_input_key.lower() == "r":
                    # Start recording in a separate thread
                    record_thread = threading.Thread(target=record_audio, args=(20,))
                    record_thread.start()
                    record_thread.join()  # Wait for the recording to complete

                    # Transcribe the recorded audio
                    transcription = transcribe_audio(WAVE_OUTPUT_FILENAME)
                    print("Transcribed Text: ", transcription)

                    # Generate chat response using the transcribed text
                    response = await generate_chat_response(
                        transcription,
                        system_prompt,
                        openai_api_key,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=0.9,
                        user_name=user_name,
                    )
                    print(f"{bot_name}: " + response)

                    # Convert the chatbot's response to speech and play it
                    chosen_voice = (
                        "your_preferred_voice"  # Set your preferred voice model
                    )
                    voice_stream(response, chosen_voice)

                elif user_input_key.lower() == "q":
                    print("Exiting voice input mode...")
                    break
                else:
                    print("Invalid input. Press 'r' to record, 'q' to quit.")
            else:
                # Existing text input mode
                user_input = input(f"{bot_name} (Ctrl-C to exit): ")
                response = await generate_chat_response(
                    user_input,
                    system_prompt,
                    openai_api_key,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    user_name=user_name,
                )
                print(f"{bot_name}: " + response)
        except KeyboardInterrupt:
            print("Exiting...")
            break


async def chat_slack():
    if os.environ.get("SLACK_BOT_TOKEN") and os.environ.get("SLACK_APP_TOKEN"):
        app = AsyncApp(token=os.environ.get("SLACK_BOT_TOKEN"))
        response = requests.post(
            "https://slack.com/api/auth.test",
            headers={"Authorization": "Bearer " + os.environ.get("SLACK_BOT_TOKEN")},
        )
        bot_user_id = response.json()["user_id"]

        @app.message(f"<@{bot_user_id}>")
        async def message_hello(message, say):
            print("Message received... ")
            user_input = message["text"]
            user = message["user"]
            response = await generate_chat_response(
                user_input,
                system_prompt,
                openai_api_key,
                max_tokens=max_tokens,
                temperature=temperature,
                user_name=user,
            )

            await say(text=f"<@{user}> {response}")

        socket_handler = AsyncSocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
        await socket_handler.start_async()
    else:
        log.error(
            "SLACK_BOT_TOKEN and SLACK_APP_TOKEN environment variables are not found. Exiting..."
        )


intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    await chat_collection.upsert_documents(system_prompt_document)
    print(f"We have logged in as {client.user}")


@client.event
async def on_message(message):
    bot_mention = f"<@{client.user.id}>"
    if message.author != client.user and bot_mention in message.content:
        print("Discord response in progress ..")
        user_input = message.content
        response = await generate_chat_response(
            user_input,
            system_prompt,
            openai_api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            user_name=message.author.name,
        )
        await message.channel.send(response)


async def run():
    """
    The `main` function connects to a database, ingests documents from a specified folder, generates
    chunks, and logs the total number of documents and chunks.
    """
    log.info("Starting pgml-chat.... ")
    await chat_collection.upsert_documents(system_prompt_document)
    # await migrate()
    if stage == "ingest":
        root_dir = args.root_dir
        await ingest_documents(root_dir)

    elif stage == "chat":
        if chat_interface == "cli":
            await chat_cli()
        elif chat_interface == "slack":
            await chat_slack()


def main():
    init_logger()
    if (
        stage == "chat"
        and chat_interface == "discord"
        and os.environ.get("DISCORD_BOT_TOKEN")
    ):
        client.run(os.environ["DISCORD_BOT_TOKEN"])
    else:
        asyncio.run(run())


main()
