import os
import asyncio
import discord
from discord.ext import commands
from pgml import Collection, Model, Splitter, Pipeline
import logging
from rich.logging import RichHandler
from rich.progress import track
from rich import print
from dotenv import load_dotenv
from tiktoken import encoding_for_model
import openai
import glob
import argparse
from time import time
import signal
import readline
from halo import Halo
import ast
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
import requests

# Load environment variables
load_dotenv()
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT")
BASE_PROMPT = os.getenv("BASE_PROMPT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Add your new code here
logging.basicConfig(level=logging.DEBUG)

# Initialize the tokenizer and encoder for GPT-4
enc = encoding_for_model("gpt-4")

INITIAL_TOKEN_LIMIT = 8100
MAX_RESPONSE_TOKENS = 7200


# System prompt placeholder
system_prompt = "How can I assist you today?"

# Load OpenAI API key from an environment variable or secret management service
openai_api_key = os.environ.get("OPENAI_API_KEY")


# Placeholder function for getting the prompt from user input (you'd replace this with your actual function)
async def get_prompt(user_input):
    return user_input


def handler(signum, frame):
    print("Exiting...")
    exit(0)


def count_message_tokens(messages):
    """Counts the number of tokens in a list of messages."""
    token_count = sum([len(enc.encode(message["content"])) for message in messages])
    print(f"[green]Token count: {token_count}[/green]")
    return token_count


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
collection = Collection(args.collection_name)
stage = args.stage
chat_interface = args.chat_interface

# The above code is retrieving environment variables and assigning their values to various variables.
database_url = os.environ.get("DATABASE_URL")
splitter_name = os.environ.get("SPLITTER", "recursive_character")
splitter_params = os.environ.get(
    "SPLITTER_PARAMS", {"chunk_size": 1500, "chunk_overlap": 40}
)
splitter = Splitter(splitter_name, splitter_params)
model_name = os.environ.get("MODEL", "intfloat/e5-small")
model_params = ast.literal_eval(os.environ.get("MODEL_PARAMS", {}))
model = Model(model_name, "pgml", model_params)
pipeline = Pipeline(args.collection_name + "_pipeline", model, splitter)
query_params = ast.literal_eval(os.environ.get("QUERY_PARAMS", {}))
system_prompt = os.environ.get("SYSTEM_PROMPT")
base_prompt = os.environ.get("BASE_PROMPT")
openai_api_key = os.environ.get("OPENAI_API_KEY")


async def upsert_documents(folder: str) -> int:
    log.info("Scanning " + folder + " for markdown files")
    md_files = []
    # root_dir needs a trailing slash (i.e. /root/dir/)
    for filename in glob.iglob(folder + "/**/*.md", recursive=True):
        md_files.append(filename)

    log.info("Found " + str(len(md_files)) + " markdown files")
    documents = []
    for md_file in track(md_files, description="Extracting text from markdown"):
        with open(md_file, "r") as f:
            documents.append({"text": f.read(), "id": md_file})

    log.info("Upserting documents into database")
    await collection.upsert_documents(documents)

    return len(md_files)


async def ingest_documents(folder: str):
    # Add the pipeline to the collection, does nothing if we have already added it
    await collection.add_pipeline(pipeline)
    # This will upsert, chunk, and embed the contents in the folder
    total_docs = await upsert_documents(folder)
    log.info("Total documents: " + str(total_docs))


async def generate_response(messages, openai_api_key, temperature=0.1):
    """Generates a response using OpenAI and ensures token limits are respected."""

    # Count the tokens of the initial messages
    initial_tokens = count_message_tokens(messages)
    print(f"[green]Initial tokens: {initial_tokens}[/green]")

    # Identify and trim or remove the most verbose message if tokens exceed the limit
    while initial_tokens > INITIAL_TOKEN_LIMIT:
        most_verbose_message = max(messages, key=lambda x: count_message_tokens([x]))
        messages.remove(most_verbose_message)
        initial_tokens = count_message_tokens(messages)
        print(
            "[green]Note: Your request is extensive, and some older messages were trimmed to process it. Please provide more specific details or break your request into smaller parts if needed.[/green]"
        )

    # Calculate available tokens for the completion
    allowed_completion_tokens = 8192 - initial_tokens
    completion_tokens = min(MAX_RESPONSE_TOKENS, allowed_completion_tokens)
    print(f"[green]Available tokens for completion: {completion_tokens}[/green]")

    try:
        openai.api_key = openai_api_key
        response = openai.ChatCompletion.create(
            model="gpt-4-0314",
            messages=messages,
            temperature=temperature,
            max_tokens=completion_tokens,
            top_p=0.4,
            frequency_penalty=0,
            presence_penalty=0,
        )
    except Exception as e:
        print(f"[red]Error while calling OpenAI API: {e}[/red]")
        return (
            "An error occurred while processing your request. Please try again later."
        )

    # Add OpenAI's response to the messages list
    messages.append(
        {"role": "ai", "content": response["choices"][0]["message"]["content"]}
    )

    # Count the tokens after adding OpenAI's response
    total_tokens_after_response = count_message_tokens(messages)
    print(
        f"[green]Total tokens after including OpenAI's response: {total_tokens_after_response}[/green]"
    )

    # Check if the total tokens after the response exceed the max limit of 8192
    if total_tokens_after_response > 8192:
        print(
            f"[red]Total tokens {total_tokens_after_response} exceed the model's maximum context length of 8192. Cancelling query.[/red]"
        )
        return (
            "Token limit exceeded after including OpenAI's response. Query cancelled."
        )

    return response["choices"][0]["message"]["content"]


def trim_text_to_fit_token_limit(text, max_tokens=8100):
    """
    Trim the text to fit within the token limit.
    """
    token_count = count_message_tokens(
        [{"content": text}]
    )  # Adjusted to match function's expected input structure
    while token_count > max_tokens:
        # Reduce text length by a percentage and check tokens again
        text = text[: int(len(text) * 0.95)]
        token_count = count_message_tokens([{"content": text}])
    return text


async def get_prompt(user_input: str = ""):
    vector_results = (
        await collection.query()
        .vector_recall(user_input, pipeline, query_params)
        .limit(2)
        .fetch_all()
    )
    log.info(vector_results)
    context = ""

    for result in vector_results:
        context += result[1] + "\n"

    query = base_prompt.format(context=context, question=user_input)

    return query


async def chat_cli():
    print("Welcome to IT Solutions Bot! How can I assist you today?")
    while True:
        try:
            user_input = input("User (Ctrl-C to exit): ")

            # This will fetch relevant data from your database based on user input.
            context_from_db = await get_prompt(user_input)

            # Start the spinner before making the request.
            with Halo(text="Processing...", spinner="dots"):
                # Construct the message to be sent to OpenAI.
                messages = [
                    {
                        "role": "system",
                        "content": "You are consulting the IT Solutions Expert Bot, fine-tuned with a vast knowledge base. This bot is tailored to deliver precise answers for IT-related queries. Accuracy and relevancy are paramount. Ensure you consider the provided context for the best response.",
                    },
                    {
                        "role": "user",
                        "content": context_from_db + "\n\nQuestion: " + user_input,
                    },
                ]

            # Generate response from OpenAI.
            response = await generate_response(
                messages, OPENAI_API_KEY, temperature=0.002
            )

            print(f"ITS Assistant: {response}")

        except KeyboardInterrupt:
            print("\nExiting chat...")
            break


async def run():
    """
    The `main` function connects to a database, ingests documents from a specified folder, generates
    chunks, and logs the total number of documents and chunks.
    """
    print("Starting KnowledgeBased-chat.... ")

    if stage == "ingest":
        root_dir = args.root_dir
        await ingest_documents(root_dir)

    elif stage == "chat":
        if chat_interface == "cli":
            await chat_cli()
        elif chat_interface == "slack":
            # Slack related code
            pass
        elif chat_interface == "discord":
            import discord
            from discord.ext import commands

            intents = discord.Intents.default()
            intents.messages = True
            intents.reactions = True
            intents.guilds = True
            bot = commands.Bot(command_prefix="!", intents=intents)

            @bot.event
            async def on_ready():
                print(f"We have logged in as {bot.user}")
                await bot.change_presence(
                    activity=discord.Game(name="Mention me with a question!")
                )

            @bot.event
            async def on_message(message):
                if message.author.bot:
                    return

                bot_mention = f"<@{bot.user.id}>"

                if bot_mention in message.content:
                    print("Processing user request...")
                    user_input = message.content.replace(bot_mention, "").strip()
                    query = await get_prompt(user_input)

                    try:
                        response = await generate_response(
                            [{"role": "user", "content": query}],
                            openai_api_key,
                            temperature=0.2,
                        )
                        # Check if the response contains the fallback message
                        if (
                            "We are working to connect you with the neccassary resources"
                            in response
                        ):
                            # If it does, remove it
                            response = response.replace("Please Try Again.", "")
                        await message.channel.send(response)
                    except Exception as e:
                        await message.channel.send(
                            "Sorry, I encountered an error. Please try again later."
                        )

                    # This line is necessary for processing commands
                    await bot.process_commands(message)

            # Run the bot
            await bot.start(os.environ.get("DISCORD_BOT_TOKEN"))


def main():
    if args.stage == "ingest":
        asyncio.run(run())
    elif args.stage == "chat":
        if args.chat_interface == "cli":
            asyncio.run(chat_cli())
        elif args.chat_interface == "slack":
            asyncio.run(chat_slack())
        elif args.chat_interface == "discord":
            # Run the bot
            asyncio.run(run())  # Note: using asyncio.run here


main()
