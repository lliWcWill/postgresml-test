import asyncio
from pgml import Database
import logging
from rich.logging import RichHandler
from rich.progress import track
from rich import print
import os
from dotenv import load_dotenv
from tiktoken import encoding_for_model
import openai

# Load environment variables
load_dotenv()
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT")
BASE_PROMPT = os.getenv("BASE_PROMPT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Other imports
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
import discord
from discord.ext import commands
import logging

logging.basicConfig(level=logging.DEBUG)

# Initialize the tokenizer and encoder for GPT-4
enc = encoding_for_model("gpt-4")

INITIAL_TOKEN_LIMIT = 8000
MAX_RESPONSE_TOKENS = 7000


def count_message_tokens(messages):
    """Counts the number of tokens in a list of messages."""
    return sum([len(enc.encode(message["content"])) for message in messages])


async def generate_response(messages, openai_api_key, temperature=0.1):
    """Generates a response using OpenAI and ensures token limits are respected."""

    # Count the tokens of the initial messages
    initial_tokens = count_message_tokens(messages)
    print(f"Initial tokens: {initial_tokens}")

    # Identify and trim or remove the most verbose message if tokens exceed the limit
    while initial_tokens > INITIAL_TOKEN_LIMIT:
        most_verbose_message = max(messages, key=lambda x: count_message_tokens([x]))
        messages.remove(most_verbose_message)
        initial_tokens = count_message_tokens(messages)
        print(
            "Note: Your request is extensive, and some older messages were trimmed to process it. Please provide more specific details or break your request into smaller parts if needed."
        )

    # Calculate available tokens for the completion
    allowed_completion_tokens = 8192 - initial_tokens
    completion_tokens = min(MAX_RESPONSE_TOKENS, allowed_completion_tokens)
    print(f"Available tokens for completion: {completion_tokens}")

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
        print(f"Error while calling OpenAI API: {e}")
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
        f"Total tokens after including OpenAI's response: {total_tokens_after_response}"
    )

    # Check if the total tokens after the response exceed the max limit of 8192
    if total_tokens_after_response > 8192:
        print(
            f"Total tokens {total_tokens_after_response} exceed the model's maximum context length of 8192. Cancelling query."
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


# Define the intents
intents = discord.Intents.default()
intents.messages = True
intents.reactions = True
intents.guilds = True

# Initialize the bot with the defined intents and a command prefix (e.g., '!')
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.command(name="kb_help")
async def _kb_help(ctx):
    help_embed = discord.Embed(
        title="Bot Helper", description="List of available commands:", color=0x42F2F5
    )
    help_embed.add_field(
        name="!kb_help", value="Displays this help message.", inline=False
    )
    help_embed.add_field(
        name="!feedback",
        value="Provide feedback on the bot's performance.",
        inline=False,
    )
    await ctx.send(embed=help_embed)


# System prompt placeholder
system_prompt = "How can I assist you today?"

# Load OpenAI API key from an environment variable or secret management service
openai_api_key = os.environ.get("OPENAI_API_KEY")


@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")
    await bot.change_presence(activity=discord.Game(name="Mention me with a question!"))


@bot.command(name="feedback")
async def feedback(ctx, *, feedback_message: str):
    # Log feedback for further analysis and improvements
    print(f"Feedback from {ctx.author}: {feedback_message}")
    await ctx.send("Thank you for your feedback!")


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
                [{"role": "user", "content": query}], openai_api_key, temperature=0.2
            )
            await message.channel.send(response)
        except Exception as e:
            await message.channel.send(
                "Sorry, I encountered an error. Please try again later."
            )

    # This line is necessary for processing commands
    await bot.process_commands(message)


# Placeholder function for getting the prompt from user input (you'd replace this with your actual function)
async def get_prompt(user_input):
    return user_input


# Your existing generate_response function and other utility functions go here

# Run the bot
bot.run(os.environ.get("DISCORD_BOT_TOKEN"))


def handler(signum, frame):
    print("Exiting...")
    exit(0)


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
collection_name = args.collection_name
stage = args.stage
chat_interface = args.chat_interface

# The above code is retrieving environment variables and assigning their values to various variables.
database_url = os.environ.get("DATABASE_URL")
db = Database(database_url)
splitter = os.environ.get("SPLITTER", "recursive_character")
splitter_params = os.environ.get(
    "SPLITTER_PARAMS", {"chunk_size": 1500, "chunk_overlap": 40}
)
model = os.environ.get("MODEL", "intfloat/e5-small")
# fmt: off
model_params = ast.literal_eval(os.environ.get("MODEL_PARAMS", '{}'))
query_params = ast.literal_eval(os.environ.get("QUERY_PARAMS", '{}'))
# fmt: on

system_prompt = os.environ.get("SYSTEM_PROMPT")
base_prompt = os.environ.get("BASE_PROMPT")
openai_api_key = os.environ.get("OPENAI_API_KEY")


async def upsert_documents(db: Database, collection_name: str, folder: str) -> int:
    log.info("Scanning " + folder + " for markdown files")
    md_files = []
    # root_dir needs a trailing slash (i.e. /root/dir/)
    for filename in glob.iglob(folder + "/**/*.md", recursive=True):
        md_files.append(filename)

    log.info("Found " + str(len(md_files)) + " markdown files")
    documents = []
    for md_file in track(md_files, description="Extracting text from markdown"):
        with open(md_file, "r") as f:
            documents.append({"text": f.read(), "filename": md_file})

    log.info("Upserting documents into database")
    collection = await db.create_or_get_collection(collection_name)
    await collection.upsert_documents(documents)

    return len(md_files)


async def generate_chunks(
    db: Database,
    collection_name: str,
    splitter: str = "recursive_character",
    splitter_params: dict = {"chunk_size": 1500, "chunk_overlap": 40},
) -> int:
    """
    The function `generate_chunks` generates chunks for a given collection in a database and returns the
    count of chunks created.

    :param db: The `db` parameter is an instance of a database connection or client. It is used to
    interact with the database and perform operations such as creating collections, executing queries,
    and fetching results
    :type db: Database
    :param collection_name: The `collection_name` parameter is a string that represents the name of the
    collection in the database. It is used to create or get the collection and perform operations on it
    :type collection_name: str
    :return: The function `generate_chunks` returns an integer, which represents the count of chunks
    generated in the specified collection.
    """
    log.info("Generating chunks")
    collection = await db.create_or_get_collection(collection_name)
    splitter_id = await collection.register_text_splitter(splitter, splitter_params)
    query_string = """SELECT count(*) from {collection_name}.chunks""".format(
        collection_name=collection_name
    )
    results = await db.query(query_string).fetch_all()
    start_chunks = results[0]["count"]
    log.info("Starting chunk count: " + str(start_chunks))
    await collection.generate_chunks(splitter_id)
    results = await db.query(query_string).fetch_all()
    log.info("Ending chunk count: " + str(results[0]["count"]))
    return results[0]["count"] - start_chunks


async def generate_embeddings(
    db: Database,
    collection_name: str,
    splitter: str = "recursive_character",
    splitter_params: dict = {"chunk_size": 1500, "chunk_overlap": 40},
    model: str = "intfloat/e5-small",
    model_params: dict = {},
) -> int:
    """
    The `generate_embeddings` function generates embeddings for text data using a specified model and
    splitter.

    :param db: The `db` parameter is an instance of a database object. It is used to interact with the
    database and perform operations such as creating or getting a collection, registering a text
    splitter, registering a model, and generating embeddings
    :type db: Database
    :param collection_name: The `collection_name` parameter is a string that represents the name of the
    collection in the database where the embeddings will be generated
    :type collection_name: str
    :param splitter: The `splitter` parameter is used to specify the text splitting method to be used
    during the embedding generation process. In this case, the value is set to "recursive_character",
    which suggests that the text will be split into chunks based on recursive character splitting,
    defaults to recursive_character
    :type splitter: str (optional)
    :param splitter_params: The `splitter_params` parameter is a dictionary that contains the parameters
    for the text splitter. In this case, the `splitter_params` dictionary has two keys:
    :type splitter_params: dict
    :param model: The `model` parameter is the name or identifier of the language model that will be
    used to generate the embeddings. In this case, the model is specified as "intfloat/e5-small",
    defaults to intfloat/e5-small
    :type model: str (optional)
    :param model_params: The `model_params` parameter is a dictionary that allows you to specify
    additional parameters for the model. These parameters can be used to customize the behavior of the
    model during the embedding generation process. The specific parameters that can be included in the
    `model_params` dictionary will depend on the specific model you are
    :type model_params: dict
    :return: an integer value of 0.
    """
    log.info("Generating embeddings")
    collection = await db.create_or_get_collection(collection_name)
    splitter_id = await collection.register_text_splitter(splitter, splitter_params)
    model_id = await collection.register_model("embedding", model, model_params)
    log.info("Splitter ID: " + str(splitter_id))
    start = time()
    await collection.generate_embeddings(model_id, splitter_id)
    log.info("Embeddings generated in %0.3f seconds" % (time() - start))

    return 0


async def ingest_documents(
    db: Database,
    collection_name: str,
    folder: str,
    splitter: str,
    splitter_params: dict,
    model: str,
    model_params: dict,
):
    total_docs = await upsert_documents(db, collection_name, folder=folder)
    total_chunks = await generate_chunks(
        db, collection_name, splitter=splitter, splitter_params=splitter_params
    )
    log.info(
        "Total documents: " + str(total_docs) + " Total chunks: " + str(total_chunks)
    )

    await generate_embeddings(
        db,
        collection_name,
        splitter=splitter,
        splitter_params=splitter_params,
        model=model,
        model_params=model_params,
    )


async def get_prompt(user_input: str = ""):
    collection = await db.create_or_get_collection(collection_name)
    model_id = await collection.register_model("embedding", model, model_params)
    splitter_id = await collection.register_text_splitter(splitter, splitter_params)

    log.info(f"Model id: {model_id} | Splitter id: {splitter_id}")

    # Dynamic top_k
    context = ""
    top_k = 1
    max_context_length = 8100  # For example, you might want to adjust this value
    while (
        len(context) < max_context_length and top_k < 7
    ):  # Limit top_k to avoid excessive database calls
        vector_results = await collection.vector_search(
            user_input,
            model_id=model_id,
            splitter_id=splitter_id,
            top_k=top_k,
            query_params=query_params,
        )
        context = "".join([result[1] + "\\n" for result in vector_results])
        top_k += 1

    # Placeholder for refining vector search
    # TODO: Adjust embeddings, model, or other parameters as needed to improve relevance

    # Enhanced Logging
    log.info(f"Context length: {len(context)} | Records fetched: {top_k - 1}")

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
            messages = [{"role": "system", "content": system_prompt}]
            user_input = message["text"]

            query = await get_prompt(user_input)
            messages.append({"role": "user", "content": query})
            response = await generate_response(
                messages, openai_api_key, max_tokens=8100, temperature=1.0
            )
            user = message["user"]

            await say(text=f"<@{user}> {response}")

        socket_handler = AsyncSocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
        await socket_handler.start_async()
    else:
        log.error(
            "SLACK_BOT_TOKEN and SLACK_APP_TOKEN environment variables are not found. Exiting..."
        )


async def run():
    """
    The `main` function connects to a database, ingests documents from a specified folder, generates
    chunks, and logs the total number of documents and chunks.
    """
    print("Starting pgml-chat.... ")

    if stage == "ingest":
        root_dir = args.root_dir
        await ingest_documents(
            db,
            collection_name,
            root_dir,
            splitter,
            splitter_params,
            model,
            model_params,
        )

    elif stage == "chat":
        if chat_interface == "cli":
            await chat_cli()
        elif chat_interface == "slack":
            await chat_slack()


def main():
    if (
        stage == "chat"
        and chat_interface == "discord"
        and os.environ.get("DISCORD_BOT_TOKEN")
    ):
        client.run(os.environ["DISCORD_BOT_TOKEN"])
    else:
        asyncio.run(run())
