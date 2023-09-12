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
import time
import signal
import readline
from halo import Halo
import ast
import requests


# At the top, after imports and before class and function definitions
TEMPERATURE = 0.2  # Default value
TOP_P = 0.5  # Default value


# Load environment variables
load_dotenv()
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT")
BASE_PROMPT = os.getenv("BASE_PROMPT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OA_model_name = "gpt-4-0314"
TOTAL_TOKENS_USED = 0


# # logging.basicConfig(level=logging.DEBUG)  # Comment out or remove this line
class NewlineFilter(logging.Filter):
    def filter(self, record):
        record.msg = record.msg.replace("\\n\\", "\n")
        return True


# Create a logger and add the filter
logger = logging.getLogger()
logger.addFilter(NewlineFilter())

# Set up the logging configuration
FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level to DEBUG
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        RichHandler(rich_tracebacks=True)
    ],  # Use RichHandler for rich, colored output
)

log = logging.getLogger("rich")  # Get the logger named 'rich'

# System prompt placeholder
system_prompt = "How can I assist you today?"

# Load OpenAI API key from an environment variable or secret management service
openai_api_key = os.environ.get("OPENAI_API_KEY")


# Placeholder function for getting the prompt from user input (you'd replace this with your actual function)
async def get_prompt(user_input):
    return user_input


def handler(signum, frame):
    global TOTAL_TOKENS_USED
    print("Exiting...")
    print(f"Total tokens used in this session: {TOTAL_TOKENS_USED}")
    price_per_token = 0.004 / 1000  # Price per token in dollars
    total_price = TOTAL_TOKENS_USED * price_per_token
    print(f"Total price of this session: ${total_price:.2f}")
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
            text = f.read().replace("\x00", "")  # Clean the text data
            documents.append({"text": text, "id": md_file})

    log.info("Upserting documents into database")
    await collection.upsert_documents(documents)

    return len(md_files)


async def ingest_documents(folder: str):
    # Add the pipeline to the collection, does nothing if we have already added it
    await collection.add_pipeline(pipeline)
    # This will upsert, chunk, and embed the contents in the folder
    total_docs = await upsert_documents(folder)
    log.info("Total documents: " + str(total_docs))


# Initialize the tokenizer and encoder for GPT-4
enc = encoding_for_model("gpt-4")

INITIAL_TOKEN_LIMIT = 6000
MAX_RESPONSE_TOKENS = 8000
MAX_TOKEN_LIMIT = 8160


def count_message_tokens(messages):
    """Counts the number of tokens in a list of messages."""
    all_messages_content = [message["content"] for message in messages]
    token_count = len(enc.encode(" ".join(all_messages_content)))
    print(f"[green]Token count: {token_count}[/green]")
    return token_count


async def generate_response(messages, openai_api_key, temperature=None, top_p=None):
    global TEMPERATURE, TOP_P
    """Generates a response using OpenAI and ensures token limits are respected."""
    temperature = temperature or TEMPERATURE
    top_p = top_p or TOP_P

    logging.debug(
        f"generate_response received temperature: {temperature}, top_p: {top_p}"
    )

    # Count the tokens of the initial messages
    initial_tokens = count_message_tokens(messages)
    print(f"[green]Initial tokens: {initial_tokens}[/green]")

    # Identify and trim or remove the most verbose message if tokens exceed the limit
    while initial_tokens > INITIAL_TOKEN_LIMIT:
        most_verbose_message = max(messages, key=lambda x: count_message_tokens([x]))
        most_verbose_message["content"] = trim_text_to_fit_token_limit(
            most_verbose_message["content"], max_tokens=INITIAL_TOKEN_LIMIT
        )
        initial_tokens = count_message_tokens(messages)
        print(
            "[green]Note: Your request is extensive, and some older messages were trimmed to process it. Please provide more specific details or break your request into smaller parts if needed.[/green]"
        )

    # Calculate available tokens for the completion
    allowed_completion_tokens = 15000 - initial_tokens
    completion_tokens = min(MAX_RESPONSE_TOKENS, allowed_completion_tokens)
    print(f"[green]Available tokens for completion: {completion_tokens}[/green]")

    openai.api_key = openai_api_key
    try:
        response = openai.ChatCompletion.create(
            model=OA_model_name,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=completion_tokens,
            top_p=TOP_P,
            frequency_penalty=0,
            presence_penalty=0,
        )

    except openai.error.APIError as e:
        print(f"[red]OpenAI API returned an API Error: {e}[/red]")
        return (
            "An error occurred while processing your request. Please try again later."
        )

    except openai.error.APIConnectionError as e:
        print(f"[red]Failed to connect to OpenAI API: {e}[/red]")
        return "Failed to connect to OpenAI API. Please try again later."

    except openai.error.RateLimitError as e:
        print(f"[red]OpenAI API request exceeded rate limit: {e}[/red]")
        return "OpenAI API request exceeded rate limit. Please try again later."

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

    # Check if the total tokens after the response exceed the max limit of 16000
    if total_tokens_after_response > MAX_TOKEN_LIMIT:
        print(
            f"[red]Total tokens {total_tokens_after_response} exceed the model's maximum context length of 16000. Cancelling query.[/red]"
        )
        return (
            "Token limit exceeded after including OpenAI's response. Query cancelled."
        )

    return response["choices"][0]["message"]["content"], total_tokens_after_response


def trim_text_to_fit_token_limit(text, max_tokens=MAX_TOKEN_LIMIT):
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
    global TEMPERATURE
    global TOTAL_TOKENS_USED
    print("Welcome to IT Solutions Bot! How can I assist you today?")
    while True:
        try:
            user_input = input("User (Ctrl-C to exit): ")

            # This will fetch relevant data from your database based on user input.
            context_from_db = await get_prompt(user_input)

            # Start the spinner before making the request.
            with Halo(text="Processing...", spinner="dots"):
                await asyncio.sleep(
                    5
                )  # simulate some processing delay with stopping asyncio

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
                response, tokens_used = await generate_response(
                    messages, OPENAI_API_KEY, temperature=TEMPERATURE
                )
                TOTAL_TOKENS_USED += tokens_used

                print(f"ITS Assistant: {response}")

        except KeyboardInterrupt:
            print("\nExiting chat...")
            break


import discord
from discord.ext import commands

intents = discord.Intents.default()
intents.messages = True
intents.reactions = True
intents.guilds = True
intents.message_content = True  # Add this line
bot = commands.Bot(command_prefix="!!", intents=intents)


class TokenCounter:
    def __init__(self):
        self.total_tokens_used = 0

    def add_tokens(self, tokens_used):
        self.total_tokens_used += tokens_used


import time


class RateLimit:
    def __init__(self, limit, remaining, reset):
        self.limit = limit
        self.remaining = remaining
        self.reset = reset


ratelimits = {}
global_limit = 50000
global_remaining = global_limit


@bot.event
async def on_ready():
    global global_remaining
    global_remaining = global_limit

    ratelimits = {}


@bot.event
async def on_message(message):
    try:
        response = await make_api_call()

    except Exception as e:
        if e.status == 429:
            ratelimit = ratelimits[response.url]
            wait = ratelimit.reset - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            response = await make_api_call()
        else:
            raise e

    # Update rate limits
    limit = int(response.headers["X-RateLimit-Limit"])
    remaining = int(response.headers["X-RateLimit-Remaining"])
    reset = int(response.headers["X-RateLimit-Reset"])

    ratelimit = RateLimit(limit, remaining, reset)
    ratelimits[response.url] = ratelimit

    global_remaining -= 1

    if ratelimit.remaining < 10:
        await asyncio.sleep(1)

    if global_remaining < 100:
        await asyncio.sleep(1)

    if ratelimit.remaining == 0 or global_remaining == 0:
        await asyncio.sleep(ratelimit.reset - time.time())


@bot.command()
async def temp(ctx, temperature: float = None, top_p: float = None, *, query=None):
    global TEMPERATURE, TOP_P

    # If no arguments provided, return the current values
    if temperature is None and top_p is None and query is None:
        await ctx.send(f"🌡️= {TEMPERATURE}\n🎯 = {TOP_P}")
        return

    # Validate and update temperature
    if temperature is not None:
        if 0 <= temperature <= 1:
            TEMPERATURE = temperature
            await ctx.send(f"🌡️= {TEMPERATURE} ✅")
        else:
            await ctx.send("Invalid temperature. It should be between 0 and 1.")
            return

    # Validate and update top_p
    if top_p is not None:
        if 0 <= top_p <= 1:
            TOP_P = top_p
            await ctx.send(f"🎯= {TOP_P} ✅")
        else:
            await ctx.send("Invalid top_p. It should be between 0 and 1.")
            return

    # If a query is provided, process and send the response
    if query:
        response, tokens_used = await generate_response(
            openai_api_key,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            messages=[{"role": "user", "content": query}],
        )
        await ctx.send("Processing your query...")
        await ctx.send(response)


async def process_user_message(message, is_reply, thread_id):
    global TOTAL_TOKENS_USED
    bot_mention = f"<@{bot.user.id}>"

    if is_reply:
        log.debug("Processing user reply...")
        user_input = message.content.strip()
    else:
        print("Processing user mention...")
        user_input = message.content.replace(bot_mention, "").strip()

    query = await get_prompt(user_input)

    try:
        response, tokens_used = await generate_response(
            bot.conversation_history[thread_id],  # Include the conversation history
            openai_api_key,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        logging.debug("Response generated")
        TOTAL_TOKENS_USED += tokens_used
        logging.debug("Tokens counted")

        # Add the bot's response to the conversation history
        bot.conversation_history[thread_id].append({"role": "ai", "content": response})
        logging.debug("Response added to conversation history")

        logging.debug(f"Generated response: {response}")

        # Split and send the response if it's too long for Discord
        while len(response) > 0:
            # Ensure we don't cut off in the middle of a word
            if len(response) > 2000:
                cut_off_index = response[:2000].rfind(" ")
                chunk = response[:cut_off_index]
                response = response[cut_off_index:].strip()
            else:
                chunk = response
                response = ""

            await message.reply(chunk)

        logging.debug("Response sent")

    except Exception as e:
        logging.exception(f"Error while sending response: {e}")
        await message.reply("Sorry, I encountered an error. Please try again later.")


@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")
    log.info(f"We have logged in as {bot.user}")


@bot.event
async def on_message(message):
    if message.author.bot:
        return
    logging.debug(f"Received a message: {message.content}")

    # Ignore commands (messages starting with the bot's prefix)
    if message.content.startswith("!!temp"):
        await bot.process_commands(message)
        return
    # Exit early to prevent further processing of command messages

    # Create a dictionary to store the conversation history for each thread
    if not hasattr(bot, "conversation_history"):
        bot.conversation_history = {}

    # Get the thread ID
    thread_id = message.reference.message_id if message.reference else message.id

    # Add the message to the conversation history
    if thread_id not in bot.conversation_history:
        bot.conversation_history[thread_id] = []
    bot.conversation_history[thread_id].append(
        {"role": "user", "content": message.content}
    )

    if message.reference and message.reference.resolved.author == bot.user:
        logging.debug("Processing a reply to the bot's message")
        await process_user_message(message, is_reply=True, thread_id=thread_id)

    elif bot.user.mentioned_in(message):
        logging.debug("Processing a mention of the bot")
        await process_user_message(message, is_reply=False, thread_id=thread_id)

    # This line is necessary for processing commands
    await bot.process_commands(message)


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
            await bot.start(os.environ.get("DISCORD_BOT_TOKEN"))


async def main():
    chat_interfaces = {"cli": chat_cli, "discord": run}
    stages = {"ingest": run, "chat": chat_interfaces[args.chat_interface]}
    await stages[args.stage]()


asyncio.run(main())
