# PostgresML Chatbot Builder

A command line tool to build and deploy a **_knowledge-based_** chatbot using PostgresML and OpenAI API.

There are two stages in building a knowledge-based chatbot:

- Build a knowledge base by ingesting documents, chunking documents, generating embeddings, and indexing these embeddings for fast query.
- Generate responses to user queries by retrieving relevant documents and generating responses using OpenAI API.

This tool automates the above two stages and provides a command line interface to build and deploy a knowledge-based chatbot.

## Prerequisites

Before you begin, ensure you have the following:

- PostgresML Database: Sign up for a free [GPU-powered database](https://postgresml.org/signup).
- Python version >=3.8.
- OpenAI API key.

## Getting Started

1. **Setup**:
    1. Create a virtual environment and install `pgml-chat` using `pip`:
        ```bash
        pip install pgml-chat
        ```

    2. Download `.env.template` file from PostgresML Github repository:
        ```bash
        wget https://github.com/postgresml/postgresml/blob/master/pgml-apps/pgml-chat/.env.template
        ```

    3. Copy the template file to `.env`.

    4. Update environment variables with your OpenAI API key and PostgresML database credentials.

        ```bash
        OPENAI_API_KEY=<OPENAI_API_KEY>
        DATABASE_URL=<POSTGRES_DATABASE_URL starts with postgres://>
        MODEL=hkunlp/instructor-xl
        MODEL_PARAMS={"instruction": "Represent the Wikipedia document for retrieval: "}
        QUERY_PARAMS={"instruction": "Represent the Wikipedia question for retrieving supporting documents: "}
        SYSTEM_PROMPT="You are an assistant to answer questions about an open-source software named PostgresML. Your name is PgBot. You are based out of San Francisco, California."
        BASE_PROMPT="Given relevant parts of a document and a question, create a final answer.\ 
                    Include a SQL query in the answer wherever possible.                     Use the following portion of a long document to see if any of the text is relevant to answer the question.                    
Return any relevant text verbatim.
{context}
Question: {question}
                     If the context is empty, then ask for clarification and suggest the user send an email to team@postgresml.org or join PostgresML [Discord](https://discord.gg/DmyJP3qJ7U)."
        ```

2. **Usage**:
    - You can get help on the command line interface by running:
        ```bash
        (pgml-bot-builder-py3.9) pgml-chat % pgml-chat --help
        ```

    - **Ingest**:
        - In this step, we ingest documents, chunk documents, generate embeddings, and index these embeddings for fast query.
            ```bash
            LOG_LEVEL=DEBUG pgml-chat --root_dir <directory> --collection_name <collection_name> --stage ingest
            ```

    - **Chat**:
        - You can interact with the bot using the command line interface, Slack, or Discord.

        - **Command Line Interface**:
            - In this step, we start chatting with the chatbot at the command line. CLI is the default chat interface.
                ```bash
                LOG_LEVEL=ERROR pgml-chat --collection_name <collection_name> --stage chat --chat_interface cli
                ```

        - **Slack**:
            - **Setup**:
                - You need SLACK_BOT_TOKEN and SLACK_APP_TOKEN to run the chatbot on Slack. Include the following environment variables in your `.env` file:
                    ```bash
                    SLACK_BOT_TOKEN=<SLACK_BOT_TOKEN>
                    SLACK_APP_TOKEN=<SLACK_APP_TOKEN>
                    ```

                - Start the chatbot on Slack:
                    ```bash
                    LOG_LEVEL=ERROR pgml-chat --collection_name <collection_name> --stage chat --chat_interface slack
                    ```

        - **Discord**:
            - **Setup**:
                - You need DISCORD_BOT_TOKEN to run the chatbot on Discord. Include the following environment variable in your `.env` file:
                    ```bash
                    DISCORD_BOT_TOKEN=<DISCORD_BOT_TOKEN>
                    ```

                - Start the chatbot on Discord:
                    ```bash
                    pgml-chat --collection_name <collection_name> --stage chat --chat_interface discord
                    ```

3. **Developer Guide**:

    - Clone this repository, start a poetry shell, and install dependencies:
        ```bash
        git clone https://github.com/postgresml/postgresml
        cd postgresml/pgml-apps/pgml-chat
        poetry shell
        poetry install
        pip install .
        ```

    - Create a `.env` file in the root directory of the project and add all the environment variables discussed in [Getting Started](#getting-started) section.
    - All the logic is in `pgml_chat/main.py`.
    - Check the [roadmap](#roadmap) for features that you'd like to work on.
    - For additional features not listed, open an issue, and they will be added to the roadmap.

4. **Options**:
    - Control the chatbot's behavior by setting the following environment variables:
        - `SYSTEM_PROMPT`
        - `BASE_PROMPT`
        - `MODEL`

5. **Roadmap**:
    - Support for file formats like rst, html, pdf, docx, etc.
    - Support for open-source models in addition to OpenAI for chat completion.
    - Support for multi-turn conversations using a conversation buffer.

