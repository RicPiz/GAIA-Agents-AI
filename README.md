# GAIA Benchmark Agent

This repository contains an AI agent designed to solve Level 1 questions from the GAIA (General AI Assistance) benchmark. The agent uses LangChain, LangGraph, and various tools to reason, search, and process information for accurate answers. It is deployed as a Gradio web interface for easy interaction and evaluation.

## Features

- **Agent Architecture**: A LangGraph-based agent that combines reasoning with tool usage (e.g., web search, Wikipedia lookup, Python execution, audio transcription, image description).
- **Tool Integration**: Custom tools for web research, file handling, multimedia processing, and more.
- **Evaluation Runner**: Fetches questions from a remote API, runs the agent, caches results, and submits for scoring.
- **Gradio Interface**: User-friendly UI to run evaluations and submit answers.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/gaia-agent.git
   cd gaia-agent
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```
   - (Optional) Set `SPACE_ID` for Hugging Face Spaces integration.

## Usage

1. Run the Gradio app:
   ```
   python app.py
   ```

2. Open the provided URL in your browser (e.g., http://127.0.0.1:7860).

3. Log in with your Hugging Face account.

4. Click "Run Evaluation (Cache Answers)" to process all questions.

5. Click "Submit Cached Answers" to send results to the scoring API.

The app will display status updates and a table of questions, answers, and tools used.

## Dependencies

See `requirements.txt` for the full list of packages. Key dependencies include:
- `langchain` and `langgraph` for agent orchestration.
- `gradio` for the web interface.
- `openai` and `whisper` for LLM and audio processing.
- `duckduckgo-search`, `wikipedia`, and others for information retrieval.

## Project Structure

- `app.py`: Main Gradio application and evaluation runner.
- `agents.py`: Core agent logic using LangGraph.
- `model.py`: LLM configuration and system prompt.
- `tools.py`: Custom tools for search, file handling, etc.
- `requirements.txt`: List of dependencies.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.