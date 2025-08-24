import os
from dotenv import load_dotenv
load_dotenv()
import gradio as gr
import requests
import pandas as pd
import time

from agents import BasicAgent, TRACE_LOGS

# Constants
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# Cached results for decoupled run/submit flow
CACHED_ANSWERS = []
CACHED_RESULTS_LOG = []
CACHED_USERNAME = None
CACHED_AGENT_CODE = None

def run_and_cache_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, caches all answers,
    and displays the results. Use a separate action to submit.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for idx, item in enumerate(questions_data, 1):
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item {idx} with missing task_id or question: {item}")
            continue
        print(f"Processing question {idx}/{len(questions_data)} (Task ID: {task_id}): {question_text[:100]}...")
        try:
            # Modified call to include task_id
            submitted_answer = agent(question_text, task_id)
            print(f"Finished question {idx}: Answer = {submitted_answer}")
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": submitted_answer,
                "Tools Used": ",".join(TRACE_LOGS[-1].get("tools_used", [])) if TRACE_LOGS else "",
                "Before Clean": TRACE_LOGS[-1].get("answer_before_clean", "") if TRACE_LOGS else "",
                "After Clean": TRACE_LOGS[-1].get("answer_after_clean", "") if TRACE_LOGS else "",
            })
        except Exception as e:
            print(f"Error on question {idx} (Task ID: {task_id}): {e}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Cache results for later submission
    global CACHED_ANSWERS, CACHED_RESULTS_LOG, CACHED_USERNAME, CACHED_AGENT_CODE
    CACHED_ANSWERS = answers_payload
    CACHED_RESULTS_LOG = results_log
    CACHED_USERNAME = username.strip()
    CACHED_AGENT_CODE = agent_code

    status_message = (
        f"Cached {len(answers_payload)} answers for user '{username}'.\n"
        f"Click 'Submit Cached Answers' to submit."
    )
    results_df = pd.DataFrame(results_log)
    return status_message, results_df

def submit_cached_answers(profile: gr.OAuthProfile | None):
    """Submits previously cached answers."""
    # Ensure there is cached data
    if not CACHED_ANSWERS:
        return "No cached answers found. Please run evaluation first.", None

    # Prefer current profile username if provided; otherwise use cached
    if profile:
        username = f"{profile.username}".strip()
    else:
        username = (CACHED_USERNAME or "").strip()
    if not username:
        return "Username not available. Please login before submitting.", None

    api_url = DEFAULT_API_URL
    submit_url = f"{api_url}/submit"

    agent_code = CACHED_AGENT_CODE or f"https://huggingface.co/spaces/{os.getenv('SPACE_ID')}/tree/main"
    submission_data = {"username": username, "agent_code": agent_code, "answers": CACHED_ANSWERS}

    print(f"Submitting {len(CACHED_ANSWERS)} cached answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(CACHED_RESULTS_LOG)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(CACHED_RESULTS_LOG)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(CACHED_RESULTS_LOG)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(CACHED_RESULTS_LOG)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(CACHED_RESULTS_LOG)
        return status_message, results_df

# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---

        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation (Cache Answers)")
    submit_button = gr.Button("Submit Cached Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_cache_all,
        outputs=[status_output, results_table]
    )
    submit_button.click(
        fn=submit_cached_answers,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)
