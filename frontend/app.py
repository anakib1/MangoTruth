import gradio as gr
import requests
import pandas as pd
from io import BytesIO
from PyPDF2 import PdfReader
import os
import uuid
import json
import mimetypes


def extract_text_from_file(file):
    if file is None:
        return "", None

    extracted_text = ""
    file_bytes = None
    file_type = None

    # Determine the type of 'file'
    if isinstance(file, dict):
        # For certain Gradio versions, 'file' might be a dict
        file_type = file.get("type")
        file_bytes = file.get("data")
    elif isinstance(file, str):
        # 'file' is a file path
        file_path = file
        if not os.path.exists(file_path):
            raise gr.Error("Uploaded file path does not exist.")

        # Guess the file type based on the file extension
        file_type, _ = mimetypes.guess_type(file_path)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
    else:
        # Assume 'file' is a file-like object
        try:
            file_type = file.type
            file_bytes = file.read()
        except AttributeError:
            raise gr.Error("Uploaded file is not in the expected format.")

    try:
        if file_type == "application/pdf":
            reader = PdfReader(BytesIO(file_bytes))
            pages = [page.extract_text() for page in reader.pages if page.extract_text() is not None]
            extracted_text = "\n".join(pages)
            if not extracted_text.strip():
                raise gr.Error("Uploaded PDF contains no extractable text.")
        elif file_type == "text/plain":
            extracted_text = file_bytes.decode("utf-8")
            if not extracted_text.strip():
                raise gr.Error("Uploaded TXT file is empty.")
        else:
            raise gr.Error("Unsupported file type. Please upload a PDF or TXT file.")
    except Exception as e:
        raise gr.Error(f"Error reading file: {e}")

    return extracted_text, file_bytes


def submit_detection(server_url, input_text, file):
    if file is not None:
        input_text, file_bytes = extract_text_from_file(file)
    else:
        file_bytes = None

    if not input_text.strip():
        raise gr.Error("Please enter some text or upload a file.")
    if not server_url.strip():
        raise gr.Error("Please enter a valid server URL.")

    try:
        # Generate a unique requestId
        request_id = str(uuid.uuid4())

        # Prepare the JSON payload
        data = {
            "requestId": request_id,
            "content": input_text
        }

        # Send PUT request to the server
        response = requests.put(server_url, json=data)

        # Check if the request was successful
        if response.status_code == 200:
            try:
                data_response = response.json()
            except json.JSONDecodeError as e:
                raise gr.Error(f"Invalid JSON response from server: {e}")

            # Validate response structure
            if data_response.get("requestId") == request_id:
                status = data_response.get("status")
                return {"Request ID": request_id, "Status": status}
            else:
                raise gr.Error("Mismatch in response requestId.")
        else:
            try:
                error_msg = response.json().get("error", "An error occurred.")
            except json.JSONDecodeError:
                error_msg = response.text
            raise gr.Error(f"Server returned status code {response.status_code}: {error_msg}")

    except requests.exceptions.RequestException as e:
        raise gr.Error(f"An error occurred: {e}")


def fetch_status_wrapper(server_url, request_id):
    if not request_id.strip():
        raise gr.Error("Please enter a valid Request ID.")

    if not server_url.strip():
        raise gr.Error("Please enter a valid server URL.")

    try:
        # Prepare the JSON payload
        data = {
            "requestId": request_id
        }

        # Send GET request to the server
        response = requests.get(server_url, json=data)

        # Check if the request was successful
        if response.status_code == 200:
            try:
                data_response = response.json()
            except json.JSONDecodeError as e:
                raise gr.Error(f"Invalid JSON response from server: {e}")

            # Extract status and verdict
            status = data_response.get("status")
            verdict = data_response.get("verdict")

            # Prepare the JSON output
            full_response = {
                "requestId": data_response.get("requestId"),
                "status": status,
                "verdict": verdict
            }

            # Handle verdict labels
            if verdict and verdict.get("labels") is not None:
                labels = verdict.get("labels", [])
                if isinstance(labels, list) and len(labels) > 0:
                    # Filter out any None or improperly formatted entries
                    valid_labels = [
                        label for label in labels
                        if isinstance(label, dict) and 'label' in label and 'probability' in label
                    ]
                    if valid_labels:
                        df = pd.DataFrame(valid_labels, columns=["Label", "Probability"])
                        verdict_message = ""
                    else:
                        df = pd.DataFrame(columns=["Label", "Probability"])
                        verdict_message = "Verdict is currently empty."
                else:
                    df = pd.DataFrame(columns=["Label", "Probability"])
                    verdict_message = "Verdict is currently empty."
            else:
                df = pd.DataFrame(columns=["Label", "Probability"])
                verdict_message = "Verdict not available yet."

            # Prepare the verdict message
            if verdict_message:
                verdict_message_md = f"**{verdict_message}**"
                verdict_message_visibility = True
            else:
                verdict_message_md = ""
                verdict_message_visibility = False

            # Return all outputs
            return full_response, df, gr.update(value=verdict_message_md, visible=verdict_message_visibility)

        elif response.status_code == 400:
            try:
                error_msg = response.json().get("error", "Bad Request")
            except json.JSONDecodeError:
                error_msg = "Bad Request"
            raise gr.Error(f"Bad Request: {error_msg}")
        elif response.status_code == 404:
            try:
                error_msg = response.json().get("error", "Request not found")
            except json.JSONDecodeError:
                error_msg = "Request not found"
            raise gr.Error(f"Not Found: {error_msg}")
        else:
            try:
                error_msg = response.json().get("error", "An error occurred.")
            except json.JSONDecodeError:
                error_msg = response.text
            raise gr.Error(f"Server returned status code {response.status_code}: {error_msg}")

    except requests.exceptions.RequestException as e:
        raise gr.Error(f"An error occurred: {e}")


if __name__ == '__main__':
    # Define the Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# Text Detection Application")

        with gr.Tab("Submit Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    submit_server_url = gr.Textbox(
                        label="Server URL",
                        value=os.getenv("SERVER_URL", "http://localhost:8080/api/v1/detection"),
                        placeholder="Enter the REST API server URL...",
                    )

            input_method_submit = gr.Radio(
                choices=["Paste Text", "Upload File"],
                value="Paste Text",
                label="Input Method",
                interactive=True
            )

            with gr.Row():
                with gr.Column(scale=1):
                    submit_input_text = gr.Textbox(
                        label="Input Text",
                        lines=15,
                        placeholder="Enter or paste your document text here...",
                        visible=True
                    )
                    submit_file_upload = gr.File(
                        label="Upload File",
                        file_types=[".pdf", ".txt"],
                        visible=False
                    )


            def toggle_input_submit(input_method):
                if input_method == "Paste Text":
                    return gr.update(visible=True), gr.update(visible=False)
                elif input_method == "Upload File":
                    return gr.update(visible=False), gr.update(visible=True)


            input_method_submit.change(
                toggle_input_submit,
                inputs=[input_method_submit],
                outputs=[submit_input_text, submit_file_upload]
            )

            submit_btn = gr.Button("Submit Detection")

            submit_output = gr.JSON(label="Submission Result")

            submit_btn.click(
                submit_detection,
                inputs=[submit_server_url, submit_input_text, submit_file_upload],
                outputs=[submit_output],
                show_progress=True
            )

        with gr.Tab("Fetch Status"):
            with gr.Row():
                with gr.Column(scale=1):
                    fetch_server_url = gr.Textbox(
                        label="Server URL",
                        value=os.getenv("SERVER_URL", "http://localhost:8080/api/v1/detection"),
                        placeholder="Enter the REST API server URL...",
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    fetch_request_id = gr.Textbox(
                        label="Request ID",
                        placeholder="Enter the Request ID...",
                    )

            fetch_btn = gr.Button("Fetch Status")

            # Define output components
            fetch_status_output = gr.JSON(label="Detection Status")
            fetch_verdict_output = gr.DataFrame(headers=["Label", "Probability"], label="Detection Verdict", visible=True)
            fetch_verdict_message = gr.Markdown("", visible=False)

            fetch_btn.click(
                fetch_status_wrapper,
                inputs=[fetch_server_url, fetch_request_id],
                outputs=[fetch_status_output, fetch_verdict_output, fetch_verdict_message],
                show_progress=True
            )

        # Launch the app
        demo.launch()
