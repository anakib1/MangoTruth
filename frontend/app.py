import json
import logging

import gradio as gr
import requests
from io import BytesIO
from PyPDF2 import PdfReader
import os
import mimetypes
from docx import Document  # Importing python-docx
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set the server URL from environment or default
SERVER_URL = os.getenv('SERVER_URL', 'http://localhost:8080')


def extract_text_from_file(file):
    """
    Extracts text from uploaded PDF, TXT, or DOCX files.
    """
    logger.info("Starting to extract text from file.")
    if file is None:
        logger.warning("No file provided for extraction.")
        return "", None

    extracted_text = ""
    file_bytes = None
    file_type = None
    file_name = ""

    # Determine the type of 'file'
    if isinstance(file, dict):
        logger.debug("File is a dictionary.")
        file_type = file.get("type")
        file_bytes = file.get("data")
        file_name = file.get("name", "")
    elif isinstance(file, str):
        logger.debug("File is a file path.")
        file_path = file
        if not os.path.exists(file_path):
            logger.error("Uploaded file path does not exist.")
            raise gr.Error("Uploaded file path does not exist.")

        # Guess the file type based on the file extension
        file_type, _ = mimetypes.guess_type(file_path)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        file_name = os.path.basename(file_path)
    else:
        logger.debug("File is a file-like object.")
        try:
            file_type = file.type
            file_bytes = file.read()
            if hasattr(file, 'name'):
                file_name = os.path.basename(file.name)
            else:
                file_name = ""
        except AttributeError:
            logger.error("Uploaded file is not in the expected format.")
            raise gr.Error("Uploaded file is not in the expected format.")

    # Log the filename for debugging
    logger.debug(f"Uploaded file name: '{file_name}'")

    # If file_type is None, attempt to guess based on the file extension
    if file_type is None:
        logger.debug("File type is None. Attempting to guess from file extension.")
        # Sanitize the filename by stripping whitespace
        file_name = file_name.strip()
        _, ext = os.path.splitext(file_name)
        file_type, _ = mimetypes.guess_type(ext)
        logger.debug(f"Guessed file type from extension: {file_type}")

    logger.info(f"File type detected: {file_type}")

    try:
        if file_type == "application/pdf":
            logger.info("Extracting text from PDF.")
            reader = PdfReader(BytesIO(file_bytes))
            pages = [page.extract_text() for page in reader.pages if page.extract_text() is not None]
            extracted_text = "\n".join(pages)
            if not extracted_text.strip():
                logger.error("Uploaded PDF contains no extractable text.")
                raise gr.Error("Uploaded PDF contains no extractable text.")
        elif file_type == "text/plain":
            logger.info("Extracting text from TXT.")
            extracted_text = file_bytes.decode("utf-8")
            if not extracted_text.strip():
                logger.error("Uploaded TXT file is empty.")
                raise gr.Error("Uploaded TXT file is empty.")
        elif file_type in [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword"
        ] or file_name[-5:] == '.docx':
            logger.info("Extracting text from DOCX.")
            with BytesIO(file_bytes) as docx_io:
                try:
                    doc = Document(docx_io)
                    extracted_text = "\n".join([para.text for para in doc.paragraphs])
                    if not extracted_text.strip():
                        logger.error("Uploaded DOCX file contains no extractable text.")
                        raise gr.Error("Uploaded DOCX file contains no extractable text.")
                except Exception as e:
                    logger.error(f"Error reading DOCX file: {e}")
                    raise gr.Error(f"Error reading DOCX file: {e}")
        else:
            logger.error(f"Unsupported file type: {file_type}")
            raise gr.Error("Unsupported file type. Please upload a PDF, TXT, or DOCX file.")
    except Exception as e:
        logger.exception("An error occurred while extracting text from the file.")
        raise gr.Error(f"Error reading file: {e}")

    logger.info("Text extraction successful.")
    return extracted_text, file_bytes




def submit_detection(input_text, file, detector_name):
    """
    Submits a detection request to the server using the selected detector.
    """
    logger.info("Submitting detection request.")
    if file is not None:
        logger.debug("File is provided. Extracting text.")
        input_text, file_bytes = extract_text_from_file(file)
    else:
        file_bytes = None
        logger.debug("No file provided. Using input text.")

    if not input_text.strip():
        logger.error("Input text is empty.")
        raise gr.Error("Please enter some text or upload a file.")

    if not detector_name:
        logger.error("Detector name not selected.")
        raise gr.Error("Please select a detector.")

    try:
        # Prepare the JSON payload with the selected detector
        data = {
            "content": input_text,
            "detector_name": detector_name
        }
        logger.debug(f"Payload for detection request: {data}")

        # Send PUT request to the server
        response = requests.put(f"{SERVER_URL}/api/v1/detection", json=data)
        logger.debug(f"Received response with status code {response.status_code}.")

        # Check if the request was successful
        if response.status_code == 200:
            try:
                data_response = response.json()
                logger.info("Detection request submitted successfully.")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from server: {e}")
                raise gr.Error(f"Invalid JSON response from server: {e}")

            # Return the full JSON response
            return data_response
        else:
            try:
                error_msg = response.json().get("error", "An error occurred.")
                logger.error(f"Server returned error: {error_msg}")
            except json.JSONDecodeError:
                error_msg = response.text
                logger.error(f"Server returned error: {error_msg}")
            raise gr.Error(f"Server returned status code {response.status_code}: {error_msg}")

    except requests.exceptions.RequestException as e:
        logger.exception("An error occurred while submitting detection request.")
        raise gr.Error(f"An error occurred: {e}")


def fetch_mass_status():
    """
    Fetches all detection statuses from the server without user_id.
    """
    logger.info("Fetching mass detection statuses.")
    try:
        # Send GET request to the mass endpoint without user_id
        response = requests.get(f"{SERVER_URL}/api/v1/detection/mass")
        logger.debug(f"Received response with status code {response.status_code}.")

        # Check if the request was successful
        if response.status_code == 200:
            try:
                data_response = response.json()
                logger.info("Fetched detection statuses successfully.")
                return data_response
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from server: {e}")
                raise gr.Error(f"Invalid JSON response from server: {e}")
        else:
            try:
                error_msg = response.json().get("error", "An error occurred.")
                logger.error(f"Server returned error: {error_msg}")
            except json.JSONDecodeError:
                error_msg = response.text
                logger.error(f"Server returned error: {error_msg}")
            raise gr.Error(f"Server returned status code {response.status_code}: {error_msg}")

    except requests.exceptions.RequestException as e:
        logger.exception("An error occurred while fetching mass detection statuses.")
        raise gr.Error(f"An error occurred: {e}")


def fetch_detectors():
    """
    Fetches the list of available detectors from the server.
    """
    logger.info("Fetching detectors from server.")
    try:
        response = requests.get(f"{SERVER_URL}/api/v1/detectors")
        logger.debug(f"Received response with status code {response.status_code}.")

        if response.status_code == 200:
            try:
                response_json = response.json()
                detectors = response_json.get("detectors", [])
                if not isinstance(detectors, list):
                    logger.error("Invalid detectors format received from server.")
                    raise gr.Error("Invalid detectors format received from server.")
                logger.info(f"Detectors fetched successfully: {detectors}")
                return detectors
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from server: {e}")
                raise gr.Error(f"Invalid JSON response from server: {e}")
        else:
            try:
                error_msg = response.json().get("error", "An error occurred while fetching detectors.")
                logger.error(f"Server returned error: {error_msg}")
            except json.JSONDecodeError:
                error_msg = response.text
                logger.error(f"Server returned error: {error_msg}")
            raise gr.Error(f"Server returned status code {response.status_code}: {error_msg}")

    except requests.exceptions.RequestException as e:
        logger.exception("An error occurred while fetching detectors.")
        raise gr.Error(f"An error occurred: {e}")


def process_fetch_response(detection_status_list, selected_request_id):
    """
    Processes the list of DetectionStatus from the mass endpoint.
    Finds the selected DetectionStatus and updates the AI percentage display.
    """
    logger.info(f"Processing fetch response for request ID: {selected_request_id}")

    # Find the selected DetectionStatus
    selected_status = next((item for item in detection_status_list if item['request_id'] == selected_request_id), None)

    if not selected_status:
        logger.warning("Request ID not found.")
        status_message = "Request ID not found."
        ai_percentage_html = ""
        logger.debug(
            f"Returning: status_message={status_message} (type: {type(status_message)}), ai_percentage_html={ai_percentage_html} (type: {type(ai_percentage_html)})")
        return status_message, ai_percentage_html

    # Extract the status
    status = selected_status.get("status", "Unknown")
    logger.debug(f"Status of request ID {selected_request_id}: {status}")

    # Create status message
    status_message = f"**Status:** {status}"

    # Extract labels and probabilities
    labels = selected_status.get("verdict", {}).get("labels")
    logger.debug(f"Labels for request ID {selected_request_id}: {labels}")

    ai_probability = None

    if labels:
        # Extract AI probability
        for label in labels:
            # Adjust the label names as per your server's response
            if label['label'].lower() in ['ai', 'ai-generated', 'generated by ai', 'ai generated']:
                ai_probability = label['probability']
                break
        # If AI label not found, default to None or handle accordingly
        if ai_probability is None:
            logger.warning("AI label not found in labels.")
            ai_probability = 0.0
    else:
        logger.warning("No labels available yet.")

    if ai_probability is not None:
        ai_percentage = ai_probability * 100  # Convert to percentage
        logger.debug(f"AI-generated percentage: {ai_percentage}%")
        # Determine color based on AI percentage
        if ai_percentage >= 50:
            color = "#f97316"  # Orange for high AI-generated content
        else:
            color = "#16a34a"  # Green for low AI-generated content

        # Create HTML structure
        ai_percentage_html = f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
            border: 2px solid {color};
            border-radius: 12px;
            background-color: #ffffff;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            max-width: 400px;
            margin: 20px auto;
            font-family: Arial, sans-serif;
        ">
            <!-- Icon -->
            <div style="
                margin-bottom: 15px;
            ">
                <!-- Replace the src with your preferred icon -->
                <img src="https://img.icons8.com/ios-filled/50/000000/artificial-intelligence.png" alt="AI Detector Icon" width="50" height="50">
            </div>
            <!-- Percentage and Message -->
            <div style="
                text-align: center;
                margin-bottom: 10px;
            ">
                <h2 style="
                    font-size: 32px;
                    margin: 0;
                    color: #374151;
                ">
                    <b style="color: #374151;">{ai_percentage:.2f}%</b> of this text appears to be AI-generated
                </h2>
            </div>
            <!-- Info Button -->
            <div style="
                margin-bottom: 20px;
            ">
                <button onclick="alert('This percentage indicates the likelihood that the text was generated by AI based on our detection algorithms.')" style="
                    background: none;
                    border: none;
                    cursor: pointer;
                ">
                    <!-- Info Icon SVG -->
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M12 2a10 10 0 100 20 10 10 0 000-20z" />
                    </svg>
                </button>
            </div>
        </div>
        """
    else:
        ai_percentage_html = ""
        logger.debug("AI percentage not available.")

    if labels:
        logger.info("AI percentage extracted from labels.")
    else:
        # Append the message to the status
        status_message += "\n\n**No labels available yet.**"
        ai_percentage_html = ""

    # Log the return values
    logger.debug(f"Returning status_message: {status_message} (type: {type(status_message)})")
    logger.debug(f"Returning ai_percentage_html: {ai_percentage_html} (type: {type(ai_percentage_html)})")

    return status_message, ai_percentage_html


def get_detection_requests():
    """
    Retrieves the list of detection requests from the server.
    """
    logger.info("Retrieving detection requests.")
    try:
        detection_status_list = fetch_mass_status()
        # Create display text for dropdown in "ID - STATUS" format
        request_list = [f"{item['request_id']} - {item['status']}" for item in detection_status_list]
        logger.debug(f"Detection requests: {request_list}")
        return request_list, detection_status_list
    except gr.Error as e:
        logger.error(f"Error retrieving detection requests: {e}")
        return [], []


def refresh_detection_list():
    """
    Refreshes the Detection Requests dropdown.
    """
    logger.info("Refreshing detection list.")
    detection_requests, detection_status_list = get_detection_requests()
    if detection_requests:
        # Set the first request as the default selected value
        updated_dropdown = gr.Dropdown(choices=detection_requests, value=detection_requests[0], interactive=True)
        logger.debug("Updated detection_list dropdown with new detection requests.")
        return updated_dropdown, detection_status_list
    else:
        # If no detection requests, set choices to empty and value to None
        updated_dropdown = gr.Dropdown(choices=[], value=None, interactive=True)
        logger.debug("No detection requests available. detection_list dropdown set to empty.")
        return updated_dropdown, detection_status_list


def on_detection_request_selected(detection_status_list, selected_item):
    """
    Handles the event when a detection request is selected from the dropdown.
    """
    logger.info(f"Detection request selected: {selected_item}")
    if not selected_item:
        logger.warning("No detection request selected.")
        status_message = "Please select a valid Request ID."
        ai_percentage_html = ""
        logger.debug(
            f"Returning: status_message={status_message} (type: {type(status_message)}), ai_percentage_html={ai_percentage_html} (type: {type(ai_percentage_html)})")
        return status_message, ai_percentage_html

    # Extract request_id from the selected_item
    request_id = selected_item.split(" - ")[0]
    logger.debug(f"Extracted request_id: {request_id}")

    # Process the selected DetectionStatus
    status_message, ai_percentage_html = process_fetch_response(detection_status_list, request_id)

    logger.debug(
        f"Processed fetch response with: status_message={status_message} (type: {type(status_message)}), ai_percentage_html={'HTML Content' if ai_percentage_html else 'None'} (type: {type(ai_percentage_html)})")

    return status_message, ai_percentage_html


def submit_and_refresh(input_text, file, detector_name):
    """
    Submits a new detection request and refreshes the Detection Requests list.
    """
    logger.info("Submitting and refreshing detection list.")
    try:
        data_response = submit_detection(input_text, file, detector_name)
        request_id = data_response.get('request_id', 'Unknown')
        logger.debug(f"Received request ID: {request_id}")
        message = f"**Request with ID `{request_id}` was submitted successfully.**"
    except gr.Error as e:
        logger.error(f"Error submitting detection request: {e}")
        return f"**Error:** {e}", gr.Dropdown(choices=[], value=None, interactive=True), []

    detection_requests, detection_status_list = get_detection_requests()
    logger.debug(f"Detection requests after submission: {detection_requests}")

    if detection_requests:
        # Set the first request as the default selected value
        updated_dropdown = gr.Dropdown(choices=detection_requests, value=detection_requests[0], interactive=True)
        logger.debug("Updated dropdown with new detection requests.")
        return message, updated_dropdown, detection_status_list
    else:
        # If no detection requests, set choices to empty and value to None
        updated_dropdown = gr.Dropdown(choices=[], value=None, interactive=True)
        logger.debug("No detection requests available. Dropdown set to empty.")
        return message, updated_dropdown, detection_status_list


def initialize_detectors():
    """
    Initializes the detectors list when the app loads.
    """
    logger.info("Initializing detectors.")
    detectors = fetch_detectors()
    if detectors:
        logger.info("Detectors initialized successfully.")
        updated_dropdown = gr.Dropdown(choices=detectors, value=detectors[0], interactive=True)
        return updated_dropdown
    else:
        logger.warning("No detectors available.")
        updated_dropdown = gr.Dropdown(choices=[], value=None, interactive=True)
        return updated_dropdown


def refresh_detectors():
    """
    Refreshes the detectors list when the refresh button is clicked.
    """
    logger.info("Refreshing detectors.")
    detectors = fetch_detectors()
    if detectors:
        logger.info(f"Detectors fetched successfully: {detectors}")
        updated_dropdown = gr.Dropdown(choices=detectors, value=detectors[0], interactive=True)
        logger.debug("Updated detector_selection dropdown with new detectors.")
        return updated_dropdown
    else:
        logger.warning("No detectors available.")
        updated_dropdown = gr.Dropdown(choices=[], value=None, interactive=True)
        logger.debug("No detectors available. detector_selection dropdown set to empty.")
        return updated_dropdown


if __name__ == '__main__':
    # Define the Gradio interface
    with gr.Blocks(css="""
#fetch_table {
    height: 500px;
    overflow-y: auto;
}

#fetch_table table {
    width: 100%;
    border-collapse: collapse;
}

#fetch_table th, #fetch_table td {
    border: 1px solid #ddd;
    padding: 8px;
}

#fetch_table th {
    background-color: #f97316;
    color: black;
    text-align: left;
}
    """) as demo:
        gr.Markdown("# 🕵️‍♂️ Text Detection Application")

        with gr.Row():
            # Left panel
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Detection Requests")
                detection_list = gr.Dropdown(label="Detection Requests", choices=[], interactive=True)
                refresh_btn = gr.Button("🔄 Refresh Requests")
                request_mapping_state = gr.State([])  # Stores the list of DetectionStatus

            # Right panel
            with gr.Column(scale=2):
                gr.Markdown("### 📄 Supported File Types: PDF, TXT, DOCX")

                # Row containing Input Method and Detector Selection aligned side-by-side
                with gr.Row():
                    # Input Method on the left
                    with gr.Column(scale=6):
                        input_method_submit = gr.Radio(
                            choices=["📝 Paste Text", "📎 Upload File"],
                            value="📝 Paste Text",
                            label="Input Method",
                            interactive=True
                        )

                    # Detector Selection on the right
                    with gr.Column(scale=6):
                        detector_selection = gr.Dropdown(
                            label="🛠️ Select Detector",
                            choices=[],  # To be populated on load
                            value=None,
                            interactive=True
                        )
                        # Refresh Detectors Button placed directly under the Select Detector Dropdown
                        refresh_detectors_btn = gr.Button("🔄 Refresh Detectors")
                        refresh_detectors_btn.click(
                            refresh_detectors,
                            inputs=[],
                            outputs=[detector_selection]
                        )

                # Input Text or File Upload below the above row
                submit_input_text = gr.Textbox(
                    label="Input Text",
                    lines=15,
                    placeholder="Enter or paste your document text here...",
                    visible=True
                )
                submit_file_upload = gr.File(
                    label="Upload File",
                    file_types=[".pdf", ".txt", ".docx"],  # Added .docx
                    visible=False
                )

                def toggle_input_submit(input_method):
                    logger.info(f"Toggling input method to: {input_method}")
                    if input_method == "📝 Paste Text":
                        logger.debug("Setting Textbox visible and File Upload hidden with value=None")
                        return gr.update(visible=True), gr.update(visible=False, value=None)
                    elif input_method == "📎 Upload File":
                        logger.debug("Setting Textbox hidden and File Upload visible with value=None")
                        return gr.update(visible=False, value=None), gr.update(visible=True)
                    else:
                        logger.warning(f"Unknown input method selected: {input_method}")
                        return gr.update(visible=True), gr.update(visible=False, value=None)

                input_method_submit.change(
                    toggle_input_submit,
                    inputs=[input_method_submit],
                    outputs=[submit_input_text, submit_file_upload]
                )

                submit_btn = gr.Button("🚀 Submit Detection")

                submit_output = gr.Markdown(label="")

                fetch_status_message = gr.Markdown(
                    label="**Status:**"
                )
                fetch_ai_percentage = gr.HTML(
                    label="📊 AI Detection Percentage"
                )

        # Connect the buttons and functions
        refresh_btn.click(
            refresh_detection_list,
            inputs=[],
            outputs=[detection_list, request_mapping_state]
        )

        detection_list.change(
            on_detection_request_selected,
            inputs=[request_mapping_state, detection_list],
            outputs=[fetch_status_message, fetch_ai_percentage]
        )

        submit_btn.click(
            submit_and_refresh,
            inputs=[submit_input_text, submit_file_upload, detector_selection],
            outputs=[submit_output, detection_list, request_mapping_state],
            show_progress=True
        )

        # Fetch detectors when the app loads
        demo.load(
            fn=initialize_detectors,
            inputs=[],
            outputs=[detector_selection]
        )

    # Launch the app with specified server name and port
    server_name = os.getenv("GRADIO_SERVER_NAME", "localhost")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    demo.launch(server_name=server_name, server_port=server_port)
