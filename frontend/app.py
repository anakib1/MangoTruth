import gradio as gr


def display_text(input_text):
    """
    This function takes input text from the Gradio text box and simply returns it.

    :param input_text: Text input from the user.
    :return: The same text input, to be displayed as output.
    """
    return f"You entered: {input_text}"


# Create Gradio interface
iface = gr.Interface(
    fn=display_text,  # The function to process input
    inputs=gr.Textbox(label="Enter your text here"),  # Textbox for input
    outputs="text",  # Display output as text
    title="Simple Text Input Interface",  # Title for the interface
    description="A simple Gradio interface to enter and display text."  # Description
)

if __name__ == '__main__':
    iface.launch()
