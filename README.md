# Event Assistant Chatbot
Event assistant app written in python leveraging Claude, Gemini and ChatGPT LLMs. It is currently designed for a school event dataset, but can be easily expanded to any organisation events database. 

## Features
- The chatbot UI is built using Gradio.
- The user can query the chatbot asking for information on an upcoming event in a school (PTA meeting, 100 days of school, etc)
- The chatbot responds with the event details, reads out the response and also displays an image generated dynamically for that particular event.
- Used each of the LLMs' tools agentic framework to route the queries to an exclusive locally written function to fetch the event details.
  
## Requirements
To run this project, you need the following:
- Google account (to run this in Google Colab)
- Python 3.11 or higher (to run it locally as python script and also for Jupyter Notebook)
- Jupyter Notebook (to run it locally using Jupyter lab)
- Your OpenAI API Secret Key. Get one in few secs from [OpenAi](https://platform.openai.com/settings/organization/api-keys)

## Installation

1. **Clone this repository:**
   Open a terminal and run:
   ```bash
   git clone https://github.com/samt07/event_assistant.git

2. **Navigate to the project directory**
    ```bash
    cd event_assistant

## Set Up Environment Variables  

1. **Create a `.env` file**  
   - Navigate to the project directory.  
   - Create a new file named `.env`.  

2. **Add the OpenAI API Key**  
   - Open the `.env` file in a text editor.  
   - Add the following line:  
     ```env
     OPENAI_API_KEY=youropenaikey
     ```
   - Ensure:  
     - No spaces before or after the `=`.  
     - No quotes around the value.  

3. **Save the file**  
   - Save it with the exact name `.env`.  
   - Verify that it is not saved as `.env.txt`, `env.env`, or any other variation.  

## Usage

## Option 1: Run with locally installed Jupyter Notebook. You must have installed Python already. 
   1. Create a .env file as mentioned above
   2. Open the Jupyter Notebook:
       ```bash
          jupyter lab event_assistant.ipynb
   3. Follow the instructions in the notebook to execute the code cell by cell, by using `Shift+Enter` key.
   4. If the Python version is 3.13 or higher, there might be a warning message for some imports. These can be ignored.

## Option 3: Run this on Google Colab (No installation Required.)

   1. Go to [Google Colab](https://colab.research.google.com/).  
   2. Click **File > Upload Notebook** and select `event_assistant_colab.ipynb` from your local cloned repository folder.
   3. Set up env variable. Use Google Colab's Keys (typically found on the left side tool bar with a Key image)
      - 3a. Add `OPENAI_API_KEY` as the Secret name and paste your Open AI API Key. Enable the Notebook access option.  
   4. Run the Notebook cell-by-cell by pressing `Shift+Enter`
   5. At the end, upload your requirement document in Gradio UI.

## Option 4: Run as a standalone .py python script
   1. If Python is not installed already, install Python 3.11 or higher version from [here](https://www.python.org/downloads/)
   2. Create a .env file as mentioned above.
   3. Install dependenices by running this command
      ```bash
       pip install -r requirements.txt
   4. Run the following command
      ```bash
       ipython event_assistant.py
   
## File Structure
- `event_assistant.ipynb`: Jupyter notebook to run in locally installed jupyter lab.
- `event_assistant_colab.ipynb`: Jupyter notebook to run in Google Colab.
-  `event_assistant.py`: To run as a standalone python script locally
- `.env`: Environment file for storing the OpenAI API Key (not included in the repository).
- `requirements.txt`: Contains the required dependencies. Needed only for running as local python script
