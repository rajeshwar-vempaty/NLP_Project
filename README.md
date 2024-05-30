# Project Title: Interactive PDF Knowledge Extraction System

Installation and Execution Guide:

•	Prerequisites
Python: The system is developed in Python, so ensure Python 3.8 or newer is installed on your system.
Visual Studio Code: Recommended as the development environment to leverage features like IntelliSense, code navigation, and integrated terminal for a seamless experience.
•	Installation Steps
Clone the Repository:
First, clone the repository containing the project code to your local machine using Git. If you don't have Git installed, download, and install it from git-scm.com.

git clone https://your-repository-url.git
cd your-project-folder

Set Up a Virtual Environment:
It is best practice to use a virtual environment for Python projects to manage dependencies efficiently. You can create one using:

python -m venv venv

Activate the virtual environment:
Windows: .\venv\Scripts\activate
MacOS/Linux: source venv/bin/activate

Install Dependencies:
Install all required packages using pip. These packages include Streamlit, PyPDF2, pdfplumber, langchain, faiss, dotenv, and sentence_transformers among others.

pip install streamlit PyPDF2 pdfplumber python-dotenv faiss-cpu sentence-transformers langchain

Note: If you are using a GPU and want to leverage it, install faiss-gpu instead of faiss-cpu for better performance.

Environment Variables:
If the project uses environment variables (e.g., API keys), ensure that the .env file is set up correctly in the project's root directory with the required variables.

•	Running the Application
Once installation is complete, you can run the application through Visual Studio Code's integrated terminal:

Start the Streamlit App:
Make sure you are still in the project's root directory and the virtual environment is activated. Run the application using Streamlit:

streamlit run app.py

Access the Web Interface:
After executing the command, Streamlit will start the server and provide a local URL (usually http://localhost:8501) which you can open in a web browser to interact with the application.
Using the Application:
Upload PDF files using the provided interface, enter your queries related to the PDF content, and receive answers dynamically generated by the system.

•	Troubleshooting
If you encounter any package dependency issues, try updating pip and retrying the installation:

pip install --upgrade pip
pip install -r requirements.txt
