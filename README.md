# LLM Chatbot for Higher School of Economics Admissions

## 📌 Overview
This project is a chatbot designed to assist prospective students in obtaining information about admissions, academic programs, and university life at the **Higher School of Economics (HSE)**. By leveraging **Large Language Models (LLMs)**, the chatbot provides real-time answers, automating and simplifying the information retrieval process.

### 🔹 Key Technologies
- **LangChain** – Framework for integrating LLMs
- **Vector Database (Pinecone)** – Efficient information retrieval
- **ReAct Framework** – Combining reasoning and action for decision-making
- **Agents** – Managing user interactions dynamically
- **Retrieval-Augmented Generation (RAG)** – Ensuring up-to-date, relevant responses

---

## 🎯 Features
✅ **Conversational Memory** – Context-aware interactions using `ConversationBufferWindowMemory`  
✅ **Semantic Search** – Retrieves accurate university-related data from a vector database  
✅ **Dynamic Query Handling** – Uses ReAct agents to provide precise responses  
✅ **Real-Time Information Retrieval** – Keeps answers relevant and updated  

---

## Installation

## Clone the repository:
   ```
   git clone https://github.com/vlada270/LLM_chat.git
   cd LLM_chat
  ```
## Running the chatbot locally:
```
python main.py
```
### Running the web-based interface (if applicable):
``` streamlit run app.py ```
### Testing the chatbot in Python:
```
from chatbot import Chatbot

bot = Chatbot()
response = bot.ask("What are the admission requirements for Computer Science?")
print(response)
```

## Project Structure
```
LLM_chat/
│── data/                  # Data files and embeddings
│── models/                # Model configurations
│── chatbot/               # Core chatbot logic
│   ├── agents.py          # Agent-based interactions
│   ├── memory.py          # Conversational memory implementation
│   ├── retriever.py       # Retrieval logic with vector database
│── app.py                 # Web application (if applicable)
│── main.py                # Main script for chatbot execution
│── requirements.txt       # Required dependencies
│── .env.example           # Environment variable template
└── README.md              # Project documentation
```

## Example Queries
```
bot.ask("What bachelor's programs in physics are available at HSE?")
bot.ask("What exams are required for the Applied Data Analysis program?")
bot.ask("How much does the HSE design program cost?")
```


