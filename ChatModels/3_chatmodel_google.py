from langchain_google_genai import ChatGoogleGenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenAI(model="gemini-1.5-flash")   

result = model.invoke("What is the capital of France?")
print(result.content)  # Output the response from the model