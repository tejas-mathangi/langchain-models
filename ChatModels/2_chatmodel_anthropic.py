from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(
    model="claude-2",
    # temperature=0.7,
    # max_tokens=1000,
    # top_p=0.9,
    # stop=["\n\nHuman:", "\n\nAssistant:"],
    # response_model="chat",
    # anthropic_api_key="your_anthropic_api_key_here"
)

result = model.invoke(
    "What is the capital of France?",  # Example query
    # stop=["\n\nHuman:", "\n\nAssistant:"]
)

print(result.content)  # Output the response from the model 