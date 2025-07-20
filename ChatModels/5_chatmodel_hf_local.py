from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline


llm = HuggingFacePipeline.from_model_id(
    model_id="tiiuae/falcon-rw-1b",
    task="text-generation",
    huggingfacehub_api_token="Your_HuggingFace_API_Token",
    pipeline_kwargs={
        "max_length": 100,
        "temperature": 0.7
    }
)

model = ChatHuggingFace(llm=llm)    

result = model.invoke("What is the capital of France?")

print(result.content)  # Output the response from the model

#pytorch was required to run this code but lesser version was needed
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/wh 

#it didnt run!