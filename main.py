import os
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    city: str = Field(description="The city of the person")

def main():
    model = init_chat_model(
        "gpt-4o-mini",
        temperature=0.7,
    )
    # for chunk in model.stream("너는 누구니?"):
    #     print(chunk.content, end="", flush=True)
    
    # full = None
    # for chunk in model.stream("너는 누구니?"):
    #     full = chunk if full is None else full + chunk
    # print(full.text)

    # responses = model.batch([
    #     "Why do parrots have colorful feathers?",
    #     "How do airplanes fly?",
    #     "What is quantum computing?"
    # ])
    # for response in responses:
    #     print(response.content)
    model_with_structure = model.with_structured_output(Person)
    response = model_with_structure.invoke("워렌 버핏에 대해 설명해줘.")
    print(response)


if __name__ == "__main__":
    main()
