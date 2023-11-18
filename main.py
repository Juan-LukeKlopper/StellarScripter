from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler    
from langchain.chains import LLMChain
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

llm = Ollama(model="mistral-openorca", 
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=0.9,
             )

pre_prompt = "You are a respected industry leader in writing and blogs with indepth knowledge of all factors including tone and storytelling, "
brainstorm_template = str(pre_prompt) + "Please brainstorm and create 10 ideas for a short to medium length blog post around the topic: {topic}?"
brainstorm_prompt = PromptTemplate(
    input_variables=["topic"],
    template=brainstorm_template,
)

brainstorm_chain = LLMChain(llm=llm, 
                 prompt=brainstorm_prompt,
                 verbose=False)

user_input = input("What do you want to write a blog about: ")

# Run the chain only specifying the input variable.
brainstorm_output = brainstorm_chain.run(user_input)
#print(brainstorm_output)

introduction_template = str(pre_prompt) + "Please output only an introduction about idea number {number} from the following list " + brainstorm_output + " Don't respond with any text othet than the introduction "

introduction_prompt = PromptTemplate(
    input_variables=["number"],
    template=introduction_template
)

introduction_chain = LLMChain(llm=llm,
                    prompt=introduction_prompt,
                    verbose=False)

choice_input = int(input("\n\nNumber to generate: "))
print("\n\nHere is the introduction's first draft: \n\n\n")

introduction_output = introduction_chain.run(choice_input)
