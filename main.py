from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler    
from langchain.chains import LLMChain
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

llm = Ollama(model="mistral-openorca", 
            #callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
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
print(brainstorm_output)

introduction_template = str(pre_prompt) + "Please output only an introduction about idea number {number} from the following list " + brainstorm_output + " Don't respond with any text other than the introduction "

introduction_prompt = PromptTemplate(
    input_variables=["number"],
    template=introduction_template
)

introduction_chain = LLMChain(llm=llm,
                    prompt=introduction_prompt,
                    verbose=False)

choice_input = int(input("\n\nNumber to generate: "))

introduction_output = introduction_chain.run(choice_input)

body_template = str(pre_prompt) + "Please read through the following introduction and write the body of the blog post: {intro} don't respond with any text other than the body of the blog "

body_prompt = PromptTemplate(
    input_variables=["intro"],
    template=body_template
)

body_chain = LLMChain(llm=llm,
                    prompt=body_prompt,
                    verbose=False)

body_output = body_chain.run(introduction_output)

end_template = str(pre_prompt) + "Please read through the following body of a blog post then write a suitable conclussion of the blog post: {intro} {body} don't respond with any text othet than the conclussion of the blog "

end_prompt = PromptTemplate(
    input_variables=["body"],
    template=end_template
)

end_chain = LLMChain(llm=llm,
                    prompt=end_prompt,
                    verbose=False)

end_output = end_chain.run(introduction_output, body_output)



print("\n\n\nHere is the final blog post: \n\n\n"+ end_output)
