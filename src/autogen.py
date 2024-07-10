from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import autogen
import openai

# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample.json

# config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

config_list = [
    {
        'model': 'gpt-3.5-turbo-16k',
        'api_key': '',
    },
]

llm_config = {"config_list": config_list, "seed": 42}

coder = autogen.AssistantAgent(
    name="Coder",
    llm_config=llm_config,
)

critic = autogen.AssistantAgent(
    name="Critic",
    system_message="""Critic. You are a helpful assistant highly skilled in evaluating the quality of a given code by providing score from 1 to 10 out of 10, based on following dimensions:
- Bugs: Are there bugs, logic errors, syntax errors, or types? Are there any reasons why the code may fail to compile? How should it be fixed?
- Goal compliance: How well the code meets the specified visualization goal?
- Aesthetics: Are the aesthetics of the visualization appropriate for the type and data?

You must provide a score for each of the above dimensions.
{Bugs, Goal compliance, Aesthetics}
Do not suggest code.
Finally, based on the critique above, suggest a concrete list of actions that the coder should take to improve the code.
""",
    llm_config=llm_config,
)

#assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})

user_proxy = UserProxyAgent(
    "user_proxy", code_execution_config={"work_dir": "coding"})

groupchat = autogen.GroupChat(
    agents=[user_proxy, coder, critic], messages=[], max_round=20)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(
    manager, message="Create a snake game")
# This initiates an automated chat between the two agents to solve the task