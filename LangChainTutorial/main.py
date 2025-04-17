from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# LLM 모델을 생성합니다.
llm = ChatOpenAI(temperature=0)

# ConversationChain을 생성합니다.
conversation = ConversationChain(
    # ConversationBufferMemory를 사용합니다.
    llm=llm,
    memory=ConversationBufferMemory(),
)
# memory.save_context(
#     inputs={
#         "human" : "안녕하세요"
#     },
#     outputs={
#         "ai" : "안녕하세요! 무엇을 도와 드릴까요"
#     }
# )
# memory.save_context(
#     inputs={
#         "human" : "안녕하세요"
#     },
#     outputs={
#         "ai" : "안녕하세요! 무엇을 도와 드릴까요"
#     }
# )
# 대화를 시작합니다.
response = conversation.predict(
    input="안녕하세요, 비대면으로 은행 계좌를 개설하고 싶습니다. 어떻게 시작해야 하나요?"
)
print(response)