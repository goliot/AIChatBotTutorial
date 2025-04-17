import streamlit as st
import os
from pathlib import Path
import base64
import pygame

# 페이지 설정 (가장 먼저 실행되어야 함)
st.set_page_config(page_title="차수민 챗봇", page_icon="💬")

from dotenv import load_dotenv
from openai import OpenAI, BadRequestError
import tempfile
import time
import hashlib
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# 환경 변수 로드
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI()

# pygame 초기화
pygame.mixer.init()

# 현재 작업 디렉토리 가져오기
current_dir = Path(__file__).parent.absolute()

# 음성 파일 저장 디렉토리 생성
speech_dir = current_dir / "speech_files"
os.makedirs(speech_dir, exist_ok=True)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = "gpt-4.1-nano"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "voice_model" not in st.session_state:
    st.session_state.voice_model = "tts-1"
if "voice_type" not in st.session_state:
    st.session_state.voice_type = "alloy"
if "voice_instructions" not in st.session_state:
    st.session_state.voice_instructions = ""

# 음성 파일 경로 생성 함수
def get_speech_file_path(message_content):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    content_hash = hashlib.md5(message_content.encode()).hexdigest()[:8]
    return str(speech_dir / f"speech_{timestamp}_{content_hash}.mp3")

# 음성 생성 및 저장 함수
def generate_and_save_speech(text, file_path):
    try:
        with client.audio.speech.with_streaming_response.create(
            model=st.session_state.voice_model,
            voice=st.session_state.voice_type,
            input=text,
            instructions=st.session_state.voice_instructions if st.session_state.voice_instructions else None,
            response_format="mp3"
        ) as response:
            # 파일 확장자가 없으면 추가
            if not file_path.endswith('.mp3'):
                file_path += '.mp3'
            response.stream_to_file(file_path)
            if not os.path.exists(file_path):
                raise Exception("음성 파일이 생성되지 않았습니다.")
            return file_path
    except Exception as e:
        st.error(f"음성 생성 중 오류가 발생했습니다: {str(e)}")
        raise

# 음성 재생 함수
def play_speech(file_path):
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()
    except Exception as e:
        st.error(f"음성 재생 중 오류가 발생했습니다: {str(e)}")

# 파일을 base64로 인코딩하는 함수
def get_binary_file_downloader_html(file_path, file_label='File'):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:audio/mp3;base64,{b64}" download="{os.path.basename(file_path)}">{file_label}</a>'

# PDF 문서 로드 및 벡터 저장소 초기화
@st.cache_resource
def initialize_pdf_rag():
    # PDF 파일들이 있는 디렉토리
    pdf_dir = "pdfs"
    
    # 모든 PDF 문서를 로드
    all_docs = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
    
    # 텍스트 분할기 초기화
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
    )
    
    # 문서 분할
    texts = text_splitter.split_documents(all_docs)
    
    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings()
    
    # 벡터 저장소 생성
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # QA 체인 초기화
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return qa_chain

# PDF RAG 초기화
qa_chain = initialize_pdf_rag()

# 사이드바 설정
with st.sidebar:
    st.title("설정 ⚙️")
    
    # AI 모델 선택
    model_options = {
        "GPT-4.1 Nano": "gpt-4.1-nano",
        "GPT-3.5 Turbo": "gpt-3.5-turbo",
        "GPT-4": "gpt-4",
        "GPT-4 Turbo": "gpt-4-turbo-preview"
    }
    selected_model = st.selectbox(
        "AI 모델 선택",
        options=list(model_options.keys()),
        index=list(model_options.keys()).index(next(k for k, v in model_options.items() if v == st.session_state.model))
    )
    st.session_state.model = model_options[selected_model]
    
    # Temperature 설정
    st.session_state.temperature = st.slider(
        "창의성 (Temperature)",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.temperature,
        step=0.1,
        help="값이 높을수록 더 창의적인 응답을 생성합니다. 낮을수록 더 결정적이고 일관된 응답을 생성합니다."
    )

    # 음성 설정 구분선
    st.divider()
    st.subheader("음성 설정 🎤")

    # 음성 모델 선택
    voice_model_options = {
        "TTS-1": "tts-1",
        "TTS-1 HD": "tts-1-hd"
    }
    selected_voice_model = st.selectbox(
        "음성 모델 선택",
        options=list(voice_model_options.keys()),
        index=list(voice_model_options.keys()).index(next(k for k, v in voice_model_options.items() if v == st.session_state.voice_model))
    )
    st.session_state.voice_model = voice_model_options[selected_voice_model]

    # 목소리 타입 선택
    voice_type_options = {
        "Alloy": "alloy",
        "Echo": "echo",
        "Fable": "fable",
        "Onyx": "onyx",
        "Nova": "nova",
        "Shimmer": "shimmer"
    }
    selected_voice_type = st.selectbox(
        "목소리 타입 선택",
        options=list(voice_type_options.keys()),
        index=list(voice_type_options.keys()).index(next(k for k, v in voice_type_options.items() if v == st.session_state.voice_type))
    )
    st.session_state.voice_type = voice_type_options[selected_voice_type]

    # 음성 지시사항 설정
    st.subheader("음성 지시사항")
    
    # 미리 정의된 지시사항 옵션
    preset_instructions = {
        "기본": "",
        "건방진": "건방진 목소리",
        "친근한": "친근하고 따뜻한 목소리",
        "진지한": "진지하고 권위있는 목소리",
        "재미있는": "재미있고 활기찬 목소리",
        "슬픈": "슬프고 감정적인 목소리",
        "조폭": "조폭같은 말투",
        "술취한 사람": "술 취한 목소리"
    }
    
    # 지시사항 선택 방식
    instruction_mode = st.radio(
        "지시사항 설정 방식",
        ["미리 정의된 옵션", "직접 입력"],
        horizontal=True
    )
    
    if instruction_mode == "미리 정의된 옵션":
        selected_preset = st.selectbox(
            "지시사항 선택",
            options=list(preset_instructions.keys()),
            index=list(preset_instructions.keys()).index(next(k for k, v in preset_instructions.items() if v == st.session_state.voice_instructions))
        )
        st.session_state.voice_instructions = preset_instructions[selected_preset]
    else:
        st.session_state.voice_instructions = st.text_input(
            "직접 지시사항 입력",
            value=st.session_state.voice_instructions,
            placeholder="예: 건방진 목소리, 친근한 톤으로 말하기 등"
        )

# 타이틀
st.title("AI 채팅방 🤖")

# 오디오 HTML 생성 함수
def get_audio_html(audio_file):
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
        <audio controls autoplay>
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
    """
    return audio_html

# 채팅 기록 표시
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # AI 응답에만 음성 재생 버튼 추가
        if message["role"] == "assistant":
            # 음성 파일 경로가 없으면 생성
            if "speech_file" not in message:
                try:
                    speech_file = get_speech_file_path(message["content"])
                    if not os.path.exists(speech_file):
                        speech_file = generate_and_save_speech(message["content"], speech_file)
                        st.success(f"음성 파일이 생성되었습니다: {speech_file}")
                    message["speech_file"] = speech_file
                except Exception as e:
                    st.error(f"음성 파일 생성 중 오류가 발생했습니다: {str(e)}")
                    continue
            
            # 음성 재생
            try:
                if "speech_file" in message and os.path.exists(message["speech_file"]):
                    # HTML 오디오 플레이어 직접 생성
                    with open(message["speech_file"], "rb") as f:
                        audio_bytes = f.read()
                    audio_b64 = base64.b64encode(audio_bytes).decode()
                    audio_html = f"""
                        <audio controls autoplay>
                            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                            Your browser does not support the audio element.
                        </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
                else:
                    st.error("음성 파일을 찾을 수 없습니다. 다시 생성해주세요.")
            except Exception as e:
                st.error(f"음성 재생 중 오류가 발생했습니다: {str(e)}")

# 사용자 입력
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # AI 응답 생성
    with st.chat_message("assistant"):
        try:
            # PDF RAG를 사용하여 답변 생성
            rag_response = qa_chain.invoke({"query": prompt})
            rag_answer = rag_response["result"]
            
            # 시스템 메시지에 음성 지시사항 반영
            system_message = {
                "role": "system",
                "content": f"당신은 {st.session_state.voice_instructions}로 대화하는 AI 어시스턴트입니다. 모든 응답은 이 톤과 스타일을 유지해야 합니다."
            } if st.session_state.voice_instructions else None

            # 메시지 목록 준비
            messages = [system_message] if system_message else []
            messages.extend([
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ])
            
            # PDF RAG 답변을 컨텍스트로 포함
            messages.append({
                "role": "system",
                "content": f"다음은 PDF 문서에서 찾은 관련 정보입니다: {rag_answer}"
            })

            # OpenAI API 호출
            response = client.chat.completions.create(
                model=st.session_state.model,
                messages=messages,
                temperature=st.session_state.temperature,
            )
            
            # AI 응답 표시
            ai_response = response.choices[0].message.content
            st.write(ai_response)
            
            # 음성 파일 생성 및 저장
            try:
                speech_file = get_speech_file_path(ai_response)
                generate_and_save_speech(ai_response, speech_file)
                st.success(f"음성 파일이 생성되었습니다: {speech_file}")
                
                # AI 응답을 메시지 히스토리에 추가
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_response,
                    "speech_file": speech_file
                })
                
                # 음성 재생
                with open(speech_file, "rb") as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/mp3")
            except Exception as e:
                st.error(f"음성 생성/재생 중 오류가 발생했습니다: {str(e)}")
            
        except BadRequestError as e:
            st.error(f"API 호출 중 오류가 발생했습니다: {str(e)}")
        except Exception as e:
            st.error(f"예기치 않은 오류가 발생했습니다: {str(e)}")
