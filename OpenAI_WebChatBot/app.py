import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI, BadRequestError
import pygame
import tempfile
import time
import hashlib
from datetime import datetime

# 환경 변수 로드
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI()

# pygame 초기화
pygame.mixer.init()

# 음성 파일 저장 디렉토리 생성
os.makedirs("speech_files", exist_ok=True)

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
if "current_speech" not in st.session_state:
    st.session_state.current_speech = None

# 음성 파일 경로 생성 함수
def get_speech_file_path(message_content):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    content_hash = hashlib.md5(message_content.encode()).hexdigest()[:8]
    return os.path.join("speech_files", f"speech_{timestamp}_{content_hash}.mp3")

# 음성 생성 및 저장 함수
def generate_and_save_speech(text, file_path):
    with client.audio.speech.with_streaming_response.create(
        model=st.session_state.voice_model,
        voice=st.session_state.voice_type,
        input=text,
        instructions=st.session_state.voice_instructions if st.session_state.voice_instructions else None
    ) as response:
        response.stream_to_file(file_path)
    return file_path

# 음성 재생 콜백 함수
def play_audio(file_path):
    st.session_state.current_speech = file_path
    
# 실제 음성 재생 함수
def play_speech(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.music.unload()

# 페이지 설정
st.set_page_config(page_title="차수민 챗봇", page_icon="💬")

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

# 채팅 기록 표시
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # AI 응답에만 음성 재생 버튼 추가
        if message["role"] == "assistant":
            # 음성 파일 경로가 없으면 생성
            if "speech_file" not in message:
                speech_file = get_speech_file_path(message["content"])
                if not os.path.exists(speech_file):
                    speech_file = generate_and_save_speech(message["content"], speech_file)
                message["speech_file"] = speech_file
            
            # 음성 재생 버튼
            st.button("🔊 음성 다시 듣기", key=f"play_{i}", on_click=play_audio, args=(message["speech_file"],))

# 현재 재생할 음성이 있으면 재생
if st.session_state.current_speech:
    play_speech(st.session_state.current_speech)
    st.session_state.current_speech = None

# 사용자 입력
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # AI 응답 생성
    with st.chat_message("assistant"):
        try:
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
            speech_file = get_speech_file_path(ai_response)
            generate_and_save_speech(ai_response, speech_file)
            
            # AI 응답을 메시지 히스토리에 추가
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_response,
                "speech_file": speech_file
            })
            
            # 음성 즉시 재생
            play_speech(speech_file)
            
        except BadRequestError as e:
            st.error(f"API 호출 중 오류가 발생했습니다: {str(e)}")
        except Exception as e:
            st.error(f"예기치 않은 오류가 발생했습니다: {str(e)}")
