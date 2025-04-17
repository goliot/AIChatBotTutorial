# 음성 받기(Speech To Text) #pip install SpeechRecognition
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound

r = sr.Recognizer()
while True:
    try:
        with sr.Microphone() as source:
            print("음성을 입력하세요.")
            audio = r.listen(source)
            txt = r.recognize_google(audio, language='ko')
            print(txt)

            #TTS
            # if '카카오' in txt:
            tts = gTTS(text=txt, lang='ko')
            path = "TTS.mp3"
            tts.save(path)
            playsound(path)
            break
    except:
        print('에러!!!')