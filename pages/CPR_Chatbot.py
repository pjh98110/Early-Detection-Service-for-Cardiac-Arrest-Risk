import streamlit as st
import openai
import google.generativeai as genai
from streamlit_chat import message
import os
import requests
from streamlit_extras.colored_header import colored_header
import pandas as pd

# 페이지 구성 설정
st.set_page_config(layout="wide")

openai.api_key = st.secrets["secrets"]["OPENAI_API_KEY"]

if "page" not in st.session_state:
    st.session_state.page = "CPR"

if "gpt_api_key" not in st.session_state:
    st.session_state.gpt_api_key = openai.api_key # gpt API Key

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = st.secrets["secrets"]["GEMINI_API_KEY"]

# GPT 프롬프트 엔지니어링 함수
def gpt_prompt(user_input):
    base_prompt = f"""
    너는 지금부터 입력된 [환자의 주변 상황]에 따라 심폐소생술 가이드라인을 알려주는 친절하고 차분한 [응급의료 지원센터의 가이드]이다. 
    사용자는 응급 상황에서 최선을 다하려는 [구조자]이며, 사용자에게 단계별로 심폐소생술 가이드라인을 제시한다. 
    내가 채팅을 입력하면 아래의 <규칙>에 따라서 답변한다.

    <규칙>
    1) 환자의 성별, 나이, 주변 상황을 분석하여, 맞춤형 심폐소생술 가이드라인을 전달한다. 
    2) 예시를 참고하여 더 발전시킨 심폐소생술 가이드라인을 답변한다.

    구조자의 성별은 {st.session_state.selected_gender}이며, 구조자의 나이는 {st.session_state.selected_age}이다.
    [환자의 주변 상황]은 {st.session_state.gpt_input_cpr}이며 이 정보를 바탕으로 <규칙>에 따라서 답변한다.

    
    예시: 심폐소생술 가이드라인
    [1단계] 반응 확인 및 안전 확보
    a) 주변 안전 확인: 환자와 주변의 안전을 먼저 확인하세요. 위험 요소(예: 차량, 불, 물 등)가 없는지 확인합니다.
    b) 의식 확인: 환자의 어깨를 가볍게 흔들며 큰 소리로 "괜찮으세요?"라고 물어보세요.
    c) 호흡 확인: 환자의 호흡을 10초 동안 확인하세요. 호흡이 없거나 비정상적인 경우, 심정지로 간주합니다.

    [2단계] 도움 및 자동심장충격기 요청
    a) 주변 사람에게 도움 요청: 큰 소리로 "여기 사람 좀 도와주세요!"라고 외치세요.
    b) 도움 요청: 주변에 도움을 줄 수 있는 사람이 있다면 역할을 분담합니다. (예: 119와의 통화, 자동심장충격기 가져오기, 주변 정리)
    c) 119 신고: 주변에 누군가에게 119에 전화를 걸도록 요청하고 신고 시 위치, 상황, 환자 상태 명확히 전달하도록 안내합니다.
    d) AED 요청: 주변에 자동심장충격기가 있는지 확인하고, 누군가에게 가져오도록 요청합니다. 

    [3단계] 가슴 압박 시작
    a) 가슴 압박 위치: 환자의 가슴 중앙(양쪽 젖꼭지 사)에 손을 놓습니다.
    b) 손 위치: 한 손을 다른 손 위에 놓고, 손가락을 깍지 낍니다.
    c) 가슴 압박: 몸을 똑바로 세워 팔꿈치을 곧게 펴고, 가슴을 5-6cm 깊이로 1분에 100-120회의 속도로 수직으로 체중을 실어 압박합니다.
    d) 성인과 청소년: 분당 100-120회 속도로, 5-6cm 깊이로 압박합니다.
    e) 어린이: 한 손 또는 두 손을 사용, 가슴 두께의 1/3 깊이로 압박합니다.
    f) 영아: 두 손가락을 사용, 가슴 두께의 1/3 깊이로 압박합니다.
    g) 30회 압박 후 2회 인공호흡을 합니다. (훈련된 구조자의 경우)
    h) 인공호흡 시 1초에 걸쳐 가슴이 올라올 정도로 불어넣습니다.

    [4단계] 자동심장충격기 사용
    a) 자동심장충격기 준비: 자동심장충격기가 도착하면 AED를 환자 옆에 놓고, 전원을 켭니다.
    b) 패드 부착: 환자의 가슴을 노출시키고, 패드 두 개를 부착합니다. 하나는 오른쪽 쇄골 아래, 다른 하나는 왼쪽 가슴 아래에 부착합니다.
    c) 음성지시 따르기: AED의 음성지시에 따라 충격을 준비하고, 충격이 필요할 경우 모두에게 환자에게서 떨어지라고 지시합니다.
    d) 충격 시행: AED가 충격을 가하라고 지시하면 주변 사람들에게 환자에게서 떨어지도록 큰 소리로 알리고 충격 버튼을 누릅니다. 
    e) CPR 재개: AED가 충격을 가한 후, 즉시 가슴 압박을 재개합니다. AED가 다시 분석할 때까지 CPR을 계속합니다.

    [5단계] 지속적인 심폐소생술 및 상태 관찰
    a) 2분마다 가슴압박 시행자 교대합니다. (피로 방지)
    b) 환자의 상태 변화 지속 관찰합니다. (호흡, 의식 회복 등)
    c) 119 도착 시까지 또는 환자가 회복될 때까지 멈추지 않고 지속합니다.

    사용자 입력: {user_input}
    """
    return base_prompt


# Gemini 프롬프트 엔지니어링 함수
def gemini_prompt(user_input):
    base_prompt = f"""
    너는 지금부터 입력된 [환자의 주변 상황]에 따라 심폐소생술 가이드라인을 알려주는 친절하고 차분한 [응급의료 지원센터의 가이드]이다. 
    사용자는 응급 상황에서 최선을 다하려는 [구조자]이며, 사용자에게 단계별로 심폐소생술 가이드라인을 제시한다. 
    내가 채팅을 입력하면 아래의 <규칙>에 따라서 답변한다.

    <규칙>
    1) 환자의 성별, 나이, 주변 상황을 분석하여, 맞춤형 심폐소생술 가이드라인을 전달한다. 
    2) 예시를 참고하여 더 발전시킨 심폐소생술 가이드라인을 자세하게 답변한다.
    3) 심폐소생술 가이드라인 전달을 다음 채팅으로 넘어가야한다면, 다음 채팅에서 이어서 다음 단계를 답변한다.

    구조자의 성별은 {st.session_state.selected_gender}이며, 구조자의 나이는 {st.session_state.selected_age}이다.
    [환자의 주변 상황]은 {st.session_state.gemini_input_cpr}이며 이 정보를 바탕으로 <규칙>에 따라서 답변한다.

    
    예시: 심폐소생술 가이드라인
    [1단계] 반응 확인 및 안전 확보
    a) 주변 안전 확인: 환자와 주변의 안전을 먼저 확인하세요. 위험 요소(예: 차량, 불, 물 등)가 없는지 확인합니다.
    b) 의식 확인: 환자의 어깨를 가볍게 흔들며 큰 소리로 "괜찮으세요?"라고 물어보세요.
    c) 호흡 확인: 환자의 호흡을 10초 동안 확인하세요. 호흡이 없거나 비정상적인 경우, 심정지로 간주합니다.

    [2단계] 도움 및 자동심장충격기 요청
    a) 주변 사람에게 도움 요청: 큰 소리로 "여기 사람 좀 도와주세요!"라고 외치세요.
    b) 도움 요청: 주변에 도움을 줄 수 있는 사람이 있다면 역할을 분담합니다. (예: 119와의 통화, 자동심장충격기 가져오기, 주변 정리)
    c) 119 신고: 주변에 누군가에게 119에 전화를 걸도록 요청하고 신고 시 위치, 상황, 환자 상태 명확히 전달하도록 안내합니다.
    d) AED 요청: 주변에 자동심장충격기가 있는지 확인하고, 누군가에게 가져오도록 요청합니다. 

    [3단계] 가슴 압박 시작
    a) 가슴 압박 위치: 환자의 가슴 중앙(양쪽 젖꼭지 사)에 손을 놓습니다.
    b) 손 위치: 한 손을 다른 손 위에 놓고, 손가락을 깍지 낍니다.
    c) 가슴 압박: 몸을 똑바로 세워 팔꿈치을 곧게 펴고, 가슴을 5-6cm 깊이로 1분에 100-120회의 속도로 수직으로 체중을 실어 압박합니다.
    d) 성인과 청소년: 분당 100-120회 속도로, 5-6cm 깊이로 압박합니다.
    e) 어린이: 한 손 또는 두 손을 사용, 가슴 두께의 1/3 깊이로 압박합니다.
    f) 영아: 두 손가락을 사용, 가슴 두께의 1/3 깊이로 압박합니다.
    g) 30회 압박 후 2회 인공호흡을 합니다. (훈련된 구조자의 경우)
    h) 인공호흡 시 1초에 걸쳐 가슴이 올라올 정도로 불어넣습니다.

    [4단계] 자동심장충격기 사용
    a) 자동심장충격기 준비: 자동심장충격기가 도착하면 AED를 환자 옆에 놓고, 전원을 켭니다.
    b) 패드 부착: 환자의 가슴을 노출시키고, 패드 두 개를 부착합니다. 하나는 오른쪽 쇄골 아래, 다른 하나는 왼쪽 가슴 아래에 부착합니다.
    c) 음성지시 따르기: AED의 음성지시에 따라 충격을 준비하고, 충격이 필요할 경우 모두에게 환자에게서 떨어지라고 지시합니다.
    d) 충격 시행: AED가 충격을 가하라고 지시하면 주변 사람들에게 환자에게서 떨어지도록 큰 소리로 알리고 충격 버튼을 누릅니다. 
    e) CPR 재개: AED가 충격을 가한 후, 즉시 가슴 압박을 재개합니다. AED가 다시 분석할 때까지 CPR을 계속합니다.

    [5단계] 지속적인 심폐소생술 및 상태 관찰
    a) 2분마다 가슴압박 시행자 교대합니다. (피로 방지)
    b) 환자의 상태 변화 지속 관찰합니다. (호흡, 의식 회복 등)
    c) 119 도착 시까지 또는 환자가 회복될 때까지 멈추지 않고 지속합니다.

    사용자 입력: {user_input}
    """
    return base_prompt

# 스트림 표시 함수
def stream_display(response, placeholder):
    text = ''
    for chunk in response:
        if parts := chunk.parts:
            if parts_text := parts[0].text:
                text += parts_text
                placeholder.write(text + "▌")
    return text

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = {
        "gpt": [
            {"role": "system", "content": "GPT가 사용자에게 상황에 맞는 심폐소생술 가이드라인을 알려드립니다."}
        ],
        "gemini": [
            {"role": "model", "parts": [{"text": "Gemini가 사용자에게 상황에 맞는 심폐소생술 가이드라인을 알려드립니다."}]}
        ]
    }

# 세션 변수 체크
def check_session_vars():
    required_vars = ['selected_gender', 'selected_age']
    for var in required_vars:
        if var not in st.session_state:
            st.warning("필요한 정보가 없습니다. 처음으로 돌아가서 정보를 입력해 주세요.")
            st.stop()

selected_chatbot = st.selectbox(
    "원하는 챗봇을 선택하세요.",
    options=["GPT를 통한 심폐소생술 교육", "Gemini를 통한 심폐소생술 교육"],
    placeholder="챗봇을 선택하세요.",
    help="선택한 LLM 모델에 따라 다른 챗봇을 제공합니다."
)

if selected_chatbot == "GPT를 통한 심폐소생술 교육":
    colored_header(
        label='GPT를 통한 심폐소생술 교육',
        description=None,
        color_name="gray-70",
    )

    # 세션 변수 체크
    check_session_vars()

    # 대화 초기화 버튼
    def on_clear_chat_gpt():
        st.session_state.messages["gpt"] = [
            {"role": "system", "content": "GPT가 사용자에게 상황에 맞는 심폐소생술 가이드라인을 알려드립니다."}
        ]

    st.button("대화 초기화", on_click=on_clear_chat_gpt)

    # 이전 메시지 표시
    if "gpt" not in st.session_state.messages:
        st.session_state.messages["gpt"] = [
            {"role": "system", "content": "GPT가 사용자에게 상황에 맞는 심폐소생술 가이드라인을 알려드립니다."}
        ]
        
    for msg in st.session_state.messages["gpt"]:
        role = 'user' if msg['role'] == 'user' else 'assistant'
        with st.chat_message(role):
            st.write(msg['content'])

    # 사용자 입력 처리
    if prompt := st.chat_input("챗봇과 대화하기:"):
        # 사용자 메시지 추가
        st.session_state.messages["gpt"].append({"role": "user", "content": prompt})
        with st.chat_message('user'):
            st.write(prompt)

        # 프롬프트 엔지니어링 적용
        enhanced_prompt = gpt_prompt(prompt)

        # 모델 호출 및 응답 처리
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": enhanced_prompt}
                ] + st.session_state.messages["gpt"],
                max_tokens=1500,
                temperature=0.8,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            text = response.choices[0]['message']['content']

            # 응답 메시지 표시 및 저장
            st.session_state.messages["gpt"].append({"role": "assistant", "content": text})
            with st.chat_message("assistant"):
                st.write(text)
        except Exception as e:
            st.error(f"OpenAI API 요청 중 오류가 발생했습니다: {str(e)}")

elif selected_chatbot == "Gemini를 통한 심폐소생술 교육":
    colored_header(
        label='Gemini를 통한 심폐소생술 교육',
        description=None,
        color_name="gray-70",
    )
    # 세션 변수 체크
    check_session_vars()

    # 사이드바에서 모델의 파라미터 설정
    with st.sidebar:
        st.header("모델 설정")
        model_name = st.selectbox(
            "모델 선택",
            ["gemini-1.5-pro", 'gemini-1.5-flash']
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, help="생성 결과의 다양성을 조절합니다.")
        max_output_tokens = st.number_input("Max Tokens", min_value=1, value=2048, help="생성되는 텍스트의 최대 길이를 제한합니다.")
        top_k = st.slider("Top K", min_value=1, value=40, help="다음 단어를 선택할 때 고려할 후보 단어의 최대 개수를 설정합니다.")
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, help="다음 단어를 선택할 때 고려할 후보 단어의 누적 확률을 설정합니다.")

    st.button("대화 초기화", on_click=lambda: st.session_state.update({
        "messages": {"gemini": [{"role": "model", "parts": [{"text": "Gemini가 사용자에게 상황에 맞는 심폐소생술 가이드라인을 알려드립니다."}]}]}
    }))

    # 이전 메시지 표시
    if "gemini" not in st.session_state.messages:
        st.session_state.messages["gemini"] = [
            {"role": "model", "parts": [{"text": "Gemini가 사용자에게 상황에 맞는 심폐소생술 가이드라인을 알려드립니다."}]}
        ]
        
    for msg in st.session_state.messages["gemini"]:
        role = 'human' if msg['role'] == 'user' else 'ai'
        with st.chat_message(role):
            st.write(msg['parts'][0]['text'] if 'parts' in msg and 'text' in msg['parts'][0] else '')

    # 사용자 입력 처리
    if prompt := st.chat_input("챗봇과 대화하기:"):
        # 사용자 메시지 추가
        st.session_state.messages["gemini"].append({"role": "user", "parts": [{"text": prompt}]})
        with st.chat_message('human'):
            st.write(prompt)

        # 프롬프트 엔지니어링 적용
        enhanced_prompt = gemini_prompt(prompt)

        # 모델 호출 및 응답 처리
        try:
            genai.configure(api_key=st.session_state.gemini_api_key)
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_k": top_k,
                "top_p": top_p
            }
            model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
            chat = model.start_chat(history=st.session_state.messages["gemini"])
            response = chat.send_message(enhanced_prompt, stream=True)

            with st.chat_message("ai"):
                placeholder = st.empty()
                
            text = stream_display(response, placeholder)
            if not text:
                if (content := response.parts) is not None:
                    text = "Wait for function calling response..."
                    placeholder.write(text + "▌")
                    response = chat.send_message(content, stream=True)
                    text = stream_display(response, placeholder)
            placeholder.write(text)

            # 응답 메시지 표시 및 저장
            st.session_state.messages["gemini"].append({"role": "model", "parts": [{"text": text}]})
        except Exception as e:
            st.error(f"Gemini API 요청 중 오류가 발생했습니다: {str(e)}")
