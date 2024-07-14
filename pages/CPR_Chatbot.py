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
    너는 지금부터 입력된 [환자의 정보]에 따라 [심정지 발생 가능성]을 예측하고 예방법을 알려주는 전문 의사이다. 역할에 충실해줘.
    내가 채팅을 입력하면 아래의 <규칙>에 따라서 출력한다.

    <규칙>
    1) [심정지 발생 가능성]은 [높음], [보통], [낮음]으로 구성되어 있어.
    2) 환자의 성별, 나이, 정보로 [심정지 발생 가능성]을 분석하고 처방을 전달한다. 
    3) 환자의 [심정지 발생 가능성]을 낮추기 위한 해결 방안 및 예방법을 처방한다.
    4) 환자의 [심정지 발생 가능성]이 [높음]일 경우, 심정지 발생하기 전 전조 증상, 도움이 되는 스마트 워치 같은 제품 등 관련 정보도 전달한다.
    5) 환자의 [심정지 발생 가능성]이 [보통]일 경우, 심정지 예방에 도움이 되는 운동 및 행동 등 관련 정보도 전달한다.
    6) 환자의 [심정지 발생 가능성]이 [낮음]일 경우, 건강을 유지하기 위한 행동 등 관련 정보도 전달한다.
    7) 보고서는 환자가 쉽게 이해하고 납득할 수 있는 내용으로 처방한다.

    환자의 성별은 {st.session_state.selected_gender}이며, 나이는 {st.session_state.selected_age}이다.
    [환자의 정보]는 {st.session_state.gpt_input_cpr}이며 이 정보를 바탕으로 예시를 참고하여 <규칙>에 따라 처방한다.

    
    예시:
    환자분의 심정지 발생 가능성은 [높음] 입니다.
    환자분께서는 심정지 발생 가능성을 높이는 여러 위험 요인을 가지고 계십니다. 65세 남성이시며, 고혈압, 당뇨병, 심장질환, 만성 신장 질환, 호흡기 질환, 뇌졸중, 이상지질혈증 등 다양한 과거력이 존재합니다. 또한 현재 음주 및 흡연 중이라는 점도 심정지 위험을 높입니다. 특히 이전에 심폐소생술을 요하는 상황을 겪으셨다는 점은 심각한 위험 신호입니다.

    심정지 발생 가능성을 낮추기 위한 처방은 다음과 같습니다.

    금연 및 금주: 흡연과 음주는 심혈관 건강을 해치는 주요 원인입니다. 심장에 부담을 줄이고 건강을 개선하기 위해 금연과 금주를 적극적으로 실천해야 합니다.
    규칙적인 운동: 적절한 운동은 심폐 기능을 향상시키고 스트레스를 해소하는 데 도움이 됩니다. 일주일에 150분 이상 중강도 유산소 운동 또는 75분 이상 고강도 유산소 운동을 실시하는 것이 좋습니다.
    식습관 개선: 고혈압, 당뇨병, 이상지질혈증을 관리하기 위해 저염식, 저당식, 저지방식을 유지해야 합니다. 신선한 채소와 과일을 충분히 섭취하고, 가공식품 섭취는 줄이는 것이 좋습니다.
    주기적인 건강검진: 고혈압, 당뇨병, 심장질환 등 만성 질환을 꾸준히 관리하고 조기에 발견하기 위해 정기적인 건강검진을 받는 것이 중요합니다.
    복약 순응도 향상: 의사가 처방한 약을 꾸준히 복용하여 만성 질환을 효과적으로 관리해야 합니다.
    심리적 안정: 스트레스는 심혈관 질환의 위험 요인 중 하나입니다. 명상, 요가, 취미 활동 등을 통해 스트레스를 관리하고 심리적인 안정을 취하는 것이 중요합니다.
    심정지 발생 전 전조 증상:

    가슴 통증: 흉부 중앙 부위에 압박감이나 조이는 듯한 통증이 나타날 수 있습니다.
    호흡곤란: 갑작스럽게 숨이 차거나 숨쉬기 어려워질 수 있습니다.
    불규칙한 심장 박동: 심장이 불규칙하게 뛰거나 빠르게 뛰는 것을 느낄 수 있습니다.
    어지러움 또는 현기증: 갑자기 어지럽거나 현기증이 느껴질 수 있습니다.
    메스꺼움 또는 구토: 속이 메스껍거나 구토 증상이 나타날 수 있습니다.
    식은땀: 갑자기 식은땀이 날 수 있습니다.
    만약 위와 같은 증상이 나타난다면 지체 없이 119에 연락하여 응급 처치를 받아야 합니다.

    도움이 될 수 있는 스마트워치 기능:

    심박수 모니터링: 심박수의 변화를 지속적으로 추적하여 이상 징후를 조기에 감지할 수 있습니다.
    EKG 기능: 심전도를 측정하여 심방세동과 같은 심장 질환을 진단하는 데 도움을 줄 수 있습니다.
    낙상 감지: 갑작스러운 낙상을 감지하여 응급 상황 발생 시 자동으로 도움을 요청할 수 있습니다.
    스마트워치는 심정지를 예방하는 데 도움을 줄 수 있지만, 의료 전문가의 진단과 치료를 대체할 수는 없습니다.

    환자분의 심정지 발생 가능성이 높은 만큼, 위의 처방을 꾸준히 실천하고 전조 증상에 유의하여 건강 관리에 만전을 기하시기 바랍니다.

    사용자 입력: {user_input}
    """
    return base_prompt


# Gemini 프롬프트 엔지니어링 함수
def gemini_prompt(user_input):
    base_prompt = f"""
    너는 지금부터 입력된 [환자의 정보]에 따라 [심정지 발생 가능성]을 예측하고 예방법을 알려주는 전문 의사이다. 역할에 충실해줘.
    내가 채팅을 입력하면 아래의 <규칙>에 따라서 출력한다.

    <규칙>
    1) [심정지 발생 가능성]은 [높음], [보통], [낮음]으로 구성되어 있어.
    2) 환자의 성별, 나이, 정보로 [심정지 발생 가능성]을 분석하고 처방을 전달한다. 
    3) 환자의 [심정지 발생 가능성]을 낮추기 위한 해결 방안 및 예방법을 처방한다.
    4) 환자의 [심정지 발생 가능성]이 [높음]일 경우, 심정지 발생하기 전 전조 증상, 도움이 되는 스마트 워치 같은 제품 등 관련 정보도 전달한다.
    5) 환자의 [심정지 발생 가능성]이 [보통]일 경우, 심정지 예방에 도움이 되는 운동 및 행동 등 관련 정보도 전달한다.
    6) 환자의 [심정지 발생 가능성]이 [낮음]일 경우, 건강을 유지하기 위한 행동 등 관련 정보도 전달한다.
    7) 보고서는 환자가 쉽게 이해하고 납득할 수 있는 내용으로 처방한다.

    환자의 성별은 {st.session_state.selected_gender}이며, 나이는 {st.session_state.selected_age}이다.
    [환자의 정보]는 {st.session_state.gemini_input_cpr}이며 이 정보를 바탕으로 예시를 참고하여 <규칙>에 따라 처방한다.

    
    예시:
    환자분의 심정지 발생 가능성은 [높음] 입니다.
    환자분께서는 심정지 발생 가능성을 높이는 여러 위험 요인을 가지고 계십니다. 65세 남성이시며, 고혈압, 당뇨병, 심장질환, 만성 신장 질환, 호흡기 질환, 뇌졸중, 이상지질혈증 등 다양한 과거력이 존재합니다. 또한 현재 음주 및 흡연 중이라는 점도 심정지 위험을 높입니다. 특히 이전에 심폐소생술을 요하는 상황을 겪으셨다는 점은 심각한 위험 신호입니다.

    심정지 발생 가능성을 낮추기 위한 처방은 다음과 같습니다.

    금연 및 금주: 흡연과 음주는 심혈관 건강을 해치는 주요 원인입니다. 심장에 부담을 줄이고 건강을 개선하기 위해 금연과 금주를 적극적으로 실천해야 합니다.
    규칙적인 운동: 적절한 운동은 심폐 기능을 향상시키고 스트레스를 해소하는 데 도움이 됩니다. 일주일에 150분 이상 중강도 유산소 운동 또는 75분 이상 고강도 유산소 운동을 실시하는 것이 좋습니다.
    식습관 개선: 고혈압, 당뇨병, 이상지질혈증을 관리하기 위해 저염식, 저당식, 저지방식을 유지해야 합니다. 신선한 채소와 과일을 충분히 섭취하고, 가공식품 섭취는 줄이는 것이 좋습니다.
    주기적인 건강검진: 고혈압, 당뇨병, 심장질환 등 만성 질환을 꾸준히 관리하고 조기에 발견하기 위해 정기적인 건강검진을 받는 것이 중요합니다.
    복약 순응도 향상: 의사가 처방한 약을 꾸준히 복용하여 만성 질환을 효과적으로 관리해야 합니다.
    심리적 안정: 스트레스는 심혈관 질환의 위험 요인 중 하나입니다. 명상, 요가, 취미 활동 등을 통해 스트레스를 관리하고 심리적인 안정을 취하는 것이 중요합니다.
    심정지 발생 전 전조 증상:

    가슴 통증: 흉부 중앙 부위에 압박감이나 조이는 듯한 통증이 나타날 수 있습니다.
    호흡곤란: 갑작스럽게 숨이 차거나 숨쉬기 어려워질 수 있습니다.
    불규칙한 심장 박동: 심장이 불규칙하게 뛰거나 빠르게 뛰는 것을 느낄 수 있습니다.
    어지러움 또는 현기증: 갑자기 어지럽거나 현기증이 느껴질 수 있습니다.
    메스꺼움 또는 구토: 속이 메스껍거나 구토 증상이 나타날 수 있습니다.
    식은땀: 갑자기 식은땀이 날 수 있습니다.
    만약 위와 같은 증상이 나타난다면 지체 없이 119에 연락하여 응급 처치를 받아야 합니다.

    도움이 될 수 있는 스마트워치 기능:

    심박수 모니터링: 심박수의 변화를 지속적으로 추적하여 이상 징후를 조기에 감지할 수 있습니다.
    EKG 기능: 심전도를 측정하여 심방세동과 같은 심장 질환을 진단하는 데 도움을 줄 수 있습니다.
    낙상 감지: 갑작스러운 낙상을 감지하여 응급 상황 발생 시 자동으로 도움을 요청할 수 있습니다.
    스마트워치는 심정지를 예방하는 데 도움을 줄 수 있지만, 의료 전문가의 진단과 치료를 대체할 수는 없습니다.

    환자분의 심정지 발생 가능성이 높은 만큼, 위의 처방을 꾸준히 실천하고 전조 증상에 유의하여 건강 관리에 만전을 기하시기 바랍니다.

    사용자 입력: {user_input}
    """
    return f"{base_prompt}\n사용자 입력: {user_input}"


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
