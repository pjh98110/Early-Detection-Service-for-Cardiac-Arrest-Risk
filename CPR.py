import streamlit as st
import pandas as pd
import numpy as np
import random
import os
from streamlit_extras.switch_page_button import switch_page
from st_pages import Page, show_pages
from streamlit_extras.colored_header import colored_header
from datetime import datetime, timedelta
import requests

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import shap
import pickle

# Streamlit의 경우 로컬 환경에서 실행할 경우 터미널 --> (폴더 경로)Streamlit run CPR.py로 실행 / 로컬 환경과 스트리밋 웹앱 환경에서 기능의 차이가 일부 있을 수 있음
# 파일 경로를 잘못 설정할 경우 오류가 발생하고 실행이 불가능하므로 파일 경로 수정 필수
# 데이터 파일의 경우 배포된 웹앱 깃허브에서 다운로드 가능함

# 페이지 구성 설정
st.set_page_config(layout="wide")


show_pages(
    [
        Page("CPR.py", "심정지 발생 시 생존여부 시뮬레이션", "👨‍⚕️"),
        Page("pages/CARE_Chatbot.py", "심정지발생 예방 챗봇", "💔"),
        Page("pages/CPR_Chatbot.py", "심폐소생술 교육 챗봇", "📝"),
        Page("pages/Tableau.py", "Tableau", "🖥️"),
    ]
)

if "page" not in st.session_state:
    st.session_state.page = "CPR"

DATA_PATH = "./data/"

@st.cache
def load_data():
    df1 = pd.read_csv(f'{DATA_PATH}급성심장정지조사_2018_1.csv')
    df2 = pd.read_csv(f'{DATA_PATH}급성심장정지조사_2018_2.csv')
    df18 = pd.concat([df1, df2], axis=1)
    return df18

# 데이터 불러오기
df18 = load_data().copy()

def reset_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

reset_seeds(42)

# 한글 폰트 설정 함수
def set_korean_font():
    font_path = f"{DATA_PATH}NanumGothic.ttf"  # 폰트 파일 경로

    from matplotlib import font_manager, rc
    font_manager.fontManager.addfont(font_path)
    rc('font', family='NanumGothic')

# 한글 폰트 설정 적용
set_korean_font()


# 웹앱 스타일 변경하기
# 사용자 정의 CSS 적용

# E2F2FD 하늘색

def apply_custom_styles():
    st.markdown("""
        <style>
            /* 전체 배경색 변경 */
            .stApp {
                background-color: #FFFFF1;  /* 흰색 */
            }
            
            /* 헤더 색상 변경 */
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
                color: #000000;  /* 검정 */
            }
            
            /* 버튼 스타일 변경 */
            .stButton > button {
                background-color: #B2EBF2;  /* 하늘색 */
                color: #000000;  /* 검정 */
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 16px;
            }
            
            .stButton > button:hover {
                background-color: #B2EBF2;  /* 하늘색 */
            }
            
            /* 인풋 텍스트 스타일 */
            .stTextInput > div > input {
                border: 2px solid #007BFF;  /* 파란색 */
                border-radius: 8px;
                padding: 10px;
            }
            
            /* 채팅 메시지 스타일 */
            .stChatMessage {
                background-color: #FFFFFF;  /* 흰색 */
                border-radius: 10px;
                margin-bottom: 10px;
                padding: 10px;
                font-size: 14px;
            }
            
            .stChatMessage.user {
                border-left: 4px solid #007BFF;  /* 사용자 메시지 왼쪽 파란색 바 */
            }
            
            .stChatMessage.ai {
                border-left: 4px solid #FFA500;  /* 모델 메시지 왼쪽 주황색 바 */
            }
        </style>
    """, unsafe_allow_html=True)

# 스타일 적용
apply_custom_styles()


# 세션 변수에 저장
if 'type_of_case' not in st.session_state:
    st.session_state.type_of_case = None

if 'selected_district' not in st.session_state:
    st.session_state.selected_district = "서울특별시"

if 'selected_gender' not in st.session_state:
    st.session_state.selected_gender = "남자"

if 'selected_age' not in st.session_state:
    st.session_state.selected_age = 20

if 'questions' not in st.session_state:
    st.session_state.questions = None

if 'gpt_input_care' not in st.session_state:
    st.session_state.gpt_input_care = None

if 'gemini_input_care' not in st.session_state:
    st.session_state.gemini_input_care = None  

if 'gpt_input_cpr' not in st.session_state:
    st.session_state.gpt_input_cpr = None

if 'gemini_input_cpr' not in st.session_state:
    st.session_state.gemini_input_cpr = None   

if 'selected_survey' not in st.session_state:
    st.session_state.selected_survey = []




# 타이틀
colored_header(
    label= '심정지 발생 시 생존여부 시뮬레이션👨‍⚕️',
    description=None,
    color_name="green-70",
)

# [사이드바]
st.sidebar.markdown(f"""
            <span style='font-size: 20px;'>
            <div style=" color: #000000;">
                <strong>사용자 정보 입력</strong>
            </div>
            """, unsafe_allow_html=True)


# 사이드바에서 지역 선택
selected_district = st.sidebar.selectbox(
    "(1) 당신의 지역을 선택하세요:",
    ('서울특별시', '경기도', '부산광역시', '인천광역시', '충청북도', '충청남도', 
     '세종특별자치시', '대전광역시', '전북특별자치도', '전라남도', '광주광역시', 
     '경상북도', '경상남도', '대구광역시', '울산광역시', '강원특별자치도', '제주특별자치도'), key="side1")
st.session_state.selected_district = selected_district

# 사이드바에서 성별 선택
selected_gender = st.sidebar.selectbox("(2) 당신의 성별을 선택하세요:", ('남성', '여성'), key="side2")
st.session_state.selected_gender = selected_gender

# 사이드바에서 나이 선택
selected_age = st.sidebar.number_input(
    "(3) 당신의 연령(만 나이)을 입력하세요:", placeholder = "나이 __살",
    min_value=1, max_value=125, value=20, key="side3")
st.session_state.selected_age = selected_age



selected_survey = st.selectbox(
    "궁금한 검사 결과를 선택하세요.",
    options=["심정지 발생 시 생존여부 시뮬레이션", "GPT를 통한 심정지위험 예방", "Gemini를 통한 심정지위험 예방", "GPT를 통한 심폐소생술 교육", "Gemini를 통한 심폐소생술 교육"],
    placeholder="하나를 선택하세요.",
    help="선택한 검사에 따라 다른 분석 결과를 제공합니다."
)

st.session_state.selected_survey = selected_survey


if selected_survey == "심정지 발생 시 생존여부 시뮬레이션":

    goldentime = int(st.number_input("심폐소생술을 시작한 시간을 입력하세요.", min_value=0, max_value=60, value=0, key="q0"))  
    questions = {
        "보험종류" : st.selectbox("1.보험종류를 선택하세요", (df18["보험종류_LABEL"].value_counts().keys()), key="q1"),
        "병원 도착 전 심폐소생술 시행 여부" : st.selectbox("2.병원 도착 전 심폐소생술 시행 여부", (df18["병원 도착 전 심폐소생술 시행 여부_LABEL"].value_counts().keys()),  key="q2"),
        "병원 도착 전 자발순환 회복 여부" : st.selectbox("3.병원 도착 전 자발순환 회복 여부", (df18["병원 도착 전 자발순환 회복 여부_LABEL"].value_counts().keys()), key="q3"),
        "병원 도착 전 급성심장정지 목격 여부" : st.selectbox("4.병원 도착 전 급성심장정지 목격 여부", (df18["병원 도착 전 급성심장정지 목격 여부_LABEL"].value_counts().keys()), key="q4"),
        "일반인 심폐소생술 시행여부" : st.selectbox("5.일반인 심폐소생술 시행여부", (df18["일반인 심폐소생술 시행여부_LABEL"].value_counts().keys()), key="q5"),
        "응급실 심폐소생술 시행여부" : st.selectbox("6.응급실 심폐소생술 시행여부", (df18["응급실 심폐소생술 시행여부_LABEL"].value_counts().keys()), key="q6"),
        "응급실 심폐소생술 후 자발순환 회복 여부" : st.selectbox("7.응급실 심폐소생술 후 자발순환 회복 여부", (df18["응급실 심폐소생술 후 자발순환 회복 여부_LABEL"].value_counts().keys()), key="q7"),
        "응급실 제세동 실시 여부" : st.selectbox("8.응급실 제세동 실시 여부", (df18["응급실 제세동 실시 여부_LABEL"].value_counts().keys()), key="q8"),
        "과거력_고혈압" : st.selectbox("9.과거력_고혈압", (df18["과거력_고혈압_LABEL"].value_counts().keys()), key="q9"),
        "과거력_당뇨병" : st.selectbox("10.과거력_당뇨병", (df18["과거력_당뇨병_LABEL"].value_counts().keys()), key="q10"),
        "과거력_심장질환" : st.selectbox("11.과거력_심장질환", (df18["과거력_심장질환_LABEL"].value_counts().keys()), key="q11"),
        "과거력_만성신장질환" : st.selectbox("12.과거력_만성신장질환", (df18["과거력_만성신장질환_LABEL"].value_counts().keys()), key="q12"),
        "과거력_호흡기질환" : st.selectbox("13.과거력_호흡기질환", (df18["과거력_호흡기질환_LABEL"].value_counts().keys()), key="q13"),
        "과거력_뇌졸중" : st.selectbox("14.과거력_뇌졸중", (df18["과거력_뇌졸중_LABEL"].value_counts().keys()), key="q14"),
        "과거력_이상지질혈증" : st.selectbox("15.과거력_이상지질혈증", (df18["과거력_이상지질혈증_LABEL"].value_counts().keys()), key="q15"),
        "음주력" : st.selectbox("16.음주력", (df18["음주력_LABEL"].value_counts().keys()), key="q16"),
        "흡연력" : st.selectbox("17.흡연력", (df18["흡연력_LABEL"].value_counts().keys()), key="q17"),
    }
                
        # 스트리밋 클라우드 서버의 데이터 크기 제한으로 인해, 현재 웹앱에서 모델을 전체적으로 
        # 실행하는 것이 불가능합니다. 이에 따라, 웹앱에서는 모델의 결과를 예시로 보여주는 샘플데이터(25mb 이하)로 분석을 제공하며, 
        # 실제로 정확한 모델 결과를 얻고자 한다면 제출된 모델의 코드를 자신의 로컬 환경에서 실행해야 합니다.
        # 전체적인 모델은 제출한 코드에 있으며, 여기에는 샘플데이터 분석 결과만 있습니다.
    

    # 검사결과 버튼을 누를 경우
    if st.button("검사결과"):
        col1, col2 = st.columns(2)

        with col1:
            # 골든타임에 따라 이미지 표시
            if 0 <= goldentime <= 4:
                st.image('./data/안전.png', width=200)
                st.write("심폐소생술을 시작한 결과, 환자의 골든타임은 **안전**입니다.")
            elif 4 < goldentime <= 6:
                st.image('./data/주의.png', width=200)
                st.write("심폐소생술을 시작한 결과, 환자의 골든타임은 **주의**입니다.")
            elif 6 < goldentime <= 10:
                st.image('./data/위험.png', width=200)
                st.write("심폐소생술을 시작한 결과, 환자의 골든타임은 **위험**입니다.")
            elif goldentime > 10:
                st.image('./data/고위험.png', width=200)
                st.write("심폐소생술을 시작한 결과, 환자의 골든타임은 **고위험**입니다.")

            df18['사망여부'] = df18.apply(
                lambda row: 0 if (
                    row['응급실 진료결과'] == '사망' or 
                    row['입원 후 결과'] == '사망' or 
                    row['2차 이송병원 응급실 진료결과'] == '사망' or 
                    row['2차 이송병원 입원 후 결과'] == '사망'
                ) else 1,
                axis=1
                )


        with col2:
            st.markdown(f"당신의 지역은 [{selected_district}]이며, 성별은 [{selected_gender}], 나이는 [{selected_age}살]입니다.")
            st.markdown(f"현재 상태를 유지할 시, {'생존' if df18['사망여부'].sample(1).values[0] == 1 else '사망'}할 확률이 높습니다.")
            st.markdown(f"추가 정보를 원하면, 심정지위험 진단 챗봇 버튼을 클릭하세요. 챗봇 페이지로 이동합니다.")



    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #B2EBF2;
            width: 100%; /
            display: inline-block;
            margin: 0; /
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def page1():
        want_to_CARE_Chatbot = st.button("심정지발생 예방 챗봇")
        if want_to_CARE_Chatbot:
            st.session_state.type_of_case = "CARE_Chatbot"
            switch_page("CARE_Chatbot")
            
    def page2():
        want_to_CPR_Chatbot = st.button("심폐소생술 교육 챗봇")
        if want_to_CPR_Chatbot:
            st.session_state.type_of_case = "CPR_Chatbot"
            switch_page("CPR_Chatbot")            

    def page3():
        want_to_Tableau = st.button("Tableau")
        if want_to_Tableau:
            st.session_state.type_of_case = "Tableau"
            switch_page("Tableau")

    col1, col2, col3 = st.columns(3)
    with col1:
        page1()
    with col2:
        page2()
    with col3:
        page3()


if selected_survey == "GPT를 통한 심정지위험 예방":


    gpt_input_care = {
        "과거력_고혈압" : st.selectbox("1.과거력_고혈압", (df18["과거력_고혈압_LABEL"].value_counts().keys()), key="q9"),
        "과거력_당뇨병" : st.selectbox("2.과거력_당뇨병", (df18["과거력_당뇨병_LABEL"].value_counts().keys()), key="q10"),
        "과거력_심장질환" : st.selectbox("3.과거력_심장질환", (df18["과거력_심장질환_LABEL"].value_counts().keys()), key="q11"),
        "과거력_만성신장질환" : st.selectbox("4.과거력_만성신장질환", (df18["과거력_만성신장질환_LABEL"].value_counts().keys()), key="q12"),
        "과거력_호흡기질환" : st.selectbox("5.과거력_호흡기질환", (df18["과거력_호흡기질환_LABEL"].value_counts().keys()), key="q13"),
        "과거력_뇌졸중" : st.selectbox("6.과거력_뇌졸중", (df18["과거력_뇌졸중_LABEL"].value_counts().keys()), key="q14"),
        "과거력_이상지질혈증" : st.selectbox("7.과거력_이상지질혈증", (df18["과거력_이상지질혈증_LABEL"].value_counts().keys()), key="q15"),
        "음주력" : st.selectbox("8.음주력", (df18["음주력_LABEL"].value_counts().keys()), key="q16"),
        "흡연력" : st.selectbox("9.흡연력", (df18["흡연력_LABEL"].value_counts().keys()), key="q17"),
    }
    st.session_state.gpt_input_care = gpt_input_care

    # 검사결과 버튼을 누를 경우
    if st.button("검사결과"):
    
        st.markdown(f"당신의 지역은 [{selected_district}]이며, 성별은 [{selected_gender}], 나이는 [{selected_age}살]입니다.")
        st.markdown(f"심정지위험 진단 챗봇 버튼을 클릭하세요. 입력된 정보로 챗봇이 맞춤형 분석 결과를 제공합니다.")


    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #B2EBF2;
            width: 100%; /
            display: inline-block;
            margin: 0; /
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    def page1():
        want_to_CARE_Chatbot = st.button("심정지발생 예방 챗봇")
        if want_to_CARE_Chatbot:
            st.session_state.type_of_case = "CARE_Chatbot"
            switch_page("CARE_Chatbot")
            
    def page2():
        want_to_CPR_Chatbot = st.button("심폐소생술 교육 챗봇")
        if want_to_CPR_Chatbot:
            st.session_state.type_of_case = "CPR_Chatbot"
            switch_page("CPR_Chatbot")            

    def page3():
        want_to_Tableau = st.button("Tableau")
        if want_to_Tableau:
            st.session_state.type_of_case = "Tableau"
            switch_page("Tableau")


    col1, col2, col3 = st.columns(3)
    with col1:
        page1()
    with col2:
        page2()
    with col3:
        page3()


if selected_survey == "Gemini를 통한 심정지위험 예방":


    gemini_input_care = {
        "과거력_고혈압" : st.selectbox("1.과거력_고혈압", (df18["과거력_고혈압_LABEL"].value_counts().keys()), key="q9"),
        "과거력_당뇨병" : st.selectbox("2.과거력_당뇨병", (df18["과거력_당뇨병_LABEL"].value_counts().keys()), key="q10"),
        "과거력_심장질환" : st.selectbox("3.과거력_심장질환", (df18["과거력_심장질환_LABEL"].value_counts().keys()), key="q11"),
        "과거력_만성신장질환" : st.selectbox("4.과거력_만성신장질환", (df18["과거력_만성신장질환_LABEL"].value_counts().keys()), key="q12"),
        "과거력_호흡기질환" : st.selectbox("5.과거력_호흡기질환", (df18["과거력_호흡기질환_LABEL"].value_counts().keys()), key="q13"),
        "과거력_뇌졸중" : st.selectbox("6.과거력_뇌졸중", (df18["과거력_뇌졸중_LABEL"].value_counts().keys()), key="q14"),
        "과거력_이상지질혈증" : st.selectbox("7.과거력_이상지질혈증", (df18["과거력_이상지질혈증_LABEL"].value_counts().keys()), key="q15"),
        "음주력" : st.selectbox("8.음주력", (df18["음주력_LABEL"].value_counts().keys()), key="q16"),
        "흡연력" : st.selectbox("9.흡연력", (df18["흡연력_LABEL"].value_counts().keys()), key="q17"),
    }
    st.session_state.gemini_input_care = gemini_input_care


    # 검사결과 버튼을 누를 경우
    if st.button("검사결과"):

        st.markdown(f"당신의 지역은 [{selected_district}]이며, 성별은 [{selected_gender}], 나이는 [{selected_age}살]입니다.")
        st.markdown(f"심정지위험 진단 챗봇 버튼을 클릭하세요. 입력된 정보로 챗봇이 맞춤형 분석 결과를 제공합니다.")


    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #B2EBF2;
            width: 100%; /
            display: inline-block;
            margin: 0; /
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    def page1():
        want_to_CARE_Chatbot = st.button("심정지발생 예방 챗봇")
        if want_to_CARE_Chatbot:
            st.session_state.type_of_case = "CARE_Chatbot"
            switch_page("CARE_Chatbot")
            
    def page2():
        want_to_CPR_Chatbot = st.button("심폐소생술 교육 챗봇")
        if want_to_CPR_Chatbot:
            st.session_state.type_of_case = "CPR_Chatbot"
            switch_page("CPR_Chatbot")            

    def page3():
        want_to_Tableau = st.button("Tableau")
        if want_to_Tableau:
            st.session_state.type_of_case = "Tableau"
            switch_page("Tableau")


    col1, col2, col3 = st.columns(3)
    with col1:
        page1()
    with col2:
        page2()
    with col3:
        page3()



if selected_survey == "GPT를 통한 심폐소생술 교육":


    gpt_input_cpr = {
        "심정지 환자를 발견한 시간(분)" : st.number_input("1.심정지 환자를 발견한 시간(분)", value=0, min_value=0, max_value=60, placeholder="__분", key="q1"),
        "심폐소생술을 시작한 시간(분)" : st.number_input("2.심폐소생술을 시작한 시간(분)", value=0, min_value=0, max_value=60, placeholder="__분", key="q2"),
        "주변에 도움을 줄 수 있는 사람의 수" : st.number_input("3.주변에 도움을 줄 수 있는 사람의 수", value=0, min_value=0, max_value=100, placeholder="__명", key="q3"),
        "자동심장충격기(AED) 사용 가능 여부" : st.selectbox("4.자동심장충격기(AED) 사용 가능 여부", ("예", "아니오"), key="q4"),
        "사용자의 CPR(심폐소생술) 경험 여부" : st.selectbox("5.사용자의 CPR(심폐소생술) 경험 여부", ("예", "아니오"), key="q5"),
        "심정지 환자의 나이" : st.selectbox("6.심정지 환자의 나이", ("성인", "어린이", "영아"), key="q6"),
        "심정지 환자의 성별" : st.selectbox("7.심정지 환자의 성별", ("남성", "여성"), key="q7"),
    }
    st.session_state.gpt_input_cpr = gpt_input_cpr

    # 검사결과 버튼을 누를 경우
    if st.button("검사결과"):
    
        st.markdown(f"당신의 지역은 [{selected_district}]이며, 성별은 [{selected_gender}], 나이는 [{selected_age}살]입니다.")
        st.markdown(f"심폐소생술 교육 챗봇 버튼을 클릭하세요. 입력된 정보로 챗봇이 맞춤형 심폐소생술 가이드라인을 제공합니다.")


    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #B2EBF2;
            width: 100%; /
            display: inline-block;
            margin: 0; /
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    def page1():
        want_to_CARE_Chatbot = st.button("심정지발생 예방 챗봇")
        if want_to_CARE_Chatbot:
            st.session_state.type_of_case = "CARE_Chatbot"
            switch_page("CARE_Chatbot")
            
    def page2():
        want_to_CPR_Chatbot = st.button("심폐소생술 교육 챗봇")
        if want_to_CPR_Chatbot:
            st.session_state.type_of_case = "CPR_Chatbot"
            switch_page("CPR_Chatbot")            

    def page3():
        want_to_Tableau = st.button("Tableau")
        if want_to_Tableau:
            st.session_state.type_of_case = "Tableau"
            switch_page("Tableau")

    col1, col2, col3 = st.columns(3)
    with col1:
        page1()
    with col2:
        page2()
    with col3:
        page3()


if selected_survey == "Gemini를 통한 심폐소생술 교육":


    gemini_input_cpr = {
        "심정지 환자를 발견한 시간(분)" : st.number_input("1.심정지 환자를 발견한 시간(분)", value=0, min_value=0, max_value=60, placeholder="__분", key="q1"),
        "심폐소생술을 시작한 시간(분)" : st.number_input("2.심폐소생술을 시작한 시간(분)", value=0, min_value=0, max_value=60, placeholder="__분", key="q2"),
        "주변에 도움을 줄 수 있는 사람의 수" : st.number_input("3.주변에 도움을 줄 수 있는 사람의 수", value=0, min_value=0, max_value=100, placeholder="__명", key="q3"),
        "자동심장충격기(AED) 사용 가능 여부" : st.selectbox("4.자동심장충격기(AED) 사용 가능 여부", ("예", "아니오"), key="q4"),
        "사용자의 CPR(심폐소생술) 경험 여부" : st.selectbox("5.사용자의 CPR(심폐소생술) 경험 여부", ("예", "아니오"), key="q5"),
        "심정지 환자의 나이" : st.selectbox("6.심정지 환자의 나이", ("성인", "어린이", "영아"), key="q6"),
        "심정지 환자의 성별" : st.selectbox("7.심정지 환자의 성별", ("남성", "여성"), key="q7"),
    }
    st.session_state.gemini_input_cpr = gemini_input_cpr


    # 검사결과 버튼을 누를 경우
    if st.button("검사결과"):

        st.markdown(f"당신의 지역은 [{selected_district}]이며, 성별은 [{selected_gender}], 나이는 [{selected_age}살]입니다.")
        st.markdown(f"심폐소생술 교육 챗봇 버튼을 클릭하세요. 입력된 정보로 챗봇이 맞춤형 심폐소생술 가이드라인을 제공합니다.")

    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #B2EBF2;
            width: 100%; /
            display: inline-block;
            margin: 0; /
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    def page1():
        want_to_CARE_Chatbot = st.button("심정지발생 예방 챗봇")
        if want_to_CARE_Chatbot:
            st.session_state.type_of_case = "CARE_Chatbot"
            switch_page("CARE_Chatbot")
            
    def page2():
        want_to_CPR_Chatbot = st.button("심폐소생술 교육 챗봇")
        if want_to_CPR_Chatbot:
            st.session_state.type_of_case = "CPR_Chatbot"
            switch_page("CPR_Chatbot")            

    def page3():
        want_to_Tableau = st.button("Tableau")
        if want_to_Tableau:
            st.session_state.type_of_case = "Tableau"
            switch_page("Tableau")


    col1, col2, col3 = st.columns(3)
    with col1:
        page1()
    with col2:
        page2()
    with col3:
        page3()
