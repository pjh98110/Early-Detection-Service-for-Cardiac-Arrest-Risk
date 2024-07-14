import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.colored_header import colored_header

# 페이지 구성 설정
st.set_page_config(layout="wide")

# 세션 상태 초기화
if "page" not in st.session_state:
    st.session_state.page = "CPR"


# 타이틀
colored_header(
    label= '심정지 발생 위험 조기 진단 서비스👨‍⚕️',
    description=None,
    color_name="green-70",
)


st.title("서울시 제세동기 및 응급실 위치 정보")
st.write("화면이 보이지 않으면 이 링크로 접속해 주세요. https://public.tableau.com/app/profile/jonghyeon.park/viz/_17209678049430/sheet2?publish=yes")

# Tableau Public에서 대시보드 임베딩
# 여기에 Tableau Public 대시보드 URL을 입력합니다.
tableau_public_url = "https://public.tableau.com/app/profile/jonghyeon.park/viz/_17209678049430/sheet2?publish=yes"
tableau_embed_code = f"""
<iframe src="{tableau_public_url}" width="100%" height="1000"></iframe>
"""
st.markdown(tableau_embed_code, unsafe_allow_html=True)

