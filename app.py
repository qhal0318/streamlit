import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os # 파일 경로 확인용

from detector import prepare_data, calculate_abuse_scores, get_blocklist

st.set_page_config(
    layout="wide",
    page_title="광고 어뷰징 탐지 센서",
    page_icon="1-794df7f8.ico"
)

# [수정 1] 경로는 무조건 '상대 경로'로! (깃허브 리포지토리 기준)
# 데이터 파일들이 깃허브의 'data' 폴더 안에 들어있다고 가정할게.
FILE_PATH_RWD = "final_ads_rwd.csv"       
FILE_PATH_LIST = "sample_ads_list.csv"      
FILE_PATH_IP = "ip_hostname_2.json"        

DEFAULT_MAPPING = {
    'dvc_idx': 'dvc_idx',       
    'user_ip': 'user_ip',       
    'ads_idx_list': 'ads_idx'   
}

# --- (제목 및 스타일 설정은 그대로 유지) ---
col1, col2 = st.columns([1, 5])
with col1:
    try: st.image("1-794df7f8.ico", width=100)
    except: pass
with col2:
    st.markdown("""
    <div style="text-align: left; margin-left: 5px;">
        <h1 style="font-size: 3rem; font-weight: bold; color: #FFFFFF; margin: 0; padding: 1rem 0; display: inline-block; vertical-align: middle;">
            광고 어뷰징 탐지 센서
        </h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- (설정값 DEFAULT_CONFIG 등은 그대로 유지) ---
# ... (중략) ...

models = load_models()

# --- [수정 2] 사이드바는 '장식용'으로만 남기고 기능 해제 ---
st.sidebar.title("⚙️ 탐지 설정")
with st.sidebar.expander("📂 파일 업로드 (자동 로드됨)", expanded=False):
    st.info("현재 서버에 저장된 데이터를 자동으로 불러옵니다.")
    # 파일 업로더는 보여주기만 하고 변수에 할당 안 함
    st.file_uploader("1. 원본 로그", disabled=True)
    st.file_uploader("2. 광고 정보", disabled=True)
    st.file_uploader("3. IP 정보", disabled=True)

# ... (민감도 설정 등은 그대로 유지) ...
sensitivity = st.sidebar.radio("탐지 민감도 프리셋", ('평균', '엄격', '완화'))
# ... (점수 조정 부분 그대로 유지) ...

# --- [수정 3] 파일 자동 읽기 로직 ---
@st.cache_data
def load_data_auto():
    # 파일이 실제로 있는지 확인
    if not os.path.exists(FILE_PATH_RWD): return None, None, None, f"파일 없음: {FILE_PATH_RWD}"
    
    try:
        rwd = pd.read_csv(FILE_PATH_RWD)
        lst = pd.read_csv(FILE_PATH_LIST)
        with open(FILE_PATH_IP, 'r', encoding='utf-8') as f:
            ip = json.load(f)
        return rwd, lst, ip, None
    except Exception as e:
        return None, None, None, str(e)

# 여기서 바로 데이터 로딩 시작!
rwd_df, list_df, ip_data, err = load_data_auto()

if err:
    st.error(f"❌ 데이터 로딩 실패: {err}")
    st.warning("깃허브에 데이터 파일이 정확한 경로에 올라갔는지 확인해주세요.")
    st.stop()

# 데이터가 로드되었으면 세션에 저장하고 바로 분석 시작
if rwd_df is not None:
    st.session_state.df_rwd = rwd_df
    st.session_state.df_list = list_df
    st.session_state.ip_cache = ip_data
    st.session_state.mapping = DEFAULT_MAPPING

    # --- 여기서부터 분석 결과 바로 출력 (기존 버튼 클릭 로직 제거) ---
    st.header("STEP 1: 데이터 자동 로드 완료")
    st.success("✅ 서버 데이터를 성공적으로 불러왔습니다.")
    
    st.markdown("---")
    st.header("STEP 2: 어뷰징 분석 결과")

    # 분석 로직 바로 실행
    with st.spinner('자동 분석 중...'):
        # ... (이후 분석 로직은 기존 코드와 동일하게 복붙) ...
        # (ads_rwd_info 복사부터 prepare_data, calculate_abuse_scores 등등)
        # ...
        
        # [중요] 기존 코드의 if st.button("🚀..."): 부분을 없애고
        # 들여쓰기를 당겨서 바로 실행되게 해야 해.
