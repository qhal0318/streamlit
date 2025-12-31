import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os

# detector.py 파일이 같은 폴더에 있어야 합니다.
from detector import prepare_data, calculate_abuse_scores, get_blocklist

# --------------------------------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="광고 어뷰징 탐지 센서",
    page_icon="1-794df7f8.ico"
)

# --------------------------------------------------------------------------
# 2. 파일 경로 및 설정 (사용자 수정 영역)
# --------------------------------------------------------------------------
# 깃허브에 올린 파일명과 정확히 일치해야 합니다. (app.py와 같은 폴더 기준)
FILE_PATH_RWD = "final_ads_rwd.csv"       
FILE_PATH_LIST = "sample_ads_list.csv"      
FILE_PATH_IP = "ip_hostname_2.json"        

# 컬럼 매핑 자동 설정
DEFAULT_MAPPING = {
    'dvc_idx': 'dvc_idx',       
    'user_ip': 'user_ip',       
    'ads_idx_list': 'ads_idx'   
}

# 기본 탐지 규칙 설정
DEFAULT_CONFIG = {
    'burst_attack': {'threshold_clicks': 15, 'score': 15, 'window_min': 5}, 
    'media_concentration': {'threshold_clicks': 20, 'threshold_mda': 2, 'score': 20}, 
    'abnormal_cvr': {'threshold_cvr': 0.90, 'threshold_clicks': 20, 'score': 45}, 
    'short_ctit': {'threshold_sec': 5, 'score': 15}, 
    'suspicious_early_hour': {'start_hour': 2, 'end_hour': 6, 'score': 10}, 
    'consistent_ctit': {'threshold_std': 3.0, 'threshold_clicks': 4, 'score': 40}, 
    'anomaly_model': {'threshold_clicks': 8, 'score': 45},
    'heavy_click_spam': {'threshold_clicks': 50, 'score': 20}, 
    'rapid_click': {'threshold_sec': 1.0, 'score': 10}, 
    'many_devices_per_ip': {'threshold_devices': 6, 'score': 25, 'carrier_ip_threshold': 10000}, 
    'many_ips_per_device': {'threshold_ips': 15, 'score': 25}, 
    'aws_ip': {'score': 25},
    'fraud_long_ctit': {'threshold_sec': 3600, 'score': 35},
    'suspicious_single_conv': {'score': 30},
    'ctit_anomaly_model': {'score': 35},
    'combo_stealth_bot': {'score': 30},
    'combo_focused_fraud': {'score': 35},
    'blocklist_method': 'percentile', 
    'blocklist_percentile': 0.95, 
    'absolute_score_threshold': 100
}

KOREAN_NAMES = {
    'burst_attack': '단기 클릭 폭주', 'media_concentration': '매체 집중', 
    'abnormal_cvr': '비정상적 전환율(CVR)', 'short_ctit': '짧은 전환 시간(CTIT)', 
    'suspicious_early_hour': '의심스러운 심야 활동', 'consistent_ctit': '일정한 전환 시간(CTIT)', 
    'anomaly_model': '이상 탐지 모델', 'heavy_click_spam': '과도한 클릭 (미전환)', 
    'rapid_click': '매우 빠른 클릭', 'many_devices_per_ip': '하나의 IP당 다수의 기기', 
    'many_ips_per_device': '하나의 기기당 다수의 IP', 'aws_ip': '서버 IP 사용 (AWS)', 
    'fraud_long_ctit': '비정상적으로 긴 CTIT', 'suspicious_single_conv': '의심스러운 단일 전환', 
    'ctit_anomaly_model': 'CTIT 패턴 모델', 'combo_stealth_bot': '콤보: 은신 봇', 
    'combo_focused_fraud': '콤보: 집중형 사기'
}

# --------------------------------------------------------------------------
# 3. 데이터 및 모델 로드 함수
# --------------------------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    try: models['anomaly_model'] = joblib.load('isolation_forest_model.joblib')
    except: models['anomaly_model'] = None
    try: models['ctit_anomaly_model'] = joblib.load('ctit_anomaly_model.joblib')
    except: models['ctit_anomaly_model'] = None
    return models

@st.cache_data
def load_data_automatically():
    """지정된 경로에서 파일을 자동으로 읽어옵니다."""
    # 파일 존재 여부 확인 (디버깅용)
    if not os.path.exists(FILE_PATH_RWD):
        return None, None, None, f"파일을 찾을 수 없습니다: {FILE_PATH_RWD}"
    
    try:
        rwd = pd.read_csv(FILE_PATH_RWD)
        lst = pd.read_csv(FILE_PATH_LIST)
        with open(FILE_PATH_IP, 'r', encoding='utf-8') as f:
            ip = json.load(f)
        return rwd, lst, ip, None
    except Exception as e:
        return None, None, None, str(e)

# --------------------------------------------------------------------------
# 4. UI 및 메인 로직
# --------------------------------------------------------------------------

# 제목 출력
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

# 모델 로드
models = load_models()

# 사이드바 (설정용)
st.sidebar.title("⚙️ 탐지 설정")
with st.sidebar.expander("📂 데이터 소스 상태", expanded=True):
    st.info("서버에 저장된 데이터를 자동으로 불러옵니다.")
    st.text(f"로그: {FILE_PATH_RWD}")
    st.text(f"광고: {FILE_PATH_LIST}")
    st.text(f"IP  : {FILE_PATH_IP}")

sensitivity = st.sidebar.radio("탐지 민감도 프리셋", ('평균', '엄격', '완화'))
with st.sidebar.expander("세부 점수 조정하기 (고급)"):
    config = DEFAULT_CONFIG.copy()
    for rule, params in config.items():
        if isinstance(params, dict) and 'score' in params:
            korean_name = KOREAN_NAMES.get(rule, rule)
            config[rule]['score'] = st.slider(f"'{korean_name}' 규칙 점수", 0, 100, params['score'], key=f"score_{rule}")

# ==========================================================================
# [핵심] 데이터 자동 로드 및 분석 실행
# ==========================================================================

# 1. 데이터 로드 시도
rwd_df, list_df, ip_data, err_msg = load_data_automatically()

if err_msg:
    st.error(f"❌ 데이터 로딩 실패: {err_msg}")
    st.warning("깃허브 리포지토리에 파일이 정확한 경로와 이름으로 업로드되었는지 확인해주세요.")
    st.stop() # 여기서 중단

if rwd_df is not None:
    # 세션에 데이터 저장
    st.session_state.df_rwd = rwd_df
    st.session_state.df_list = list_df
    st.session_state.ip_cache = ip_data
    st.session_state.mapping = DEFAULT_MAPPING

    # Step 1 UI (보여주기용)
    st.header("STEP 1: 데이터 자동 로드 완료")
    with st.expander("✅ 컬럼 매핑 정보 확인", expanded=False):
        st.write(f"- Device ID: {DEFAULT_MAPPING['dvc_idx']}")
        st.write(f"- User IP: {DEFAULT_MAPPING['user_ip']}")
        st.write(f"- Ads ID: {DEFAULT_MAPPING['ads_idx_list']}")
    
    st.markdown("---")
    st.header("STEP 2: 어뷰징 분석 결과")

    # 버튼 클릭 없이 바로 분석 로직 실행!
    with st.spinner('데이터 분석 중입니다...'):
        
        # -------------------------------------------------------
        # 분석 로직 시작
        # -------------------------------------------------------
        ads_rwd_info = st.session_state.df_rwd.copy()
        ads_list = st.session_state.df_list.copy()
        mapping = st.session_state.mapping
        
        # 컬럼명 통일
        try:
            ads_rwd_info.rename(columns={mapping['dvc_idx']: 'dvc_idx', mapping['user_ip']: 'user_ip'}, inplace=True)
            ads_list.rename(columns={mapping['ads_idx_list']: 'ads_idx'}, inplace=True)
        except KeyError:
            st.error("컬럼 매핑 오류: CSV 파일의 컬럼명이 코드 상단의 DEFAULT_MAPPING과 일치하는지 확인하세요.")
            st.stop()

        # 민감도 설정 적용
        if sensitivity == '엄격': config['blocklist_percentile'] = 0.97
        elif sensitivity == '완화': config['blocklist_percentile'] = 0.85
        
        # 데이터 전처리 (prepare_data)
        df_original, df_complete, df_incomplete, clicks_per_mda, cvr_per_mda = prepare_data(
            ads_rwd_info, ads_list, st.session_state.ip_cache, config
        )
        
        if df_original.empty:
            st.error("분석할 데이터가 없습니다 (데이터가 비어있음).")
            st.stop()
        
        # 스코어링 (calculate_abuse_scores)
        complete_scored = calculate_abuse_scores(df_complete, 'conversion', clicks_per_mda, cvr_per_mda, anomaly_model=models['anomaly_model'], ctit_anomaly_model=models['ctit_anomaly_model'], config=config)
        incomplete_scored = calculate_abuse_scores(df_incomplete, 'click', clicks_per_mda, cvr_per_mda, anomaly_model=models['anomaly_model'], ctit_anomaly_model=models['ctit_anomaly_model'], config=config)
        
        # 결과 병합
        all_scored_df = pd.concat([complete_scored, incomplete_scored], ignore_index=True)
        final_block_list, device_scores = get_blocklist(all_scored_df, "통합 분석")
        
        # Threshold 계산
        if not device_scores.empty:
            if config.get('blocklist_method', 'percentile') == 'percentile':
                threshold = device_scores.quantile(config.get('blocklist_percentile', 0.95))
            else:
                threshold = config.get('absolute_score_threshold', 100)
        else:
            threshold = 0
        
        st.success("✅ 분석이 완료되었습니다!")

        # -------------------------------------------------------
        # 결과 리포트 출력
        # -------------------------------------------------------

        # 날짜 범위 계산
        if 'done_date' in df_original.columns and df_original['done_date'].notna().any():
            min_date = pd.to_datetime(df_original['done_date'].dropna()).min()
            max_date = pd.to_datetime(df_original['done_date'].dropna()).max()
            date_standard = "전환 완료 시점 기준"
        else:
            min_date = pd.to_datetime(df_original['click_date']).min()
            max_date = pd.to_datetime(df_original['click_date']).max()
            date_standard = "클릭 시점 기준"

        # 분석 기준 표 출력
        st.markdown("### 📋 분석 기준")
        rules_data = []
        for rule, params in config.items():
            if isinstance(params, dict) and 'score' in params:
                rules_data.append({
                    '기준명': KOREAN_NAMES.get(rule, rule),
                    '점수': params['score']
                })
        rules_df = pd.DataFrame(rules_data).sort_values(by='점수', ascending=False)
        
        # 표 스타일링 함수
        def bold_high_score(row):
            return ['font-weight: bold'] * len(row) if row['점수'] >= 30 else ['font-weight: normal'] * len(row)

        st.markdown("""
        <style>
        .dataframe td:nth-child(2), th:nth-child(2) { text-align: left !important; width: 60px !important; }
        </style>
        """, unsafe_allow_html=True)
        
        st.dataframe(rules_df.style.apply(bold_high_score, axis=1), use_container_width=True, height=300)
        st.markdown("---")

        # 결과 요약 헤더
        col_title, col_date = st.columns([0.7, 0.3])
        with col_title: st.subheader("📊 분석 결과 요약")
        with col_date:
            st.markdown(f"""
            <div style="text-align: right; padding-top: 10px;">
                <p style="font-size: 1.1rem; font-weight: 500; margin: 0;">{min_date.strftime('%Y.%m.%d')} ~ {max_date.strftime('%Y.%m.%d')}</p>
                <p style="font-size: 0.8rem; color: #8A8B94; margin: 0;">({date_standard})</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 메트릭 계산 및 표시
        total_devices = df_original['dvc_idx'].nunique()
        abusive_devices_count = len(final_block_list)
        device_abuse_ratio = (abusive_devices_count / total_devices * 100) if total_devices > 0 else 0
        
        total_logs = len(df_original)
        abusive_logs = len(df_original[df_original['dvc_idx'].isin(final_block_list)])
        log_abuse_ratio = (abusive_logs / total_logs * 100) if total_logs > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1: st.markdown(f"""<div style="padding: 10px; border-radius: 5px; background-color: #262730;"><p style="font-size: 16px; color: #FAFAFA; margin-bottom: 5px;">전체 디바이스 중 어뷰징 비율</p><p style="font-size: 28px; color: #FAFAFA; font-weight: bold;">{device_abuse_ratio:.2f}%</p><p style="font-size: 18px; color: #8A8B94;">{abusive_devices_count:,} / {total_devices:,} 개</p></div>""", unsafe_allow_html=True)
        with col2: st.markdown(f"""<div style="padding: 10px; border-radius: 5px; background-color: #262730;"><p style="font-size: 16px; color: #FAFAFA; margin-bottom: 5px;">전체 로그 중 어뷰징 비율</p><p style="font-size: 28px; color: #FAFAFA; font-weight: bold;">{log_abuse_ratio:.2f}%</p><p style="font-size: 18px; color: #8A8B94;">{abusive_logs:,} / {total_logs:,} 건</p></div>""", unsafe_allow_html=True)
        st.metric("차단 임계 점수", f"{threshold:.2f} 점")
        st.divider()

        if not final_block_list:
            st.info("탐지된 어뷰징 의심 디바이스가 없습니다.")
        else:
            abusive_df = all_scored_df[all_scored_df['dvc_idx'].isin(final_block_list)].copy()
            
            # 매체 리포트
            st.subheader("📊 어뷰징 유저가 가장 많이 이용한 매체 Top 10")
            mda_abuse_counts = abusive_df.groupby('mda_idx')['dvc_idx'].nunique().sort_values(ascending=False).head(10)
            mda_abuse_df = mda_abuse_counts.reset_index()
            mda_abuse_df.columns = ['매체 ID (mda_idx)', '어뷰징 유저 수']
            mda_abuse_df['전체 어뷰징 중 비율 (%)'] = (mda_abuse_df['어뷰징 유저 수'] / abusive_devices_count * 100).map('{:.2f}%'.format)
            st.dataframe(mda_abuse_df, use_container_width=True)
            
            # CSV 다운로드용 함수
            @st.cache_data
            def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8-sig')
            
            st.download_button("📈 매체 리포트 다운로드", convert_df_to_csv(mda_abuse_df), "abuse_media_report.csv", "text/csv")
            st.divider()

            # 디바이스별 요약 리포트
            st.subheader("📄 어뷰징 요약 리포트 (디바이스별)")
            
            def translate_reasons(row):
                if pd.isna(row['abuse_reasons']) or row['abuse_reasons'] == '': return '정보 없음'
                reasons = []
                parts = row['abuse_reasons'].split(']')
                for part in parts:
                    if part.strip():
                        name = part.replace('[', '').strip()
                        reasons.append(KOREAN_NAMES.get(name.lower(), name))
                return ', '.join(reasons) if reasons else '정보 없음'
            
            summary_df = abusive_df.groupby('dvc_idx').agg({'abuse_score': 'max', 'abuse_reasons': 'first'}).reset_index()
            summary_df['주요 어뷰징 사유'] = summary_df.apply(translate_reasons, axis=1)
            summary_df = summary_df[['dvc_idx', 'abuse_score', '주요 어뷰징 사유']]
            summary_df.columns = ['디바이스 ID', '어뷰징 점수', '주요 어뷰징 사유']
            summary_df = summary_df.sort_values('어뷰징 점수', ascending=False).reset_index(drop=True)
            
            st.dataframe(summary_df)
            st.download_button("✅ 요약 리포트 다운로드", convert_df_to_csv(summary_df), "abuse_summary_report.csv", "text/csv", type="primary")
