import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib

from detector import prepare_data, calculate_abuse_scores, get_blocklist

st.set_page_config(
    layout="wide",
    page_title="광고 어뷰징 탐지 센서",
    page_icon="1-794df7f8.ico"
)

FILE_PATH_RWD = "/Users/jyj/Desktop/vs_code/streamlit 프로젝트/data/ads_info_rwd.csv"       # 원본 로그 파일명
FILE_PATH_LIST = "/Users/jyj/Desktop/vs_code/streamlit 프로젝트/ads_list.csv"     # 광고 리스트 파일명
FILE_PATH_IP = "/Users/jyj/Desktop/vs_code/streamlit 프로젝트/data/ip_hostname_2.json"        # IP 정보 파일명

DEFAULT_MAPPING = {
    'dvc_idx': 'dvc_idx',       # 디바이스 ID 컬럼명
    'user_ip': 'user_ip',       # IP 주소 컬럼명
    'ads_idx_list': 'ads_idx'   # 광고 ID 컬럼명
}

# 제목을 왼쪽 정렬로 예쁘게 만들기
col1, col2 = st.columns([1, 5])

with col1:
    st.image("1-794df7f8.ico", width=100)

with col2:
    st.markdown("""
    <div style="text-align: left; margin-left: 5px;">
        <h1 style="font-size: 3rem; font-weight: bold; color: #FFFFFF; margin: 0; padding: 1rem 0; display: inline-block; vertical-align: middle;">
            광고 어뷰징 탐지 센서
        </h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- 기본 설정값, 한글 번역, 설명 (이전과 동일) ---
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
    # 추가 규칙들
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
RULE_DESCRIPTIONS = {
    'burst_attack': "단시간(예: 5분) 내에 비정상적으로 많은 클릭을 발생시키는 패턴입니다. 주로 단기 보상을 노리는 어뷰저들이 사용합니다.", 
    'media_concentration': "한 명의 사용자가 소수의 광고 매체(채널)에만 집중적으로 참여하는 패턴입니다. 일반적인 사용자는 다양한 매체를 이용하는 경향이 있습니다.", 
    'abnormal_cvr': "전환율(CVR)이 비정상적으로 높은 매체에서 발생하는 클릭을 탐지합니다. 해당 매체가 어뷰징에 취약하거나 이미 어뷰저에게 점령되었을 수 있습니다.", 
    'short_ctit': "광고 클릭부터 전환까지 걸리는 시간(CTIT)이 비정상적으로 짧은 경우입니다. 사람이 인지하고 행동하기 어려운 속도의 전환을 탐지합니다.", 
    'suspicious_early_hour': "사용자 활동이 드문 심야나 새벽 시간대에 의심스러운 활동(예: 초고속 클릭, 짧은 전환 시간)이 발생하는 패턴입니다.", 
    'consistent_ctit': "한 사용자의 여러 전환 활동에서 CTIT 값이 기계적으로 거의 일정한 패턴을 보입니다. 이는 자동화된 스크립트(봇)일 가능성이 높습니다.", 
    'anomaly_model': "클릭 간격 패턴을 기반으로 한 머신러닝 모델을 통해 이상 패턴을 탐지합니다.",
    'heavy_click_spam': "전환은 전혀 발생시키지 않으면서, 광고 클릭만 대량으로 발생시키는 패턴입니다. 경쟁사 광고 예산을 소진시키려는 목적일 수 있습니다.", 
    'rapid_click': "클릭 간격이 1초 미만으로, 사람의 행동이라고 보기 어려운 매우 빠른 연속 클릭을 탐지합니다.", 
    'many_devices_per_ip': "하나의 IP 주소에서 너무 많은 기기가 접속하는 경우입니다. 어뷰징 작업을 위한 특정 공간(작업장)일 가능성이 있습니다.", 
    'many_ips_per_device': "하나의 기기에서 너무 많은 IP 주소를 바꿔가며 접속하는 패턴입니다. VPN 등을 이용해 여러 사용자인 척 위장하는 경우를 탐지합니다.", 
    'aws_ip': "데이터 센터나 클라우드 서버에서 사용하는 IP(예: AWS)에서의 접속입니다. 일반 사용자보다는 전문 어뷰저일 가능성이 높습니다.",
    # 추가 규칙 설명
    'fraud_long_ctit': "광고 클릭부터 전환까지 걸리는 시간(CTIT)이 비정상적으로 긴 경우입니다. 1시간(3600초) 초과 시 유령 클릭으로 판단합니다.",
    'suspicious_single_conv': "클릭 수가 1개이면서 심야 시간대에 발생하는 의심스러운 단일 전환을 탐지합니다.",
    'ctit_anomaly_model': "CTIT 패턴을 기반으로 한 머신러닝 모델을 통해 이상 패턴을 탐지합니다.",
    'combo_stealth_bot': "AWS IP 사용과 심야 활동이 결합된 은신 봇 패턴을 탐지합니다.",
    'combo_focused_fraud': "매체 집중과 다수 IP 사용이 결합된 집중형 사기 패턴을 탐지합니다."
}

# --- 모델 로딩 함수 (캐시 사용) ---
@st.cache_resource
def load_models():
    models = {}
    try: models['anomaly_model'] = joblib.load('isolation_forest_model.joblib')
    except FileNotFoundError: models['anomaly_model'] = None
    try: models['ctit_anomaly_model'] = joblib.load('ctit_anomaly_model.joblib')
    except FileNotFoundError: models['ctit_anomaly_model'] = None
    return models

models = load_models()

# --- 사이드바 UI 구성 ---
st.sidebar.title("⚙️ 탐지 설정")
with st.sidebar.expander("📂 파일 업로드", expanded=True):
    st.info("분석에 필요한 세 개의 파일을 모두 업로드하세요.")
    uploaded_file_rwd = st.file_uploader("1. 원본 로그 (광고참여정보 등)", type=['csv'])
    uploaded_file_list = st.file_uploader("2. 광고 정보 (광고 리스트 등)", type=['csv'])
    uploaded_file_ip_cache = st.file_uploader("3. IP 정보 (IP 별 AWS 탐지 파일 등)", type=['json'])

sensitivity = st.sidebar.radio("탐지 민감도 프리셋", ('평균', '엄격', '완화'))
with st.sidebar.expander("세부 점수 조정하기 (고급)"):
    config = DEFAULT_CONFIG.copy()
    for rule, params in config.items():
        if isinstance(params, dict) and 'score' in params:
            korean_name = KOREAN_NAMES.get(rule, rule)
            description = RULE_DESCRIPTIONS.get(rule, "설명이 없습니다.")
            config[rule]['score'] = st.slider(f"'{korean_name}' 규칙 점수", 0, 100, params['score'], key=f"score_{rule}", help=description)

# --- 메인 로직 시작 ---
if all([uploaded_file_rwd, uploaded_file_list, uploaded_file_ip_cache]):
    try:
        st.session_state.df_rwd = pd.read_csv(uploaded_file_rwd)
        st.session_state.df_list = pd.read_csv(uploaded_file_list)
        st.session_state.ip_cache = json.load(uploaded_file_ip_cache)
    except Exception as e:
        st.error(f"❌ 파일을 읽는 중 오류가 발생했습니다: {e}"); st.stop()

    st.header("STEP 1: 핵심 컬럼 확인하기")
    # ... (이하 컬럼 매핑 부분은 변경 없음, 생략)
    mapping_form = st.form(key="mapping_form")
    cols = mapping_form.columns(3)
    file_columns_rwd = ["(선택 안 함)"] + st.session_state.df_rwd.columns.tolist()
    file_columns_list = ["(선택 안 함)"] + st.session_state.df_list.columns.tolist()
    with cols[0]: dvc_idx_col = st.selectbox("**디바이스 ID**", file_columns_rwd, index=file_columns_rwd.index('dvc_idx') if 'dvc_idx' in file_columns_rwd else 0)
    with cols[1]: user_ip_col = st.selectbox("**IP 주소**", file_columns_rwd, index=file_columns_rwd.index('user_ip') if 'user_ip' in file_columns_rwd else 0)
    with cols[2]: ads_idx_col_list = st.selectbox("**광고 ID (연결 키)**", file_columns_list, index=file_columns_list.index('ads_idx') if 'ads_idx' in file_columns_list else 0)
    submitted = mapping_form.form_submit_button("✅ 확인 완료, 분석 준비하기")
    if submitted:
        st.session_state.mapping = {'dvc_idx': dvc_idx_col, 'user_ip': user_ip_col, 'ads_idx_list': ads_idx_col_list}
        st.success("핵심 컬럼 확인 완료. 이제 분석을 시작할 수 있습니다.")

if 'mapping' in st.session_state:
    st.markdown("---")
    st.header("STEP 2: 어뷰징 분석 실행하기")
    
    if st.button("🚀 어뷰징 분석 시작하기", type="primary"):
        with st.spinner('데이터를 분석중입니다...'):
            ads_rwd_info = st.session_state.df_rwd.copy()
            ads_list = st.session_state.df_list.copy()
            mapping = st.session_state.mapping
            ads_rwd_info.rename(columns={mapping['dvc_idx']: 'dvc_idx', mapping['user_ip']: 'user_ip'}, inplace=True)
            ads_list.rename(columns={mapping['ads_idx_list']: 'ads_idx'}, inplace=True)
            
            if sensitivity == '엄격': config['blocklist_percentile'] = 0.97
            elif sensitivity == '완화': config['blocklist_percentile'] = 0.85
            
            df_original, df_complete, df_incomplete, clicks_per_mda, cvr_per_mda = prepare_data(
                ads_rwd_info, ads_list, st.session_state.ip_cache, config
            )
            if df_original.empty: st.error("분석할 데이터가 없습니다."); st.stop()
            
            complete_scored = calculate_abuse_scores(df_complete, 'conversion', clicks_per_mda, cvr_per_mda, anomaly_model=models['anomaly_model'], ctit_anomaly_model=models['ctit_anomaly_model'], config=config)
            incomplete_scored = calculate_abuse_scores(df_incomplete, 'click', clicks_per_mda, cvr_per_mda, anomaly_model=models['anomaly_model'], ctit_anomaly_model=models['ctit_anomaly_model'], config=config)
            all_scored_df = pd.concat([complete_scored, incomplete_scored], ignore_index=True)
            final_block_list, device_scores = get_blocklist(all_scored_df, "통합 분석")
            
            # threshold 계산
            if not device_scores.empty:
                if config.get('blocklist_method', 'percentile') == 'percentile':
                    threshold = device_scores.quantile(config.get('blocklist_percentile', 0.95))
                else:
                    threshold = config.get('absolute_score_threshold', 100)
            else:
                threshold = 0
            
            st.success("✅ 분석이 완료되었습니다!")

            # 날짜 계산 (나중에 사용)
            if 'done_date' in df_original.columns and df_original['done_date'].notna().any():
                min_date = pd.to_datetime(df_original['done_date'].dropna()).min()
                max_date = pd.to_datetime(df_original['done_date'].dropna()).max()
                date_standard = "전환 완료 시점 기준"
            else:
                min_date = df_original['click_date'].min()
                max_date = df_original['click_date'].max()
                date_standard = "클릭 시점 기준"

            # 분석 기준 표시 (표 형태)
            st.markdown("### 📋 분석 기준")
            
            # 규칙들을 점수순으로 정렬
            rules_data = []
            for rule, params in config.items():
                if isinstance(params, dict) and 'score' in params:
                    score = params['score']
                    korean_name = KOREAN_NAMES.get(rule, rule)
                    rules_data.append({
                        '기준명': korean_name,
                        '점수': score,
                        '중요도': '높음' if score >= 30 else '낮음'
                    })
            
            # 점수순으로 정렬
            rules_data.sort(key=lambda x: x['점수'], reverse=True)
            
            # 표 생성
            rules_df = pd.DataFrame(rules_data)
            
            # 중요도 컬럼 제거 후 표시
            display_df = rules_df.drop(columns=['중요도'])
            
            # 30점 이상인 항목들의 글씨를 굵게 만들기
            def bold_high_score(row):
                if row['점수'] >= 30:
                    return ['font-weight: bold'] * len(row)
                else:
                    return ['font-weight: normal'] * len(row)
            
            # CSS를 사용하여 점수 컬럼을 왼쪽 정렬하고 너비 조정
            st.markdown("""
            <style>
            .dataframe td:nth-child(2) {
                text-align: left !important;
                padding-left: 8px !important;
                width: 60px !important;
                max-width: 60px !important;
            }
            .dataframe th:nth-child(2) {
                text-align: left !important;
                width: 60px !important;
                max-width: 60px !important;
            }
            div[data-testid="stDataFrame"] table td:nth-child(2) {
                text-align: left !important;
                width: 60px !important;
                max-width: 60px !important;
            }
            /* 더 강력한 선택자들 */
            table td:nth-child(2) {
                text-align: left !important;
            }
            .stDataFrame table td:nth-child(2) {
                text-align: left !important;
            }
            [data-testid="stDataFrame"] table tbody tr td:nth-child(2) {
                text-align: left !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # 스타일 적용
            styled_df = display_df.style.apply(bold_high_score, axis=1)
            
            st.dataframe(styled_df, use_container_width=True, height=300)
            
            st.markdown("---")

            # 분석 결과 요약 (원래 위치로 복원)
            # 1. 제목과 날짜 표시를 위한 영역 분리
            col_title, col_date = st.columns([0.7, 0.3])
            
            with col_title:
                st.subheader("📊 분석 결과 요약")
            
            # 2. 날짜를 오른쪽에 더 큰 글씨로 표시
            with col_date:
                st.markdown(f"""
                <div style="text-align: right; padding-top: 10px;">
                    <p style="font-size: 1.1rem; font-weight: 500; margin: 0;">{min_date.strftime('%Y.%m.%d')} ~ {max_date.strftime('%Y.%m.%d')}</p>
                    <p style="font-size: 0.8rem; color: #8A8B94; margin: 0;">({date_standard})</p>
                </div>
                """, unsafe_allow_html=True)
            
            total_devices = df_original['dvc_idx'].nunique()
            abusive_devices_count = len(final_block_list)
            device_abuse_ratio = (abusive_devices_count / total_devices) * 100 if total_devices > 0 else 0
            total_logs = len(df_original)
            abusive_logs = len(df_original[df_original['dvc_idx'].isin(final_block_list)])
            log_abuse_ratio = (abusive_logs / total_logs) * 100 if total_logs > 0 else 0
            
            col1, col2 = st.columns(2)
            with col1: st.markdown(f"""<div style="padding: 10px; border-radius: 5px; background-color: #262730;"><p style="font-size: 16px; color: #FAFAFA; margin-bottom: 5px;">전체 디바이스 중 어뷰징 비율</p><p style="font-size: 28px; color: #FAFAFA; font-weight: bold;">{device_abuse_ratio:.2f}%</p><p style="font-size: 18px; color: #8A8B94;">{abusive_devices_count:,} / {total_devices:,} 개</p></div>""", unsafe_allow_html=True)
            with col2: st.markdown(f"""<div style="padding: 10px; border-radius: 5px; background-color: #262730;"><p style="font-size: 16px; color: #FAFAFA; margin-bottom: 5px;">전체 로그 중 어뷰징 비율</p><p style="font-size: 28px; color: #FAFAFA; font-weight: bold;">{log_abuse_ratio:.2f}%</p><p style="font-size: 18px; color: #8A8B94;">{abusive_logs:,} / {total_logs:,} 건</p></div>""", unsafe_allow_html=True)
            st.metric("차단 임계 점수", f"{threshold:.2f} 점")
            st.divider()

            # ... (이하 나머지 코드는 변경 없음)
            if not final_block_list: 
                st.info("탐지된 어뷰징 의심 디바이스가 없습니다.")
            else:
                abusive_df = all_scored_df[all_scored_df['dvc_idx'].isin(final_block_list)].copy()
                @st.cache_data
                def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8-sig')

                st.subheader("📊 어뷰징 유저가 가장 많이 이용한 매체 Top 10")
                
                # 어뷰징 디바이스가 가장 많이 이용한 mda_idx 계산
                mda_abuse_counts = abusive_df.groupby('mda_idx')['dvc_idx'].nunique().sort_values(ascending=False).head(10)
                
                # 결과 데이터프레임 생성
                mda_abuse_df = mda_abuse_counts.reset_index()
                mda_abuse_df.columns = ['매체 ID (mda_idx)', '어뷰징 유저 수']
                mda_abuse_df['전체 어뷰징 중 비율 (%)'] = (mda_abuse_df['어뷰징 유저 수'] / abusive_devices_count * 100).map('{:.2f}%'.format)
                
                st.dataframe(mda_abuse_df, use_container_width=True)
                st.download_button("📈 매체 리포트 다운로드", convert_df_to_csv(mda_abuse_df), "abuse_media_report.csv", "text/csv")
                st.divider()
                
                # 교차 분석 부분 삭제
                
                st.subheader("📄 어뷰징 요약 리포트 (디바이스별)")
                
                # 어뷰징 사유를 한글로 변환하는 함수
                def translate_reasons(row):
                    if pd.isna(row['abuse_reasons']) or row['abuse_reasons'] == '':
                        return '정보 없음'
                    
                    reasons = []
                    reason_parts = row['abuse_reasons'].split(']')
                    for part in reason_parts:
                        if part.strip():
                            # [Reason_Name] 형태에서 Reason_Name만 추출
                            reason_name = part.replace('[', '').strip()
                            if reason_name:
                                # 영어 이름을 한글 이름으로 변환
                                korean_name = KOREAN_NAMES.get(reason_name.lower(), reason_name)
                                reasons.append(korean_name)
                    return ', '.join(reasons) if reasons else '정보 없음'
                
                # 디바이스별로 그룹화하여 최고 점수와 사유 추출
                summary_df = abusive_df.groupby('dvc_idx').agg({
                    'abuse_score': 'max',
                    'abuse_reasons': 'first'
                }).reset_index()
                
                # 어뷰징 사유를 한글로 변환
                summary_df['주요 어뷰징 사유'] = summary_df.apply(translate_reasons, axis=1)
                
                # 컬럼명 변경 및 정렬
                summary_df = summary_df[['dvc_idx', 'abuse_score', '주요 어뷰징 사유']]
                summary_df.columns = ['디바이스 ID', '어뷰징 점수', '주요 어뷰징 사유']
                summary_df = summary_df.sort_values('어뷰징 점수', ascending=False).reset_index(drop=True)
                
                st.dataframe(summary_df)
                st.download_button("✅ 요약 리포트 다운로드", convert_df_to_csv(summary_df), "abuse_summary_report.csv", "text/csv", type="primary")

else:
    st.header("STEP 1: 데이터 파일 업로드하기")
    st.info("⬆️ 사이드바에서 분석에 필요한 3개 파일을 모두 업로드하면 다음 단계가 나타납니다.")
