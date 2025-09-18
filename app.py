import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from detector import prepare_data, calculate_abuse_scores, get_blocklist
from PIL import Image

try:
    # app.py와 같은 폴더에 아이콘 파일이 있어야 해
    icon = Image.open("1-794df7f8.ico")
except FileNotFoundError:
    st.error("아이콘 파일을 찾을 수 없습니다. app.py와 같은 폴더에 있는지 확인해주세요.")
    icon = "🍀" # 파일을 못 찾으면 대신 이모티콘을 사용

# --- 2. 페이지 기본 설정 ---
st.set_page_config(
    page_title="광고 어뷰징 탐지 센서",
    page_icon=icon, # 브라우저 탭 아이콘 설정
    layout="wide"
)

# --- 3. CSS 코드로 아이콘과 제목의 세로 위치(중앙) 맞추기 ---
st.markdown("""
    <style>
    [data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. 컬럼을 사용해 아이콘과 제목 배치 ---
col1, col2 = st.columns([1, 10], gap="small") # gap="small"로 간격 좁히기

with col1:
    st.image(icon, width=70) # 아이콘 크기 70으로 키우기

with col2:
    # anchor=False 옵션으로 깔끔한 URL 유지
    st.title("광고 어뷰징 탐지 센서", anchor=False)

# --- 5. 제목 아래 구분선 ---
st.markdown("---")
# --- 설정값, 한글 번역, 설명: 17개 규칙에 맞게 모두 업데이트 ---
DEFAULT_CONFIG = {
    # --- 기존 12개 ---
    'burst_attack': {'threshold_clicks': 15, 'score': 15, 'window_min': 5}, 
    'media_concentration': {'threshold_clicks': 20, 'threshold_mda': 2, 'score': 20}, 
    'abnormal_cvr': {'threshold_cvr': 0.90, 'threshold_clicks': 20, 'score': 45}, 
    'short_ctit': {'threshold_sec': 5, 'score': 15}, 
    'suspicious_early_hour': {'start_hour': 2, 'end_hour': 6, 'score': 10}, 
    'consistent_ctit': {'threshold_std': 3.0, 'threshold_clicks': 4, 'score': 40}, 
    'heavy_click_spam': {'threshold_clicks': 50, 'score': 20}, 
    'rapid_click': {'threshold_sec': 1.0, 'score': 10}, 
    'many_devices_per_ip': {'threshold_devices': 6, 'score': 25, 'carrier_ip_threshold': 10000}, 
    'many_ips_per_device': {'threshold_ips': 15, 'score': 25}, 
    'aws_ip': {'score': 25}, # aws_ip_conditional 대신 aws_ip 사용

    # --- 새로 추가된 5개 ---
    'fraud_long_ctit': {'score': 35},
    'suspicious_single_conv': {'score': 30},
    'ctit_anomaly_model': {'score': 35},
    'anomaly_model': {'score': 45}, # consistent_click 점수와 동일하게 설정
    'combo_stealth_bot': {'score': 30},
    'combo_focused_fraud': {'score': 35},
    # combo_suspicious_conv는 결과에 없었으므로 제외
    
    # --- 제재 방식 설정 ---
    'blocklist_method': 'percentile', 
    'blocklist_percentile': 0.95, 
    'absolute_score_threshold': 100
}

KOREAN_NAMES = {
    'burst_attack': '단기 클릭 폭주', 
    'media_concentration': '매체 집중', 
    'abnormal_cvr': '비정상적 전환율(CVR)', 
    'short_ctit': '짧은 전환 시간(CTIT)', 
    'suspicious_early_hour': '의심스러운 심야 활동', 
    'consistent_ctit': '일정한 전환 시간(CTIT)', 
    'heavy_click_spam': '과도한 클릭 (미전환)', 
    'rapid_click': '매우 빠른 클릭', 
    'many_devices_per_ip': '다수 기기/IP', 
    'many_ips_per_device': '다수 IP/기기', 
    'aws_ip': '서버 IP 사용 (AWS)', 
    'fraud_long_ctit': '비정상적으로 긴 CTIT', 
    'suspicious_single_conv': '의심스러운 단일 전환', 
    'ctit_anomaly_model': 'CTIT 패턴 모델', 
    'anomaly_model': '클릭 간격 패턴 모델', 
    'aws_ip_conditional': '서버 IP 사용 (조건부)', 
    'combo_stealth_bot': '콤보: 은신 봇', 
    'combo_focused_fraud': '콤보: 집중형 사기', 
}

RULE_DESCRIPTIONS = {
    'burst_attack': "단시간(예: 5분) 내에 비정상적으로 많은 클릭을 발생시키는 패턴입니다. 주로 단기 보상을 노리는 어뷰저들이 사용합니다.", 
    'media_concentration': "한 명의 사용자가 소수의 광고 매체(채널)에만 집중적으로 참여하는 패턴입니다. 일반적인 사용자는 다양한 매체를 이용하는 경향이 있습니다.", 
    'abnormal_cvr': "전환율(CVR)이 비정상적으로 높은 매체에서 발생하는 클릭을 탐지합니다. 해당 매체가 어뷰징에 취약하거나 이미 어뷰저에게 점령되었을 수 있습니다.", 
    'short_ctit': "광고 클릭부터 전환까지 걸리는 시간(CTIT)이 비정상적으로 짧은 경우입니다. 사람이 인지하고 행동하기 어려운 속도의 전환을 탐지합니다.", 
    'suspicious_early_hour': "사용자 활동이 드문 심야나 새벽 시간대에 의심스러운 활동(예: 초고속 클릭, 짧은 전환 시간)이 발생하는 패턴입니다.", 
    'consistent_ctit': "한 사용자의 여러 전환 활동에서 CTIT 값이 기계적으로 거의 일정한 패턴을 보입니다. 이는 자동화된 스크립트(봇)일 가능성이 높습니다.", 
    'heavy_click_spam': "전환은 전혀 발생시키지 않으면서, 광고 클릭만 대량으로 발생시키는 패턴입니다. 경쟁사 광고 예산을 소진시키려는 목적일 수 있습니다.", 
    'rapid_click': "클릭 간격이 1초 미만으로, 사람의 행동이라고 보기 어려운 매우 빠른 연속 클릭을 탐지합니다.", 
    'many_devices_per_ip': "하나의 IP 주소에서 너무 많은 기기가 접속하는 경우입니다. 어뷰징 작업을 위한 특정 공간(작업장)일 가능성이 있습니다.", 
    'many_ips_per_device': "하나의 기기에서 너무 많은 IP 주소를 바꿔가며 접속하는 패턴입니다. VPN 등을 이용해 여러 사용자인 척 위장하는 경우를 탐지합니다.", 
    'aws_ip': "데이터 센터나 클라우드 서버에서 사용하는 IP(예: AWS)에서의 접속입니다. 일반 사용자보다는 전문 어뷰저일 가능성이 높습니다.",
    'fraud_long_ctit': "CTIT가 1시간을 초과하는 유령 클릭 패턴입니다. 비정상적인 전환으로 간주됩니다.",
    'suspicious_single_conv': "총 클릭 수가 단 1회이면서 심야 시간대에 발생한 의심스러운 전환입니다.",
    'ctit_anomaly_model': "머신러닝 모델이 기존 데이터와 다른 비정상적인 CTIT 분포 패턴을 보이는 기기를 탐지합니다.",
    'anomaly_model': "머신러닝 모델이 기존 데이터와 다른 비정상적인 클릭 간격 패턴을 보이는 기기를 탐지합니다.",
    'combo_stealth_bot': "서버 IP(AWS)를 사용하면서 동시에 심야에 활동하는 은신형 봇 패턴입니다.",
    'combo_focused_fraud': "소수의 매체에 클릭을 집중시키면서 다수의 IP를 사용하는 복합적인 사기 패턴입니다."
}

# --- 모델 로딩 함수 (캐시 사용) ---
@st.cache_resource
def load_models():
    models = {}
    try: models['anomaly_model'] = joblib.load('models/isolation_forest_model.joblib')
    except FileNotFoundError: models['anomaly_model'] = None
    try: models['ctit_anomaly_model'] = joblib.load('models/ctit_anomaly_model.joblib')
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
    # 루프가 업데이트된 DEFAULT_CONFIG를 기반으로 동적으로 슬라이더를 생성
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
    mapping_form = st.form(key="mapping_form")
    cols = mapping_form.columns(3)
    file_columns_rwd = ["(선택 안 함)"] + st.session_state.df_rwd.columns.tolist()
    file_columns_list = ["(선택 안 함)"] + st.session_state.df_list.columns.tolist()
    
    # 사용자가 업로드한 파일의 컬럼명을 기반으로 동적으로 기본값 설정
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
            
            # 원본 컬럼 이름이 mapping에 있는지 확인 후 rename
            if mapping['dvc_idx'] in ads_rwd_info.columns:
                ads_rwd_info.rename(columns={mapping['dvc_idx']: 'dvc_idx'}, inplace=True)
            if mapping['user_ip'] in ads_rwd_info.columns:
                ads_rwd_info.rename(columns={mapping['user_ip']: 'user_ip'}, inplace=True)
            if mapping['ads_idx_list'] in ads_list.columns:
                ads_list.rename(columns={mapping['ads_idx_list']: 'ads_idx'}, inplace=True)

            if sensitivity == '엄격': config['blocklist_percentile'] = 0.97
            elif sensitivity == '완화': config['blocklist_percentile'] = 0.85
            
            df_original, df_complete, df_incomplete, clicks_per_mda, cvr_per_mda = prepare_data(
                ads_rwd_info, ads_list, st.session_state.ip_cache, config
            )
            if df_original.empty: st.error("분석할 데이터가 없습니다."); st.stop()
            
            complete_scored = calculate_abuse_scores(df_complete, config, 'conversion', clicks_per_mda, cvr_per_mda, **models)
            incomplete_scored = calculate_abuse_scores(df_incomplete, config, 'click', clicks_per_mda, cvr_per_mda, **models)
            all_scored_df = pd.concat([complete_scored, incomplete_scored], ignore_index=True)
            final_block_list, device_scores, threshold = get_blocklist(all_scored_df, config, "통합 분석")
            
            st.success("✅ 분석이 완료되었습니다!")

            col_title, col_date = st.columns([0.7, 0.3])
            
            with col_title:
                st.subheader("📊 분석 결과 요약")
            
            if 'done_date' in df_original.columns and pd.to_datetime(df_original['done_date'], errors='coerce').notna().any():
                min_date = pd.to_datetime(df_original['done_date'].dropna()).min()
                max_date = pd.to_datetime(df_original['done_date'].dropna()).max()
                date_standard = "전환 완료 시점 기준"
            else:
                min_date = df_original['click_date'].min()
                max_date = df_original['click_date'].max()
                date_standard = "클릭 시점 기준"
            
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
            with col1: st.markdown(f"""<div style="padding: 10px; border-radius: 5px; background-color: #262730;"><p style="font-size: 16px; color: #FAFAFA; margin-bottom: 5px;">전체 유저 중 어뷰징 유저 비율</p><p style="font-size: 28px; color: #FAFAFA; font-weight: bold;">{device_abuse_ratio:.2f}%</p><p style="font-size: 18px; color: #8A8B94;">{abusive_devices_count:,} / {total_devices:,} 개</p></div>""", unsafe_allow_html=True)
            with col2: st.markdown(f"""<div style="padding: 10px; border-radius: 5px; background-color: #262730;"><p style="font-size: 16px; color: #FAFAFA; margin-bottom: 5px;">전체 로그 중 어뷰징 로그 비율</p><p style="font-size: 28px; color: #FAFAFA; font-weight: bold;">{log_abuse_ratio:.2f}%</p><p style="font-size: 18px; color: #8A8B94;">{abusive_logs:,} / {total_logs:,} 건</p></div>""", unsafe_allow_html=True)
            st.metric("차단 임계 점수", f"{threshold:.2f} 점")
            st.divider()

            if not final_block_list: 
                st.info("탐지된 어뷰징 의심 디바이스가 없습니다.")
            else:
                abusive_df = all_scored_df[all_scored_df['dvc_idx'].isin(final_block_list)].copy()
                @st.cache_data
                def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8-sig')

                st.subheader("📊 어뷰징 주요 발생 매체 Top 10")
                if 'mda_idx' in abusive_df.columns:
                    mda_abuse_counts = abusive_df.groupby('mda_idx')['dvc_idx'].nunique().sort_values(ascending=False).head(10)
                    mda_abuse_df = mda_abuse_counts.reset_index()
                    mda_abuse_df.columns = ['매체 ID (mda_idx)', '어뷰징 디바이스 수']
                    mda_abuse_df['비율 (%)'] = (mda_abuse_df['어뷰징 디바이스 수'] / abusive_devices_count * 100).map('{:.2f}%'.format)
                    st.dataframe(mda_abuse_df, use_container_width=True)
                    st.download_button("📈 매체 리포트 다운로드", convert_df_to_csv(mda_abuse_df), "abuse_media_report.csv", "text/csv")
                    st.divider()
                
                st.subheader("📄 어뷰징 요약 리포트 (디바이스별 점수 높은 순)")
                
                # 'score_'로 시작하는 모든 컬럼을 동적으로 찾음
                score_cols = [col for col in abusive_df.columns if col.startswith('score_')]
                
                def translate_reasons(row):
                    reasons = []
                    for col in score_cols:
                        # score가 0보다 큰 경우에만 해당 규칙의 이름을 가져옴
                        if row[col] > 0:
                            reason_key = col.replace('score_', '')
                            # 'aws_ip_conditional' 같은 키를 'aws_ip'으로 일반화하여 KOREAN_NAMES에서 찾음
                            if reason_key == 'aws_ip_conditional':
                                reason_key = 'aws_ip'
                            reasons.append(KOREAN_NAMES.get(reason_key, reason_key))
                    return ', '.join(sorted(list(set(reasons))))

                abusive_df['suspicion_reasons_kr'] = abusive_df.apply(translate_reasons, axis=1)
                
                summary_df = abusive_df.groupby('dvc_idx').agg(
                    abuse_score=('abuse_score', 'max'),
                    suspicion_reasons=('suspicion_reasons_kr', lambda reasons: ', '.join(sorted(list(set(reason for reason in ', '.join(reasons).split(', ') if reason and reason.strip())))))
                ).reset_index()
                
                summary_df.rename(columns={'dvc_idx': '디바이스 ID', 'abuse_score': '어뷰징 점수', 'suspicion_reasons': '주요 어뷰징 사유'}, inplace=True)
                summary_df = summary_df.sort_values(by='어뷰징 점수', ascending=False)
                summary_df = summary_df.reset_index(drop=True)
                
                st.dataframe(summary_df)
                st.download_button("✅ 요약 리포트 다운로드", convert_df_to_csv(summary_df), "abuse_summary_report.csv", "text/csv", type="primary")

else:
    st.header("STEP 1: 데이터 파일 업로드하기")
    st.info("⬆️ 사이드바에서 분석에 필요한 3개 파일을 모두 업로드하면 다음 단계가 나타납니다.")