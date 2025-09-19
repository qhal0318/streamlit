# 파일 이름: detector.py

import pandas as pd
import numpy as np
import json
import joblib

# --- 설정값 (CONFIG) ---
CONFIG = {
    'burst_attack': {'threshold_clicks': 15, 'score': 15, 'window_min': 5},
    'media_concentration': {'threshold_clicks': 20, 'threshold_mda': 2, 'score': 20},
    'abnormal_cvr': {'threshold_cvr': 0.90, 'threshold_clicks': 20, 'score': 45}, # threshold_clicks 하향
    'short_ctit': {'threshold_sec': 5, 'score': 15},
    'suspicious_early_hour': {'start_hour': 2, 'end_hour': 6, 'score': 10},
    'consistent_ctit': {'threshold_std': 3.0, 'threshold_clicks': 4, 'score': 40}, # threshold_clicks 하향
    'anomaly_model': {'threshold_clicks': 8, 'score': 45},                      # 이상 탐지 모델 기반
    'heavy_click_spam': {'threshold_clicks': 50, 'score': 20},
    'rapid_click': {'threshold_sec': 1.0, 'score': 10},
    'many_devices_per_ip': {'threshold_devices': 6, 'score': 25, 'carrier_ip_threshold': 10000}, # threshold_devices 하향
    'many_ips_per_device': {'threshold_ips': 15, 'score': 25},
    'aws_ip': {'score': 25},
    
    # ▼▼▼ 추가 규칙들 ▼▼▼
    'fraud_long_ctit': {'threshold_sec': 3600, 'score': 35},  # 1시간 초과
    'suspicious_single_conv': {'score': 30},
    'ctit_anomaly_model': {'score': 35},
    'combo_stealth_bot': {'score': 30},
    'combo_focused_fraud': {'score': 35},
    
    # ▼▼▼ 제재 방식 선택 및 절대 점수 기준 추가 ▼▼▼
    'blocklist_method': 'percentile',     # 'percentile' 또는 'absolute' 선택
    'blocklist_percentile': 0.95,       # 'percentile' 방식일 때 사용 (상위 1%)
    'absolute_score_threshold': 100
}

def prepare_data(ads_rwd_info, ads_list, ip_cache_data, config):
    """
    데이터를 읽고 병합하며 기본적인 전처리를 수행합니다.
    (원본 스크립트의 '0단계' 로직과 100% 동일)
    """
    # 원본처럼 ads_list의 중복을 제거하지 않아 데이터 뻥튀기 현상을 재현합니다.
    ads_rwd_info['hostname'] = ads_rwd_info['user_ip'].map(ip_cache_data).fillna('N/A')
    ads_rwd_info['is_aws'] = ads_rwd_info['hostname'].str.contains('amazonaws.com|AMAZON|AWS', case=False, na=False)
    df_original = pd.merge(ads_rwd_info, ads_list[['ads_idx', 'ads_type', 'ads_category','ads_name']], on='ads_idx', how='left')
    df_original = df_original.loc[:, ~df_original.columns.duplicated()]
    df_original['dvc_idx'] = pd.to_numeric(df_original['dvc_idx'], errors='coerce')
    df_original.dropna(subset=['dvc_idx'], inplace=True)
    df_original = df_original[df_original['dvc_idx'] != 0].copy()
    df_original['dvc_idx'] = df_original['dvc_idx'].astype(int)
    df_original['click_date'] = pd.to_datetime(df_original['click_date'])
    
    # Burst Attack 피쳐 계산 (원본과 동일)
    df_original.sort_values(by=['dvc_idx', 'click_date'], inplace=True)
    window = f"{config['burst_attack']['window_min']}min"  # CONFIG에서 가져온 값
    clicks_in_Nmin = df_original.set_index('click_date').groupby('dvc_idx').rolling(window)['ads_idx'].count().reset_index(name='clicks_in_Nmin')
    df_original = pd.merge(df_original, clicks_in_Nmin, on=['dvc_idx', 'click_date'], how='left')
    df_original['clicks_in_Nmin'].fillna(1, inplace=True)
    
    # 완전한 데이터와 불완전한 데이터 분리
    df_complete = df_original.dropna(subset=['ctit', 'user_ip', 'dvc_idx']).copy()
    df_incomplete = df_original[~df_original.index.isin(df_complete.index)].copy()
    
    # CVR 계산
    clicks_per_mda = df_original.groupby('mda_idx').size()
    conversions_per_mda = df_original.dropna(subset=['done_date']).groupby('mda_idx').size()
    cvr_per_mda = (conversions_per_mda / clicks_per_mda).fillna(0)
    
    return df_original, df_complete, df_incomplete, clicks_per_mda, cvr_per_mda

def calculate_abuse_scores(df, analysis_type='conversion', clicks_per_mda_series=None, cvr_per_mda_series=None, anomaly_model=None, ctit_anomaly_model=None, config=None):
    """
    어뷰징 점수를 계산합니다.
    (원본 스크립트의 '1단계' 로직과 100% 동일)
    """
    if df.empty:
        return df
    
    # config가 None이면 기본 CONFIG 사용
    if config is None:
        config = CONFIG

    df['abuse_score'] = 0
    df['abuse_reasons'] = ''
    dvc_per_ip = df.groupby('user_ip')['dvc_idx'].nunique()
    ip_per_dvc = df.groupby('dvc_idx')['user_ip'].nunique()
    df['dvc_count_per_ip'] = df['user_ip'].map(dvc_per_ip)
    df['ip_count_per_dvc'] = df['dvc_idx'].map(ip_per_dvc)
    df.sort_values(by=['dvc_idx', 'click_date'], inplace=True)
    df['time_diff_sec'] = df.groupby('dvc_idx')['click_date'].diff().dt.total_seconds()
    df['total_clicks_per_dvc'] = df.groupby('dvc_idx')['dvc_idx'].transform('count')
    df['click_hour'] = df['click_date'].dt.hour

    # (기존 규칙 적용 로직은 동일)
    burst_attack_mask = df['clicks_in_Nmin'] > config['burst_attack']['threshold_clicks']
    df.loc[burst_attack_mask, 'abuse_score'] += config['burst_attack']['score']; df.loc[burst_attack_mask, 'abuse_reasons'] += '[Burst_Attack] '
    unique_mda_per_dvc = df.groupby('dvc_idx')['mda_idx'].nunique()
    df['unique_mda_count'] = df['dvc_idx'].map(unique_mda_per_dvc)
    media_concentration_mask = (df['total_clicks_per_dvc'] > config['media_concentration']['threshold_clicks']) & (df['unique_mda_count'] < config['media_concentration']['threshold_mda'])
    df.loc[media_concentration_mask, 'abuse_score'] += config['media_concentration']['score']; df.loc[media_concentration_mask, 'abuse_reasons'] += '[Media_Concentration] '
    if cvr_per_mda_series is not None and not cvr_per_mda_series.empty:
        df['mda_cvr'] = df['mda_idx'].map(cvr_per_mda_series)
        abnormal_cvr_mask = (df['mda_cvr'] > config['abnormal_cvr']['threshold_cvr']) & (df['mda_idx'].map(clicks_per_mda_series) > config['abnormal_cvr']['threshold_clicks'])
        df.loc[abnormal_cvr_mask, 'abuse_score'] += config['abnormal_cvr']['score']; df.loc[abnormal_cvr_mask, 'abuse_reasons'] += '[Abnormal_CVR] '
    
    if analysis_type == 'conversion':
        df['click_interval_std'] = df.groupby('dvc_idx')['time_diff_sec'].transform('std').fillna(0)
        df['ctit_std'] = df.groupby('dvc_idx')['ctit'].transform('std').fillna(0)
        conditions = [(df['ads_type'] == 4) | (df['ads_category'] == 4), (df['ads_type'].isin([1,2,3,5,6,7,10,11])) | (df['ads_category'].isin([1,2,3,5,6,7,8,10,13])), (df['ads_type'] == 12) | (df['ads_category'].isin([11,12]))]
        choices = [0.05, 0.5, 1.0]; df['dynamic_consistency_threshold'] = np.select(conditions, choices, default=0.1)
        short_ctit_mask = df['ctit'] < config['short_ctit']['threshold_sec']
        df.loc[short_ctit_mask, 'abuse_score'] += config['short_ctit']['score']; df.loc[short_ctit_mask, 'abuse_reasons'] += '[Short_CTIT] '
        early_hour_mask = df['click_hour'].between(config['suspicious_early_hour']['start_hour'], config['suspicious_early_hour']['end_hour'])
        suspicious_in_early_hour_mask = early_hour_mask & ((df['time_diff_sec'] < 2) | (df['ctit'] < 10))
        df.loc[suspicious_in_early_hour_mask, 'abuse_score'] += config['suspicious_early_hour']['score']; df.loc[suspicious_in_early_hour_mask, 'abuse_reasons'] += '[Suspicious_Early_Hour] '
        consistent_ctit_mask = (df['ctit_std'] < config['consistent_ctit']['threshold_std']) & (df['total_clicks_per_dvc'] > config['consistent_ctit']['threshold_clicks'])
        df.loc[consistent_ctit_mask, 'abuse_score'] += config['consistent_ctit']['score']; df.loc[consistent_ctit_mask, 'abuse_reasons'] += '[Consistent_CTIT] '

        # ▼▼▼ 여기에 새로운 "맞춤형 저격 룰" 추가 ▼▼▼
        # --- (신규) 맞춤형 저격 룰 ---
        # 1. 유령 클릭 (비정상적으로 긴 CTIT)
        long_ctit_mask = df['ctit'] > config['fraud_long_ctit']['threshold_sec']
        df.loc[long_ctit_mask, 'abuse_score'] += config['fraud_long_ctit']['score']
        df.loc[long_ctit_mask, 'abuse_reasons'] += '[Fraud_Long_CTIT] '

        # 2. 의심스러운 단일 전환 (클릭 수가 1개 & 심야 활동)
        single_click_mask = df['total_clicks_per_dvc'] == 1
        suspicious_single_conv_mask = single_click_mask & early_hour_mask
        df.loc[suspicious_single_conv_mask, 'abuse_score'] += config['suspicious_single_conv']['score']
        df.loc[suspicious_single_conv_mask, 'abuse_reasons'] += '[Suspicious_Single_Conversion] '
        # ▲▲▲ 여기까지 새로운 "맞춤형 저격 룰" 추가 ▲▲▲

    elif analysis_type == 'click':
        heavy_clicker_mask = df['total_clicks_per_dvc'] > config['heavy_click_spam']['threshold_clicks']
        df.loc[heavy_clicker_mask, 'abuse_score'] += config['heavy_click_spam']['score']; df.loc[heavy_clicker_mask, 'abuse_reasons'] += '[Heavy_Click_Spam] '
        if anomaly_model:
            device_features = df.groupby('dvc_idx')['time_diff_sec'].agg(['mean', 'std', 'median', 'count']).dropna()
            if not device_features.empty:
                predictions = anomaly_model.predict(device_features)
                anomalous_dvc_ids = device_features.index[predictions == -1]
                model_based_mask = df['dvc_idx'].isin(anomalous_dvc_ids)
                df.loc[model_based_mask, 'abuse_score'] += config['anomaly_model']['score']
                df.loc[model_based_mask, 'abuse_reasons'] += '[Anomaly_Model_Flag] '
    
    rapid_click_mask = df['time_diff_sec'] < config['rapid_click']['threshold_sec']
    df.loc[rapid_click_mask, 'abuse_score'] += config['rapid_click']['score']; df.loc[rapid_click_mask, 'abuse_reasons'] += '[Rapid_Click] '
    
    lower_bound = config['many_devices_per_ip']['threshold_devices']
    upper_bound = config['many_devices_per_ip']['carrier_ip_threshold']
    many_dvc_mask = df['dvc_count_per_ip'].between(lower_bound + 1, upper_bound)
    df.loc[many_dvc_mask, 'abuse_score'] += config['many_devices_per_ip']['score']; df.loc[many_dvc_mask, 'abuse_reasons'] += '[Many_Devices_Per_IP] '
    many_ip_mask = df['ip_count_per_dvc'] > config['many_ips_per_device']['threshold_ips']
    df.loc[many_ip_mask, 'abuse_score'] += config['many_ips_per_device']['score']; df.loc[many_ip_mask, 'abuse_reasons'] += '[Many_IPs_Per_Device] '
    aws_abuse_mask = (df['is_aws']) & (df['abuse_score'] > 0)
    df.loc[aws_abuse_mask, 'abuse_score'] += config['aws_ip']['score']; df.loc[aws_abuse_mask, 'abuse_reasons'] += '[AWS_IP_Used] '
    
    if analysis_type == 'conversion' and ctit_anomaly_model:
        device_ctit_features = df.dropna(subset=['ctit']).groupby('dvc_idx')['ctit'].agg(
            ['mean', 'std', 'median', 'count', 'min', 'max']
        ).dropna()
        device_ctit_features = device_ctit_features[device_ctit_features['count'] >= 3]
        if not device_ctit_features.empty:
            predictions = ctit_anomaly_model.predict(device_ctit_features)
            anomalous_ctit_dvc_ids = device_ctit_features.index[predictions == -1]
            ctit_model_mask = df['dvc_idx'].isin(anomalous_ctit_dvc_ids)
            df.loc[ctit_model_mask, 'abuse_score'] += config['ctit_anomaly_model']['score']
            df.loc[ctit_model_mask, 'abuse_reasons'] += '[CTIT_Anomaly_Model] '

    if 'aws_abuse_mask' in locals() and 'early_hour_mask' in locals():
        combo_stealth_bot_mask = aws_abuse_mask & early_hour_mask
        df.loc[combo_stealth_bot_mask, 'abuse_score'] += config['combo_stealth_bot']['score']
        df.loc[combo_stealth_bot_mask, 'abuse_reasons'] += '[Combo_Stealth_Bot] '

    if 'media_concentration_mask' in locals() and 'many_ip_mask' in locals():
        combo_focused_fraud_mask = media_concentration_mask & many_ip_mask
        df.loc[combo_focused_fraud_mask, 'abuse_score'] += config['combo_focused_fraud']['score']
        df.loc[combo_focused_fraud_mask, 'abuse_reasons'] += '[Combo_Focused_Fraud] '

    return df

# ▼▼▼ get_blocklist 함수 수정 ▼▼▼
def get_blocklist(df_scored, name=""):
    method = CONFIG.get('blocklist_method', 'percentile')
    
    if df_scored.empty:
        print(f"--- [{name}] 분석 대상 데이터가 없어 건너뜁니다. ---")
        return [], pd.Series() # 빈 리스트와 빈 Series 반환
        
    high_score_events = df_scored[df_scored['abuse_score'] > 0]
    device_scores = high_score_events.groupby('dvc_idx')['abuse_score'].max()
    
    if not device_scores.empty:
        if method == 'percentile':
            percentile = CONFIG['blocklist_percentile']
            threshold = device_scores.quantile(percentile)
            print(f"--- [{name}] 상위 {(1-percentile)*100:.1f}% 커트라인 점수(상대): {threshold:.2f} ---")
        elif method == 'absolute':
            threshold = CONFIG['absolute_score_threshold']
            print(f"--- [{name}] 커트라인 점수(절대): {threshold:.2f} ---")
        else:
            print(f"--- [{name}] 잘못된 threshold 방식입니다. 'percentile' 또는 'absolute'를 사용하세요. ---")
            return [], device_scores
            
        abusive_devices = device_scores[device_scores >= threshold]
        return abusive_devices.index.tolist(), device_scores # 리스트와 함께 device_scores도 반환
    else:
        return [], pd.Series()

def run_detection(ads_rwd_info, ads_list, ip_cache_data, config):
    """
    전체 어뷰징 탐지 프로세스를 실행합니다.
    """
    print("--- 0단계: 데이터 준비 ---")
    
    # 모델 로딩
    try:
        anomaly_model = joblib.load('isolation_forest_model.joblib')
        print("✅ 이상 탐지 모델 로딩 완료.")
    except FileNotFoundError:
        anomaly_model = None
        print("⚠️ 이상 탐지 모델 파일을 찾을 수 없습니다.")
    
    try:
        ctit_anomaly_model = joblib.load('ctit_anomaly_model.joblib')
        print("✅ CTIT 이상 탐지 모델 로딩 완료.")
    except FileNotFoundError:
        ctit_anomaly_model = None
        print("⚠️ CTIT 이상 탐지 모델 파일을 찾을 수 없습니다.")
    
    # 데이터 준비
    df_original, df_complete, df_incomplete, clicks_per_mda, cvr_per_mda = prepare_data(ads_rwd_info, ads_list, ip_cache_data, config)
    
    print("✅ 데이터 준비 및 정제 완료.")
    print(f"df_complete 크기: {df_complete.shape}, df_incomplete 크기: {df_incomplete.shape}")
    
    print("\n--- 2단계: 분석 실행 ---")
    df_complete_scored = calculate_abuse_scores(df_complete, 'conversion', clicks_per_mda, cvr_per_mda, anomaly_model=anomaly_model, ctit_anomaly_model=ctit_anomaly_model, config=config)
    df_incomplete_scored = calculate_abuse_scores(df_incomplete, 'click', clicks_per_mda, cvr_per_mda, anomaly_model=anomaly_model, ctit_anomaly_model=ctit_anomaly_model, config=config)
    
    print("\n--- 3단계: 결과 추출 ---")
    # 통합된 데이터 전체에서 제재 대상을 한 번에 추출
    all_scored_df = pd.concat([df_complete_scored, df_incomplete_scored], ignore_index=True)
    final_block_list, device_scores = get_blocklist(all_scored_df, "통합 어뷰징")
    
    print(f"\n✅ 최종 통합 제재 디바이스: {len(final_block_list)}개")
    
    return final_block_list, all_scored_df, device_scores