import pandas as pd
import numpy as np
import json
import joblib

def prepare_data(ads_rwd_info, ads_list, ip_cache_data, config):
    # 이 함수는 이제 당신의 원본 스크립트와 100% 동일하게 작동하며, 데이터 뻥튀기 현상을 막습니다.
    
    # 1. ads_list의 중복을 먼저 제거 (안정성 확보)
    ads_list.drop_duplicates(subset=['ads_idx'], keep='first', inplace=True)

    # 2. 초기 데이터 로드 및 병합
    ads_rwd_info['hostname'] = ads_rwd_info['user_ip'].map(ip_cache_data).fillna('N/A')
    ads_rwd_info['is_aws'] = ads_rwd_info['hostname'].str.contains('amazonaws.com|AMAZON|AWS', case=False, na=False)
    df_original = pd.merge(ads_rwd_info, ads_list[['ads_idx', 'ads_type', 'ads_category']], on='ads_idx', how='left')

    # 3. 데이터 클리닝
    df_original = df_original.loc[:, ~df_original.columns.duplicated()]
    df_original['dvc_idx'] = pd.to_numeric(df_original['dvc_idx'], errors='coerce')
    df_original.dropna(subset=['dvc_idx'], inplace=True)
    df_original = df_original[df_original['dvc_idx'] != 0].copy()
    df_original['dvc_idx'] = df_original['dvc_idx'].astype(int)
    df_original['click_date'] = pd.to_datetime(df_original['click_date'])
    
    # 4. 통계 피쳐 계산
    clicks_per_mda = df_original.groupby('mda_idx').size()
    conversions_per_mda = df_original.dropna(subset=['done_date']).groupby('mda_idx').size()
    cvr_per_mda = (conversions_per_mda / clicks_per_mda).fillna(0)

    # 5. Burst Attack 피쳐 계산 및 병합
    df_original.sort_values(by=['dvc_idx', 'click_date'], inplace=True)

    window = f"{config['burst_attack']['window_min']}min"
    clicks_in_Nmin = df_original.set_index('click_date').groupby('dvc_idx').rolling(window)['ads_idx'].count().reset_index(name='clicks_in_Nmin')
    df_original = pd.merge(df_original, clicks_in_Nmin, on=['dvc_idx', 'click_date'], how='left')
    df_original['clicks_in_Nmin'].fillna(1, inplace=True)

    # 6. 최종 데이터 분리
    df_complete = df_original.dropna(subset=['ctit', 'user_ip', 'dvc_idx']).copy()
    df_incomplete = df_original[~df_original.index.isin(df_complete.index)].copy()
    
    return df_original, df_complete, df_incomplete, clicks_per_mda, cvr_per_mda

def calculate_abuse_scores(df, config, analysis_type='conversion', clicks_per_mda_series=None, cvr_per_mda_series=None, anomaly_model=None, ctit_anomaly_model=None):
    if df.empty: return df
    df = df.copy()
    df.sort_values(by=['dvc_idx', 'click_date'], inplace=True)
    df['time_diff_sec'] = df.groupby('dvc_idx')['click_date'].diff().dt.total_seconds()
    df['total_clicks_per_dvc'] = df.groupby('dvc_idx')['dvc_idx'].transform('count')
    df['click_hour'] = df['click_date'].dt.hour
    
    # --- 기본 규칙 점수 및 마스크 생성 ---
    df['score_burst_attack'] = np.where(df['clicks_in_Nmin'] > config['burst_attack']['threshold_clicks'], config['burst_attack']['score'], 0)
    unique_mda_per_dvc = df.groupby('dvc_idx')['mda_idx'].nunique()
    df['unique_mda_count'] = df['dvc_idx'].map(unique_mda_per_dvc)
    media_concentration_mask = (df['total_clicks_per_dvc'] > config['media_concentration']['threshold_clicks']) & (df['unique_mda_count'] < config['media_concentration']['threshold_mda'])
    df['score_media_concentration'] = np.where(media_concentration_mask, config['media_concentration']['score'], 0)
    df['score_abnormal_cvr'] = 0
    if cvr_per_mda_series is not None:
        df['mda_cvr'] = df['mda_idx'].map(cvr_per_mda_series)
        cvr_mask = (df['mda_cvr'] > config['abnormal_cvr']['threshold_cvr']) & (df['mda_idx'].map(clicks_per_mda_series) > config['abnormal_cvr']['threshold_clicks'])
        df['score_abnormal_cvr'] = np.where(cvr_mask, config['abnormal_cvr']['score'], 0)
    rapid_click_mask = df['time_diff_sec'] < config['rapid_click']['threshold_sec']
    df['score_rapid_click'] = np.where(rapid_click_mask, config['rapid_click']['score'], 0)
    dvc_per_ip = df.groupby('user_ip')['dvc_idx'].nunique()
    ip_per_dvc = df.groupby('dvc_idx')['user_ip'].nunique()
    df['dvc_count_per_ip'] = df['user_ip'].map(dvc_per_ip)
    df['ip_count_per_dvc'] = df['dvc_idx'].map(ip_per_dvc)
    many_dvc_mask = df['dvc_count_per_ip'].between(config['many_devices_per_ip']['threshold_devices'] + 1, config['many_devices_per_ip']['carrier_ip_threshold'])
    df['score_many_devices_per_ip'] = np.where(many_dvc_mask, config['many_devices_per_ip']['score'], 0)
    many_ip_mask = df['ip_count_per_dvc'] > config['many_ips_per_device']['threshold_ips']
    df['score_many_ips_per_device'] = np.where(many_ip_mask, config['many_ips_per_device']['score'], 0)

    if analysis_type == 'conversion':
        short_ctit_mask = df['ctit'] < config['short_ctit']['threshold_sec']
        df['score_short_ctit'] = np.where(short_ctit_mask, config['short_ctit']['score'], 0)
        early_hour_mask = df['click_hour'].between(config['suspicious_early_hour']['start_hour'], config['suspicious_early_hour']['end_hour'])
        early_hour_cond_mask = early_hour_mask & ((df.get('time_diff_sec', pd.Series(dtype='float')) < 2) | (df.get('ctit', pd.Series(dtype='float')) < 10))
        df['score_suspicious_early_hour'] = np.where(early_hour_cond_mask, config['suspicious_early_hour']['score'], 0)
        df['ctit_std'] = df.groupby('dvc_idx')['ctit'].transform('std').fillna(0)
        ctit_mask = (df['ctit_std'] < config['consistent_ctit']['threshold_std']) & (df['total_clicks_per_dvc'] > config['consistent_ctit']['threshold_clicks'])
        df['score_consistent_ctit'] = np.where(ctit_mask, config['consistent_ctit']['score'], 0)
        df['score_fraud_long_ctit'] = np.where(df['ctit'] > 3600, 35, 0)
        single_click_mask = df['total_clicks_per_dvc'] == 1
        suspicious_single_conv_mask = single_click_mask & early_hour_mask
        df['score_suspicious_single_conv'] = np.where(suspicious_single_conv_mask, 30, 0)
        
        # ▼▼▼ consistent_click 규칙 관련 코드 블록 전체 삭제됨 ▼▼▼

        df['score_ctit_anomaly_model'] = 0
        if ctit_anomaly_model:
            features = df.dropna(subset=['ctit']).groupby('dvc_idx')['ctit'].agg(['mean', 'std', 'median', 'count', 'min', 'max']).dropna()
            features = features[features['count'] >= 3]
            if not features.empty:
                predictions = ctit_anomaly_model.predict(features)
                anomalous_ids = features.index[predictions == -1]
                df.loc[df['dvc_idx'].isin(anomalous_ids), 'score_ctit_anomaly_model'] = 35
                
    elif analysis_type == 'click':
        df['score_heavy_click_spam'] = np.where(df['total_clicks_per_dvc'] > config['heavy_click_spam']['threshold_clicks'], config['heavy_click_spam']['score'], 0)
        df['score_anomaly_model'] = 0
        if anomaly_model:
            features = df.groupby('dvc_idx')['time_diff_sec'].agg(['mean', 'std', 'median', 'count']).dropna()
            if not features.empty:
                predictions = anomaly_model.predict(features)
                anomalous_ids = features.index[predictions == -1]
                # 'consistent_click' 점수를 참조하던 부분을 상수(45)로 변경 또는 다른 값으로 대체 필요. 여기서는 45로 하드코딩.
                df.loc[df['dvc_idx'].isin(anomalous_ids), 'score_anomaly_model'] = 45
    
    base_score_cols = [col for col in df.columns if col.startswith('score_')]
    df['base_score'] = df[base_score_cols].sum(axis=1)
    aws_mask = (df['is_aws']) & (df['base_score'] > 0)
    df['score_aws_ip_conditional'] = np.where(aws_mask, config['aws_ip']['score'], 0)
    
    if 'early_hour_mask' in locals():
        df['score_combo_stealth_bot'] = np.where(aws_mask & early_hour_mask, 30, 0)
    df['score_combo_focused_fraud'] = np.where(media_concentration_mask & many_ip_mask, 35, 0)
    
    # ▼▼▼ score_combo_suspicious_conv 규칙 라인 삭제됨 ▼▼▼

    final_score_cols = [col for col in df.columns if col.startswith('score_')]
    df['abuse_score'] = df[final_score_cols].sum(axis=1)
    df.drop(columns=['base_score'], inplace=True, errors='ignore')
    return df

def get_blocklist(df_scored, config, name=""):
    method = config.get('blocklist_method', 'percentile')
    threshold = 0
    if df_scored.empty or 'abuse_score' not in df_scored.columns or df_scored['abuse_score'].sum() == 0: return [], pd.Series(), threshold
    high_score_events = df_scored[df_scored['abuse_score'] > 0]
    if high_score_events.empty: return [], pd.Series(), threshold
    device_scores = high_score_events.groupby('dvc_idx')['abuse_score'].max()
    if not device_scores.empty:
        if method == 'percentile':
            percentile = config.get('blocklist_percentile', 0.95)
            threshold = device_scores.quantile(percentile)
        elif method == 'absolute':
            threshold = config.get('absolute_score_threshold', 100)
        abusive_devices = device_scores[device_scores >= threshold]
        return abusive_devices.index.tolist(), device_scores, threshold
    else: return [], pd.Series(), threshold