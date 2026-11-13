"""TM5 MAIN 분석. 실행: python src/run_analysis.py

원본은 읽기만 하며 결과와 학습 모델은 outputs에 저장합니다.
환자 식별자는 메모리 안의 분할·검증에만 사용하고 결과 파일에는 기록하지 않습니다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['KERAS_TORCH_DEVICE'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['KERAS_HOME'] = str(ROOT / 'outputs' / '.keras')

import hashlib
import importlib.metadata
import json
import platform
import warnings
import zipfile

import joblib
import keras
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import torch
from scipy.special import expit
from scipy.stats import beta
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, auc, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from label_audit import audit_labels, token
from chart_layout import write_charts

SEED = 42
BOOTSTRAPS = 2000
OUT = ROOT / 'outputs'
Y, GROUP, TIME, KEY, TEXT = 'is_important_alert', '환자번호', '점검일시', '점검ID', '경고내용'
MODELS = ['LR', 'LightGBM', 'Keras']
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
px.defaults.template = 'plotly_white'
warnings.filterwarnings('ignore', message='X does not have valid feature names')


def save_csv(frame, name):
    frame.to_csv(OUT / f'{name}.csv', index=False, encoding='utf-8-sig')


def save_json(obj, name):
    (OUT / f'{name}.json').write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def load_data():
    """패키지 안의 최종 파일만 선택하고 ZIP과 SHA256 일치 여부를 검증합니다."""
    package = Path(os.environ.get('SMARTDUR_PACKAGE_DIR', str(ROOT / 'data' / 'package')))
    if not (package / '01_최종결과').is_dir():
        raise FileNotFoundError('승인된 SmartDUR 패키지를 data/package에 준비하거나 SMARTDUR_PACKAGE_DIR을 설정하세요.')
    paths = {k: next((package / '01_최종결과').glob(pattern)) for k, pattern in {
        'main': '*MAIN*.csv', 'master': '*MASTER*.csv', 'features': '*모델입력*.csv'}.items()}
    archive = Path(os.environ.get('SMARTDUR_ARCHIVE', str(ROOT / 'data' / 'package.zip')))
    provenance = {}
    with zipfile.ZipFile(archive) as z:
        for k, p in paths.items():
            member = next(n for n in z.namelist() if n.endswith(p.name))
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            assert hashlib.sha256(z.read(member)).hexdigest() == digest
            provenance[k] = {'file': p.name, 'sha256': digest, 'zip_matches': True}
    main = pd.read_csv(paths['main'])
    master = pd.read_csv(paths['master'])
    features = pd.read_csv(paths['features']).iloc[:, 0].tolist()
    assert main.shape == (458, 46) and master.shape == (1030, 104)
    assert main[KEY].is_unique and master[KEY].is_unique
    assert main[Y].value_counts().to_dict() == {0.0: 398, 1.0: 60}
    assert len(features) == 35 and len(set(features)) == 35
    assert not set(features) & {Y, GROUP, KEY, TIME, 'y_source', 'analysis_population', 'red_rule_flag',
                                'y_reviewed', 'y_audit_candidate', 'label_issue', 'label_confidence'}
    main = main.merge(master[[KEY, TEXT]], on=KEY, how='left', validate='one_to_one')
    assert main[TEXT].notna().all() and main[GROUP].notna().all()
    main[TIME] = pd.to_datetime(main[TIME])
    label_audit, label_summary = audit_labels(package, master, main, OUT,
                                             OUT / 'review_decisions_PRIVATE.csv')
    main['case_token'] = main[KEY].map(token)
    main = main.merge(label_audit[['case_token', 'y_reviewed', 'y_audit_candidate', 'decision_status',
                                 'label_issue', 'label_confidence', 'review_note']], on='case_token', validate='one_to_one')
    save_csv(main[['case_token'] + features + [Y, 'y_source', 'y_reviewed', 'label_issue',
                                             'label_confidence', 'decision_status', 'review_note']],
             'MAIN_458_label_review_PRIVATE')
    # 장표의 77건은 MAIN 마지막 시각에서 2개월을 뺀 기준으로 재현됩니다.
    cutoff = main[TIME].max() - pd.DateOffset(months=2)
    hold = main[TIME].gt(cutoff)
    overlap = set(main.loc[hold, GROUP]) & set(main.loc[~hold, GROUP])
    purge = ~hold & main[GROUP].isin(overlap)
    # 주 분석은 첨부안의 377건입니다. 14일 관찰기간 분리는 별도 민감도로 비교합니다.
    embargo = ~hold & ~purge & main[TIME].gt(cutoff - pd.Timedelta(days=14))
    dev = main.loc[~hold & ~purge].reset_index(drop=True)
    test = main.loc[hold].reset_index(drop=True)
    assert len(test) == 77 and test[Y].sum() == 9
    assert not set(dev[GROUP]) & set(test[GROUP])
    assert dev[TIME].max() <= cutoff < test[TIME].min()
    assert len(dev) == 377 and dev[Y].sum() == 50
    split = {'cutoff': cutoff, 'reference': 'MAIN maximum alert timestamp minus two calendar months',
             'main_max': main[TIME].max(), 'master_max': pd.to_datetime(master[TIME]).max(),
             'dev_n': len(dev), 'dev_positive': int(dev[Y].sum()), 'dev_patients': dev[GROUP].nunique(),
             'holdout_n': len(test), 'holdout_positive': int(test[Y].sum()), 'holdout_patients': test[GROUP].nunique(),
             'purged_rows': int(purge.sum()), 'purged_positive': int(main.loc[purge, Y].sum()),
             'primary_embargo_days': 0, 'embargo_rows': 0,
             'temporal_sensitivity_embargo_days': 14,
             'temporal_sensitivity_removed_rows': int(embargo.sum()),
             'temporal_sensitivity_removed_positive': int(main.loc[embargo, Y].sum()),
             'label_audit': label_summary,
             'overlapping_patients_removed_from_training': len(overlap)}
    save_json({'files': provenance, 'split': split}, 'data_audit')
    stages = pd.DataFrame([
        {'stage': '원래 시간순 학습 후보', 'n': int((~hold).sum()), 'positive': int(main.loc[~hold, Y].sum()), 'overlapping_patients': len(overlap)},
        {'stage': '환자 중복 제거: 주 분석', 'n': len(dev), 'positive': int(dev[Y].sum()), 'overlapping_patients': 0},
        {'stage': '14일 관찰기간 분리: 민감도', 'n': int((~hold & ~purge & ~embargo).sum()), 'positive': int(main.loc[~hold & ~purge & ~embargo, Y].sum()), 'overlapping_patients': 0},
        {'stage': '고정 hold-out', 'n': len(test), 'positive': int(test[Y].sum()), 'overlapping_patients': 0}])
    save_csv(stages, 'split_stages')
    for name, frame in [('train', dev), ('holdout', test)]:
        save_csv(frame[['case_token'] + features + [Y, 'y_reviewed', 'label_issue', 'label_confidence']], name + '_PRIVATE')
    save_csv(pd.DataFrame({'feature': features, 'missing_n': [int(main[f].isna().sum()) for f in features],
                           'dtype': [str(main[f].dtype) for f in features]}), 'input_audit')
    # MAIN 전체 561건에서 라벨 유무에 따른 선택 편향을 점검합니다.
    population = master[master.analysis_population.eq('MAIN')].copy()
    population['label_status'] = np.where(population[Y].notna(), '판정 있음', '판정 보류')
    selection = population.groupby('label_status').agg(n=(KEY, 'size'), age_mean=('age', 'mean'),
                lab_tested_pre_rate=('lab_tested_pre', 'mean'), active_drugs_mean=('n_active_drugs', 'mean'))
    save_csv(selection.reset_index(), 'selection_bias')
    save_csv(population.groupby(['label_status', '경고위험유형']).size().reset_index(name='n'), 'selection_by_type')
    return dev, test, features, split


def preprocessor(frame, features):
    cats = [c for c in features if not pd.api.types.is_numeric_dtype(frame[c])]
    nums = [c for c in features if c not in cats]
    return ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                          ('scale', StandardScaler())]), nums),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cats)
    ], verbose_feature_names_out=False)


def clean_x(frame, features):
    # pandas 문자열 결측도 sklearn이 처리할 수 있는 np.nan으로 맞춥니다.
    x = frame[features].copy()
    for c in x:
        if not pd.api.types.is_numeric_dtype(x[c]):
            x[c] = x[c].astype(object).where(x[c].notna(), np.nan)
    return x


def fit_tabular(train, features, model_name):
    prep = preprocessor(train, features)
    x = prep.fit_transform(clean_x(train, features))
    if model_name == 'LR':
        estimator = LogisticRegression(C=0.1, class_weight='balanced', max_iter=3000, random_state=SEED)
    else:
        estimator = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.03, max_depth=2,
                    num_leaves=4, min_child_samples=20, reg_lambda=5, verbosity=-1,
                    class_weight='balanced', random_state=SEED, n_jobs=2, deterministic=True, force_col_wise=True)
    estimator.fit(x, train[Y].astype(int))
    return {'preprocessor': prep, 'model': estimator, 'features': features}


def text_encoder(train):
    """외부 API 없이 문자 TF-IDF와 SVD 임베딩을 훈련 부분에서만 학습합니다."""
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=1500, min_df=2)
    x = vectorizer.fit_transform(train[TEXT])
    svd = TruncatedSVD(n_components=min(16, x.shape[0] - 1, x.shape[1] - 1), random_state=SEED)
    embedding = svd.fit_transform(x)
    scale = StandardScaler().fit(embedding)
    return vectorizer, svd, scale


def keras_inputs(bundle, frame):
    tab = bundle['preprocessor'].transform(clean_x(frame, bundle['features'])).astype('float32')
    vec, svd, scale = bundle['text_encoder']
    text = scale.transform(svd.transform(vec.transform(frame[TEXT]))).astype('float32')
    return [tab, text]


def build_keras(ntab, ntext, seed):
    keras.utils.set_random_seed(seed)
    structured = keras.Input(shape=(ntab,), name='structured')
    text = keras.Input(shape=(ntext,), name='alert_text_embedding')
    a = keras.layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(structured)
    b = keras.layers.Dense(8, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(text)
    x = keras.layers.Concatenate()([a, b])
    x = keras.layers.Dropout(0.25)(x)
    output = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model([structured, text], output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy')
    return model


def fit_keras(train, features, seed, tag, histories):
    """내부 환자 분할로 epoch를 선택하고 외부 훈련 전체에서 다시 학습합니다."""
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    tr, va = next(splitter.split(train, train[Y], train[GROUP]))
    inner, val = train.iloc[tr], train.iloc[va]
    assert not set(inner[GROUP]) & set(val[GROUP])
    bundle = {'preprocessor': preprocessor(inner, features).fit(clean_x(inner, features)),
              'text_encoder': text_encoder(inner), 'features': features}
    xi, xv = keras_inputs(bundle, inner), keras_inputs(bundle, val)
    model = build_keras(xi[0].shape[1], xi[1].shape[1], seed)
    y = inner[Y].to_numpy(dtype='int32')
    weights = {0: len(y) / (2 * (y == 0).sum()), 1: len(y) / (2 * y.sum())}
    h = model.fit(xi, y, validation_data=(xv, val[Y].to_numpy(dtype='int32')),
                  epochs=60, batch_size=32, class_weight=weights, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)])
    best_epoch = int(np.argmin(h.history['val_loss']) + 1)
    for epoch, (loss, val_loss) in enumerate(zip(h.history['loss'], h.history['val_loss']), 1):
        histories.append({'fold': tag, 'epoch': epoch, 'loss': loss, 'val_loss': val_loss, 'selected_epoch': best_epoch})
    bundle = {'preprocessor': preprocessor(train, features).fit(clean_x(train, features)),
              'text_encoder': text_encoder(train), 'features': features}
    xf = keras_inputs(bundle, train)
    model = build_keras(xf[0].shape[1], xf[1].shape[1], seed)
    y = train[Y].to_numpy(dtype='int32')
    assert xf[0].dtype == xf[1].dtype == np.float32 and xf[0].shape[0] == len(y)
    assert np.isfinite(xf[0]).all() and np.isfinite(xf[1]).all()
    weights = {0: len(y) / (2 * (y == 0).sum()), 1: len(y) / (2 * y.sum())}
    model.fit(xf, y, epochs=best_epoch, batch_size=32, class_weight=weights, verbose=0)
    bundle.update(model=model, best_epoch=best_epoch)
    return bundle


def predict(bundle, frame):
    if isinstance(bundle['model'], keras.Model):
        p = bundle['model'].predict(keras_inputs(bundle, frame), verbose=0).ravel()
    else:
        p = bundle['model'].predict_proba(bundle['preprocessor'].transform(clean_x(frame, bundle['features'])))[:, 1]
    assert p.shape == (len(frame),) and np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    return p


def metric_values(y, p, threshold):
    positive = p >= threshold
    tp = int((positive & (y == 1)).sum())
    fp = int((positive & (y == 0)).sum())
    fn = int((~positive & (y == 1)).sum())
    return {'threshold': float(threshold), 'precision': tp / (tp + fp) if tp + fp else np.nan,
            'recall': tp / (tp + fn) if tp + fn else np.nan,
            'popup_reduction': float((~positive).mean()), 'TP': tp, 'FP': fp, 'FN': fn,
            'TN': int((~positive & (y == 0)).sum()), 'shown': int(positive.sum()), 'n': len(y)}


def cluster_weights(groups, seed=SEED):
    """환자를 복원 추출하고 같은 환자의 경고를 같은 횟수만큼 복제합니다."""
    codes, unique = pd.factorize(groups)
    counts = np.random.default_rng(seed).multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=BOOTSTRAPS)
    return counts[:, codes]


def bounds(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return (float(np.quantile(a, .025)), float(np.quantile(a, .975)), len(a)) if len(a) else (np.nan, np.nan, 0)


def bootstrap_ap(y, p, weights):
    result = np.full(len(weights), np.nan)
    for i, w in enumerate(weights):
        if (w @ y) > 0 and (w @ (1 - y)) > 0:
            result[i] = average_precision_score(y, p, sample_weight=w)
    return result


def threshold_ci(y, p, threshold, weights):
    row = metric_values(y, p, threshold)
    alert = (p >= threshold).astype(int)
    tp, shown, positives = weights @ (alert * y), weights @ alert, weights @ y
    with np.errstate(invalid='ignore', divide='ignore'):
        metrics = {'precision': tp / shown, 'recall': tp / positives,
                   'popup_reduction': 1 - shown / weights.sum(axis=1)}
    for name, vals in metrics.items():
        lo, hi, valid = bounds(vals)
        row.update({name + '_low': lo, name + '_high': hi, name + '_valid_bootstraps': valid})
    # 관측 위음성 0건이면 비모수 bootstrap은 재현율 [1,1]로 퇴화합니다.
    # 독립 경고 가정의 보조 exact 구간을 함께 기록하되 환자 상관은 반영하지 않습니다.
    npos, success = row['TP'] + row['FN'], row['TP']
    row['recall_exact_low_independent'] = (float(beta.ppf(.025, success, npos-success+1)) if success else 0.0) if npos else np.nan
    row['recall_exact_high_independent'] = (float(beta.ppf(.975, success+1, npos-success)) if success < npos else 1.0) if npos else np.nan
    return row


def select_threshold(y, p):
    """OOF 재현율 90% 이상에서 팝업을 가장 많이 줄이는 임계값을 고릅니다."""
    candidates = [metric_values(y, p, t) for t in np.unique(np.r_[0, p])]
    feasible = [r for r in candidates if r['recall'] >= 0.9]
    return max(feasible, key=lambda r: (r['popup_reduction'], r['threshold']))['threshold']


def explain_lightgbm(bundle, test):
    """SHAP을 원래 변수로 합칩니다. 모델의 연관성을 설명하며 인과를 뜻하지 않습니다."""
    prep, model = bundle['preprocessor'], bundle['model']
    x = prep.transform(clean_x(test, bundle['features']))
    explainer = shap.TreeExplainer(model, model_output='raw', feature_perturbation='tree_path_dependent')
    values = explainer(x).values
    assert values.shape == x.shape
    base = float(np.asarray(explainer.expected_value).ravel()[-1])
    raw = model.predict(x, raw_score=True)
    assert np.allclose(values.sum(axis=1) + base, raw, atol=1e-5)
    assert np.allclose(expit(raw), predict(bundle, test))
    # 원-핫 열의 부호 있는 기여도를 원래 변수별로 합칩니다.
    nums = prep.transformers_[0][2]
    cats = prep.transformers_[1][2]
    encoder = prep.named_transformers_['cat'].named_steps['onehot']
    origins = list(nums) + [c for c, categories in zip(cats, encoder.categories_) for _ in categories]
    agg = pd.DataFrame(values.T).groupby(origins, sort=False).sum().T
    importance = agg.abs().mean().sort_values(ascending=False).rename('mean_abs_shap').rename_axis('feature').reset_index()
    labels = {'n_active_drugs': '점검 당시 유효 처방 성분 수', 'lab_tested_pre': '경고 전 관련 검사 보고 여부',
              '경고위험유형': '경고가 다루는 위험 유형', '점검구분': 'DUR 점검 구분', 'age': '연령',
              'cr_latest': '경고 전 마지막 크레아티닌', 'hb_latest': '경고 전 마지막 혈색소',
              'k_latest': '경고 전 마지막 칼륨', 'drug_class': '처방 약효군', 'bun_latest': '경고 전 마지막 BUN',
              '진료과': '진료과', 'los_at_alert': '경고 시점까지 입원 기간', 'alert_repeat_cnt': '이전 경고 반복 횟수',
              'n_diagnoses': '이전 진단 수', 'n_transfers_before': '이전 전실 횟수', 'route': '투여 경로',
              'ckd_flag': '만성 신장질환 기록', 'crp_latest': '경고 전 마지막 CRP', 'na_latest': '경고 전 마지막 나트륨',
              'sbp_latest': '경고 전 마지막 수축기 혈압', 'hr_latest': '경고 전 마지막 심박수',
              'rr_latest': '경고 전 마지막 호흡수', 'spo2_latest': '경고 전 마지막 산소포화도',
              'bt_latest': '경고 전 마지막 체온', 'o2flow_latest': '경고 전 마지막 산소 유량',
              'loc_latest': '경고 전 마지막 의식 상태', 'in_icu': '중환자실 여부', '성별': '성별',
              '입원경로': '입원 경로', 'is_night': '야간 경고 여부', 'is_weekend': '주말 경고 여부',
              'diabetes_flag': '당뇨병 기록', 'copd_flag': 'COPD 기록', 'recent_surgery_7d': '최근 7일 수술 기록',
              'n_prior_allergies': '이전 알레르기 기록 수'}
    importance['screen_label'] = importance.feature.map(labels).fillna(importance.feature)
    importance['screen_guidance'] = importance.screen_label.map(lambda s: f'의사·약사는 다음 정보를 원자료와 대조합니다: {s}.')
    importance['interpretation'] = '양의 SHAP만 해당 경고 점수를 올린 요인. 정상/비정상 및 인과관계를 뜻하지 않음.'
    save_csv(importance, 'shap_top_features')
    reasons = []
    for i in range(len(test)):
        for rank, (f, contribution) in enumerate(agg.iloc[i][agg.iloc[i] > 0].sort_values(ascending=False).head(3).items(), 1):
            value = test.iloc[i][f]
            reasons.append({'case_index': i, 'rank': rank, 'feature': f,
                            'observed_value': '미측정' if pd.isna(value) else str(value),
                            'shap_log_odds': contribution,
                            'screen_reason': f'{labels.get(f, f)}: {"미측정" if pd.isna(value) else value}. 이 정보가 모델의 중요 경고 점수를 높였습니다. 의사·약사는 관련 처방과 경고 전 자료를 확인해 주세요.'})
    save_csv(pd.DataFrame(reasons), 'local_screen_reasons_PRIVATE')
    fig = px.bar(importance.head(12).sort_values('mean_abs_shap'), x='mean_abs_shap', y='screen_label', orientation='h',
                 title='LightGBM: 중요 경고 점수에 기여한 변수 (hold-out)', labels={'mean_abs_shap': '평균 |SHAP| (log-odds)', 'screen_label': '변수'})
    return fig


def run():
    if os.environ.get("SMARTDUR_ALLOW_TRAINING") != "1":
        raise RuntimeError("학습하려면 SMARTDUR_ALLOW_TRAINING=1을 직접 설정하세요.")
    # Keras는 설치된 CUDA를 자동 선택할 수 있어 실행 장치를 명시합니다.
    with keras.device('cpu'):
        return _run()


def _run():
    OUT.mkdir(parents=True, exist_ok=True)
    dev, test, features, split = load_data()
    save_json({'seed': SEED, 'bootstrap_repeats': BOOTSTRAPS, 'backend': 'torch/CPU', 'reference_ap': 0.29,
               'threshold_policy': 'maximize reduction subject to development OOF recall >= 0.90; exploratory',
               'holdout_used_for_selection': False, 'features': features,
               'protocol_version': 'v2_label_audit_patient_split_377',
               'holdout_previously_reported': True,
               'target_label': 'original_proxy_label_A',
               'models': {'LR': 'C=.1 balanced', 'LightGBM': '120 trees, depth2, leaves4, lambda5 balanced',
                          'Keras': 'X35 + train-only char TF-IDF/SVD16; Dense16+8; dropout .25; L2 .01; inner grouped early stopping'}} , 'analysis_config')
    print('분할 검증:', json.dumps(split, ensure_ascii=False, default=str), flush=True)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(dev, dev[Y], dev[GROUP]))
    oof = {m: np.full(len(dev), np.nan) for m in MODELS}
    histories, fold_audit = [], []
    sensitivity_features = {'LR_no_active_drugs': [f for f in features if f != 'n_active_drugs'],
                            'LR_no_type': [f for f in features if f not in {'점검구분', '경고위험유형'}]}
    sensitivity_oof = {m: np.full(len(dev), np.nan) for m in sensitivity_features}
    for fold, (tr, va) in enumerate(folds, 1):
        train, valid = dev.iloc[tr], dev.iloc[va]
        assert not set(train[GROUP]) & set(valid[GROUP])
        fold_audit.append({'fold': fold, 'train_n': len(tr), 'valid_n': len(va),
                           'train_positive': int(train[Y].sum()), 'valid_positive': int(valid[Y].sum()), 'patient_overlap': 0})
        for name in MODELS:
            print(f'학습 fold {fold}/5: {name}', flush=True)
            bundle = fit_keras(train, features, SEED + fold, str(fold), histories) if name == 'Keras' else fit_tabular(train, features, name)
            oof[name][va] = predict(bundle, valid)
        for name, fs in sensitivity_features.items():
            sensitivity_oof[name][va] = predict(fit_tabular(train, fs, 'LR'), valid)
    assert all(np.isfinite(p).all() for p in oof.values())
    save_csv(pd.DataFrame(fold_audit), 'fold_audit')
    thresholds = {m: select_threshold(dev[Y].to_numpy(int), oof[m]) for m in MODELS}
    winner = max(MODELS, key=lambda m: average_precision_score(dev[Y], oof[m]))
    # 최종 평가 전에 모델 후보와 임계값을 고정합니다.
    save_json({'selected_by_dev_oof': winner, 'thresholds': thresholds,
               'oof_targets': {m: {'recall_target_met': metric_values(dev[Y].to_numpy(int), oof[m], thresholds[m])['recall'] >= .9,
                                  'reduction_30pct_target_met': metric_values(dev[Y].to_numpy(int), oof[m], thresholds[m])['popup_reduction'] >= .3}
                               for m in MODELS}}, 'locked_decisions')
    predictions, fitted = {}, {}
    for name in MODELS:
        print('개발 전체 재학습·최종 평가:', name, flush=True)
        fitted[name] = fit_keras(dev, features, SEED, 'final_inner', histories) if name == 'Keras' else fit_tabular(dev, features, name)
        predictions[name] = predict(fitted[name], test)
        if name == 'Keras':
            fitted[name]['model'].save(OUT / 'keras_model.keras')
            joblib.dump({k: v for k, v in fitted[name].items() if k != 'model'}, OUT / 'keras_preprocessing.joblib')
        else:
            joblib.dump(fitted[name], OUT / f'{name}_model.joblib')
    save_csv(pd.DataFrame(histories), 'keras_training_history')
    metrics, threshold_rows, selected_rows, pr_rows, paired = [], [], [], [], []
    for split_name, frame, scores in [('development_OOF', dev, oof), ('holdout', test, predictions)]:
        y = frame[Y].to_numpy(int)
        weights = cluster_weights(frame[GROUP])
        bootstrap_scores = {}
        export = pd.DataFrame({'case_index': np.arange(len(frame)), 'case_token': frame.case_token,
                               'y': y, 'risk_type': frame['경고위험유형'], **scores})
        for name, p in scores.items():
            export[name + '_display_tier'] = np.where(p >= thresholds[name], 'Yellow', 'Grey')
        save_csv(export, split_name + '_predictions_PRIVATE')
        for name, p in scores.items():
            bootstrap_scores[name] = bootstrap_ap(y, p, weights)
            lo, hi, valid = bounds(bootstrap_scores[name])
            prec, rec, _ = precision_recall_curve(y, p)
            metrics.append({'split': split_name, 'model': name, 'PR_AUC_AP': average_precision_score(y, p),
                            'ci_low': lo, 'ci_high': hi, 'valid_bootstraps': valid,
                            'PR_AUC_trapezoid': auc(rec, prec), 'n': len(y), 'positive': int(y.sum()), 'prevalence': y.mean()})
            pr_rows.extend({'split': split_name, 'model': name, 'recall': r, 'precision': v} for r, v in zip(rec, prec))
            # 한 행의 세 지표는 같은 모델·임계값·평가 집단에서 계산합니다.
            for t in np.unique(np.r_[np.linspace(0, 1, 41), thresholds[name]]):
                threshold_rows.append({'split': split_name, 'model': name, **threshold_ci(y, p, t, weights)})
            selected_rows.append({'split': split_name, 'model': name, **threshold_ci(y, p, thresholds[name], weights)})
        for a, b in [('LightGBM', 'LR'), ('Keras', 'LR'), ('Keras', 'LightGBM')]:
            lo, hi, n = bounds(bootstrap_scores[a] - bootstrap_scores[b])
            paired.append({'split': split_name, 'comparison': f'{a} - {b}',
                           'delta_AP': average_precision_score(y, scores[a]) - average_precision_score(y, scores[b]),
                           'ci_low': lo, 'ci_high': hi, 'valid_bootstraps': n})
    metric_df, threshold_df, selected_df = pd.DataFrame(metrics), pd.DataFrame(threshold_rows), pd.DataFrame(selected_rows)
    for data, name in [(metric_df, 'model_metrics'), (threshold_df, 'threshold_metrics'), (selected_df, 'selected_threshold_metrics'),
                       (pd.DataFrame(paired), 'paired_model_differences')]:
        save_csv(data, name)
    sensitivity = []
    for name, fs in sensitivity_features.items():
        p = sensitivity_oof[name]
        lo, hi, valid = bounds(bootstrap_ap(dev[Y].to_numpy(int), p, cluster_weights(dev[GROUP])))
        sensitivity.append({'model': name, 'split': 'development_OOF', 'PR_AUC_AP': average_precision_score(dev[Y], p),
                            'ci_low': lo, 'ci_high': hi, 'valid_bootstraps': valid})
    save_csv(pd.DataFrame(sensitivity), 'sensitivity_LR')
    run_quality_sensitivity(dev, test, features, folds, oof, predictions, split)
    subgroup, false_negatives = [], []
    for name, p in predictions.items():
        for risk in ['BLEED', 'QT']:
            mask = test['경고위험유형'].eq(risk).to_numpy()
            y = test.loc[mask, Y].to_numpy(int)
            if len(y):
                subgroup.append({'model': name, 'risk_type': risk, 'positive': int(y.sum()),
                                  **threshold_ci(y, p[mask], thresholds[name], cluster_weights(test.loc[mask, GROUP]))})
        for i in np.flatnonzero((test[Y].to_numpy() == 1) & (p < thresholds[name])):
            false_negatives.append({'model': name, 'case_index': int(i), 'risk_type': test.iloc[i]['경고위험유형'],
                                    'score': p[i], 'threshold': thresholds[name],
                                    'review': '의사·약사가 경고 당시 처방·검사와 정답 판정 경로를 원자료에서 전수 검토해야 합니다.'})
    save_csv(pd.DataFrame(subgroup), 'subgroup_metrics')
    save_csv(pd.DataFrame(false_negatives, columns=['model', 'case_index', 'risk_type', 'score', 'threshold', 'review']), 'false_negative_review_PRIVATE')
    label_summary = split['label_audit']
    figures = [('label_evidence_audit', px.bar(
        pd.read_csv(OUT / 'label_basis_summary.csv'), x='original_label_basis', y='n', color='analysis_population',
        title='기존 대리 라벨의 판정 근거 (라벨 확정과 임상적 참값은 다름)'))]
    chart = metric_df.assign(error_plus=metric_df.ci_high - metric_df.PR_AUC_AP,
                             error_minus=metric_df.PR_AUC_AP - metric_df.ci_low)
    fig = px.bar(chart, x='model', y='PR_AUC_AP', color='model', facet_col='split', error_y='error_plus', error_y_minus='error_minus',
                 title='LR · LightGBM · Keras: PR-AUC (AP)와 환자 부트스트랩 95% CI', range_y=[0, 1], hover_data=['n', 'positive', 'ci_low', 'ci_high'])
    fig.add_hline(y=.29, line_dash='dash', annotation_text='이전 X40 참고선 0.29 (동일 조건 비교 아님)')
    figures.append(('model_comparison', fig))
    for split_name, frame in [('development_OOF', dev), ('holdout', test)]:
        curve = pd.DataFrame(pr_rows).query('split == @split_name')
        fig = px.line(curve, x='recall', y='precision', color='model', line_shape='hv', title=f'{split_name}: Precision–Recall 곡선', range_x=[0, 1], range_y=[0, 1])
        fig.add_hline(y=frame[Y].mean(), line_dash='dot', annotation_text=f'전수 표시 {frame[Y].mean():.3f}')
        figures.append((split_name + '_pr', fig))
    long = threshold_df.melt(id_vars=['split', 'model', 'threshold'], value_vars=['precision', 'recall', 'popup_reduction'], var_name='metric', value_name='value')
    figures.append(('threshold_tradeoffs', px.line(long, x='threshold', y='value', color='metric', facet_col='model', facet_row='split', range_y=[0, 1],
                     title='같은 임계값의 정밀도·재현율·MAIN 팝업 감소율')))
    figures.append(('pareto', px.line(threshold_df, x='popup_reduction', y='recall', color='model', facet_col='split', markers=True,
                                     hover_data=['threshold', 'precision'], title='팝업 감소와 중요 경고 포착의 교환 관계', range_x=[0, 1], range_y=[0, 1])))
    hist = pd.DataFrame(histories).melt(id_vars=['fold', 'epoch', 'selected_epoch'], value_vars=['loss', 'val_loss'], var_name='dataset', value_name='BCE_plus_regularization')
    figures.append(('keras_learning_curves', px.line(hist, x='epoch', y='BCE_plus_regularization', color='dataset', facet_col='fold', facet_col_wrap=3,
                                                     title='Keras 내부 환자 분할: 학습·검증 손실 (학습에만 클래스 가중치 적용)')))
    figures.append(('shap_importance', explain_lightgbm(fitted['LightGBM'], test)))
    status_html = (f'<h2>세 가지 검증 절차</h2><ul><li>정답: 기존 Y 보존. 약사중재 수용 전체 {label_summary["accepted_source_all"]}건 중 엄격 연결 미확인 {label_summary["accepted_source_without_strict_link_all"]}건. 전문가 검토 확정 {label_summary["confirmed_reviews"]}건.</li>'
                   '<li>환자 분리: 학습 후보 381 → 환자 중복 제거 377건(양성 50), hold-out 77건(양성 9). 환자 중복 0명. 14일 관찰기간 분리 351건은 별도 민감도.</li>'
                   '<li>평가: OOF로 임계값 고정. Yellow/Grey는 표시 시나리오. 전체 1,030건 서비스 감소율과 고위험 재현율은 별도 검증.</li></ul>'
                   '<p>검사 악화 없음은 불필요한 경고라는 뜻이 아닙니다. 기존 대리 라벨 기준의 탐색 결과입니다. 이 hold-out은 이전에도 보고되어 새 독립 시험으로 간주하지 않습니다.</p>')
    write_charts(figures, status_html + '<p>환자 단위 95% CI. hold-out 양성 9건. 감소율의 분모는 판정된 MAIN입니다.</p>', OUT)
    versions = {m: importlib.metadata.version(m) for m in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'keras', 'torch', 'shap', 'plotly', 'joblib', 'nbformat', 'nbclient', 'ipykernel']}
    save_json({'python': platform.python_version(), 'platform': platform.platform(), 'packages': versions}, 'environment')
    (OUT / 'requirements-lock.txt').write_text('# Actual build is recorded in outputs/environment.json.\n' + '\n'.join(f'{m}=={v.split("+")[0] if m == "torch" else v}' for m, v in versions.items()) + '\n', encoding='utf-8')
    write_report(metric_df, selected_df, pd.DataFrame(paired), split, winner, sensitivity)
    print(metric_df.to_string(index=False), flush=True)
    print('분석 완료:', OUT, flush=True)
    return metric_df, selected_df


def run_quality_sensitivity(dev, test, features, folds, baseline_oof, baseline_test, split):
    """수정 정답을 발명하지 않고, 근거 제외·관찰기간 분리의 영향을 비교합니다."""
    rows, statuses = [], []

    def evaluate(frame, score, threshold, scenario, model, stage, cohort):
        y = frame[Y].to_numpy(int)
        if len(np.unique(y)) < 2:
            statuses.append({'scenario': scenario, 'model': model, 'split': stage, 'status': '두 클래스 부족'})
            return
        weights = cluster_weights(frame[GROUP])
        lo, hi, valid = bounds(bootstrap_ap(y, score, weights))
        rows.append({'scenario': scenario, 'model': model, 'split': stage, 'evaluation_cohort': cohort,
                     'PR_AUC_AP': average_precision_score(y, score), 'ci_low': lo, 'ci_high': hi,
                     'valid_bootstraps': valid, 'positive': int(y.sum()),
                     **threshold_ci(y, score, threshold, weights)})

    # 공통 평가 집단에서 A와 자동 연결감사 제외 민감도를 나란히 평가합니다.
    # 자동 제외는 y_reviewed를 채우거나 기존 Y를 0으로 바꾸지 않습니다.
    for scenario, column in [('linkage_audit_exclusion_NOT_clinical_Aprime', 'y_audit_candidate'),
                              ('expert_confirmed_only', 'y_reviewed')]:
        train_mask, test_mask = dev[column].notna(), test[column].notna()
        if scenario == 'expert_confirmed_only':
            train_mask &= dev.decision_status.eq('confirmed')
            test_mask &= test.decision_status.eq('confirmed')
        counts = dev.loc[train_mask, column].value_counts()
        if len(counts) != 2 or counts.min() < 5:
            statuses.append({'scenario': scenario, 'status': '분석 불가: 확정된 두 클래스가 충분하지 않음',
                             'train_n': int(train_mask.sum()), 'holdout_n': int(test_mask.sum())})
            continue
        target_dev = dev.copy()
        target_dev[Y] = target_dev[column]
        target_test = test.loc[test_mask].copy()
        target_test[Y] = target_test[column]
        for model in ['LR', 'LightGBM']:
            candidate_oof = np.full(len(dev), np.nan)
            feasible = True
            for tr, va in folds:
                usable = np.asarray(tr)[train_mask.iloc[tr].to_numpy()]
                train = target_dev.iloc[usable]
                if train[Y].nunique() < 2:
                    feasible = False
                    break
                bundle = fit_tabular(train, features, model)
                candidate_oof[va] = predict(bundle, dev.iloc[va])
            if not feasible:
                statuses.append({'scenario': scenario, 'model': model, 'status': '일부 fold 훈련에 두 클래스가 없음'})
                continue
            eval_dev = target_dev.loc[train_mask]
            p = candidate_oof[train_mask]
            threshold = select_threshold(eval_dev[Y].to_numpy(int), p)
            reference_threshold = select_threshold(dev[Y].to_numpy(int), baseline_oof[model])
            # 각 전략의 임계값은 해당 개발 OOF로만 결정합니다.
            evaluate(eval_dev, baseline_oof[model][train_mask], reference_threshold,
                     scenario + '__A_original_common', model, 'development_OOF', '동일 근거 확인 가능 부분집단')
            evaluate(eval_dev, p, threshold, scenario, model, 'development_OOF', '동일 근거 확인 가능 부분집단')
            fitted = fit_tabular(target_dev.loc[train_mask], features, model)
            score = predict(fitted, target_test) if len(target_test) else np.array([])
            evaluate(target_test, baseline_test[model][test_mask], reference_threshold,
                     scenario + '__A_original_common', model, 'holdout', '동일 근거 확인 가능 부분집단')
            evaluate(target_test, score, threshold, scenario, model, 'holdout', '동일 근거 확인 가능 부분집단')
        statuses.append({'scenario': scenario, 'status': '계산 완료', 'train_n': int(train_mask.sum()),
                         'holdout_n': int(test_mask.sum()), 'clinical_relabel': scenario == 'expert_confirmed_only'})

    # 377건과 14일 gap의 351건 비교: hold-out 77건과 기존 Y를 그대로 유지합니다.
    cutoff = pd.Timestamp(split['cutoff'])
    temporal = dev[dev[TIME].le(cutoff - pd.Timedelta(days=14))].reset_index(drop=True)
    for model in ['LR', 'LightGBM']:
        temporal_oof = np.full(len(temporal), np.nan)
        for tr, va in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(temporal, temporal[Y], temporal[GROUP]):
            assert not set(temporal.iloc[tr][GROUP]) & set(temporal.iloc[va][GROUP])
            temporal_oof[va] = predict(fit_tabular(temporal.iloc[tr], features, model), temporal.iloc[va])
        threshold = select_threshold(temporal[Y].to_numpy(int), temporal_oof)
        evaluate(temporal, temporal_oof, threshold, 'temporal_14d_embargo_351', model, 'development_OOF', '14일 gap의 개발 집단')
        temporal_score = predict(fit_tabular(temporal, features, model), test)
        evaluate(test, temporal_score, threshold, 'temporal_14d_embargo_351', model, 'holdout', '기존 Y의 고정 77건')
    save_csv(pd.DataFrame(rows), 'label_and_time_sensitivity')
    save_json({'scenarios': statuses, 'warning': '자동 연결감사 제외는 전문가 재판정 A′가 아닙니다. 평가 분모와 양성 수를 함께 비교하세요.'},
              'reviewed_label_comparison_status')


def write_report(metrics, selected, paired, split, winner, sensitivity):
    audit = split['label_audit']
    lines = ['# 의사·약사의 중요 DUR 경고 검토를 위한 모델링 결과', '',
             '## 분석 대상과 평가 설계', '',
             f'- MAIN 458건 중 양성 60건. 개발 {split["dev_n"]}건(양성 {split["dev_positive"]}), hold-out 77건(양성 9건).',
             f'- 시간 기준: MAIN 마지막 시각 {split["main_max"]}에서 2개월을 뺀 {split["cutoff"]} 이후를 hold-out으로 사용했습니다.',
             f'- 환자 중복 {split["overlapping_patients_removed_from_training"]}명의 과거 {split["purged_rows"]}건(양성 {split["purged_positive"]}건)을 개발 집단에서 제외했습니다.',
             '- 주 분석은 첨부안과 같은 377건(양성 50)입니다. 14일 관찰기간 분리로 26건을 더 제외한 351건(양성 49)은 별도 시간 민감도로 비교합니다. 주 분석은 라벨 관찰기간이 경계에 걸칠 수 있는 한계가 있습니다.',
             '- 이 hold-out은 이전 실행에서 이미 보고됐습니다. 이번 재실행은 새 독립 외부 검증을 뜻하지 않습니다. 모델·임계값은 여전히 개발 OOF로만 선택했습니다.',
             '- 개발 집단에서 환자 단위 StratifiedGroupKFold 5-fold OOF를 계산했습니다. 이 CV는 시간순 CV가 아니며 최종 hold-out만 미래 구간입니다.',
             '- 전처리와 텍스트 임베딩은 각 훈련 부분에서만 fit했습니다. Keras epoch 선택도 내부 환자 분할로 수행했습니다.',
             '- PR-AUC의 주 지표는 Average Precision(AP)입니다. 사다리꼴 적분은 별도 열로 저장해 두 정의를 혼동하지 않게 했습니다.',
             '- 95% CI는 환자를 2,000회 복원 추출한 percentile 구간입니다. 학습된 예측을 고정하므로 재학습·분할 선택의 불확실성까지 포함하지 않습니다.',
             '- 이전 X40 분석의 0.29는 참고선입니다. 같은 분할·변수·AP 정의가 보장되지 않아 0.29 대비 유의한 개선으로 주장하지 않습니다.', '',
             '## ① 정답의 근거 감사: 임상 확정과 구분', '',
             f'- 기존 is_important_alert는 수정하지 않았습니다. 약사중재 수용 경로 전체 {audit["accepted_source_all"]}건, MAIN {audit["accepted_source_main"]}건을 전수 감사했습니다.',
             f'- 처방번호 또는 상대처방번호 교집합, 동일 입원, 중재일시 ≥ 점검일시, 동일 DUR 유형, 수용·부분수용 조건을 동시에 충족하는 연결이 없는 사례는 전체 {audit["accepted_source_without_strict_link_all"]}건, MAIN {audit["accepted_source_without_strict_link_main"]}건입니다.',
             '- 연결 미확인은 잘못된 임상 정답의 확정이나 Y=0 근거가 아닙니다. 원본 중재에 점검ID가 없으므로 엄격 연결에 통과해도 정확한 경고 귀속을 보장하지 않습니다.',
             f'- Y=0 398건 중 자동 충돌 표지 {audit["zero_label_issues"]}건입니다. 원래 판정 순서가 수용중재·변경취소를 먼저 처리하므로 표지가 0건이어도 라벨의 임상 타당성이 입증되지 않습니다.',
             f'- Y=0 검토 표본 {audit["zero_sample_n"]}건: 출혈·QT 층별 표본과 K·CK 소수 유형 전수. 이는 실무 시작 표본이며 통계적 충분성을 뜻하지 않습니다.',
             f'- 전문가 검토 확정 {audit["confirmed_reviews"]}건, 보류 결정 {audit["deferred_reviews"]}건, 원래 Y에서 변경 {audit["changed_labels"]}건. y_reviewed는 서명·시각·근거를 갖춘 전문가 결정에만 기록합니다.',
             '- review_decisions_PRIVATE.csv에 decision_status(confirmed/deferred), y_reviewed(1/0/결측), 검토자·시각·근거·노트를 입력합니다. 자동으로 원래 Y를 복사하지 않습니다.',
             '- 의사·약사는 검사 악화가 없었다는 사실과 경고가 불필요했다는 판단을 구분해야 합니다. 현재 성능은 임상적 참값이 아니라 팀이 정의한 대리 라벨 A의 예측 성능입니다.',
             '- 자동 연결감사 제외 민감도는 y_audit_candidate를 사용하며 전문가 확정 A′로 부르지 않습니다. 전문가 검토가 부족하면 수정 A′ 성능은 산출 대기 상태로 남깁니다.', '',
             '## ② 환자 분리와 ③ 모델 평가', '',
             '원래 학습 후보 381건(양성 51)에서 중복 환자의 4건(양성 1)을 제외해 377건(양성 50), hold-out 77건(양성 9), 중복 환자 0명입니다.', '',
             '## PR-AUC와 95% 신뢰구간', '', '| 평가 | 모델 | AP [95% CI] | 양성/전체 |', '|---|---|---|---|']
    for r in metrics.itertuples():
        lines.append(f'| {r.split} | {r.model} | {r.PR_AUC_AP:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] | {r.positive}/{r.n} |')
    lines += ['', '## 개발 데이터에서 고정한 임계값의 최종 평가', '',
              '개발 OOF 재현율 90% 이상을 만족하는 임계값 중 팝업 감소율이 가장 큰 값을 고정했습니다. 이 90%는 탐색 목표이며 팀에서 정의할 고위험 분모의 안전성 KPI를 대체하지 않습니다.', '',
              '| 모델 | 임계값 | 정밀도 [95% CI] | 재현율 [95% CI] | MAIN 감소율 [95% CI] | TP / FN / 표시 |', '|---|---|---|---|---|---|']
    for _, r in selected[selected.split.eq('holdout')].iterrows():
        spans = [f'{r[k]:.1%} [{r[k+"_low"]:.1%}, {r[k+"_high"]:.1%}]' for k in ['precision', 'recall', 'popup_reduction']]
        lines.append(f'| {r.model} | {r.threshold:.4f} | ' + ' | '.join(spans) + f' | {int(r.TP)} / {int(r.FN)} / {int(r.shown)} |')
    lines += ['', '표시 경고가 0건인 경우 정밀도는 정의할 수 없어 NaN으로 둡니다. CI에서도 분모가 0인 복원 표본은 제외하고 유효 반복 수를 CSV에 기록했습니다.',
              '위음성이 0건이면 관측 재현율 9/9의 환자 부트스트랩 구간은 [100%, 100%]로 퇴화합니다. 이는 모집단 재현율 100%의 증거가 아닙니다. 독립 경고를 가정한 9/9의 보조 exact 95% 구간은 약 66.4–100%이며 환자 내 상관을 반영하지 않습니다. 보조 구간도 CSV에 별도 기록했습니다.',
              '점수 ≥ 고정 임계값은 Yellow, 미만은 Grey로 저장합니다. 이는 표시 시나리오이며 자동 처방 변경·실서비스 억제 결정이 아닙니다. OOF 팝업 감소 30% 충족 여부는 locked_decisions.json에 별도 기록합니다.', '',
              '## 의사·약사가 해석할 결론', '',
              f'의사·약사는 개발 OOF AP로 선택한 {winner} 후보를 추가 검증할 수 있습니다. hold-out 결과를 보고 모델이나 임계값을 다시 선택하지 않았습니다.',
              '의사·약사는 hold-out 양성이 9건뿐이므로 점수 한 개보다 위 신뢰구간과 놓친 경고 수를 함께 확인해야 합니다. 한 건의 차이만으로도 재현율이 약 11.1%p 달라집니다.',
              '의사·약사는 LightGBM의 SHAP 상위 변수를 처방 화면의 확인 항목으로 활용할 수 있습니다. 개별 경고에서 양의 SHAP인 변수만 점수를 높인 요인으로 제시하며, 기여도는 log-odds 단위입니다.',
              '의사·약사는 SHAP을 약물 위해의 인과관계나 임상적 이상 판정으로 받아들이지 않고 경고 당시 처방·검사 원자료를 확인해야 합니다. 처방 화면 문구는 검토용 초안입니다.',
              '의사·약사는 false_negative_review_PRIVATE.csv의 모든 위음성 경고를 원자료에서 검토해야 합니다. 이 분석은 누락 건을 추출했으며 전문가의 임상 검토 완료를 뜻하지 않습니다.', '',
              '## 적용 범위와 남은 검증', '',
              '- 감소율 = 1 − 표시 MAIN 경고 / 평가 MAIN 경고입니다. 정답이 있는 MAIN만 평가하므로 전체 DUR 1,030건의 실제 팝업 감소율과 다릅니다.',
              '- RED_RULE 61건은 모델에 넣지 않으며 항상 표시하는 기존 정책을 유지합니다. EXPLORE 333건·TEAM_DECISION 75건·MAIN 보류 103건에 대해서는 억제 결정을 내리지 않았습니다.',
              '- 검사 여부에 따른 선택 편향과 약사중재·처방변경 판정 비대칭이 남아 있습니다. selection_bias.csv와 selection_by_type.csv는 편향 점검용 요약입니다.',
              '- 전문가 확정 A′·B안 라벨과 고위험 확정 정의가 부족하므로 수정 정답 성능이나 고위험 재현율을 임의로 만들지 않았습니다. label_and_time_sensitivity.csv는 자동 연결감사 제외와 14일 gap의 영향을 별도로 비교합니다.',
              '- n_active_drugs의 사후 종료시각 수정 가능성을 확인해야 합니다. 제외한 LR 민감도와 경고 유형을 제외한 LR 민감도를 별도로 계산했습니다. 모든 모델의 시점 누수 해소를 보증하지 않습니다.',
              '- LR·LightGBM은 주 비교이고 Keras/NLP는 추가 실험입니다. 모두 같은 377건·환자 fold·평가지표로 비교하지만, Keras는 텍스트가 추가되어 알고리즘만의 순수 효과 비교는 아닙니다.',
              '- 첨부 문서의 임시 LR OOF AP 0.238·임계값 0.08 등은 참고 결과이며 정확한 설정·예측 파일이 없어 검증값으로 복사하지 않았습니다. 여기에는 이 코드로 실제 재실행한 값만 제시했습니다.',
              '- 클래스 가중치로 학습한 출력은 보정된 위해 확률이 아닙니다. risk_score를 상대적인 중요 경고 점수로 해석하며 임계값은 배포 승인값이 아닙니다.',
              '- 출혈·QT 하위군 CI는 subgroup_metrics.csv에 저장했습니다. 소수 하위군의 구간은 특히 불안정하며 다른 유형으로 일반화하지 않습니다.',
              '- 0.29와의 비교, 작은 표본, CI 중첩만으로 모델 간 우열을 확정하지 않습니다. 같은 환자 복원 표본의 AP 차이는 paired_model_differences.csv에서 확인합니다.', '',
              '## 재현과 파일 사용', '',
              '```powershell', 'python src/run_analysis.py', '```', '',
              '대화형 시각화는 dashboard.html을 브라우저에서 열어 확인합니다. 모든 그래프는 plotly.express로 생성하고 Plotly JS를 포함해 오프라인으로 열 수 있습니다.',
              'PRIVATE 표기가 있는 예측·위음성·개별 설명 파일과 학습 모델은 승인된 팀 내부에서만 사용합니다. 원본 환자번호·처방번호·점검ID는 결과 파일에 기록하지 않았지만 의료 특성과 모델 자체는 보호 대상입니다.', '',
              '## API 정의 참고', '',
              '- [scikit-learn Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)',
              '- [Keras 백엔드 설정](https://keras.io/getting_started/)',
              '- [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)', '']
    quality = pd.read_csv(OUT / 'label_and_time_sensitivity.csv')
    comparison = ['', '## 연결감사·시간 민감도 비교', '',
                  '연결감사 제외와 원래 A는 같은 근거 확인 가능 부분집단에서 비교합니다. 351건 시간 민감도는 기존 Y의 hold-out 77건으로 평가합니다. 서로 다른 평가 집단의 AP 차이를 라벨 개선 효과로 단정하지 않습니다.', '',
                  '| 시나리오 | 모델 | 평가 | 양성/전체 | AP [95% CI] |', '|---|---|---|---|---|']
    scenario_labels = {
        'linkage_audit_exclusion_NOT_clinical_Aprime__A_original_common': '기존 A: 공통 평가 집단',
        'linkage_audit_exclusion_NOT_clinical_Aprime': '연결감사 제외: 공통 평가 집단',
        'temporal_14d_embargo_351': '14일 관찰기간 분리: 351건 학습',
        'expert_confirmed_only': '전문가 확정 정답',
        'expert_confirmed_only__A_original_common': '기존 A: 전문가 검토 집단'}
    for row in quality.itertuples():
        comparison.append(f'| {scenario_labels.get(row.scenario, row.scenario)} | {row.model} | {row.split} | {row.positive}/{row.n} | {row.PR_AUC_AP:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}] |')
    comparison += ['', '전문가 확정 A′ 비교의 준비 상태는 reviewed_label_comparison_status.json에 기록했습니다. 확인된 정답이 부족하면 성능을 산출하지 않습니다.', '']
    at = lines.index('## 의사·약사가 해석할 결론')
    lines[at:at] = comparison
    (OUT / '분석결과_보고서.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    run()
