"""원본 Y는 보존하고 약사중재의 연결 근거와 Y=0 검토 대상을 감사합니다.

자동 감사는 임상 재판정이 아닙니다. y_reviewed는 서명·시각·근거가 있는
전문가 결정 파일을 읽었을 때만 채웁니다. 감사 정보는 모델 입력에서 제외합니다.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

KEY = '점검ID'
Y = 'is_important_alert'
SOURCE_NAMES = {1: '약사중재 수용', 2: '14일 내 ADR', 3: '7일 내 관련 검사 악화',
                4: '처방 변경·취소로 보류', 5: '관련 검사 후 악화 없음', 9: '기타 보류'}


def token(alert_id):
    """원본 ID 대신 사용하는 연결 토큰입니다. 완전한 익명화를 뜻하지 않습니다."""
    return hashlib.sha256(('tm5-review-v1:' + str(alert_id)).encode()).hexdigest()[:20]


def order_ids(row):
    return {str(row[c]).strip() for c in ['처방번호', '상대처방번호']
            if pd.notna(row[c]) and str(row[c]).strip()}


def normalize_kind(value):
    return ''.join(str(value).split()) if pd.notna(value) else ''


def linkage_flags(candidates, alert):
    """세 조건은 같은 중재 행에서 동시에 성립해야 합니다."""
    admission = candidates['입원번호'].eq(alert['입원번호'])
    after = candidates['중재일시'].ge(pd.Timestamp(alert['점검일시']))
    kind = candidates['관련DUR구분'].map(normalize_kind).eq(normalize_kind(alert['점검구분']))
    return admission, after, kind


def validate_decisions(decisions, known_tokens):
    required = {'case_token', 'decision_status', 'y_reviewed', 'reviewed_by',
                'reviewed_at', 'evidence_reference', 'review_note'}
    if not required.issubset(decisions.columns):
        raise ValueError(f'검토 결정 파일에 필요한 열이 없습니다: {sorted(required - set(decisions.columns))}')
    if decisions.case_token.duplicated().any() or not set(decisions.case_token).issubset(known_tokens):
        raise ValueError('검토 결정에 중복 또는 알 수 없는 case_token이 있습니다.')
    if not decisions.decision_status.isin(['pending', 'confirmed', 'deferred']).all():
        raise ValueError('decision_status는 pending, confirmed, deferred만 허용합니다.')
    numeric = pd.to_numeric(decisions.y_reviewed, errors='coerce')
    if (decisions.y_reviewed.notna() & ~numeric.isin([0, 1])).any():
        raise ValueError('y_reviewed는 0, 1 또는 결측(보류)만 허용합니다.')
    confirmed = decisions.decision_status.eq('confirmed')
    if numeric[confirmed].isna().any() or numeric[~confirmed].notna().any():
        raise ValueError('confirmed에만 0/1 정답을 기록하고 pending/deferred는 결측으로 두세요.')
    completed = decisions.decision_status.isin(['confirmed', 'deferred'])
    for col in ['reviewed_by', 'reviewed_at', 'evidence_reference', 'review_note']:
        if decisions.loc[completed, col].fillna('').astype(str).str.strip().eq('').any():
            raise ValueError(f'완료한 검토에는 {col} 근거가 필요합니다.')
    if pd.to_datetime(decisions.loc[completed, 'reviewed_at'], errors='coerce', utc=True).isna().any():
        raise ValueError('검토 시각을 해석할 수 없습니다. ISO 8601 형식을 사용하세요.')
    result = decisions.copy()
    result['y_reviewed'] = numeric
    return result


def audit_labels(package: Path, master: pd.DataFrame, main: pd.DataFrame,
                 output: Path, decisions_path: Path):
    """MASTER 1,030건을 감사하고, MAIN 토큰별 감사 결과와 요약을 반환합니다."""
    with zipfile.ZipFile(next((package / '05_원본데이터').glob('*.zip'))) as archive:
        raw = {prefix: pd.read_csv(archive.open(next(n for n in archive.namelist()
                    if n.startswith(prefix + '_')))) for prefix in ['15', '16']}
    alerts, interventions = raw['15'], raw['16']
    assert alerts[KEY].is_unique
    frame = master.merge(alerts[[KEY, '조치구분']], on=KEY, how='left', validate='one_to_one')
    interventions['중재일시'] = pd.to_datetime(interventions['중재일시'], errors='coerce')
    assert interventions['중재일시'].notna().all(), '중재일시 파싱 오류'
    accepted = interventions[interventions['의사수용여부'].isin(['수용', '부분수용'])]
    indexed = {}
    for index, row in accepted.iterrows():
        for oid in order_ids(row):
            indexed.setdefault(oid, set()).add(index)

    records = []
    for _, alert in frame.iterrows():
        timestamp = pd.Timestamp(alert['점검일시'])
        candidate_ids = set().union(*(indexed.get(oid, set()) for oid in order_ids(alert)))
        candidates = accepted.loc[sorted(candidate_ids)]
        admission, after, kind = linkage_flags(candidates, alert)
        valid = admission & after & kind
        # 원본에는 중재와 경고를 직접 연결하는 점검ID가 없습니다.
        # 이후 중재라도 정확한 같은 경고의 결과인지 전문가 확인이 필요합니다.
        issues = []
        source = int(alert.y_source)
        c1 = source == 1
        if c1 and not valid.any():
            if candidates.empty:
                issues.append('수용중재_처방연결없음')
            else:
                if not admission.any():
                    issues.append('입원불일치')
                if (admission & ~after).any():
                    issues.append('경고전_수용중재')
                if (admission & after & ~kind).any():
                    issues.append('DUR유형불일치또는누락')
                issues.append('동일입원_경고후_동일유형_수용중재없음')
        if alert[Y] == 0:
            if (admission & after).any():
                issues.append('Y0_경고후_수용중재있음')
            if (admission & ~after).any():
                issues.append('Y0_경고전_수용중재있음')
            if alert['조치구분'] in ['처방변경', '처방취소']:
                issues.append('Y0_처방변경취소와충돌')
        records.append({'case_token': token(alert[KEY]), 'analysis_population': alert.analysis_population,
                        'risk_type': alert['경고위험유형'], 'original_y': alert[Y], 'y_source': source,
                        'original_label_basis': SOURCE_NAMES[source], 'accepted_order_matches': len(candidates),
                        'accepted_before_same_admission': int((admission & ~after).sum()),
                        'accepted_after_same_admission': int((admission & after).sum()),
                        'accepted_after_same_type': int(valid.sum()),
                        'label_issue': ';'.join(issues), 'label_confidence': '연결근거_재검토필요' if issues else '자동감사만_임상미확정',
                        'y_audit_candidate': np.nan if c1 and not valid.any() else alert[Y],
                        'y_reviewed': np.nan, 'decision_status': 'pending', 'review_note': '',
                        'reviewed_by': '', 'reviewed_at': '', 'evidence_reference': ''})
    audit = pd.DataFrame(records)
    assert len(audit) == 1030 and audit.case_token.is_unique
    main_tokens = {token(v) for v in main[KEY]}
    audit['in_main_458'] = audit.case_token.isin(main_tokens)

    # 검토 파일은 최초에만 만듭니다. 재실행으로 사람이 입력한 결정을 덮어쓰지 않습니다.
    decision_cols = ['case_token', 'decision_status', 'y_reviewed', 'reviewed_by',
                     'reviewed_at', 'evidence_reference', 'review_note']
    if not decisions_path.exists():
        audit[decision_cols].to_csv(decisions_path, index=False, encoding='utf-8-sig')
    decisions = validate_decisions(pd.read_csv(decisions_path), set(audit.case_token))
    # 검토 파일이 바뀔 때마다 내용 해시별 스냅샷을 남겨 반복 수정 이력을 보존합니다.
    content = decisions_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    history = decisions_path.parent / 'review_history_PRIVATE'
    history.mkdir(exist_ok=True)
    snapshot = history / f'{digest}.csv'
    if not snapshot.exists():
        snapshot.write_bytes(content)
    else:
        assert snapshot.read_bytes() == content, '검토 이력의 내용 해시 불일치'
    for col in decision_cols[1:]:
        mapped = audit.case_token.map(decisions.set_index('case_token')[col])
        audit[col] = mapped if col == 'y_reviewed' else mapped.fillna('pending' if col == 'decision_status' else '')
    audit['label_changed'] = (audit.decision_status.eq('confirmed')
                              & (audit.original_y.isna() | audit.y_reviewed.ne(audit.original_y)))
    audit.loc[audit.decision_status.eq('confirmed'), 'label_confidence'] = '전문가_검토확정'
    audit.loc[audit.decision_status.eq('deferred'), 'label_confidence'] = '전문가_판정보류'

    # Y=0 소수 유형은 전수, 출혈·QT는 고정 시드로 총 40건의 층별 표본을 선정합니다.
    zero = audit[audit.in_main_458 & audit.original_y.eq(0)]
    rare = zero[zero.risk_type.isin(['K', 'CK'])]
    sampled = set(rare.case_token)
    remaining = max(0, 40 - len(sampled))
    common = zero[zero.risk_type.isin(['BLEED', 'QT'])]
    for risk, group in common.groupby('risk_type'):
        count = min(len(group), max(1, round(remaining * len(group) / max(len(common), 1))))
        sampled.update(group.sample(n=count, random_state=42).case_token)
    audit['zero_sample_review'] = audit.case_token.isin(sampled)
    audit['review_required'] = audit.y_source.eq(1) | audit.label_issue.ne('') | audit.zero_sample_review
    queue = audit[audit.review_required].copy()
    queue['priority'] = np.select([queue.in_main_458 & queue.y_source.eq(1),
                                  queue.label_issue.ne(''), queue.y_source.eq(1)], [1, 2, 3], default=4)
    queue = queue.sort_values(['priority', 'risk_type', 'case_token'])
    for data, name in [(audit, 'label_audit_PRIVATE'), (queue, 'label_review_queue_PRIVATE'),
                       (audit[audit.label_changed], 'label_change_log_PRIVATE')]:
        data.to_csv(output / f'{name}.csv', index=False, encoding='utf-8-sig')
    basis = audit.groupby(['analysis_population', 'y_source', 'original_label_basis'], dropna=False).size().reset_index(name='n')
    basis.to_csv(output / 'label_basis_summary.csv', index=False, encoding='utf-8-sig')
    summary = {'master_n': 1030, 'original_main_n': 458, 'original_main_positive': 60,
               'accepted_source_all': int(audit.y_source.eq(1).sum()),
               'accepted_source_main': int((audit.in_main_458 & audit.y_source.eq(1)).sum()),
               'accepted_source_without_strict_link_all': int((audit.y_source.eq(1) & audit.accepted_after_same_type.eq(0)).sum()),
               'accepted_source_without_strict_link_main': int((audit.in_main_458 & audit.y_source.eq(1) & audit.accepted_after_same_type.eq(0)).sum()),
               'zero_label_issues': int((audit.in_main_458 & audit.original_y.eq(0) & audit.label_issue.ne('')).sum()),
               'zero_sample_n': int(audit.zero_sample_review.sum()), 'review_queue_n': len(queue),
               'confirmed_reviews': int(audit.decision_status.eq('confirmed').sum()),
               'deferred_reviews': int(audit.decision_status.eq('deferred').sum()),
               'changed_labels': int(audit.label_changed.sum()),
               'clinical_validation_complete': False,
               'limitations': ['엄격 연결은 동일 입원·경고 이후·동일 DUR 유형·수용/부분수용 기준의 기계적 점검입니다.',
                              '원본 중재에 점검ID가 없어 정확한 경고 귀속과 임상적 타당성은 전문가 검토가 필요합니다.',
                              '검사 악화 없음은 불필요한 경고 또는 위해 부재의 임상적 확정이 아닙니다.',
                              '처방변경 조치에 별도 시각이 없어 변경 시점은 원자료 추가 확인이 필요합니다.',
                              'y_audit_candidate는 불확실한 C1 근거를 제외하는 민감도용이며 전문가 수정 A′ 라벨이 아닙니다.']}
    (output / 'label_audit_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return audit, summary
