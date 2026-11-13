"""원래 650px 크기와 Plotly 기본 글꼴·분할 배치로 그래프를 저장합니다."""
from pathlib import Path

import pandas as pd
import plotly.express as px


def standard_figure(fig):
    fig.update_layout(autosize=True, width=None, height=650,
                      font=dict(family='Malgun Gothic, Arial', size=None),
                      title=dict(font=None, x=None, xanchor=None, automargin=None),
                      hoverlabel=dict(font=None),
                      margin=dict(t=110, b=80, l=80, r=80), legend=None)
    fig.update_xaxes(title_font=None, tickfont=None, automargin=None)
    fig.update_yaxes(title_font=None, tickfont=None, automargin=None)
    fig.update_annotations(font=None)
    for trace in fig.data:
        if trace.type == 'scatter':
            trace.update(line=dict(width=None), marker=dict(size=None))
    return fig


def chart_panels(name, figure, output):
    """기존 결과로 원래의 다중 패널 그래프를 재구성합니다."""
    if name == 'model_comparison':
        frame = pd.read_csv(output / 'model_metrics.csv')
        frame = frame.assign(error_plus=frame.ci_high-frame.PR_AUC_AP,
                             error_minus=frame.PR_AUC_AP-frame.ci_low)
        figure = px.bar(frame, x='model', y='PR_AUC_AP', color='model', facet_col='split',
                        error_y='error_plus', error_y_minus='error_minus', range_y=[0, 1],
                        hover_data=['n', 'positive', 'ci_low', 'ci_high'],
                        title='LR · LightGBM · Keras: PR-AUC (AP)와 환자 부트스트랩 95% CI')
        figure.add_hline(y=.29, line_dash='dash', annotation_text='이전 X40 참고선 0.29 (동일 조건 비교 아님)')
    elif name == 'threshold_tradeoffs':
        frame = pd.read_csv(output / 'threshold_metrics.csv')
        long = frame.melt(id_vars=['split', 'model', 'threshold'],
                          value_vars=['precision', 'recall', 'popup_reduction'],
                          var_name='metric', value_name='value')
        figure = px.line(long, x='threshold', y='value', color='metric', facet_col='model',
                         facet_row='split', range_y=[0, 1],
                         title='같은 임계값의 정밀도·재현율·MAIN 팝업 감소율')
    elif name == 'pareto':
        frame = pd.read_csv(output / 'threshold_metrics.csv')
        figure = px.line(frame, x='popup_reduction', y='recall', color='model', facet_col='split',
                         markers=True, hover_data=['threshold', 'precision'], range_x=[0, 1],
                         range_y=[0, 1], title='팝업 감소와 중요 경고 포착의 교환 관계')
    elif name == 'keras_learning_curves':
        frame = pd.read_csv(output / 'keras_training_history.csv')
        long = frame.melt(id_vars=['fold', 'epoch', 'selected_epoch'], value_vars=['loss', 'val_loss'],
                          var_name='dataset', value_name='BCE_plus_regularization')
        figure = px.line(long, x='epoch', y='BCE_plus_regularization', color='dataset', facet_col='fold',
                         facet_col_wrap=3, title='Keras 내부 환자 분할: 학습·검증 손실 (학습에만 클래스 가중치 적용)')
    yield standard_figure(figure)


def write_charts(figures, status_html, output: Path):
    html_parts = []
    for index, (name, source) in enumerate(figures):
        fig = next(chart_panels(name, source, output))
        fig.write_html(output / f'{name}.html', include_plotlyjs=True, auto_open=False)
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=True if index == 0 else False))
    header = '<h1>의사·약사의 중요 경고 검토를 돕는 탐색 분석</h1>' + status_html
    (output / 'dashboard.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>TM5 모델링</title><body>' + header + ''.join(html_parts) + '</body></html>', encoding='utf-8')
