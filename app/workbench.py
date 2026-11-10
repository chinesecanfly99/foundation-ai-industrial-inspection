"""Shared presentation and navigation for the two inspection workbenches."""
import gradio as gr
import csv
import html
import json
import logging
import os
import sqlite3
import uuid

from industrial_inspection.workflow import inspect_and_store, WorkflowError
from industrial_inspection.workspace import results_dir

NAV_LABELS = {
    'en': {'lab': 'Demo Lab', 'records': 'Batch & Records'},
    'zh': {'lab': '演示实验室', 'records': '批量与记录'},
}

CSS = """
body { background:linear-gradient(145deg,#e4effa 0%,#f2f6fc 52%,#e4f3ef 100%) fixed!important; }
.gradio-container { width:100%!important; max-width:1440px!important; min-width:0!important; margin:auto; box-sizing:border-box;
  background:transparent!important; --background-fill-primary:#f4f8fc; --background-fill-secondary:#e6eff9;
  --body-text-size:17px; --input-text-size:16px; --block-label-text-size:15px; --button-large-text-size:19px; --button-medium-text-size:16px; }
.gradio-container main.app, .gradio-container main.app > .wrap { min-width:0; }
.gradio-container main.app { padding-top:12px; }
#workbench-header { align-items:center; gap:10px; }
#workbench-brand { border:0; }
.project-hero { color:#fff; background:linear-gradient(110deg,#112040 0%,#23468f 72%,#007f87 100%);
  border-radius:18px; border-left:6px solid #50e1d3; padding:18px 24px 20px; }
.project-eyebrow { color:#8df4df; font-size:13px; font-weight:800; letter-spacing:.13em; margin-bottom:4px; }
.project-hero h1 { color:#fff!important; font-size:clamp(27px,3vw,42px)!important; line-height:1.13; margin:0 0 7px!important; }
.project-hero p { color:#e3f1ff; font-size:16px; margin:0; line-height:1.45; }
#language-toggle { width:100%; min-width:96px; font-weight:700; border-color:#aec6e8; }
#inspection-tabs > [role=tablist], #lab-tabs > [role=tablist], #record-tabs > [role=tablist] { gap:7px; flex-wrap:wrap; }
#inspection-tabs [role=tab] { font-size:16px!important; font-weight:700; padding:10px 15px; min-height:42px; }
#inspection-tabs [role=tab][aria-selected=true] { color:#174592!important; background:#e8f1ff!important; border-radius:9px 9px 0 0; border-bottom-color:#245bb8!important; }
#lab-tabs [role=tab][aria-selected=true] { color:#005d58!important; background:#ddf7ef!important; border-bottom-color:#008272!important; }
#record-tabs [role=tab][aria-selected=true] { color:#805205!important; background:#fff1d4!important; border-bottom-color:#ae6900!important; }
#inspection-tabs h3 { font-size:23px!important; margin:6px 0!important; line-height:1.25; }
#single-tab > div > .row > .column { padding:12px 14px; border:1px solid #cddded; border-radius:14px; background:#edf5ff; }
#single-tab > div > .row > .column:nth-child(2) { background:#eaf6f3; border-color:#cbe3dc; }
#inspection-tabs label, #inspection-tabs .label-wrap, #inspection-tabs th, #inspection-tabs td { font-size:15px!important; }
#inspection-tabs input, #inspection-tabs textarea { font-size:16px!important; }
#rule-trace, #inspection-json { max-width:100%; min-width:0!important; overflow:auto!important; }
.decision-card { border:1px solid #d9e5f2; border-left:5px solid #3472ce; border-radius:13px; background:#edf5ff; color:#1b3e73; padding:12px 16px; }
.decision-card h2 { font-size:26px!important; margin:0 0 3px!important; line-height:1.2; color:inherit!important; }
.decision-card p { margin:0; line-height:1.45; font-size:16px; }
.decision-pass { background:#eaf9f0; border-color:#3da464; color:#14522b; }
.decision-review { background:#fff5dd; border-color:#c17c10; color:#714700; }
.decision-reject { background:#ffefee; border-color:#cb5347; color:#842620; }
.decision-undetermined { background:#f1efff; border-color:#8664ca; color:#4f3584; }
.metrics-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; }
.metric { background:#fff; border:1px solid #d7e1ee; border-top:3px solid #62bcb6; border-radius:10px; padding:9px 11px; }
.metric .label { display:block; color:#455b73; font-size:13px; margin-bottom:4px; }
.metric .value { color:#16344d; font-size:22px; font-weight:800; }
.metric .unit { font-size:13px; font-weight:400; color:#455b73; }
.rule-reasons { color:#3a5067; line-height:1.5; padding:3px 1px; }
.rule-reasons ul { margin:5px 0; padding-left:23px; }
#inspection-evidence { border:1px solid #cfe3ed; border-left:5px solid #008272; padding:12px 16px; border-radius:13px; }
.evidence-saved { color:#145943; font-weight:700; font-size:17px; margin:0 0 7px; }
.evidence-saved a { color:#174592; text-decoration:underline; margin-left:12px; }
.evidence-comparison { border-collapse:collapse; width:100%; color:#183452; }
.evidence-comparison th,.evidence-comparison td { text-align:left; padding:8px 10px; border-bottom:1px solid #d6e3ec; }
.evidence-comparison th { background:#e8f3fc; }
@media(max-width:760px) {
  #workbench-header { align-items:stretch; }
  #workbench-brand { flex-basis:100%; }
  #inspection-tabs [role=tab] { padding:9px 10px; font-size:15px!important; }
  .metrics-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
}
"""


def build_header(language, text):
    zh = language == 'zh'
    subtitle = ('查看瓶口、金属螺母与瓷砖的异常位置和处理建议。' if zh else
                'Find visual anomalies in bottle openings, metal nuts and tile surfaces.')
    title = '工业视觉质检工作台' if zh else 'Industrial Inspection Workbench'
    with gr.Row(elem_id='workbench-header'):
        with gr.Column(scale=15, min_width=250, elem_id='workbench-brand'):
            gr.HTML(f'<div class="project-hero"><div class="project-eyebrow">FOUNDATION OF AI</div><h1>{title}</h1><p>{subtitle}</p></div>')
        with gr.Column(scale=1, min_width=110):
            button=gr.Button(text['switch'], elem_id='language-toggle')
            route='/' if zh else '/zh/'
            button.click(fn=None, js=f"() => {{ window.location.assign('{route}' + location.search); return []; }}", queue=False, api_name=False)
            with gr.Accordion('说明' if zh else 'About', open=False, elem_id='language-note'):
                gr.Markdown(text['switch_note'])
                if os.environ.get("INSPECTION_CLOUD_EDITION") == "1":
                    gr.Markdown("线上上传和记录只属于当前浏览器，服务器休眠或重启后会清空。需要保留时请先下载。"
                                if zh else "Online uploads and records belong to this browser. Download your results before the service restarts or goes idle.")
                gr.Markdown('热图表示视觉异常，分数不是缺陷概率。' if zh else 'Heatmaps show visual anomalies. Scores are not defect probabilities.')


def build_evidence(language):
    zh=language=='zh'
    with gr.Group(visible=False, elem_id='inspection-evidence') as group:
        gr.Markdown('### 3 · 模型对比与记录' if zh else '### 3 · Comparison and saved record')
        summary=gr.HTML(elem_id='comparison-summary')
        with gr.Accordion('查看第二个模型的热图' if zh else 'Second model heatmap',open=False):
            secondary=gr.Image(label='对照模型热图' if zh else 'Comparison heatmap', interactive=False, height=294, elem_id='secondary-overlay')
    return group, summary, secondary


def reset_evidence():
    return gr.update(visible=False), '', None


def attach_evidence(report, image, product, root, language):
    """Reuse the primary result and store the companion under its record ID."""
    from app.localization import locale, localize_report

    if not isinstance(report, dict) or not report.get('inspection_id'):
        return reset_evidence(), None
    zh=language=='zh'
    t=locale(language)['ui']
    first_id=report['inspection_id']
    first_method='pixel' if report['model_id'].endswith('_pixel') else 'patch'
    method='patch' if first_method=='pixel' else 'pixel'
    route='/zh/' if zh else '/'
    link='打开历史与复核' if zh else 'Open records and review'
    records=[report]
    secondary=None
    note=''
    exports=None
    if product in ('bottle','metal_nut','tile'):
        try:
            secondary, other=inspect_and_store(root,image,product,method,source='comparison',
                filename=report.get('filename','uploaded_image.png'),
                metadata={'type':'workbench_comparison','primary_inspection_id':first_id},
                report_transform=lambda r:localize_report(r,language))
            records.append(other)
            out=results_dir(root)/'ui_exports'/uuid.uuid4().hex
            out.mkdir(parents=True,exist_ok=False)
            report_path=out/'inspection_with_comparison.json'
            report_path.write_text(json.dumps({'primary':report,'comparison':other},ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            csv_path=out/'inspection_with_comparison.csv'
            fields=['role','inspection_id','model_id','decision','score','image_threshold','anomaly_area_ratio','critical_overlap_ratio']
            with csv_path.open('w',encoding='utf-8-sig',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
                writer.writeheader()
                writer.writerows({'role':role,**record} for role,record in zip(('primary','comparison'),records))
            exports=(str(report_path),str(csv_path))
        except (WorkflowError,OSError,RuntimeError,ImportError,ValueError,sqlite3.Error):
            logging.getLogger(__name__).exception('Companion inspection or report could not be prepared')
            note='<p>对照检测或合并报告未完成；仍可查看主模型结果。</p>' if zh else '<p>Comparison or combined export could not finish. The primary result is still available.</p>'
    ids=' / '.join(html.escape(r['inspection_id'][:8]) for r in records)
    saved='已保存记录' if zh else 'Saved records'
    markup=f'<p class="evidence-saved">{saved}: {ids}<a href="{route}?demo=history&amp;record={html.escape(first_id)}">{link}</a></p>'
    if len(records)==2:
        headings=('模型','异常分数 / 阈值','处理建议') if zh else ('Model','Score / threshold','Recommendation')
        rows=[]
        for selected,r in zip((first_method,method),records):
            score,threshold=r['score'],r['image_threshold']
            ratio=f'{score/threshold:.2f}×' if threshold else '—'
            rows.append((t[selected],ratio,t[r['decision']]))
        markup+='<table class="evidence-comparison"><thead><tr>'+''.join(f'<th scope="col">{html.escape(s)}</th>' for s in headings)+'</tr></thead><tbody>'
        markup+=''.join('<tr>'+''.join(f'<td>{html.escape(s)}</td>' for s in row)+'</tr>' for row in rows)+'</tbody></table>'
    return (gr.update(visible=True),markup+note,secondary),exports
