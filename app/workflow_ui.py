"""Gradio workflow tabs shared by both language views and both local copies."""
from collections import Counter
from datetime import datetime
import html
import json
import sqlite3
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image

from industrial_inspection.workflow import (InspectionStore, WorkflowError, run_batch,
    compare_models, quality_challenge, transform_image, inspect_and_store)
from app.localization import locale, localize_report


def build_workflow_tabs(root: Path, language: str):
    data=locale(language);t=data['ui'];products=data['products']
    choices=[(products[c]['label'],c) for c in ('bottle','metal_nut','tile')]
    store=InspectionStore(root)
    translate=lambda r:localize_report(r,language)
    inference_queue={'concurrency_id':'inspection-selection','concurrency_limit':1,'api_name':False}

    def bind_run(button,callback,inputs,outputs,controls,clear):
        started=button.click(lambda:(*[gr.update(interactive=False) for _ in controls],*clear()),
            outputs=[*controls,*outputs],queue=False,api_name=False)
        completed=started.then(callback,inputs,outputs,**inference_queue)
        completed.then(lambda:[gr.update(interactive=True) for _ in controls],outputs=controls,queue=False,api_name=False)

    def error(exc):
        code=exc.code if isinstance(exc,WorkflowError) else 'inference_failed'
        return t['error_prefix']+t['errors'].get(code,code)

    def format_number(value):
        return f'{value:.4f}' if isinstance(value,(int,float)) else ''

    def ratio(report):
        score,threshold=report.get('score'),report.get('image_threshold')
        return score/threshold if isinstance(score,(int,float)) and isinstance(threshold,(int,float)) and threshold>0 else None

    def example(product):
        files=sorted((root/'samples'/products[product]['sample_dir']).glob('*.png'))
        chosen=next((p for p in files if not p.name.startswith('good_')),None)
        if chosen is None:return None
        with Image.open(chosen) as image:
            picture=image.convert('RGB').copy();picture.info['inspection_filename']=chosen.name
            return picture

    def reference_choices(product):
        files=sorted((root/'samples'/products[product]['sample_dir']).glob('*.png'))
        first={}
        for path in files:first.setdefault(path.stem.rsplit('_',1)[0],path.stem)
        return [(products[product]['defect_labels'].get(kind,kind),stem) for kind,stem in sorted(first.items(),key=lambda pair:(pair[0]!='good',pair[0]))]

    def load_reference(product,stem):
        allowed={value for _,value in reference_choices(product)}
        if stem not in allowed:return None
        path=root/'samples'/products[product]['sample_dir']/(stem+'.png')
        with Image.open(path) as opened:
            image=opened.convert('RGB').copy();image.info['inspection_filename']=path.name
            return image

    def demo_batch(product):
        paths=sorted((root/'samples'/products[product]['sample_dir']).glob('*.png'))
        chosen=[p for p in paths if p.name.startswith('good_')][:2]+[p for p in paths if not p.name.startswith('good_')][:2]
        return [str(p) for p in chosen]

    with gr.Tab(t['batch'],elem_id='batch-tab'):
        gr.Markdown(t['batch_help'])
        with gr.Row():
            batch_product=gr.Dropdown(choices,value='bottle',label=t['product'],elem_id='batch-product')
            batch_method=gr.Dropdown([(t['patch'],'patch'),(t['pixel'],'pixel')],value='patch',label=t['method'],elem_id='batch-method')
        batch_files=gr.File(file_count='multiple',file_types=['image'],type='filepath',label=t['files'],elem_id='batch-files')
        with gr.Row():
            batch_demo=gr.Button(t['demo_batch'],elem_id='batch-demo')
            batch_run=gr.Button(t['run_batch'],variant='primary',elem_id='batch-run')
        batch_status=gr.Markdown(t['batch_ready'],elem_id='batch-status')
        batch_table=gr.Dataframe(headers=[t[k] for k in ('id','filename','decision','score','threshold','ratio','status')],type='array',interactive=False,wrap=True,label=t['batch_results'],elem_id='batch-results')
        with gr.Row():
            batch_csv=gr.DownloadButton(t['batch_csv'],interactive=False,elem_id='batch-csv')
            batch_zip=gr.DownloadButton(t['batch_zip'],interactive=False,elem_id='batch-zip')

        def batch_callback(files,product,method,progress=gr.Progress()):
            try:
                result=run_batch(root,files,product,method,progress=progress,report_transform=translate)
                rows=[]
                for r in result['reports']:
                    decision=t.get(r.get('decision'),t['failed'])
                    if r['error']:decision=t['errors'].get(r['error'],r['error'])
                    rows.append([r['inspection_id'][:8],r['filename'],decision,format_number(r.get('score')),
                        format_number(r.get('image_threshold')),format_number(ratio(r)),t[r['status']]])
                success=sum(r['status']=='success' for r in result['reports'])
                summary=t['batch_done'].format(batch=result['batch_id'][:8],success=success,failed=len(rows)-success,total=len(rows))
                return rows,summary,gr.update(value=result['csv'],interactive=True),gr.update(value=result['zip'],interactive=True)
            except (WorkflowError,OSError,RuntimeError,ImportError,sqlite3.Error) as exc:
                return [],error(exc),gr.update(value=None,interactive=False),gr.update(value=None,interactive=False)
        batch_demo.click(demo_batch,batch_product,batch_files,api_name=False)
        def clear_batch():
            return [],t['batch_ready'],gr.update(value=None,interactive=False),gr.update(value=None,interactive=False)
        bind_run(batch_run,batch_callback,[batch_files,batch_product,batch_method],
            [batch_table,batch_status,batch_csv,batch_zip],
            [batch_files,batch_product,batch_method,batch_demo,batch_run],clear_batch)
        batch_files.change(clear_batch,outputs=[batch_table,batch_status,batch_csv,batch_zip],queue=False,api_name=False)
        for component in (batch_product,batch_method):
            component.input(clear_batch,outputs=[batch_table,batch_status,batch_csv,batch_zip],queue=False,api_name=False)

    with gr.Tab(t['history'],elem_id='history-tab') as history_tab:
        gr.Markdown(t['history_help'])
        with gr.Row():
            history_product=gr.Dropdown([(t['all_products'],'all'),*choices,(products['synthetic']['label'],'synthetic')],value='all',label=t['product'],elem_id='history-product')
            history_review=gr.Dropdown([(t[k],k) for k in ('all','unreviewed','reviewed')],value='all',label=t['review_filter'],elem_id='history-filter')
            history_source=gr.Dropdown([(t['all'],'all'),*[(t[k],v) for k,v in [('single','single'),('batch','batch'),('compare','comparison'),('challenge','challenge'),('daily','daily_case')]]],value='all',label=t['source_filter'],elem_id='history-source')
        with gr.Row():
            refresh=gr.Button(t['refresh'],elem_id='history-refresh')
            export=gr.Button(t['prepare_export'],elem_id='history-export')
            history_download=gr.DownloadButton(t['download_history'],interactive=False,elem_id='history-download')
        metrics=gr.HTML(elem_id='history-metrics')
        summary=gr.Markdown(elem_id='history-summary')
        with gr.Row():
            decisions=gr.BarPlot(x='decision',y='count',title=t['decision_chart'],x_title=t['chart_decision'],y_title=t['chart_count'],height=250,elem_id='decision-chart')
            days=gr.BarPlot(x='day',y='count',title=t['day_chart'],x_title=t['chart_day'],y_title=t['chart_count'],height=250,elem_id='day-chart')
        history_table=gr.Dataframe(headers=[t[k] for k in ('id','filename','product','model','decision','status','human')],interactive=False,type='array',wrap=True,label=t['history_rows'],elem_id='history-table')
        record_choice=gr.Dropdown(choices=[],label=t['record_choice'],interactive=True,elem_id='record-choice')
        loaded_id=gr.State(None)
        with gr.Row():
            record_input=gr.Image(label=t['record_input'],interactive=False,height=300,elem_id='record-input')
            record_overlay=gr.Image(label=t['record_overlay'],interactive=False,height=300,elem_id='record-overlay')
        with gr.Accordion(t['record_report'],open=False):
            record_report=gr.JSON(label=t['record_report'],elem_id='record-report')
        gr.Markdown(t['review_help'])
        with gr.Row():
            reviewer=gr.Textbox(label=t['reviewer'],max_lines=1,elem_id='reviewer-name')
            verdict=gr.Dropdown([(t[k],k) for k in ('pass','review','reject','retake')],value='review',label=t['verdict'],elem_id='review-verdict')
        comment=gr.Textbox(label=t['comment'],lines=2,max_lines=5,elem_id='review-comment')
        save=gr.Button(t['save_review'],variant='primary',elem_id='review-save')
        review_status=gr.Markdown(t['review_pending'],elem_id='review-status')
        review_log=gr.Dataframe(headers=[t['review_time'],t['reviewer'],t['human'],t['comment']],interactive=False,type='array',wrap=True,label=t['review_log'],elem_id='review-log')

        def refresh_history(product,review_filter,source,current):
            records=store.query(product=product,review=review_filter,source=source)
            counts={'count_total':len(records),'count_success':sum(r['status']=='success' for r in records),
                    'count_failed':sum(r['status']=='failed' for r in records),'count_reviewed':sum(r['verdict'] is not None for r in records)}
            cards='<div class="metrics-grid">'+''.join(f'<div class="metric"><span class="label">{html.escape(t[key])}</span><span class="value">{value}</span></div>' for key,value in counts.items())+'</div>'
            dc=Counter(r['decision'] for r in records if r['status']=='success')
            daycounts=Counter(r['created_at'][:10] for r in records)
            decision_df=pd.DataFrame([{'decision':t[k],'count':dc[k]} for k in ('pass','review','reject','undetermined')])
            day_df=pd.DataFrame([{'day':day,'count':count} for day,count in sorted(daycounts.items())],columns=['day','count'])
            visible=records[:100]
            rows=[[r['id'][:8],r['filename'],products.get(r['product'],{}).get('label',r['product']),r['model_id'],
                   t.get(r['decision'],''),t[r['status']],t.get(r['verdict'],t['unreviewed'])] for r in visible]
            options=[(f"{r['id'][:8]} | {r['filename']} | {r['model_id']}",r['id']) for r in visible]
            chosen=current if any(value==current for _,value in options) else (options[0][1] if options else None)
            return cards,t['summary'].format(visible=len(rows),total=len(records)),rows,decision_df,day_df,gr.update(choices=options,value=chosen)

        def load_record(inspection_id):
            if not inspection_id:return None,None,None,[],None
            try:
                item=store.get(inspection_id);folder=store.assets(inspection_id)
                images=[]
                for name in ('input.png','overlay.png'):
                    path=folder/name;images.append(str(path) if path.exists() else None)
                log=[[r['created_at'],r['reviewer'],t[r['verdict']],r['comment']] for r in item['reviews']]
                return *images,translate(item['report']),log,inspection_id
            except WorkflowError as exc:
                gr.Warning(error(exc));return None,None,None,[],None
        filters=[history_product,history_review,history_source,record_choice]
        history_outputs=[metrics,summary,history_table,decisions,days,record_choice]
        for event in (refresh.click,history_tab.select):
            event(refresh_history,filters,history_outputs,api_name=False)
        for component in (history_product,history_review,history_source):
            component.input(refresh_history,filters,history_outputs,api_name=False)
            component.input(lambda:gr.update(value=None,interactive=False),outputs=history_download,api_name=False,queue=False)
        record_choice.change(load_record,record_choice,[record_input,record_overlay,record_report,review_log,loaded_id],api_name=False)
        export.click(lambda p,r,s:gr.update(value=str(store.export_history(product=p,review=r,source=s)),interactive=True),
                     [history_product,history_review,history_source],history_download,api_name=False)
        def save_review(loaded,selected,name,result,notes):
            try:
                if loaded!=selected or not loaded:raise WorkflowError('selection_changed')
                item=store.review(loaded,name,result,notes)
                rows=[[r['created_at'],r['reviewer'],t[r['verdict']],r['comment']] for r in item['reviews']]
                return t['review_saved'],rows
            except WorkflowError as exc:return error(exc),gr.skip()
        saved=save.click(save_review,[loaded_id,record_choice,reviewer,verdict,comment],[review_status,review_log],api_name=False)
        saved.then(refresh_history,filters,history_outputs,api_name=False)

    with gr.Tab(t['compare'],elem_id='compare-tab'):
        gr.Markdown(t['compare_help'])
        compare_product=gr.Dropdown(choices,value='bottle',label=t['product'],elem_id='compare-product')
        reference_options=reference_choices('bottle')
        selected_reference=next((value for _,value in reference_options if not value.startswith('good_')),None)
        compare_case=gr.Dropdown(reference_options,value=selected_reference,label=t['compare_case'],elem_id='compare-case')
        compare_input=gr.Image(value=load_reference('bottle',selected_reference),type='pil',sources=['upload'],label=t['compare_image'],format='png',height=260,elem_id='compare-input')
        with gr.Row():
            compare_demo=gr.Button(t['demo_image'],elem_id='compare-demo')
            compare_run=gr.Button(t['run_compare'],variant='primary',elem_id='compare-run')
        with gr.Row():
            pixel_image=gr.Image(label=t['pixel_overlay'],interactive=False,height=300,elem_id='compare-pixel')
            patch_image=gr.Image(label=t['patch_overlay'],interactive=False,height=300,elem_id='compare-patch')
        compare_table=gr.Dataframe(headers=[t[k] for k in ('model','score','threshold','ratio','decision')],type='array',interactive=False,label=t['compare_table'],elem_id='compare-results')
        compare_status=gr.Markdown(elem_id='compare-status')
        def compare_callback(image,product,reference):
            try:
                if reference and image is not None:
                    expected=load_reference(product,reference)
                    actual=transform_image(image,1,0)
                    expected=transform_image(expected,1,0) if expected is not None else None
                    if expected is None or actual.size!=expected.size or actual.tobytes()!=expected.tobytes():
                        raise WorkflowError('no_image')
                outputs=compare_models(root,image,product,report_transform=translate,filename=(reference+'.png') if reference else 'uploaded_image.png')
                rows=[[t[m],format_number(r['score']),format_number(r['image_threshold']),format_number(ratio(r)),t[r['decision']]] for m,(_,r) in zip(('pixel','patch'),outputs)]
                return outputs[0][0],outputs[1][0],rows,t['compare_done'].format(count=2)
            except (WorkflowError,OSError,RuntimeError,ImportError,ValueError,sqlite3.Error) as exc:return None,None,[],error(exc)
        compare_demo.click(lambda product,reference:load_reference(product,reference) if reference else example(product),[compare_product,compare_case],compare_input,api_name=False)
        compare_case.input(load_reference,[compare_product,compare_case],compare_input,api_name=False)
        compare_product.input(lambda product:(gr.update(choices=reference_choices(product),value=None),None),compare_product,[compare_case,compare_input],api_name=False)
        compare_input.upload(lambda:gr.update(value=None),outputs=compare_case,api_name=False,queue=False)
        bind_run(compare_run,compare_callback,[compare_input,compare_product,compare_case],[pixel_image,patch_image,compare_table,compare_status],
            [compare_input,compare_product,compare_case,compare_demo,compare_run],lambda:(None,None,[],''))
        for component in (compare_input,compare_product):
            component.change(lambda:(None,None,[],''),outputs=[pixel_image,patch_image,compare_table,compare_status],api_name=False,queue=False)

    with gr.Tab(t['challenge'],elem_id='challenge-tab'):
        gr.Markdown(t['challenge_help'])
        challenge_product=gr.Dropdown(choices,value='bottle',label=t['product'],elem_id='challenge-product')
        with gr.Row():
            challenge_input=gr.Image(type='pil',sources=['upload'],label=t['challenge_image'],format='png',height=260,elem_id='challenge-input')
            challenge_preview=gr.Image(label=t['preview'],interactive=False,height=260,elem_id='challenge-preview')
        with gr.Row():
            brightness=gr.Slider(.1,1.8,value=1,step=.1,label=t['brightness'],elem_id='challenge-brightness')
            blur=gr.Slider(0,5,value=0,step=.5,label=t['blur'],elem_id='challenge-blur')
        with gr.Row():
            challenge_demo=gr.Button(t['demo_image'],elem_id='challenge-demo')
            challenge_run=gr.Button(t['run_challenge'],variant='primary',elem_id='challenge-run')
        with gr.Row():
            original_image=gr.Image(label=t['original_overlay'],interactive=False,height=280,elem_id='challenge-original')
            changed_image=gr.Image(label=t['changed_overlay'],interactive=False,height=280,elem_id='challenge-changed')
        challenge_table=gr.Dataframe(headers=[t[k] for k in ('variant','brightness_value','sharpness','ratio','decision')],type='array',interactive=False,label=t['challenge_table'],elem_id='challenge-results')
        challenge_status=gr.Markdown(elem_id='challenge-status')
        def preview_callback(image,b,r):
            try:preview=transform_image(image,b,r) if image is not None else None
            except (WorkflowError,ValueError):preview=None
            return preview,None,None,[],''
        for component in (challenge_input,brightness,blur,challenge_product):
            component.change(preview_callback,[challenge_input,brightness,blur],
                [challenge_preview,original_image,changed_image,challenge_table,challenge_status],api_name=False,
                concurrency_id='inspection-selection',concurrency_limit=1,trigger_mode='always_last')
        challenge_demo.click(example,challenge_product,challenge_input,api_name=False)
        def challenge_callback(image,product,b,r):
            try:
                changed,outputs=quality_challenge(root,image,product,b,r,report_transform=translate)
                rows=[[t[v],format_number(report['brightness']),format_number(report['sharpness']),format_number(ratio(report)),t[report['decision']]] for v,(_,report) in zip(('original','transformed'),outputs)]
                return changed,outputs[0][0],outputs[1][0],rows,t['challenge_done']
            except (WorkflowError,OSError,RuntimeError,ImportError,ValueError,sqlite3.Error) as exc:return None,None,None,[],error(exc)
        bind_run(challenge_run,challenge_callback,[challenge_input,challenge_product,brightness,blur],
            [challenge_preview,original_image,changed_image,challenge_table,challenge_status],
            [challenge_input,challenge_product,brightness,blur,challenge_demo,challenge_run],lambda:(gr.skip(),None,None,[],''))

    with gr.Tab(t['daily'],elem_id='daily-tab'):
        gr.Markdown('### '+t['daily_title'])
        gr.Markdown(t['daily_help'])
        gr.Markdown(t['daily_steps'])
        daily_input=gr.Image(value=example('tile'),type='pil',sources=['upload'],format='png',height=300,label=t['daily_source'],elem_id='daily-input')
        daily_origin=gr.State('reference_example')
        daily_source=gr.Markdown(t['daily_reference'],elem_id='daily-source')
        with gr.Row():
            daily_demo=gr.Button(t['daily_load'],elem_id='daily-demo')
            daily_run=gr.Button(t['daily_run'],variant='primary',elem_id='daily-run')
        with gr.Row():
            daily_pixel=gr.Image(label=t['pixel_overlay'],interactive=False,height=300,elem_id='daily-pixel')
            daily_patch=gr.Image(label=t['patch_overlay'],interactive=False,height=300,elem_id='daily-patch')
        daily_table=gr.Dataframe(headers=[t[k] for k in ('model','score','threshold','ratio','decision')],type='array',interactive=False,label=t['compare_table'],elem_id='daily-results')
        daily_status=gr.Markdown(elem_id='daily-status')
        def load_daily():return example('tile'),'reference_example',t['daily_reference']
        daily_demo.click(load_daily,outputs=[daily_input,daily_origin,daily_source],api_name=False)
        daily_input.upload(lambda:('uploaded_photo',t['daily_uploaded']),outputs=[daily_origin,daily_source],api_name=False,queue=False)
        daily_input.change(lambda:(None,None,[],''),outputs=[daily_pixel,daily_patch,daily_table,daily_status],api_name=False,queue=False)
        def daily_callback(image,origin):
            try:
                results=[inspect_and_store(root,image,'tile',method,source='daily_case',
                    metadata={'type':'daily_tile_case','image_source':origin,'benchmark_result':False},
                    report_transform=translate) for method in ('pixel','patch')]
                rows=[[t[m],format_number(r['score']),format_number(r['image_threshold']),format_number(ratio(r)),t[r['decision']]] for m,(_,r) in zip(('pixel','patch'),results)]
                ids=', '.join(r['inspection_id'][:8] for _,r in results)
                return results[0][0],results[1][0],rows,t['daily_done'].format(ids=ids)
            except (WorkflowError,OSError,RuntimeError,ImportError,ValueError,sqlite3.Error) as exc:return None,None,[],error(exc)
        bind_run(daily_run,daily_callback,[daily_input,daily_origin],[daily_pixel,daily_patch,daily_table,daily_status],
            [daily_input,daily_demo,daily_run],lambda:(None,None,[],''))
