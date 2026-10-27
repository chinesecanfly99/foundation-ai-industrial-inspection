"""Persistent workflow invariants, failure accounting and immutable AI results."""
import csv
import json
from pathlib import Path
import sqlite3
import zipfile

import numpy as np
from PIL import Image
import pytest

from industrial_inspection import pipeline
from industrial_inspection.workflow import (InspectionStore, WorkflowError, run_batch,
    compare_models, transform_image, quality_challenge)


def card(root,product='bottle',method='patch',source=None):
    folder=root/'models'/f'{product}_{method}';folder.mkdir(parents=True,exist_ok=True)
    (folder/'model.npz').write_bytes(b'frozen numeric artifact fixture')
    (folder/'model_card.json').write_text(json.dumps({'dataset_source':source or f'mvtec_ad/{product}'}))


def report(product='bottle',model='bottle_patch'):
    return {'product_category':product,'model_id':model,'decision':'review','score':.5,
            'image_threshold':.4,'pixel_threshold':.3,'brightness':80.,'sharpness':8.,
            'trace':[{'rule':'R3','selected':True,'condition':True}],'reasons':['test']}


@pytest.fixture
def fake_model(monkeypatch):
    def inspect(image,directory):
        product=directory.name.rsplit('_',1)[0]
        return image.copy(),report(product,directory.name)
    monkeypatch.setattr(pipeline,'inspect_image',inspect)


def test_reviews_are_append_only_and_never_change_model_report(tmp_path):
    store=InspectionStore(tmp_path);image=Image.new('RGB',(64,64),'gray')
    saved=store.save(report(),image,image)
    path=store.assets(saved['inspection_id'])/'report.json';before=path.read_bytes()
    store.review(saved['inspection_id'],'Reviewer one','reject','A visible defect')
    item=store.review(saved['inspection_id'],'Reviewer two','retake','Image quality check')
    assert len(item['reviews'])==2
    assert path.read_bytes()==before
    assert item['report']['decision']=='review'
    assert store.query()[0]['verdict']=='retake'
    assert store.query(review='unreviewed')==[]
    assert len(store.query(review='reviewed'))==1
    assert InspectionStore(tmp_path).get(saved['inspection_id'])['report']==saved


def test_two_project_stores_are_independent(tmp_path):
    first,second=InspectionStore(tmp_path/'one'),InspectionStore(tmp_path/'two')
    saved=first.save(report(),None,None)
    assert len(first.query())==1 and second.query()==[]
    with pytest.raises(WorkflowError,match='record_missing'):second.get(saved['inspection_id'])


def test_failed_records_cannot_receive_review_and_ids_cannot_escape_store(tmp_path):
    store=InspectionStore(tmp_path)
    saved=store.save({'error':'invalid_image'},None,None,error='invalid_image')
    with pytest.raises(WorkflowError,match='failed_record_review'):
        store.review(saved['inspection_id'],'Tester','pass','')
    with pytest.raises(WorkflowError,match='record_missing'):store.get('../../outside')
    with pytest.raises(WorkflowError,match='invalid_review'):store.review(saved['inspection_id'],'','pass','')


def test_batch_counts_invalid_image_and_still_inspects_later_images(tmp_path,fake_model):
    card(tmp_path)
    paths=[]
    for name in ['one.png','broken.png','two.png']:
        path=tmp_path/name
        if name=='broken.png':path.write_bytes(b'not an image')
        else:Image.new('RGB',(64,64),'gray').save(path)
        paths.append(path)
    result=run_batch(tmp_path,paths,'bottle')
    assert [r['status'] for r in result['reports']]==['success','failed','success']
    assert len(InspectionStore(tmp_path).query())==3
    with zipfile.ZipFile(result['zip']) as archive:
        assert archive.testzip() is None
        manifest=json.loads(archive.read('batch_manifest.json'))
        assert manifest['count']==3 and manifest['failed']==1 and manifest['success']==2
        assert len([n for n in archive.namelist() if n.endswith('report.json')])==3
        assert all('..' not in Path(n).parts for n in archive.namelist())
    with Path(result['csv']).open(encoding='utf-8-sig',newline='') as stream:
        assert len(list(csv.DictReader(stream)))==3


def test_invalid_product_model_rejected_before_writing_records(tmp_path):
    card(tmp_path,source='mvtec_ad/tile')
    with pytest.raises(WorkflowError,match='invalid_model'):run_batch(tmp_path,[],'bottle')
    assert not (tmp_path/'results/workflow').exists()


def test_batch_limit_rejects_entire_request_without_partial_writes(tmp_path):
    card(tmp_path)
    with pytest.raises(WorkflowError,match='batch_count'):run_batch(tmp_path,[tmp_path/'x.png']*21,'bottle')
    assert not (tmp_path/'results/workflow').exists()


def test_csv_exports_user_text_safely_and_filters_all_records(tmp_path):
    store=InspectionStore(tmp_path)
    first=store.save(report(),None,None,filename='=formula.png')
    store.save(report('tile','tile_patch'),None,None,source='challenge')
    store.review(first['inspection_id'],'=reviewer','reject','@unsafe formula')
    path=store.export_history(product='bottle',review='reviewed',source='single')
    with path.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==1 and rows[0]['filename']=="'=formula.png"
    assert rows[0]['reviewer']=="'=reviewer" and rows[0]['comment']=="'@unsafe formula"


def test_comparison_uses_same_image_two_frozen_models(tmp_path,fake_model):
    for method in ['pixel','patch']:card(tmp_path,method=method)
    files=list((tmp_path/'models').glob('*/model.npz'));before=[p.read_bytes() for p in files]
    image=Image.new('RGB',(64,64),'gray')
    output=compare_models(tmp_path,image,'bottle')
    assert [r['model_id'] for _,r in output]==['bottle_pixel','bottle_patch']
    assert output[0][1]['image_sha256']==output[1][1]['image_sha256']
    assert all(r['source']=='comparison' for r in InspectionStore(tmp_path).query())
    assert [p.read_bytes() for p in files]==before


def test_quality_change_alters_pixels_keeps_original_and_records_parameters(tmp_path,fake_model):
    card(tmp_path)
    pixels=np.indices((64,64)).sum(axis=0)%2*255
    image=Image.fromarray(np.repeat(pixels[:,:,None],3,axis=2).astype('uint8'))
    before=image.tobytes()
    changed,outputs=quality_challenge(tmp_path,image,'bottle',.2,2)
    assert image.tobytes()==before and changed.tobytes()!=before
    assert outputs[0][1]['image_sha256']!=outputs[1][1]['image_sha256']
    assert outputs[1][1]['experiment']['brightness_multiplier']==.2
    assert outputs[1][1]['experiment']['benchmark_result'] is False
    assert len(InspectionStore(tmp_path).query(source='challenge'))==2
    with pytest.raises(WorkflowError,match='invalid_transform'):transform_image(image,0,0)


def test_report_localization_changes_only_presentational_fields():
    from app.localization import localize_report
    original={**report(),'trace':[{'rule':'R3','selected':True,'condition':True,'reason':'old'}]}
    en=localize_report(original,'en');zh=localize_report(en,'zh')
    assert en['label_en']=='Manual review' and 'label_zh' not in en
    assert 'label_zh' in zh and 'label_en' not in zh
    for key in ['decision','score','image_threshold','pixel_threshold','model_id','product_category']:
        assert en[key]==zh[key]==original[key]
    assert original['trace'][0]['reason']=='old'


@pytest.mark.parametrize('default_language',['en','zh'])
def test_language_routes_redirect_before_root_mount_and_preserve_root_api(tmp_path,default_language):
    import gradio as gr
    from fastapi.testclient import TestClient
    from app.server import create_server
    with gr.Blocks() as primary:gr.Markdown('Primary')
    with gr.Blocks() as alternate:gr.Markdown('Alternate')
    path='/zh' if default_language=='en' else '/en'
    app=create_server(primary,alternate,default_language=default_language)
    with TestClient(app) as client:
        redirect=client.get(path,follow_redirects=False)
        assert redirect.status_code in (307,308) and redirect.headers['location']==path+'/'
        assert client.get('/').status_code==200
        assert client.get(path+'/').status_code==200
        assert client.get('/config').status_code==200
        assert client.get(path+'/config').status_code==200


@pytest.mark.parametrize('view_name',['app.app','app.alt_ui'])
def test_each_language_view_keeps_four_bottle_cases_and_workflow_tabs(tmp_path,view_name):
    import importlib
    folder=tmp_path/'samples/bottle';folder.mkdir(parents=True)
    for stem in ['good_000','broken_large_000','broken_small_000','contamination_000']:
        Image.new('RGB',(64,64),'gray').save(folder/(stem+'.png'))
    view=importlib.import_module(view_name)
    demo=view.create_app(tmp_path)
    components={c['props'].get('elem_id'):c for c in demo.config['components'] if c['props'].get('elem_id')}
    assert all(key in components for key in ['batch-tab','history-tab','compare-tab','challenge-tab','daily-tab'])
    options=components['compare-case']['props']['choices']
    assert {value for _,value in options}=={'good_000','broken_large_000','broken_small_000','contamination_000'}
    assert len([d for d in demo.config['dependencies'] if d.get('api_name')=='inspect'])==1
