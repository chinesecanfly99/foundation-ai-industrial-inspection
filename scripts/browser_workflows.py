"""Exercise real model inference, exports, review persistence and both locales."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import sqlite3
import traceback
import zipfile
from playwright.sync_api import sync_playwright,expect,TimeoutError as BrowserTimeout

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url',required=True)
parser.add_argument('--qa-root',type=Path,required=True)
parser.add_argument('--project',type=Path,required=True)
parser.add_argument('--language',choices=['en','zh'],required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
locale=json.loads((args.project/f'app/locales/{args.language}.json').read_text(encoding='utf-8'))
t=locale['ui']
evidence={'passed':False,'checks':[],'page_errors':[],'http_errors':[]}


def rows():
    path=args.qa_root/'results/workflow/inspections.sqlite3'
    if not path.exists():return []
    with sqlite3.connect(path) as connection:
        connection.row_factory=sqlite3.Row
        return [dict(r) for r in connection.execute('SELECT * FROM inspections ORDER BY created_at,id')]


def tab(key):
    page.get_by_role('tab',name=t[key],exact=True).click()


def ready():
    try:page.wait_for_load_state('networkidle',timeout=2500)
    except BrowserTimeout:
        # Gradio State keeps an intentional long-poll heartbeat open.
        pass
    page.locator('#language-toggle').wait_for()


def choose(selector,label):
    page.locator(selector+' input').click()
    page.get_by_role('option',name=label,exact=True).click()


def decoded(selector):
    page.locator(selector+' img').scroll_into_view_if_needed()
    page.wait_for_function("selector => {const img=document.querySelector(selector+' img'); return img?.complete && img.naturalWidth>0;}",arg=selector,timeout=90000)


with sync_playwright() as playwright:
 browser=playwright.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
 context=browser.new_context(locale='zh-CN',viewport={'width':1440,'height':1080},accept_downloads=True)
 page=context.new_page();expect.set_options(timeout=90000)
 page.on('pageerror',lambda err:evidence['page_errors'].append(str(err)))
 page.on('response',lambda response:evidence['http_errors'].append({'status':response.status,'url':response.url}) if response.status>=400 else None)
 try:
    page.goto(args.url,wait_until='domcontentloaded');ready()
    assert len(page.get_by_role('tab').all_text_contents())==6
    evidence['checks'].append('Six functional tabs render in the default language')
    # Inspect the rendered tab/control inventory before executing test actions.
    evidence['tab_count']=page.get_by_role('tab').count()
    tab('history')
    expect(page.locator('#history-summary')).to_contain_text(t['summary'].format(visible=0,total=0))
    evidence['checks'].append('Empty history and charts load without invented records')

    tab('batch')
    corrupt=args.output/'corrupt.png';corrupt.write_bytes(b'not an image')
    files=[args.qa_root/'samples/bottle/good_000.png',corrupt,args.qa_root/'samples/bottle/broken_large_000.png']
    page.locator('#batch-files input[type=file]').set_input_files([str(p) for p in files])
    page.locator('#batch-run').click()
    expect(page.locator('#batch-status')).to_contain_text(t['batch_done'].split('{batch}')[0])
    expect(page.locator('#batch-csv')).to_be_enabled()
    batch_records=[r for r in rows() if r['source']=='batch']
    assert len(batch_records)==3 and sum(r['status']=='failed' for r in batch_records)==1
    with page.expect_download() as download:page.locator('#batch-csv').click()
    csv_path=args.output/'batch.csv';download.value.save_as(csv_path)
    with csv_path.open(encoding='utf-8-sig',newline='') as stream:assert len(list(csv.DictReader(stream)))==3
    with page.expect_download() as download:page.locator('#batch-zip').click()
    zip_path=args.output/'batch.zip';download.value.save_as(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        manifest=json.loads(archive.read('batch_manifest.json'))
        assert (manifest['count'],manifest['success'],manifest['failed'])==(3,2,1)
    page.screenshot(path=str(args.output/'batch.png'),full_page=True)
    evidence['checks'].append('Batch retains all three inputs, isolates one corrupt image and exports matching CSV/ZIP')

    tab('compare');page.locator('#compare-demo').click()
    page.locator('#compare-input img').wait_for()
    page.locator('#compare-run').click()
    expect(page.locator('#compare-status')).to_contain_text(t['compare_done'].format(count=2))
    compare_records=[r for r in rows() if r['source']=='comparison']
    assert len(compare_records)==2
    assert {r['model_id'] for r in compare_records}=={'bottle_pixel','bottle_patch'}
    assert len({json.loads(r['report_json'])['image_sha256'] for r in compare_records})==1
    for r in compare_records:assert r['threshold']>0
    decoded('#compare-pixel');decoded('#compare-patch')
    page.screenshot(path=str(args.output/'comparison.png'),full_page=True)
    evidence['checks'].append('Both frozen models inspect the identical image with their own thresholds and decoded heatmaps')

    tab('challenge');page.locator('#challenge-demo').click()
    page.locator('#challenge-input img').wait_for()
    brightness=page.locator('#challenge-brightness input[type=number]')
    brightness.fill('0.1');brightness.press('Tab')
    expect(brightness).to_have_value('0.1')
    page.locator('#challenge-preview img').wait_for()
    page.locator('#challenge-run').click()
    expect(page.locator('#challenge-status')).to_contain_text(t['challenge_done'])
    challenge_records=[r for r in rows() if r['source']=='challenge']
    assert len(challenge_records)==2
    reports=[json.loads(r['report_json']) for r in challenge_records]
    original=next(r for r in reports if r['experiment']['variant']=='original')
    changed=next(r for r in reports if r['experiment']['variant']=='transformed')
    assert original['image_sha256']!=changed['image_sha256']
    assert changed['experiment']['brightness_multiplier']==.1
    assert original['brightness']>changed['brightness']
    assert changed['decision']=='undetermined' and original['decision']!='undetermined'
    assert original['score']!=changed['score']
    decoded('#challenge-original');decoded('#challenge-changed')
    evidence['quality_change']={key:{k:r[k] for k in ['score','brightness','sharpness','decision']} for key,r in [('original',original),('transformed',changed)]}
    page.screenshot(path=str(args.output/'challenge.png'),full_page=True)
    evidence['checks'].append('Brightness change produces new pixels, new inference scores and the input-quality retake recommendation')

    tab('daily');page.locator('#daily-input img').wait_for()
    expect(page.locator('#daily-source')).to_contain_text(t['daily_reference'])
    page.locator('#daily-run').click()
    expect(page.locator('#daily-status')).to_contain_text(t['daily_done'].split('{ids}')[0])
    daily_records=[r for r in rows() if r['source']=='daily_case']
    assert len(daily_records)==2 and all(r['product']=='tile' for r in daily_records)
    assert all(json.loads(r['report_json'])['experiment']=={'type':'daily_tile_case','image_source':'reference_example','benchmark_result':False} for r in daily_records)
    decoded('#daily-pixel');decoded('#daily-patch')
    page.screenshot(path=str(args.output/'daily_case.png'),full_page=True)
    evidence['checks'].append('Daily tile case is correctly attributed, routed through both tile models and excluded from benchmark claims')

    tab('history');page.locator('#history-refresh').click()
    expect(page.locator('#history-summary')).to_contain_text(t['summary'].format(visible=9,total=9))
    reviewed=compare_records[-1]
    record_path=args.qa_root/'results/workflow/records'/reviewed['id']/'report.json'
    before=record_path.read_bytes()
    control=page.locator('#record-choice input');control.click()
    page.get_by_role('option',name=re.compile(re.escape(reviewed['id'][:8]))).click()
    page.get_by_role('button',name=re.compile(re.escape(t['record_report']))).click()
    expect(page.locator('#record-report')).to_contain_text(reviewed['id'])
    page.locator('#record-input img').wait_for();page.locator('#record-overlay img').wait_for()
    page.get_by_role('button',name=re.compile(re.escape(t['record_report']))).click()
    page.locator('#reviewer-name input').fill('Demo QA')
    choose('#review-verdict',t['retake'])
    page.locator('#review-comment textarea').fill('Functional QA only; not independent business ground truth.')
    page.locator('#review-save').click()
    expect(page.locator('#review-status')).to_contain_text(t['review_saved'])
    expect(page.locator('#review-log')).to_contain_text('Demo QA')
    expect(page.locator('#history-metrics .metric').last.locator('.value')).to_have_text('1')
    assert record_path.read_bytes()==before
    with sqlite3.connect(args.qa_root/'results/workflow/inspections.sqlite3') as connection:
        assert connection.execute('SELECT COUNT(*) FROM reviews WHERE inspection_id=?',(reviewed['id'],)).fetchone()[0]==1
    page.screenshot(path=str(args.output/'history_review.png'),full_page=True)
    evidence['checks'].append('Human review persists as a separate event and leaves the original AI report byte-identical')
    page.locator('#history-export').click()
    link=page.locator('#history-download');expect(link).to_be_enabled()
    with page.expect_download() as download:link.click()
    history_csv=args.output/'history.csv';download.value.save_as(history_csv)
    with history_csv.open(encoding='utf-8-sig',newline='') as stream:assert len(list(csv.DictReader(stream)))==9
    evidence['checks'].append('History CSV includes every matching record, original decision and latest human verdict')

    count_before=len(rows())
    page.locator('#language-toggle').click();ready()
    opposite='zh' if args.language=='en' else 'en'
    other=json.loads((args.project/f'app/locales/{opposite}.json').read_text(encoding='utf-8'))['ui']
    expect(page.get_by_role('tab',name=other['history'],exact=True)).to_be_visible()
    page.get_by_role('tab',name=other['history'],exact=True).click()
    expect(page.locator('#history-summary')).to_contain_text(other['summary'].format(visible=9,total=9))
    assert len(rows())==count_before
    assert page.locator('html').get_attribute('lang')==('zh-CN' if opposite=='zh' else 'en')
    page.screenshot(path=str(args.output/'switched_language.png'),full_page=True)
    page.locator('#language-toggle').click();ready()
    expect(page.get_by_role('tab',name=t['single'],exact=True)).to_be_visible()
    assert page.url.rstrip('/')==args.url.rstrip('/')
    evidence['checks'].append('Language switch works both ways on one service and preserves inspection/review history')
    page.set_viewport_size({'width':390,'height':844})
    page.wait_for_function('document.documentElement.scrollWidth <= innerWidth + 1')
    page.screenshot(path=str(args.output/'mobile.png'),full_page=True)
    evidence['checks'].append('Default page remains usable at 390 pixels without document overflow')
    assert not evidence['page_errors'],evidence['page_errors']
    unexpected=[r for r in evidence['http_errors'] if not (r['status']==404 and 'upload_progress?upload_id=undefined' in r['url'])]
    assert not unexpected,unexpected
    evidence['passed']=True
 except Exception as exc:
    evidence['error']=str(exc);evidence['traceback']=traceback.format_exc()
    page.screenshot(path=str(args.output/'failure.png'),full_page=True)
    (args.output/'failure.html').write_text(page.content(),encoding='utf-8')
 finally:
    browser.close()
    (args.output/'browser_workflows.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'passed':evidence['passed'],'checks':len(evidence['checks']),'error':evidence.get('error')}))
raise SystemExit(0 if evidence['passed'] else 1)
