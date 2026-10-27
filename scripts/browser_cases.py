"""Verify four reference cases, frozen scores and presentation deep links."""
from pathlib import Path
import argparse,csv,json,sqlite3,re,traceback
from playwright.sync_api import sync_playwright,expect,TimeoutError

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',required=True);p.add_argument('--project',type=Path,required=True)
p.add_argument('--qa-root',type=Path,required=True);p.add_argument('--language',choices=['en','zh'],required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
resources=json.loads((a.project/f'app/locales/{a.language}.json').read_text(encoding='utf-8'));t=resources['ui']
e={'passed':False,'checks':[],'page_errors':[],'matched_scores':[]}

def records():
 with sqlite3.connect(a.qa_root/'results/workflow/inspections.sqlite3') as c:
  c.row_factory=sqlite3.Row;return [dict(r) for r in c.execute('SELECT * FROM inspections')]

def navigate(url):
 page.goto(url,wait_until='domcontentloaded')
 try:page.wait_for_load_state('networkidle',timeout=2000)
 except TimeoutError:pass
 page.locator('#language-toggle').wait_for()

with sync_playwright() as play:
 browser=play.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
 page=browser.new_page(viewport={'width':1440,'height':1080},locale='zh-CN',accept_downloads=True)
 expect.set_options(timeout=90000);page.on('pageerror',lambda err:e['page_errors'].append(str(err)))
 try:
  navigate(a.url+'/?demo=compare')
  expect(page.get_by_role('tab',name=t['compare'],exact=True)).to_have_attribute('aria-selected','true')
  e['checks'].append('PowerPoint comparison URL opens the intended tab automatically')
  page.locator('#compare-case input').click()
  options=page.get_by_role('option').all_text_contents();assert len(options)==4
  page.locator('#compare-case input').press('Escape')
  for kind in ['good','broken_large','broken_small','contamination']:
   stem=kind+'_000';label=resources['products']['bottle']['defect_labels'][kind]
   before={r['id'] for r in records()}
   if page.locator('#compare-case input').input_value()!=label:
    old=page.locator('#compare-input img').get_attribute('src')
    page.locator('#compare-case input').click();page.locator('#compare-case input').fill(label)
    page.locator('#compare-case input').press('ArrowDown')
    page.get_by_role('option').filter(has_text=label).click()
    page.wait_for_function("old => {const i=document.querySelector('#compare-input img');return i && i.getAttribute('src')!==old && i.complete && i.naturalWidth>0;}",arg=old,timeout=90000)
   page.locator('#compare-run').click()
   expect(page.locator('#compare-status')).to_contain_text(t['compare_done'].format(count=2))
   added=[r for r in records() if r['id'] not in before]
   assert len(added)==2 and all(r['filename']==stem+'.png' for r in added)
   for r in added:
    method=r['model_id'].rsplit('_',1)[1]
    with (a.project/f'results/three_products/bottle/{method}/predictions.csv').open(newline='') as stream:
     expected=next(v for v in csv.DictReader(stream) if v['image_path']==f'bottle/test/{kind}/000.png')
    assert abs(r['score']-float(expected['score']))<1e-6
    assert abs(r['threshold']-float(expected['image_threshold']))<1e-6
    e['matched_scores'].append({'case':stem,'model_id':r['model_id'],'score':r['score'],'threshold':r['threshold']})
  page.screenshot(path=str(a.output/'four_cases.png'),full_page=True)
  e['checks'].append('All four bottle cases run both real models and reproduce all eight frozen score/threshold pairs')
  for key in ['challenge','daily','batch','history']:
   navigate(a.url+'/?demo='+key)
   expect(page.get_by_role('tab',name=t[key],exact=True)).to_have_attribute('aria-selected','true')
  e['checks'].append('Presentation deep links also open image challenge, daily case, batch and history')
  opposite='zh' if a.language=='en' else 'en';target='/zh/' if a.language=='en' else '/en/'
  other=json.loads((a.project/f'app/locales/{opposite}.json').read_text(encoding='utf-8'))['ui']
  navigate(a.url+target)
  page.locator('#inspect-button').click()
  expect(page.locator('#download-json')).to_be_enabled()
  with page.expect_download() as download:page.locator('#download-json').click()
  path=a.output/'opposite_language_report.json';download.value.save_as(path)
  report=json.loads(path.read_text(encoding='utf-8'))
  assert report['language']==opposite
  assert ('label_en' in report)==(opposite=='en')
  assert ('label_zh' in report)==(opposite=='zh')
  if opposite=='en':assert not re.search('[\u3400-\u9fff]',json.dumps(report,ensure_ascii=False))
  assert abs(report['score']-.0530223473906517)<1e-6
  e['checks'].append('Opposite-language single inspection localizes its report while preserving the frozen numeric result')
  assert not e['page_errors'];e['passed']=True
 except Exception as exc:
  e['error']=str(exc);e['traceback']=traceback.format_exc()
  page.screenshot(path=str(a.output/'failure.png'),full_page=True)
 finally:
  browser.close();(a.output/'browser_cases.json').write_text(json.dumps(e,indent=2),encoding='utf-8')
print(json.dumps({'passed':e['passed'],'checks':len(e['checks']),'matched_score_pairs':len(e['matched_scores']),'error':e.get('error')}))
raise SystemExit(0 if e['passed'] else 1)
