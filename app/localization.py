"""Presentation language is independent of detector calibration and storage."""
from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=2)
def locale(language):
    if language not in {"en", "zh"}:
        raise ValueError("Unsupported interface language")
    return json.loads((Path(__file__).parent / "locales" / f"{language}.json").read_text(encoding="utf-8"))


def localized_products(language):
    return locale(language)["products"]


def localize_report(report, language):
    data=locale(language);ui=data["ui"];report=dict(report)
    report["language"]=language
    decision=report.get("decision")
    if decision in {"pass","review","reject","undetermined"}:
        report.pop("label_zh" if language=="en" else "label_en",None)
        report["label_en" if language=="en" else "label_zh"]=ui[decision]
    category=report.get("product_category")
    if category in data["products"]:
        report["product_label"]=data["products"][category]["label"]
        report["policy_description"]=data["policies"][category]
    translated=[]
    for original in report.get("trace",[]):
        step=dict(original);rule=step.get("rule")
        if rule in data["rules"]:
            step["reason"]=data["rules"][rule]
            step["suggestion"]=ui[{"R1":"undetermined","R2":"pass","R3":"review","R4":"reject","R5":"review"}[rule]]
        translated.append(step)
    report["trace"]=translated
    selected=next((s for s in translated if s.get("selected")),None)
    if selected:report["reasons"]=[selected["reason"]]
    if language=="en":
        report["notice"]="Anomaly scores are not probabilities. Heatmaps show visual differences. Human review events do not alter detector results or model weights."
    else:
        # The text is loaded from the language resource so this module remains language-neutral.
        report["notice"]=ui["review_help"]
    return report


def locale_startup_js(language):
    import gradio as gr
    assets=Path(gr.__file__).resolve().parent/'templates/frontend/assets'
    modules=[p for p in assets.glob('init-*.js') if 'fallbackLocale' in p.read_text(encoding='utf-8')]
    if len(modules)!=1:raise RuntimeError('Use the pinned Gradio 5.49.1 frontend assets')
    url=json.dumps('./assets/'+modules[0].name)
    target=json.dumps('en' if language=='en' else 'zh-CN')
    names=json.dumps({k:locale(language)['ui'][v] for k,v in {'compare':'compare','challenge':'challenge','daily':'daily','batch':'batch','history':'history'}.items()})
    return f'''async () => {{
        const locale = await import(new URL({url}, document.baseURI).href);
        locale.e({target});
        const names = {names};
        const wanted = names[new URL(location.href).searchParams.get("demo")];
        if (wanted) {{
            let attempts = 0;
            const selectTab = () => {{
                const tab = [...document.querySelectorAll('[role="tab"]')].find(el => el.offsetParent !== null && el.textContent.trim() === wanted);
                if (tab) {{ tab.click(); tab.scrollIntoView({{block: "start"}}); }}
                else if (++attempts < 40) setTimeout(selectTab, 50);
            }};
            selectTab();
        }}
    }}'''
