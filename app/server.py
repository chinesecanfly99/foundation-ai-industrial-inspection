"""Mount a second language view beside the unchanged root API."""
import gradio as gr
from fastapi import FastAPI
import uvicorn
from starlette.responses import RedirectResponse


def create_server(primary, alternate, *, default_language, host='127.0.0.1', port=7860):
    app=FastAPI(docs_url=None,redoc_url=None)
    @app.get('/health')
    def health():
        return {'status':'ok','default_language':default_language,'workflow_version':'0.3.0'}
    options=dict(server_name=host,server_port=port,show_api=False,show_error=False,
                 max_file_size='20mb',ssr_mode=False,enable_monitoring=False)
    alternate_path='/en' if default_language=='zh' else '/zh'
    @app.get(alternate_path,include_in_schema=False)
    def alternate_redirect():
        # A root mount otherwise captures the slashless URL before Starlette's
        # automatic slash redirect can run.
        return RedirectResponse(alternate_path+'/')
    app=gr.mount_gradio_app(app,alternate,alternate_path,**options)
    app=gr.mount_gradio_app(app,primary,'/',**options)
    return app


def launch(primary, alternate, *, default_language, host, port):
    app=create_server(primary,alternate,default_language=default_language,host=host,port=port)
    uvicorn.run(app,host=host,port=port)
