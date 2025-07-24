from django.apps import AppConfig
import os
import torch
import gc

class ChatApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chat_api'

    def ready(self):
        print("🔧 앱 초기화 중: 캐시 정리 수행")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ❗ 코드 변경 감지 프로세스에서는 모델 로딩 생략
        if os.environ.get('RUN_MAIN') != 'true':
            print("⚠️ RUN_MAIN 아님: 모델 로딩 생략")
            return

        print("🧠 모델 로딩 시작")
        from . import llama_loader
        llama_loader.load_model()
