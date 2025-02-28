import settings
import neptune
from neptune.types import File
import logging
import json

# Custom Neptune Logger class
logging.getLogger("neptune").setLevel(logging.CRITICAL)

class NeptuneLogger:
    def __init__(self):
        self.run = neptune.init_run(project=settings.NEPTUNE_PROJECT_NAME, api_token=settings.NEPTUNE_API_TOKEN)

    def log_init(self, prefix, **params):
        for name, value in params.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            self.run[f"{prefix}/{name}"] = value


    def log_df(self, prefix, name, df):
        self.run[f'{prefix}/{name}'].upload(File.as_html(df))
        
    def update_stats(self, prefix, step, **stats):
        for name, value in stats.items():
            if name in ['batch', 'size']:
                continue
            self.run[f"{prefix}/{name}"].log(value, step=step)

