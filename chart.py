import random
from PIL import Image
from lida import Manager, TextGenerationConfig, llm as lidallm
from dotenv import load_dotenv
load_dotenv()

import helper 


def generate_chart(prompt, df) -> Image:
    lida = Manager(text_gen = lidallm("openai"))
    textgen_config = TextGenerationConfig(n=1, temperature=0.1, use_cache=True)
    summary = lida.summarize(df, summary_method="default", textgen_config=textgen_config)
    charts = lida.visualize(summary=summary,
                            goal=prompt,
                            textgen_config=textgen_config,
                            library="matplotlib",
                            return_error=True)  
    image_base64 = charts[0].raster
    chart = helper.base64_to_image(image_base64)
    return chart

def generate_suggestions(df):
    lida = Manager(text_gen = lidallm("openai"))
    textgen_config = TextGenerationConfig(n=1, temperature=0.1, use_cache=True)
    summary = lida.summarize(df, summary_method="default", textgen_config=textgen_config)
    goals = lida.goals(summary, n=10, textgen_config=textgen_config)
    return random.sample(goals, 2)