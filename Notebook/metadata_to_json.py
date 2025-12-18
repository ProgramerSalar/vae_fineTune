import json 
import pandas as pd 
import re 

def deep_clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Normalize unicode characters
    # Map smart quotes, ellipses, and dashes to standard ASCII
    replacements = {
        '“': '"', '”': '"', '‘': "'", '’': "'",
        '…': '...', '–': '-', '—': '-'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
        
    # 2. Whitespace cleanup
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Capitalization
    # Ensure the sentence starts with an uppercase letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
        
    # 4. Terminal Punctuation
    # If the text ends with a letter, number, or closing parenthesis, add a period.
    if text and (text[-1].isalnum() or text[-1] in [')', ']']):
        text += '.'
        
    return text


if __name__ == "__main__":

    metadata_file_path = "/home/manish/Desktop/projects/FineTune/annotation/annotation_with_text/metadata.csv"


    json_list = []
    datas = pd.read_csv(metadata_file_path)
    for index, data in datas.iterrows():
        # print(data['text'])
        video_file = data['file_name']
        video_text = data['text']
        video_text = deep_clean_text(video_text)
        # print(video_text)


        json_video_data = {
            "video":video_file,
            "text": video_text,
            "video_latent":"",
            "text_latent":""
        }
        json_list.append(json_video_data)


    video_text_json = "./annotation/annotation_with_text/video_text_json.json"
    with open(video_text_json, 'w', encoding='utf-8') as f:
        json.dump(json_list, f, indent=2)
    
        
