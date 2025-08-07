import json
import os

def convert_to_prompt_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            instruction = data['instruction']
            output = data['output']
            
            # Gemma의 공식 프롬프트 템플릿 형식에 맞춰 문자열을 생성합니다.
            prompt_text = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"
            
            # 새로운 json 객체를 생성합니다.
            new_data = {"prompt": prompt_text}
            
            # 새로운 json 객체를 파일에 씁니다.
            outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(CURRENT_DIR, "../data/ncube_finetune_data.jsonl")
    output_path = os.path.join(CURRENT_DIR, "../data/ncube_prompt_data.jsonl")
    
    convert_to_prompt_format(input_path, output_path)
    
    print(f"파일 변환이 완료되었습니다. 결과가 {os.path.abspath(output_path)} 에 저장되었습니다.")
