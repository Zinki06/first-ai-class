!pip install accelerate

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from huggingface_hub import login
import random

# load_model_and_tokenizer 함수는 그대로 유지

def generate_important_words(model, tokenizer, text, num_words, device):
    prompt = f"""다음 텍스트에서 가장 중요한 단어 {num_words}개를 선택하세요.
    선택한 단어들은 반드시 텍스트에 정확히 나타나는 형태여야 합니다.
    각 단어는 한 단어여야 하며, 문장이나 구절을 선택하지 마세요.
    텍스트 전체에서 고르게 분포된 단어를 선택하세요. 처음, 중간, 끝 부분에서 골고루 선택하세요.
    선택한 단어들은 쉼표로 구분하여 나열해주세요.

예시:
텍스트: The quick brown fox jumps over the lazy dog. The dog was surprised but didn't move. The fox continued on its way.
중요한 단어 3개: quick, dog, continued

텍스트:
{text}

중요한 단어 {num_words}개:"""

    print("토크나이징 중...")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, return_token_type_ids=False).to(device)
    print("토크나이징 완료.")
    print("모델 생성 중...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
    
    print("모델 생성 완료.")
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    words_match = re.search(r'중요한 단어 \d+개:(.+)', response, re.DOTALL)
    if words_match:
        selected_words = re.findall(r'\b\w+\b', words_match.group(1))
    else:
        selected_words = []
    
    # 선택된 단어가 실제로 텍스트에 있는지 확인하고 위치 정보 저장
    validated_words = []
    for word in selected_words:
        matches = list(re.finditer(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE))
        if matches:
            validated_words.append((word, matches[0].start()))
    
    # 위치에 따라 정렬하고 균형있게 선택
    validated_words.sort(key=lambda x: x[1])
    step = len(validated_words) // num_words
    balanced_words = [word for word, _ in validated_words[::step]][:num_words]
    
    return balanced_words

def create_blank_filled_text(text, important_words):
    for word in important_words:
        pattern = r'\b' + re.escape(word) + r'\b'
        text = re.sub(pattern, '[        ]', text, count=1, flags=re.IGNORECASE)
    return text

def main():
    login(token="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")
    model_name = "EleutherAI/polyglot-ko-1.3b"
    tokenizer, model, device = load_model_and_tokenizer(model_name)
    
    if not all([tokenizer, model, device]):
        print("프로그램을 종료합니다.")
        return

    print("빈칸 채우기 문제 생성기입니다.")
    print("종료하려면 '종료'를 입력하세요.")

    while True:
        text = input("\n지문을 입력하세요: ")
        if text.lower() == '종료':
            print("프로그램을 종료합니다.")
            break
        
        num_blanks = int(input("생성할 빈칸의 개수를 입력하세요 (1-10): "))
        num_blanks = max(1, min(10, num_blanks))

        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                print(f"중요 단어 선택 중... (시도 {attempt + 1}/{max_attempts})")
                important_words = generate_important_words(model, tokenizer, text, num_blanks * 2, device)
                if len(important_words) >= num_blanks:
                    important_words = important_words[:num_blanks]
                    break
                print(f"충분한 중요 단어를 선택하지 못했습니다. ({len(important_words)}/{num_blanks})")
            except Exception as e:
                print(f"단어 선택 중 오류 발생: {e}")
        
        if len(important_words) < num_blanks:
            print("요청한 수의 중요 단어를 찾는 데 실패했습니다.")
            continue

        print("중요 단어 선택 완료.")
        
        blank_filled_text = create_blank_filled_text(text, important_words)

        print("\n생성된 문제:")
        print(blank_filled_text)
        print("\n정답:")
        for word in important_words:
            print(word)

        if device.type == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
