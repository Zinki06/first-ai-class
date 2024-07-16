!pip install accelerate


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import re
from tqdm import tqdm
from huggingface_hub import login

def load_model_and_tokenizer(model_name):
    try:
        print(f"토크나이저 로딩 중: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, return_token_type_ids=False)

        print(f"모델 로딩 중: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        if tokenizer.pad_token is None:
            print("패딩 토큰 설정 중...")
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"사용 중인 디바이스: {device}")

        if device.type == "cuda":
            print("GPU로 모델 이동 중...")
            model = model.to(device)

        print("모델 로딩 완료!")
        return tokenizer, model, device
    except OSError as e:
        print(f"모델 또는 토크나이저 파일을 찾을 수 없습니다: {e}")
    except ValueError as e:
        print(f"모델 구성 오류: {e}")
    except RuntimeError as e:
        print(f"모델 로딩 중 런타임 오류 (CUDA 메모리 부족 가능성): {e}")
    except Exception as e:
        print(f"예기치 못한 오류 발생: {e}")

    print("모델 로딩 실패. 프로그램을 종료합니다.")
    return None, None, None

def generate_important_words(model, tokenizer, text, num_words, device):
    prompt = f"""다음 텍스트에서 가장 중요한 단어 {num_words}개를 선택하세요:

{text}

중요한 단어들:"""

    print("토크나이징 중...")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, return_token_type_ids=False).to(device)
    print("토크나이징 완료.")

    print("모델 생성 중...")
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,  # 이전에는 50이었음
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
        except Exception as e:
            print(f"모델 생성 중 오류 발생: {e}")
            return []
    print("모델 생성 완료.")

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    important_words = response.split("중요한 단어들:")[-1].strip().split(", ")
    return important_words[:num_words]

def create_blank_filled_text(text, important_words):
    for word in important_words:
        text = re.sub(r'\b' + re.escape(word) + r'\b', '[     ]', text, count=1)
    return text

def main():
    # Hugging Face Hub 로그인
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

        try:
            print("중요 단어 선택 중...")
            important_words = generate_important_words(model, tokenizer, text, num_blanks, device)
            if not important_words:
                continue
            print("중요 단어 선택 완료.")
            blank_filled_text = create_blank_filled_text(text, important_words)

            print("\n생성된 빈칸 채우기 문제:")
            print(blank_filled_text)
            print("\n정답:")
            for i, word in enumerate(important_words, 1):
                print(f"{i}. {word}")
        except Exception as e:
            print(f"문제 생성 중 오류 발생: {e}")
            print("다시 시도해주세요.")

        if device.type == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
