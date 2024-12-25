import os
from tqdm import tqdm
from data_utils.screenshot import take_screenshot
from .gpt4v_utils import cleanup_response, encode_image, gpt_cost, extract_text_from_html, index_text_from_html
import json
from openai import OpenAI, AzureOpenAI
import argparse
import retry
import shutil 
from vllm_controller import stop_vllm_server, start_vllm_server
from time import sleep
import time
MODEL = 'Qwen/Qwen2-VL-7B-Instruct'
def extract_html_from_output(output: str) -> str:
    """
    Extracts the HTML code from a given output string.

    Args:
        output (str): The output string containing HTML and additional explanations.

    Returns:
        str: The extracted HTML code.
    """
    import re
    
    # Use regex to find the content between the <html> and </html> tags, inclusive
    html_pattern = r'<!DOCTYPE html>.*<\/html>'
    
    match = re.search(html_pattern, output, re.DOTALL)
    if match:
        return match.group(0).strip()
    match = re.search(r"(<html>.*</html>)", output, re.DOTALL)
    if match:
        return match.group(1).strip()
    html_pattern = r'```(.*?)```'
    match = re.search(html_pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "No HTML code found in the output."
def chat_with_openai(openai_client, MODEL, conversation_history):
    # Append the user's message to the conversation history

    # Call the OpenAI Chat Completion API
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=conversation_history,
        max_tokens=4096,
        temperature=0.0,
        frequency_penalty=0.1,
        seed=2024
    )

    # Get the assistant's reply from the response
    assistant_reply = response.choices[0].message.content.strip()
    print('Stop reason', response.choices[0].finish_reason)
    # Append the assistant's reply to the conversation history
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply
def visual_revision_prompting_v2(vlm_client, llm_client, input_image_file, original_output_image):
    '''
    {input image + initial output image + initial output html + oracle extracted text} -> {revised output html}
    '''

    ## load the original output
    

    with open(original_output_image.replace(".png", ".html"), "r") as f:
        original_output_html = f.read()

    ## encode the image 
    input_image = encode_image(input_image_file)
    original_output_image = encode_image(original_output_image)
    conversation_history = [
        
    ]
    compare_prompt = '''You are provided the screenshots of two webpages. The first one is the current version of the implemented webpage, and the second one is the target webpage. Analyze each UI component (e.g., pararaph, button, header, footer, image, etc) in the current webpage and compare it with the corresponding UI component in the target webpage. Find and categorize the mistakes in the implemented webpage.
Return a detailed YAML list of mistakes, where each mistake includes:
```yaml
- type: <what is the type of the mistake?>
  what_is_wrong: <describe what is wrong and where it is located in the current webpage>
  correct_version: 
    color: <the color of the element in the correct version>
    size: <the size of the element in the correct version, i.e., absolute size or relative size to other elements>
    location: <the location of the element in the correct version, e.g., header section, center of the page, adjacent to another element>
    text: <the text of the element in the correct version>
    ```
```'''
    compare_prompt = compare_prompt.strip()
    conversation_history.append({
        "role": "system",
        "content": [{
                "type": "text",
                "text": '''You are a tester, responsible for analyzing the implemented webpage. Your task is to find and categorize the mistakes in the implemented webpage. The mistake types include:
- Element Omission: Missing elements in the implemented version.
- Element Distortion: Inaccurate reproduction of elements (shape, size, color).
- Element Misarrangement: Incorrect positioning or order of elements.
- Element Redundancy: Extra elements in the implemented version.
'''
            }
        ]
    })
    conversation_history.append({
       "role": "user",
        "content": [
            
            {
                "type": "text", 
                "text": compare_prompt
            },
            
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{original_output_image}",
                    "detail": "high"
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{input_image}",
                    "detail": "high"
                },
            },
            
        ]
    })
    conversation_history.append({
       "role": "user",
        "content": [
            
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{input_image}",
                    "detail": "high"
                },
            },
            
        ]
    })
    json_output = chat_with_openai(vlm_client, MODEL, conversation_history)
    json_output = json_output.strip()
    json_output = json_output.replace("first_version", "correct_version")
    json_output = json_output.replace("second_version", "current_version")

    
    gen_prompt = '''You are provided with a wrong HTML + CSS code, a ground truth webpage and the issues detected from the comparison.
The wrong HTML + CSS code:
```html
''' + original_output_html + '''
```
The issues detected from the comparison:
''' + json_output + '''
Your task is to generate a new HTML + CSS code that resolves all issues.
Requirements:
- Include all CSS within the HTML file.
- Use "rick.jpg" as a placeholder for all images, including those represented by blue rectangles.
- Avoid external dependencies or JavaScript; focus only on HTML and CSS.
- Ensure correct size, text, position, color, and overall layout for all elements.
First, explain how you resolved each issue. Then, respond with a single HTML file containing the fixed HTML and CSS. .
'''
    conversation_history = []
    conversation_history.append({
        "role": "system",
        "content": [{
                "type": "text",
                "text": '''You are an expert web developer AI. Analyze the provided HTML and CSS code, and make precise edits to fix the described issues.

For each issue:

- Understand the problem, its location, and the suggested fix.
- Edit the code to resolve the issue while following web development best practices.
- Ensure the revised code is functional, visually accurate, and consistent with the correct version.

Respond with the updated HTML and CSS code, properly formatted. Add brief comments to explain significant edits if needed. Your goal is to deliver error-free, standards-compliant code that resolves all issues.
'''
            }
        ]
    })
    conversation_history.append({
        "role": "user",
        "content": [{
                "type": "text",
                "text": gen_prompt
            }
        ]
    })
    html = chat_with_openai(llm_client, 'Qwen/Qwen2.5-Coder-7B-Instruct', conversation_history)
    html = extract_html_from_output(html)
    return html
def visual_revision_prompting_v3(vlm_client, llm_client, input_image_file, original_output_image, mask_image):
    '''
    {input image + initial output image + initial output html + oracle extracted text} -> {revised output html}
    '''

    ## load the original output
    

    with open(original_output_image.replace(".png", ".html"), "r") as f:
        original_output_html = f.read()

    ## encode the image 
    input_image = encode_image(input_image_file)
    original_output_image = encode_image(mask_image)
    conversation_history = [
        
    ]
    compare_prompt = '''You are provided the screenshots of two webpages. The first one is the current version of the implemented webpage, and the second one is the target webpage. Analyze each UI component (e.g., pararaph, button, header, footer, image, etc) in the current webpage and compare it with the corresponding UI component in the target webpage. Find and categorize the mistakes in the implemented webpage.
Return a detailed YAML list of mistakes, where each mistake includes:
```yaml
- type: <what is the type of the mistake?>
  what_is_wrong: <describe what is wrong and where it is located in the current webpage>
  correct_version: 
    color: <the color of the element in the correct version>
    size: <the size of the element in the correct version, i.e., absolute size or relative size to other elements>
    location: <the location of the element in the correct version, e.g., header section, center of the page, adjacent to another element>
    text: <the text of the element in the correct version>
```
Note that: for each mistake, give an extreme detailed description of the correct version so that the developer doesn't have to look up the target webpage again to reimplement the correct version.
'''
    compare_prompt = compare_prompt.strip()
    conversation_history.append({
        "role": "system",
        "content": [{
                "type": "text",
                "text": '''You are a tester, responsible for analyzing the implemented webpage. Your task is to find and categorize the mistakes in the implemented webpage. The mistake types include:
- Element Omission: Missing elements in the implemented version.
- Element Distortion: Inaccurate reproduction of elements (shape, size, color).
- Element Misarrangement: Incorrect positioning or order of elements.
- Element Redundancy: Extra elements in the implemented version.
'''
            }
        ]
    })
    conversation_history.append({
       "role": "user",
        "content": [
            
            {
                "type": "text", 
                "text": compare_prompt
            },
            
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{original_output_image}",
                    "detail": "high"
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{input_image}",
                    "detail": "high"
                },
            },
            
        ]
    })
    conversation_history.append({
       "role": "user",
        "content": [
            
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{input_image}",
                    "detail": "high"
                },
            },
            
        ]
    })
    json_output = chat_with_openai(vlm_client, MODEL, conversation_history)
    json_output = json_output.strip()
    json_output = json_output.replace("first_version", "correct_version")
    json_output = json_output.replace("second_version", "current_version")

    
    gen_prompt = '''You are provided with a wrong HTML + CSS code, a ground truth webpage and the issues detected from the comparison.
The wrong HTML + CSS code:
```html
''' + original_output_html + '''
```
The issues detected from the comparison:
''' + json_output + '''
Your task is to generate a new HTML + CSS code that resolves all issues.
Requirements:
- Include all CSS within the HTML file.
- Use "rick.jpg" as a placeholder for all images, including those represented by blue rectangles.
- Avoid external dependencies or JavaScript; focus only on HTML and CSS.
- Ensure correct size, text, position, color, and overall layout for all elements.
First, explain how you resolved each issue. Then, respond with a single HTML file containing the fixed HTML and CSS. .
'''
    conversation_history = []
    conversation_history.append({
        "role": "system",
        "content": [{
                "type": "text",
                "text": '''You are an expert web developer AI. Analyze the provided HTML and CSS code, and make precise edits to fix the described issues.

For each issue:

- Understand the problem, its location, and the suggested fix.
- Edit the code to resolve the issue while following web development best practices.
- Ensure the revised code is functional, visually accurate, and consistent with the correct version.

Respond with the updated HTML and CSS code, properly formatted. Add brief comments to explain significant edits if needed. Your goal is to deliver error-free, standards-compliant code that resolves all issues.
'''
            }
        ]
    })
    conversation_history.append({
        "role": "user",
        "content": [{
                "type": "text",
                "text": gen_prompt
            }
        ]
    })
    html = chat_with_openai(llm_client, 'Qwen/Qwen2.5-Coder-7B-Instruct', conversation_history)
    html = extract_html_from_output(html)
    if html is None:
        return original_output_html
    return html
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--orig_output_dir', type=str, default='vllm_text_augmented_prompting', help='directory of the original output that will be further revised')
    parser.add_argument('--file_name', type=str, default='all', help='any particular file to be tested')
    parser.add_argument('--subset', type=str, default='testset_100', help='evaluate on the full testset or just a subset (choose from: {testset_100, testset_full})')
    parser.add_argument('--take_screenshot', action="store_true", help='whether to render and take screenshot of the webpages')
   
    parser.add_argument('--rerun', default=False, action="store_true")
    parser.add_argument('--port', type=int, default=18999)
    parser.add_argument('--model', type=str, default='Qwen/Qwen2-VL-7B-Instruct')
    parser.add_argument('--model_name', type=str, default='qwenvl')
    args = parser.parse_args()
    print(args)
    MODEL = args.model
    ## track usage
    if os.path.exists("usage.json"):
        with open("usage.json", 'r') as f:
            usage = json.load(f)
        total_prompt_tokens = usage["total_prompt_tokens"]
        total_completion_tokens = usage["total_completion_tokens"]
        total_cost = usage["total_cost"]
    else:
        total_prompt_tokens = 0 
        total_completion_tokens = 0 
        total_cost = 0

    log_file = open(f'workdirs/vllm_{args.model_name}_{time.time()}.log', 'w', buffering=1)
    log_file_llm = open(f'workdirs/vllm_{args.model_name}_llm_{time.time()}.log', 'w', buffering=1)
    vllm_process = start_vllm_server(log_file, port=args.port, model= args.model)
    llm_process = start_vllm_server(log_file_llm, port=args.port + 1, model= 'Qwen/Qwen2.5-Coder-7B-Instruct', is_llm=True, cuda_visible_devices='1')
    api_key = "token-abc123s"
        
        
    vlm_client = OpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key=api_key,
    )
    llm_client = OpenAI(
        base_url=f"http://localhost:{args.port + 1}/v1",
        api_key=api_key,
    )
    MODEL = args.model
    while(True):
        try:
            response = vlm_client.models.list()
            response = llm_client.models.list()
            break
        except Exception as e:
            print("Server is not live. Error:", e)
            continue
    try:
        ## OpenAI API Key
        
        ## specify file directory 
        if args.subset == "testset_final":
            test_data_dir = "../testset_final"
            cache_dir = "../predictions_final/"
        elif args.subset == "testset_100":
            test_data_dir = "../testset_100"
            cache_dir = "../predictions_100/"
        elif args.subset == "testset_full":
            test_data_dir = "../testset_full"
            cache_dir = "../predictions_full/"
        else:
            print ("Invalid subset!")
            exit()

        predictions_dir = cache_dir + f"{args.model_name}_visual_revision_prompting_llm"
        orig_data_dir = cache_dir + args.orig_output_dir
        
        ## create cache directory if not exists
        os.makedirs(predictions_dir, exist_ok=True)
        shutil.copy(test_data_dir + "/rick.jpg", os.path.join(predictions_dir, "rick.jpg"))
        
        test_files = []
        if args.file_name == "all":
            test_files = [item for item in os.listdir(test_data_dir) if item.endswith(".png") and "_marker" not in item]
        else:
            test_files = [args.file_name]

        for filename in tqdm(test_files):
            if filename.endswith(".png"):
                print (filename)
                prediction_html_path = os.path.join(predictions_dir, filename.replace(".png", ".html"))
                if not args.rerun and os.path.exists(prediction_html_path):
                    continue

                # try:
                html = visual_revision_prompting_v2(vlm_client, llm_client, os.path.join(test_data_dir, filename), os.path.join(orig_data_dir, filename))
                # cur_output_path = os.path.join(orig_data_dir, filename) 
                # html = visual_revision_prompting_v3(vlm_client, llm_client, os.path.join(test_data_dir, filename), cur_output_path, cur_output_path.replace(".png", "_masked.png"))

                
                
                with open(os.path.join(predictions_dir, filename.replace(".png", ".html")), "w") as f:
                    f.write(html)
                if args.take_screenshot:
                    take_screenshot(os.path.join(predictions_dir, filename.replace(".png", ".html")), os.path.join(predictions_dir, filename), do_it_again=True)
                # except:
                #     continue 

        ## save usage
        
    except KeyboardInterrupt:
        print("Program interrupted.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        stop_vllm_server(vllm_process)
        stop_vllm_server(llm_process)
        log_file.close()
        log_file_llm.close()
