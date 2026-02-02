import argparse
import json
import random
import os
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from keye_vl_utils import process_vision_info

def parse_args():
    parser = argparse.ArgumentParser(description="KEYE Video Captioning")
    parser.add_argument("--video_paths_json", type=str, required=True, help="Path to the JSON file containing video paths.")
    parser.add_argument("--prompts_json", type=str, required=True, help="Path to the JSON file containing text prompts.")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second for video processing.")
    parser.add_argument("--max_frames", type=int, default=1024, help="Maximum frames to process.")
    parser.add_argument("--output_file", type=str, default="captions.jsonl", help="Path to the output JSONL file.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load prompts
    try:
        with open(args.prompts_json, 'r') as f:
            prompts_data = json.load(f)
            prompts_list = prompts_data.get("prompts", [])
            if not prompts_list:
                raise ValueError("No prompts found in prompts.json")
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return

    # Load video paths
    try:
        with open(args.video_paths_json, 'r') as f:
            video_data = json.load(f)
            video_paths = video_data.get("video_paths", [])
    except Exception as e:
        print(f"Error loading video paths: {e}")
        return

    # Load model
    model_path = "Kwai-Keye/Keye-VL-1_5-8B"
    print(f"Loading model from {model_path}...")
    model = AutoModel.from_pretrained(
    model_path,
    torch_dtype="auto",
    trust_remote_code=True,
    # flash_attention_2 is recommended for better performance
    attn_implementation="flash_attention_2",
    ).eval()

    model.to("cuda")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print(f"Starting captioning for {len(video_paths)} videos...")
    
    # Ensure output directory exists (if path provided)
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_file, 'a') as outfile:
        for video_path in video_paths:
            # Check if video exists
            if not os.path.exists(video_path):
                 print(f"File not found: {video_path}, skipping.")
                 continue

            try:
                # Randomly select a prompt
                selected_prompt = random.choice(prompts_list)
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,
                                "fps": args.fps,
                                "max_frames": args.max_frames
                            },
                            {"type": "text", "text": selected_prompt},
                        ],
                    }
                ]

                # Preparation for inference
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs, mm_processor_kwargs = process_vision_info(messages)
                
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **mm_processor_kwargs
                )
                inputs = inputs.to("cuda")

                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, max_new_tokens=1024)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                result = {
                    "model_name": model_path,
                    "text_prompt": selected_prompt,
                    "response": output_text,
                    "video_path": video_path
                }
                
                outfile.write(json.dumps(result) + "\n")
                outfile.flush()
                print(f"Processed: {video_path}")

            except Exception as e:
                print(f"Error processing {video_path}: {e}")

if __name__ == "__main__":
    main()
