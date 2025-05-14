import argparse
import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    Gemma3ForConditionalGeneration,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList
)
import chromadb

# ----------- Utility Functions -----------


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()

# ----------- Data Loading -----------


def load_data_files(qa_file, neutral_file=None):
    if neutral_file:
        ds = load_dataset('json', data_files={'train': qa_file})['train']
        neutral_ds = load_dataset(
            'json', data_files={'train': neutral_file})['train']
        return ds, neutral_ds
    else:
        with open(qa_file, 'r') as f:
            qa_data = json.load(f)
        return qa_data

# ----------- Formatting Examples -----------


def format_example(example, tokenizer):
    prompt = (
        "System: For the purposes of this interaction, you are J Robert Oppenheimer. "
        "Your output will be read aloud exactly as written. "
        "DO NOT INCLUDE any voice noise markers. "
        "Answer in a paragraph or less. Use <STOP> when done.\n"
        f"User: {example['question']}\nAssistant: "
    )
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids[0]
    response_ids = tokenizer(
        example['response'], return_tensors='pt').input_ids[0]
    input_ids = torch.cat([prompt_ids, response_ids])
    labels = torch.cat([torch.full_like(prompt_ids, -100), response_ids])
    return {"input_ids": input_ids.tolist(), "labels": labels.tolist()}

# ----------- Stopping Criteria -----------


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids_list):
        self.stop_ids_list = stop_ids_list

    def __call__(self, input_ids, scores, **kwargs):
        for stop_ids in self.stop_ids_list:
            if stop_ids and input_ids[0][-len(stop_ids):].tolist() == stop_ids:
                return True
        return False

# ----------- Style Vector Methods -----------


def compute_style_vector(styled_text, neutral_text, tokenizer, model, layer_index=20):
    styled_inputs = tokenizer(styled_text, return_tensors='pt').to(DEVICE)
    neutral_inputs = tokenizer(neutral_text, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        styled_out = model(**styled_inputs, output_hidden_states=True)
        neutral_out = model(**neutral_inputs, output_hidden_states=True)
    styled_hidden = styled_out.hidden_states[layer_index][0, -1, :]
    neutral_hidden = neutral_out.hidden_states[layer_index][0, -1, :]
    return styled_hidden - neutral_hidden


def compute_average_style_vector(train_data, neutral_data, tokenizer, model, layer_index=20):
    vectors = []
    for styled_ex, neutral_ex in zip(train_data, neutral_data):
        # extract response tokens
        response_tokens = [t for t, lbl in zip(
            styled_ex['input_ids'], styled_ex['labels']) if lbl != -100]
        styled_text = tokenizer.decode(
            response_tokens, skip_special_tokens=True)
        neutral_text = neutral_ex['response']
        vec = compute_style_vector(
            styled_text, neutral_text, tokenizer, model, layer_index)
        vectors.append(vec)
    return torch.stack(vectors).mean(dim=0)

# ----------- Generation Methods -----------


def style_vector_generation(args):
    # Load data
    train_ds, neutral_ds = load_data_files(args.qa_file, args.neutral_file)
    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_name).to(DEVICE)

    # Prepare training examples
    hf_ds = train_ds.map(lambda x: format_example(
        x, tokenizer), remove_columns=['question', 'response'])
    avg_style_vec = compute_average_style_vector(
        hf_ds, neutral_ds, tokenizer, model, layer_index=args.layer_index)
    print("Computed average style vector.")

    # Build prompt
    prompt = (
        "System: For the purposes of this interaction, you are J Robert Oppenheimer. "
        "Answer in a paragraph or less. Use <STOP> when done.\n"
        f"User: {args.question}\nAssistant: "
    )
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)

    # Stopping criteria
    stop_tokens = ["\nUser:", "\nAssistant:", "<STOP>", "\nSystem:"]
    stop_ids = [
        tokenizer(t, add_special_tokens=False).input_ids for t in stop_tokens]
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])

    # Hook style injection
    # detect transformer layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        raise RuntimeError(
            "Cannot locate transformer layers for style injection.")

    def hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        hidden[:, -1, :] += args.scale * avg_style_vec.to(DEVICE)
        return (hidden,) + out[1:] if isinstance(out, tuple) else hidden

    handle = layers[args.layer_index].register_forward_hook(hook)

    # Generate
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        stopping_criteria=stopping_criteria
    )
    handle.remove()

    gen = tokenizer.decode(
        output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    # trim at stop token
    for tok in stop_tokens:
        if tok in gen:
            gen = gen.split(tok)[0]
    print("Generated styled output (Style Vectors):")
    print(gen)


def few_shot_generation(args):
    qa_data = load_data_files(args.qa_file)
    # Setup ChromaDB
    client = chromadb.Client()
    try:
        coll = client.create_collection("oppenheimer-qa2")
    except Exception:
        coll = client.get_collection("oppenheimer-qa2")
    if coll.count() == 0:
        docs = [item['response'] for item in qa_data]
        qns = [item['question'] for item in qa_data]
        ids = [f"qa_{i}" for i in range(len(docs))]
        coll.add(documents=docs, metadatas=[
                 {"question": q} for q in qns], ids=ids)

    def get_context(query, n=3):
        res = coll.query(query_texts=[query], n_results=n)
        ctx = []
        for _id in res['ids'][0]:
            idx = int(_id.split('_')[1])
            ctx.append(qa_data[idx])
        return ctx

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_name).to(DEVICE)

    user_q = args.question
    examples = get_context(user_q)
    context_str = ''.join(
        [f"Q: {e['question']}\nA: {e['response']}\n\n" for e in examples])
    prompt = (
        "System: For the purposes of this interaction, you are J Robert Oppenheimer. "
        "Answer in a paragraph at most.\n\n"
        "Here are some examples of your style:\n\n"
        f"{context_str}"
        f"User: {user_q}\nAssistant: "
    )
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)

    stop_tokens = ["\nUser:", "\nAssistant:", "<STOP>", "\nSystem:"]
    stop_ids = [
        tokenizer(t, add_special_tokens=False).input_ids for t in stop_tokens]
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])

    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        stopping_criteria=stopping_criteria
    )
    gen = tokenizer.decode(
        output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    for tok in stop_tokens:
        if tok in gen:
            gen = gen.split(tok)[0]
    print("Generated styled output (Few-Shot):")
    print(gen)

# ----------- Main CLI -----------


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to generate styled output as J Robert Oppenheimer"
    )
    parser.add_argument("--question", type=str,
                        required=True, help="User question")
    parser.add_argument(
        "--method", type=str, choices=["style_vectors", "few_shot"], default="style_vectors",
        help="Stylization method"
    )
    parser.add_argument("--qa-file", type=str, required=True,
                        help="Path to QA JSON file")
    parser.add_argument("--neutral-file", type=str,
                        help="Path to Neutral JSON file (for style_vectors)")
    parser.add_argument("--model-name", type=str,
                        default="google/gemma-3-4b-it", help="Transformers model name")
    parser.add_argument("--layer-index", type=int, default=20,
                        help="Layer index for style vector")
    parser.add_argument("--max-new-tokens", type=int,
                        default=200, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float,
                        default=0.7, help="Sampling temperature")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Style vector scale")
    args = parser.parse_args()

    if args.method == "style_vectors":
        if not args.neutral_file:
            parser.error("--neutral-file is required for style_vectors method")
        style_vector_generation(args)
    else:
        few_shot_generation(args)


if __name__ == "__main__":
    main()
