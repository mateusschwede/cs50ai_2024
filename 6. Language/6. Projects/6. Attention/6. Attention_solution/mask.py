import sys
from typing import Any, Tuple, Union
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoTokenizer,
    TFBertForMaskedLM,
    BatchEncoding,
)
from transformers.modeling_tf_outputs import TFMaskedLMOutput

MODEL = "bert-base-uncased"
K = 3
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200

AttentionWeightsMatrix = list[list[float]]
TensorData = list[list[AttentionWeightsMatrix]]


def main():
    text = input("Text: ")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)

    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    model: Any = TFBertForMaskedLM.from_pretrained(MODEL)
    result: TFMaskedLMOutput = model(**inputs, output_attentions=True)
    mask_token_logits = result.logits[0, mask_token_index]  # type: ignore
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    visualize_attentions(inputs.tokens(), result.attentions)  # type: ignore


def get_mask_token_index(mask_token_id: int | None, inputs: BatchEncoding):
    if mask_token_id is None:
        return

    input_ids: list[int] = inputs["input_ids"][0].numpy().tolist()  # type: ignore
    
    if mask_token_id in input_ids:
        return input_ids.index(mask_token_id)


def get_color_for_attention_score(attention_score: float):
    colour_weight = int(attention_score * 255)
    return (colour_weight, colour_weight, colour_weight)


def visualize_attentions(tokens: list[str], attentions: list[TensorData]):
    layers_count = len(attentions)
    heads_count = len(attentions[0][0])
    beam_index = 0
    
    for layer_index in range(layers_count):
        for head_index in range(heads_count):
            generate_diagram(
                layer_number=layer_index + 1,
                head_number=head_index + 1,
                tokens=tokens,
                attention_weights=attentions[layer_index][beam_index][head_index],
            )


def generate_diagram(
    layer_number: int,
    head_number: int,
    tokens: list[str],
    attention_weights: AttentionWeightsMatrix,
):

    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    for i, token in enumerate(tokens):
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT,
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT,
        )

    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()