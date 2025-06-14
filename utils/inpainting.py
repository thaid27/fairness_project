    

# Inpainting of the different objects of the scene for diversity scoring
def inpainting_for_scoring(image, mask, pipe):

    image = image.convert("RGB").resize((512,512))
    mask = mask.convert("L").resize((512,512))

    positive_prompt = "Full HD, 4K, high quality, high resolution, photorealistic"
    negative_prompt = ("bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, duplicate, "
                    "error, extra limbs, gross proportions, jpeg artifacts, long neck, low quality, lowres, "
                    "malformed, morbid, mutated, mutilated, out of frame, ugly, worst quality")

    result = pipe(
    prompt=positive_prompt,
    negative_prompt=negative_prompt,
    image=image,
    mask_image=mask,
    guidance_scale=7.5,
    num_inference_steps=50,
    )
    inpainted_image = result.images[0].resize((224,224))

    return inpainted_image


# Inpainting of to remove the targeted object 
def inpainting_remove_object(image, mask, pipe, inpaint_prompt, output_path):
    image_width,image_height = image.size
    original_size = [image_width,image_height]

    image = image.convert("RGB").resize((512,512))
    mask = mask.convert("L").resize((512,512))

    result = pipe(
    prompt=inpaint_prompt,
    image=image,
    mask_image=mask,
    guidance_scale=7.5,
    num_inference_steps=50,
    )
    inpainted_image = result.images[0].resize(original_size)
    if output_path:
        inpainted_image.save(output_path)

    return inpainted_image