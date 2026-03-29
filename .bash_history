pip install torch torchvision diffusers transformers accelerate pillow
python infer.py --mode mr     --person_image person.jpg     --garment_images top.jpg pants.jpg shoes.jpg     --garment_categories upper_body lower_body shoes     --output result.png --seed 42
--mode mr     --person_image person.jpg     --garment_images top.jpg pants.jpg     --garment_categories upper_body lower_body
