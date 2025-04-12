import base64

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def count2group(counts: list[int]) -> list[str]:
    groups = []
    for count in counts:
        if count == 0:
            groups.append('empty')
        elif count < 5:
            groups.append('few')
        elif count < 15:
            groups.append('medium')
        else:
            groups.append('many')
    return groups
