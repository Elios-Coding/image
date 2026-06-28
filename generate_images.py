from flask import Flask, render_template, request
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        images = []
        
        # Generate 2 unrestricted images
        for _ in range(2):
            image = pipe(prompt, height=768, width=768, num_inference_steps=50).images[0]
            images.append(image)
        
        # Convert images to base64 for display
        image_data = []
        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            image_data.append(base64.b64encode(buffer.getvalue()).decode())
        
        return render_template('index.html', images=image_data)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
