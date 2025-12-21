import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
import io

class ColorAnalysisModel:
    def __init__(self):
        self.classes = {
            0: "Spring",
            1: "Summer",
            2: "Autumn",
            3: "Winter"
        }
        
        self.color_palettes = {
            "Spring": ["#639E3F", "#DF5485", "#DD9A29", "#215380", "#EBDDCC", "#FDAA63", "#008EAA", "#963CBD"],
            "Summer": ["#EC9AAC", "#6F9987", "#72A8BA", "#9F8B84", "#7A749B", "#F1BDC8", "#9CAF88", "#484A51"],
            "Autumn": ["#591D2D", "#1A2042", "#143831", "#BE4D00", "#4D9E9A", "#5C462B", "#890C58", "#DAAA00"],
            "Winter": ["#7A2942", "#5C068C", "#1A3A47", "#20334A", "#341902", "#00594C", "#AA0061", "#0057B8"]
        }
        
        self.descriptions = {
            "Spring": "Your beauty is bright, warm, and fresh. Light, luminous colors accentuate you best, adding radiance to your complexion. Warm shades are ideal: apricot, coral, golden, light turquoise, mint, or warm beige. Avoid colors that are too cool and muted - they can rob your face of its natural radiance.",
            "Summer": "Your beauty is delicate, cool, and subtle. You look most beautiful in soft, pastel shades that complement the natural harmony of your features. Cool, smoky colors are best, such as lavender, powder pink, sky blue, dove gray, or cool raspberry. Avoid very bright and vibrant colors - they can overpower your delicate color palette.",
            "Autumn": "You have a warm, expressive, and deep beauty. Earthy colors suit you perfectly, as they emphasize your natural intensity. You look best in shades like terracotta, olive, mustard, cinnamon, dark green, and warm chocolate. Avoid very cool and neon colors, which can create an unfavorable contrast.",
            "Winter": "Your beauty is strong, contrasting, and cool. You look best in pure, bold colors that complement the intensity of your features. You look great in snow white, black, fuchsia, cobalt, ruby, and cool emerald. Avoid shades that are too warm or muted - they can weaken your natural contrast."
        }

        self.jewelery_recommendations = {
            "Spring": "Jewelry recommended for spring-like individuals includes gold pieces with pastel gemstones, such as peridot or light pink tourmaline, which accentuate the natural vitality of spring tones.",
            "Summer": "The best jewelry choices for summer-themed individuals include silver and white gold, with gemstones, and electronics like aquamarine and rose quartz. When pairing the season with your outfits, you should incorporate pieces that complement the cool and calm nature of summer tones, creating a harmonious and elegant look.",
            "Autumn": "The ideal jewelry for autumnal types includes gold and rose gold, as well as gemstones like amber, citrine, and garnet. These choices emphasize the natural warmth and depth of autumnal colors.",
            "Winter": "Jewelry choices for winter skin tones include platinum or white gold, along with statement gemstones like sapphire and amethyst. You can enhance your winter wardrobe with statement jewelry, choosing statement pieces that will add a touch of luxury and sophistication to your look."
        }
    
    def predict(self, features):
        """Predict color season from extracted features"""
        # tutaj zaimplementuj jak model robi predykcję
        predicted_class_id = np.random.randint(0, len(self.classes))
        confidence = np.random.uniform(0.75, 0.98)
        predicted_season = self.classes[predicted_class_id]
        return predicted_season, confidence

    def get_palette_info(self, season):
        return self.color_palettes.get(season)

    def get_recommendations(self, season):
        return self.descriptions.get(season)
    
    def create_color_palette_image(self, season):
        colors = self.color_palettes[season]["colors"]
        
        # Create image with color swatches
        swatch_size = 100
        padding = 10
        cols = 4
        rows = 2
        
        width = cols * swatch_size + (cols + 1) * padding
        height = rows * swatch_size + (rows + 1) * padding
        
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        for idx, color in enumerate(colors):
            row = idx // cols
            col = idx % cols
            
            x = col * swatch_size + (col + 1) * padding
            y = row * swatch_size + (row + 1) * padding
            
            draw.rectangle(
                [x, y, x + swatch_size, y + swatch_size],
                fill=color,
                outline='#CCCCCC',
                width=2
            )
        
        return img