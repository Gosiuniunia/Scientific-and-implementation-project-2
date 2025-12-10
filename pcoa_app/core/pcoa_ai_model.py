import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
import io
from utils.enums import *


class ColorAnalysisModel:
    def __init__(self):
        self.classes = {
            0: "Spring",
            1: "Summer",
            2: "Autumn",
            3: "Winter"
        }
        
        self.color_palettes = {
            "Spring": {
                "colors": ["#FFB6C1", "#FFDAB9", "#98FB98", "#87CEEB", "#F0E68C", "#FFE4B5", "#DDA0DD", "#F5DEB3"],
                "description": "Warm, bright, and clear colors with yellow undertones"
            },
            "Summer": {
                "colors": ["#B0C4DE", "#E6E6FA", "#F0F8FF", "#D8BFD8", "#FFB6D9", "#C1E1C1", "#B0E0E6", "#FADADD"],
                "description": "Cool, soft, and muted colors with blue undertones"
            },
            "Autumn": {
                "colors": ["#D2691E", "#CD853F", "#B8860B", "#8B4513", "#A0522D", "#BC8F8F", "#DAA520", "#556B2F"],
                "description": "Warm, rich, and earthy colors with golden undertones"
            },
            "Winter": {
                "colors": ["#000000", "#FFFFFF", "#DC143C", "#4169E1", "#8B008B", "#2F4F4F", "#FF1493", "#191970"],
                "description": "Cool, vivid, and high-contrast colors with blue undertones"
            }
        }
        
        self.recommendations = {
            "Spring": [
                "Wear warm, bright colors like peach, coral, and warm pink",
                "Avoid dark, heavy colors that can overwhelm your natural brightness",
                "Gold jewelry complements your warm undertones better than silver",
                "Choose clear, vibrant shades over muted tones"
            ],
            "Summer": {
                "Wear soft, cool colors like lavender, soft pink, and powder blue",
                "Avoid warm, golden tones that clash with your cool undertones",
                "Silver and white gold jewelry suit you best",
                "Choose muted, dusty shades over bright, vivid colors"
            },
            "Autumn": [
                "Wear warm, earthy colors like rust, olive, and camel",
                "Avoid icy, bright colors that can wash you out",
                "Gold and bronze jewelry enhance your warm coloring",
                "Choose rich, muted tones over pastel shades"
            ],
            "Winter": [
                "Wear cool, vivid colors like true red, royal blue, and pure white",
                "Avoid warm, muted colors that can make you look dull",
                "Silver and platinum jewelry complement your cool undertones",
                "Choose high-contrast combinations and jewel tones"
            ]
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
        return self.recommendations.get(season, [])
    
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