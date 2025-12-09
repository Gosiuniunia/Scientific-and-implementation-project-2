"""
This module creates a Gradio interface to display the results of a color analysis.
It shows the seasonal color type, description, and a palette of recommended colors.
"""

# Parameter defining the season to display
season = "autumn"

import gradio as gr

data = {"spring": {"description": "Your beauty is bright, warm, and fresh. Light, luminous colors accentuate you best, adding radiance to your complexion. Warm shades are ideal: apricot, coral, golden, light turquoise, mint, or warm beige. Avoid colors that are too cool and muted - they can rob your face of its natural radiance.",
                "colors": ["#639E3F", "#DF5485", "#DD9A29", "#215380", "#EBDDCC", "#FDAA63", "#008EAA", "#963CBD"],
                "jewellery":"Jewelry recommended for spring-like individuals includes gold pieces with pastel gemstones, such as peridot or light pink tourmaline, which accentuate the natural vitality of spring tones."},
        "summer": {"description": "Your beauty is delicate, cool, and subtle. You look most beautiful in soft, pastel shades that complement the natural harmony of your features. Cool, smoky colors are best, such as lavender, powder pink, sky blue, dove gray, or cool raspberry. Avoid very bright and vibrant colors - they can overpower your delicate color palette.",
                "colors": ["#EC9AAC", "#6F9987", "#72A8BA", "#9F8B84", "#7A749B", "#F1BDC8", "#9CAF88", "#484A51"],
                "jewellery":"The best jewelry choices for summer-themed individuals include silver and white gold, with gemstones, and electronics like aquamarine and rose quartz. When pairing the season with your outfits, you should incorporate pieces that complement the cool and calm nature of summer tones, creating a harmonious and elegant look."},
        "autumn": {"description": "You have a warm, expressive, and deep beauty. Earthy colors suit you perfectly, as they emphasize your natural intensity. You look best in shades like terracotta, olive, mustard, cinnamon, dark green, and warm chocolate. Avoid very cool and neon colors, which can create an unfavorable contrast.",
                "colors": ["#591D2D", "#1A2042", "#143831", "#BE4D00", "#4D9E9A", "#5C462B", "#890C58", "#DAAA00"],
                "jewellery":"The ideal jewelry for autumnal types includes gold and rose gold, as well as gemstones like amber, citrine, and garnet. These choices emphasize the natural warmth and depth of autumnal colors."},
        "winter": {"description": "Your beauty is strong, contrasting, and cool. You look best in pure, bold colors that complement the intensity of your features. You look great in snow white, black, fuchsia, cobalt, ruby, and cool emerald. Avoid shades that are too warm or muted - they can weaken your natural contrast.",
                "colors": ["#7A2942", "#5C068C", "#1A3A47", "#20334A", "#341902", "#00594C", "#AA0061", "#0057B8"],
                "jewellery":"Jewelry choices for winter skin tones include platinum or white gold, along with statement gemstones like sapphire and amethyst. You can enhance your winter wardrobe with statement jewelry, choosing statement pieces that will add a touch of luxury and sophistication to your look."}}

def show_season():
    colors = data[season]["colors"]
    html_colors = ""

    for i, c in enumerate(colors):
        size = 150 if i == 0 else 80
        display_hex = "block" if i == 0 else "none"
        html_colors += f"""
        <div style="text-align:center; width:150px; height:180px; margin-right:5px; display:flex; flex-direction:column; align-items:center;">
            <div class='color-box' 
                 style='background:{c}; width:{size}px; height:{size}px; border-radius:6px; cursor:pointer; transition: all 0.3s;'
                 onclick="document.querySelectorAll('.color-box').forEach(b => {{
                            b.style.width='80px'; 
                            b.style.height='80px'; 
                            b.nextElementSibling.style.display='none';
                        }}); 
                        this.style.width='150px'; 
                        this.style.height='150px'; 
                        this.nextElementSibling.style.display='block';">
            </div>
            <div class='color-hex' style='margin-top:5px; font-weight:bold; display:{display_hex};'>{c}</div>
        </div>
        """

    html = f"""
    <div style='display:flex;align-items:flex-start;'>
        {html_colors}
    </div>
    """
    return html

def dummy_back():
    print("Funkcja powrotu zostanie dodana później.")
    return

with gr.Blocks() as demo:
    gr.Markdown("## Your Seasonal Color Analysis Result")
    gr.Markdown("Based on the analysis, your seasonal color is:")
    gr.Markdown(f"# {season.capitalize()}")
    gr.Markdown("## Description:")
    gr.Markdown(f'{data[season]["description"]}')

    colors_output = gr.HTML()
    demo.load(fn=show_season, outputs=[colors_output])

    gr.Markdown("## Recommended Jewellery:")
    gr.Markdown(f'{data[season]["jewellery"]}')

    back_button = gr.Button("Back to Home")
    back_button.click(dummy_back)

demo.launch()
