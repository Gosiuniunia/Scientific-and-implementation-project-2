from PCoA_prediction import predict_class

class ColorAnalysisModel:
    """
    Class representing the AI module and result visualisation modules for PCOA app.
    Provides method to predict color season using provided model. 
    """
    def __init__(self):
        """
        Initializes the ColorAnalysisModel with predefined classes, color palettes, descriptions, and jewelry recommendations.
        classes: dict: Mapping of class indices to color seasons.
        color_palettes (dict): Mapping of color seasons to their respective color palettes.
        descriptions (dict): Mapping of color seasons to their descriptions.
        jewelery_recommendations (dict): Mapping of color seasons to jewelry recommendations.
        """

        self.model_path = "svc.pkl"
    
    def predict(self, features):
        """
        Predicts color season from extracted features
        Args:
            features (list): Extracted facial features.
        Returns:
            str: Predicted color season.
        """
        predicted_season = predict_class(self.model_path, features)
        return predicted_season
    