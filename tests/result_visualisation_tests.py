import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pytest
from core.pcoa_result_visualisation import ResultVisualizer

class TestResultVisualizer:

    @pytest.fixture
    def viz(self):
        """
        Creates a new ResultVisualizer instance before every test.
        """
        return ResultVisualizer()

    def test_get_palette_valid(self, viz):
        """
        Checks color palette retrieval for spring season as example.
        Checks:
            - if function returns a list of colors
            - if there are colors in returned list
            - if a specific color is inside the palette
        """
        palette = viz.get_palette_info("spring")
        
        assert isinstance(palette, list)
        assert len(palette) > 0
        assert "#639E3F" in palette 

    def test_get_palette_invalid(self, viz):
        """
        Tests a case when non-valid color season if provided to get_palette function.
        """
        assert viz.get_palette_info("dummy_season") is None

    def test_get_description_valid(self, viz):
        """
        Tests correctness od retrieved season description. Test is conducted for winter.
        Checks:
        - if obtained desctription is string
        - if given words are present in description
        """
        desc = viz.get_description("winter")
        assert isinstance(desc, str)
        assert "strong, contrasting" in desc
        assert "snow white" in desc

    def test_get_jewelry_valid(self, viz):
        """
        Tests correctness od retrieved jewerly recommendation description. Test is conducted for autumn.
        Checks:
        - if obtained desctription is string
        - if given words are present in description
        """
        recommendation = viz.get_jewelry_recommendation("autumn")
        assert isinstance(recommendation, str)
        # Verify specific jewelry keywords for Autumn
        assert "gold" in recommendation
        assert "amber" in recommendation

    def test_all_seasons_exist(self, viz):
        """
        Checks if its possible to retrieve details for every valid seasonal color type.
        """
        seasons = ["spring", "summer", "autumn", "winter"]
        
        for season in seasons:
            assert viz.get_palette_info(season) is not None
            assert viz.get_description(season) is not None
            assert viz.get_jewelry_recommendation(season) is not None