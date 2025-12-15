from enum import Enum, unique

@unique
class ColorType(Enum):
    SPRING = "Spring"
    SUMMER = "Summer"
    AUTUMN = "Autumn"
    WINTER = "Winter"

@unique
class PredictionStatus(Enum):
    NOT_DONE = 0
    DONE = 1

@unique
class PhotoUploadStatus(Enum):
    NOT_UPLOADED = 0
    UPLOADED = 1

@unique
class PhotoValidationStatus(Enum):
    NOT_VALIDATED = 0
    VALIDATED = 1

@unique
class PhotoPreprocessingStatus(Enum):
    NOT_VALIDATED = 0
    VALIDATED = 1

@unique 
class RecommendationStatus(Enum):
    NOT_GENERATED = 0
    GENERATED = 1

@unique 
class TermsAccepted(Enum):
    NOT_ACCEPTED = 0
    ACCEPTED = 1