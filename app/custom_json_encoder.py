import json
from fastapi.encoders import jsonable_encoder

class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, SomeCustomType):
            return obj.to_dict()  # Ensure the custom type has a method to convert it to a dictionary
        return super().default(obj)