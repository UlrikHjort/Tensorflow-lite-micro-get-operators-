#######################################################################
#
# Get tflite operators
#
# By Ulrik Hørlyk Hjort 2022
#
# Usage: python get_ops.py <tflite model>
#
# Print out a list of used operators
#
#Requirements: Tensorflow version >= 2.5  
#
#######################################################################
import sys
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema_fb

def FlatbufferToDict(fb, preserve_as_numpy):
  if isinstance(fb, int) or isinstance(fb, float) or isinstance(fb, str):
    return fb
  elif hasattr(fb, "__dict__"):
    result = {}
    for attribute_name in dir(fb):
      attribute = fb.__getattribute__(attribute_name)
      if not callable(attribute) and attribute_name[0] != "_":
        preserve = True if attribute_name == "buffers" else preserve_as_numpy
        result[attribute_name] = FlatbufferToDict(attribute, preserve)
    return result
  elif isinstance(fb, np.ndarray):
    return fb if preserve_as_numpy else fb.tolist()
  elif hasattr(fb, "__len__"):
    return [FlatbufferToDict(entry, preserve_as_numpy) for entry in fb]
  else:
    return fb

def CreateDictFromFlatbuffer(buffer_data):  
  model = schema_fb.ModelT.InitFromObj(schema_fb.Model.GetRootAsModel(buffer_data, 0))
  return FlatbufferToDict(model, preserve_as_numpy=False)


tflite_input = sys.argv[1]
with open(tflite_input, "rb") as file_handle:
    file_data = bytearray(file_handle.read())
data = CreateDictFromFlatbuffer(file_data)

for i in data['operatorCodes']:
    for name , value in schema_fb.BuiltinOperator.__dict__.items():
        if value == i['builtinCode']:
            print(name)
