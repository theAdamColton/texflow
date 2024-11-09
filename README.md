# texflow

texflow is a Blender extension that bridges ComfyUI and Blender. 
It allows artists to use AI image generation models to paint textures onto the surface of
3D objects. 

# design and coding guidelines

* Only relative imports are allowed
* texflow/client/__init__.py is the blender entry
* All inheritors of the Async Mixin class must have a task name that starts with the prefix "texflow"

