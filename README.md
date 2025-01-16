# texflow

texflow is a Blender extension that bridges ComfyUI and Blender. 
It allows artists to use AI image generation models to paint textures onto the surface of
3D objects. 

![image](https://github.com/user-attachments/assets/f12dac59-39a4-45e2-87aa-c6733b5ec5cd)


# about texflow

I have used other Blender extensions that allow you to generate textures from depth images. 
The main problem with other extensions is that they assume too much responsibility for the generation workflow. 
Texflow is designed to mesh with ComfyUI without needing to control all of the generation settings from a dinky Blender user interface.  

Unlike other extensions such as [Shaamallow/texture-diffusion](https://github.com/Shaamallow/texture-diffusion) and [carson-katri/dream-textures](https://github.com/carson-katri/dream-textures), texflow lets you use your own custom ComfyUI workflows and assumes nothing about what models or prompts you are using.


# installation

texflow supports MacOS (Arm), Windows, and Linux. You need to have an existing installation of ComfyUI.

First, install the ComfyUI extension by navigating to your ComfyUI directory, and then the `custom_nodes` directory. Clone [ComfyUI-texflow-extension](https://github.com/theAdamColton/ComfyUI-texflow-extension) into this directory:

`git clone https://github.com/theAdamColton/ComfyUI-texflow-extension`

You can verify that the ComfyUI extension installed correctly by checking that there is a new node submenu called "texflow"
<img src=https://github.com/user-attachments/assets/9ebdc9ad-607c-47c8-bb87-19f55b602a41 width=300px />


Next, download the [latest Blender extension zip](https://github.com/theAdamColton/texflow/releases/latest), picking the corresponding one for your operating system. For example if your system is Windows, you should download the file the looks like adam_colton_texflow-x.x.x-windows_x64.zip.  Install this extension in Blender by navigating to Preferences->Get Extensions->arrow in top right corner->Install From Disk

<img src=https://github.com/user-attachments/assets/8bee46a6-5d53-43b6-ba7c-dd0a5e258d59 width=400px />

Then find the zip file you downloaded and install it.

# usage

* Launch ComfyUI and take note of the server url, which is printed to the console and should look like this: `Starting server To see the GUI go to: http://127.0.0.1:8188`
* Go to the layout tab and open the 'texflow' panel
* Make sure that the ComfyUI server URL is the correct address where ComfyUI can be found.
* Select a mesh object, go into edit mode and select all of the faces of the object that you want to be textured.
* Select a camera object in the texflow panel.
* Click the "Render Depth Image" button. This will capture a depth image and upload it to the ComfyUI server.
* Go to ComfyUI. You can load the following png as a workflow. All that is required is the LoadTexflowDepthImage and SaveTexflowImage nodes. These nodes will allow the Blender extension to communicate properly with ComfyUI.
![Result_00041_](https://github.com/user-attachments/assets/0e412ae2-d7d1-4ab6-a8b4-e7406c508167)
![image](https://github.com/user-attachments/assets/1bee1c70-0abb-497d-b1e0-96d2ccf2ad96)
* Generate an image.
* Once you have one or more generated images, you can go back to Blender. Click on the "Load ComfyUI Images" button in the texflow panel. You should get a message that says "Added 1 new generated materials".
* You can find all generated materials in the shading tab. 
![image](https://github.com/user-attachments/assets/505a8dca-bc94-4c4e-9411-2be90406d51a)


# tips and tricks

* Use an orthographic camera to get a wider view of the object
* stretch out your object in 3D away and towards the camera to play with the effect of strengthing/weakening the depth

# how it works (for coders)

The code is split into a Blender extension, and a ComfyUI extension. 

The Blender extension provides two functions to the blender api, `texflow.render_depth_image` and `texflow.load_comfyui_images`. 
`render_depth_image` uses a camera to render a depth image of the selected faces of an object. A unique render_id is saved in as a non persistant Blender variable. The depth image and the occupancy image is uploaded to the ComfyUI server at the `upload/image` endpoint. 
The two images are always uploaded with the filenames, `texflow_depth_image.tiff` and `texflow_occupancy_image.png`. The render_id is saved in the exif data of the occupancy png. Note that the depth image is saved as a uint16 instead of uint8.

The ComfyUI extension provides two new nodes, `LoadTexflowDepthImage`, and `SaveTexflowImage`. The loader node loads the `texflow_depth_image.tiff` and `texflow_occupancy_image.png`, reading the render_id from the png. The saver node saves the output image with a special prefix that includes the render_id. The output filename looks like this: `texflow_a9cb714f-de4a-4cc5-8d38-392ec73e2a6d_00001_.png`. 

Going back to the Blender client, the `load_comfyui_images` will check the `history` endpoint and look for images that were saved with the matching special prefix. All matching images are downloaded and added to Blender as new diffuse textures.

# design and coding guidelines

* Only relative imports are allowed
* texflow/client/__init__.py is the blender entry
* texflow/__init__.py is deleted when building
