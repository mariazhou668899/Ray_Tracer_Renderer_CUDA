# Ray_Tracer_Renderer_CUDA
This is a small ray tracer renderer system using CUDA. 
This Ray Tracing System uses a custom data type Vec3 as a basic data to support constructing camera and ray, and to represent a location point in space and color. System gets the results of the surface normal by calculating the point of contact between the ray and an object. From there, it uses a model called Phong shading model to do the coloring for every pixel of the surface and get a snapshot. The x, y, z of snapshots’ points are transformed into PPM format and is stored as a PPM file for viewing. Each PPM file is a frame corresponding to a snapshot. Besides, it renders multiple frames to simulate an animated scene, including the light surrounding on the surface of 3 spheres and the cube moving along Y axis. The rendering process has been accelerated on GPU by CUDA, allowing massive parallelism in the rendering process and ultimately cutting down on the long rendering time.

Besides, this renderer is implemented by C++ on CPU, which is used to compare the perfermance with the GPU version.

![output](https://github.com/mariazhou668899/Ray_Tracer_Renderer_CUDA/assets/121517781/d52990f9-481c-47af-bb8a-812781cc0543)
