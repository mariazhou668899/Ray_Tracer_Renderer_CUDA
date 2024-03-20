/*
  ** In order to simplify compiling and code checking, all the Classes and functions are combined into one file "animation_CPU.cpp".
  
  
  1. Program Compiling
  
  This program runs on UWB GPU lab machine like test the GPU way implementation, the steps:
  step 1: Download and store "animation_CPU.cu" in a folder on the UWB GPU lab machine.
  step 2: Navigate to the directory where the "animation_CPU.cu" file is located.
  step 3: bash input: g++  -o cpu_anima animation_CPU.cpp
  step 4: bash input: ./cpu_anima
          
  
  2. Program Introduction
  
  This c++ program, written in C/C++, running using CPU, demonstrates ray tracing using the Phong reflection model in a simple 3D scene. 
  The scene consists of three spheres and a cube, illuminated by a moving light source. The animation of the scene includes both the 
  movement of the light source and the oscillation of the cube along the y-axis.
  
  Overview of the components:
  
  - Vec3: Represents a 3D vector and provides various vector operations.
  - Ray: Represents a ray in 3D space.
  - Camera: Represents a simple camera with basic properties.
  - Phong Shading: Implements shading using the Phong reflection model.
  - hit_object: Checks if a ray intersects with a sphere.
  - hit_sphere: Calculates the intersection of a ray with a sphere.
  - hit_object_or_floor: Checks if a ray intersects either a sphere or the floor.
  - hit_cube: Calculates the intersection of a ray with a cube.
  - calculate_cube_normal: Calculates the normal vector of a cube at a given point.
  - color: Determines the color at a specific pixel based on ray-sphere intersections.
  - save_image: Saves the rendered image to a PPM file.
  - animate_cube: Animates the movement of the cube along the y-axis.
  - animate_scene: Animates the movement of the light source in a circular path.
  - render: Function to render the scene and perform ray tracing.
  
*/


#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <random>
#include <cstdlib>

using namespace std;

// Constants about Frames
constexpr size_t width = 1000;
constexpr size_t height = 900;
constexpr size_t num_pixels = width * height;
constexpr size_t num_samples = 100;


//=========================================== Class Part ==============================================

// Vec3 Class - Represents a 3D vector
class Vec3 {
public:

    float x, y, z;

    // Constructors
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    Vec3(float value) : x(value), y(value), z(value) {}

    // Vector addition
    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    // Scalar multiplication (friend function)
    friend Vec3 operator*(float scalar, const Vec3& vec) {
        return Vec3(scalar * vec.x, scalar * vec.y, scalar * vec.z);
    }

    // Vector dot product
    float dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    // Vector cross product
    Vec3 cross(const Vec3& other) const {
        return Vec3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
    }

    // Length squared of the vector
    float lengthSquared() const {
        return x * x + y * y + z * z;
    }

    // Length of the vector
    float length() const {
        return sqrtf(lengthSquared());
    }

    // Normalize the vector
    Vec3 normalize() const {
        float len = length();
        return Vec3(x / len, y / len, z / len);
    }

    // Compound addition operator
    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    // Reflection of the vector about a normal
    Vec3 reflect(const Vec3& normal) const {
        return *this - 2.0f * this->dot(normal) * normal;
    }

    // Scalar multiplication operator
    Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    // Compound division operator
    Vec3& operator/=(size_t divisor) {
        float invDivisor = 1.0f / static_cast<float>(divisor);
        x *= invDivisor;
        y *= invDivisor;
        z *= invDivisor;
        return *this;
    }

    // Subtraction operator
    Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }

    // Unary negation operator
    Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }

    // Vec3 multiplication operator
    Vec3 operator*(const Vec3& other) const {
        return Vec3(x * other.x, y * other.y, z * other.z);
    }

};


// Ray Class - Represents a ray in 3D space
class Ray {
public:
    Vec3 origin, direction;

    // Constructor
    Ray(const Vec3& origin, const Vec3& direction) : origin(origin), direction(direction) {}
};


// Camera Class - Represents a simple camera in 3D space
class Camera {
public:
    Vec3 origin, lower_left_corner, horizontal, vertical, light_position;
    float aspect_ratio;

    // Constructor
    Camera(float aspect_ratio = 1.0f) : aspect_ratio(aspect_ratio) {
        origin = Vec3(0.0f, 0.0f, 0.0f);
        float viewport_height = 2.0f;
        float viewport_width = aspect_ratio * viewport_height;
        float focal_length = 1.0f;

        lower_left_corner = origin - 0.5f * viewport_width * Vec3(1, 0, 0) - 0.5f * viewport_height * Vec3(0, 1, 0) - focal_length * Vec3(0, 0, 1);
        horizontal = viewport_width * Vec3(1, 0, 0);
        vertical = viewport_height * Vec3(0, 1, 0);
        light_position = Vec3(1.0f, 1.0f, -1.0f);
    }

    // Generate a ray for a given pixel (u, v)
    Ray get_ray(float u, float v) const {
        Vec3 adjusted_origin = origin + 0.001f * Vec3(1, 1, 1);
        return Ray(adjusted_origin, lower_left_corner + u * horizontal + v * vertical - adjusted_origin);
    }
};




//=========================================== Tool Functions Declearing Part ==============================================

// Phong Shading - Calculates shading using the Phong reflection model
Vec3 phong_shading(const Vec3& normal, const Vec3& light_direction, const Vec3& view_direction,
                   const Vec3& ambient_color, const Vec3& diffuse_color, const Vec3& specular_color,
                   float shininess);
                   

// Sphere Intersection - Calculates the intersection of a ray with a sphere
float hit_sphere(const Vec3& center, float radius, const Ray& ray);


// Calculate cube normal at a given point
Vec3 calculate_cube_normal(const Vec3& point, const Vec3& center, float side_length);

// Color Calculation - Determines the color at a specific pixel based on ray-sphere intersections
Vec3 color(const Ray& ray, const Vec3& light_position, Vec3& cube_center);

// Save Image to PPM in text format (P3)
void save_image(const std::string& filename, Vec3* framebuffer, size_t width, size_t height);


// Animation function to move only the cube
void animate_cube(float animation_time, Vec3& cube_position, const Vec3& camera_origin, float min_y_boundary, float max_y_boundary);


// Main animation function to move the light source
void animate_scene(float animation_time, Vec3& light_position, const Vec3& camera_origin);


//Renders the scene by tracing rays through each pixel of the image.
void render(Vec3* framebuffer, size_t width, size_t height, size_t num_samples, float animation_time, Vec3 cube_position_local, float min_y_boundary, float max_y_boundary);




//=========================================== Main() Part ==============================================

int main() {
    // Allocate static memory for the framebuffer
    static Vec3 framebuffer[width * height];

    // Get start time
    const auto start_time = std::chrono::high_resolution_clock::now();
    const size_t num_frames = 5;

    // Dummy camera initialization
    struct Camera {
        Vec3 light_position;
        Vec3 origin;
    };
    
    Camera camera;

    Vec3 cube_position(0.72, 0, -1); // Adjust the position of the cube

    float min_y_boundary = -0.2;
    float max_y_boundary = 0.75;

    for (size_t frame = 0; frame < num_frames; ++frame) {
        float animation_time = static_cast<float>(frame) / num_frames;

        // Light animate
        animate_scene(animation_time, camera.light_position, camera.origin);

        // Cube animate
        animate_cube(animation_time, cube_position, camera.origin, min_y_boundary, max_y_boundary);

        // Render the current frame with updated light position
        render(framebuffer, width, height, num_samples, animation_time, cube_position, min_y_boundary, max_y_boundary);
        std::string filename = "CPUframe" + std::to_string(frame) + ".ppm";
        save_image(filename, framebuffer, width, height);
    }

    // Get end time
    const auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate and print elapsed time
    const auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "\nGenerated Frames Number: " << num_frames << std::endl;
    std::cout << "CPU totally took: " << time_elapsed << " ms" << std::endl << std::endl;

    return 0;
}



//=========================================== Tool Functions Inplimentation Part ==============================================

// Phong Shading - Calculates shading using the Phong reflection model
Vec3 phong_shading(const Vec3& normal, const Vec3& light_direction, const Vec3& view_direction,
                                        const Vec3& ambient_color, const Vec3& diffuse_color, const Vec3& specular_color,
                                        float shininess) {
    // Initialize the result color with ambient color
    Vec3 result_color = ambient_color;

    // Calculate diffuse intensity using Lambert's law
    float diffuse_intensity = max(normal.dot(light_direction), 0.0f);
    
    // Add diffuse component to the result color
    result_color += diffuse_intensity * diffuse_color;

    // Calculate reflect direction using the light direction and normal
    Vec3 reflect_direction = -light_direction.reflect(normal);

    // Calculate specular intensity using the reflect direction and view direction
    float specular_intensity = pow(max(view_direction.dot(reflect_direction), 0.0f), shininess);
    
    // Add specular component to the result color
    result_color += specular_intensity * specular_color;

    // Return the final color after combining ambient, diffuse, and specular components
    return result_color;
}


// Sphere Intersection - Calculates the intersection of a ray with a sphere
float hit_sphere(const Vec3& center, float radius, const Ray& ray) {
    Vec3 oc = ray.origin - center;
    float a = ray.direction.dot(ray.direction);
    float b = 2.0 * oc.dot(ray.direction);
    float c = oc.dot(oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant > 0) {
        // Calculate the closest intersection point
        float t = (-b - sqrt(discriminant)) / (2.0 * a);
        return t;
    }

    return -1.0f; // No intersection
}


// Cube Intersection - Calculates the intersection of a ray with a cube
float hit_cube(const Vec3& center, float side_length, const Ray& ray) {
    float half_side = 0.5 * side_length;

    float tmin = (center.x - half_side - ray.origin.x) / ray.direction.x;
    float tmax = (center.x + half_side - ray.origin.x) / ray.direction.x;

    float tymin = (center.y - half_side - ray.origin.y) / ray.direction.y;
    float tymax = (center.y + half_side - ray.origin.y) / ray.direction.y;

    float tzmin = (center.z - half_side - ray.origin.z) / ray.direction.z;
    float tzmax = (center.z + half_side - ray.origin.z) / ray.direction.z;

    // Find the minimum and maximum intersection distances
    float t_near = fmax(fmax(fmin(tmin, tmax), fmin(tymin, tymax)), fmin(tzmin, tzmax));
    float t_far = fmin(fmin(fmax(tmin, tmax), fmax(tymin, tymax)), fmax(tzmin, tzmax));

    // Check if the intersection is valid
    if (t_near <= t_far && t_far > 0.0f) {
        // Valid intersection
        return t_near;
    }

    // No intersection
    return -1.0f;
}


// Calculate cube normal at a given point
Vec3 calculate_cube_normal(const Vec3& point, const Vec3& center, float side_length) {
    float half_side = side_length / 2.0f;
    Vec3 outward_normal(0, 0, 0);

    if (fabs(point.x - center.x + half_side) < 1e-4) outward_normal.x = 1;
    else if (fabs(point.x - center.x - half_side) < 1e-4) outward_normal.x = -1;
    else if (fabs(point.y - center.y + half_side) < 1e-4) outward_normal.y = 1;
    else if (fabs(point.y - center.y - half_side) < 1e-4) outward_normal.y = -1;
    else if (fabs(point.z - center.z + half_side) < 1e-4) outward_normal.z = 1;
    else if (fabs(point.z - center.z - half_side) < 1e-4) outward_normal.z = -1;

    return outward_normal;
}


// Color Calculation - Determines the color at a specific pixel based on ray-sphere intersections
Vec3 color(const Ray& ray, const Vec3& light_position, Vec3& cube_center) {
    // Sphere parameters
    Vec3 sphere1_center(0, -0.3, -1);       // Center of the first sphere
    Vec3 sphere2_center(0, 0.45, -1);        // Center of the second sphere
    Vec3 sphere3_center(-0.61, 0, -1);       // Center of the third sphere
    float sphere_radius1 = 0.45;             // Radius of the first sphere
    float sphere_radius2 = 0.3;              // Radius of the second sphere
    float sphere_radius3 = 0.2;              // Radius of the third sphere

    // Cube parameters
    float cube_side_length = 0.2;            // Side length of the cube

    // Ray-sphere intersections
    float t1 = hit_sphere(sphere1_center, sphere_radius1, ray);   // Intersection distance with the first sphere
    float t2 = hit_sphere(sphere2_center, sphere_radius2, ray);   // Intersection distance with the second sphere
    float t3 = hit_sphere(sphere3_center, sphere_radius3, ray);   // Intersection distance with the third sphere

    // Ray-cube intersection
    float t_cube = hit_cube(cube_center, cube_side_length, ray);  // Intersection distance with the cube

    // Find the closest intersection point
    float t = -1.0f;  // Initialize the closest intersection distance
    if (t1 > 0.0f && (t < 0.0f || t1 < t)) t = t1;                // Update t if t1 is valid and closer
    if (t2 > 0.0f && (t < 0.0f || t2 < t)) t = t2;                // Update t if t2 is valid and closer
    if (t3 > 0.0f && (t < 0.0f || t3 < t)) t = t3;                // Update t if t3 is valid and closer
    if (t_cube > 0.0f && (t < 0.0f || t_cube < t)) t = t_cube;    // Update t if t_cube is valid and closer

    // Check if an intersection occurred
    if (t > 0.0f) {
        // Calculate the intersection point and normal vector
        Vec3 intersection_point = ray.origin + t * ray.direction;  // Calculate the intersection point

        // Handle sphere intersections
        if (t == t1 || t == t2 || t == t3) {
            // Determine the normal vector based on which sphere was intersected
            Vec3 normal = (t == t1) ? (intersection_point - sphere1_center).normalize() :
                          (t == t2) ? (intersection_point - sphere2_center).normalize() :
                          (intersection_point - sphere3_center).normalize();

            // Calculate the direction of the light from the intersection point
            Vec3 light_direction = (light_position - intersection_point).normalize();

            // Calculate the view direction (direction from intersection point to the camera)
            Vec3 view_direction = -ray.direction.normalize();

            // Material properties for spheres
            Vec3 ambient_color(0.1, 0.1, 0.1);                       // Ambient color of the material
            Vec3 diffuse_color = (t == t1) ? Vec3(1.0, 0.0, 0.0) :   // Diffuse color for each sphere
                                 (t == t2) ? Vec3(1.0, 1.0, 0.0) :
                                             Vec3(0.0, 0.0, 1.0);
            Vec3 specular_color(1.0, 1.0, 1.0);                      // Specular color of the material
            float shininess = 32.0;                                  // Shininess factor for specular highlights

            // Apply Phong shading to calculate the pixel color
            return phong_shading(normal, light_direction, view_direction, ambient_color, diffuse_color, specular_color, shininess);
        }
        // Handle cube intersection
        else if (t == t_cube) {
            // Calculate the normal vector of the cube at the intersection point
            Vec3 normal = calculate_cube_normal(intersection_point, cube_center, cube_side_length);

            // Calculate the direction of the light from the intersection point
            Vec3 light_direction = (light_position - intersection_point).normalize();

            // Calculate the view direction (direction from intersection point to the camera)
            Vec3 view_direction = -ray.direction.normalize();

            // Material properties for the cube
            Vec3 cube_specular_color(0.5, 0.5, 0.5);  // Specular color of the cube
            Vec3 cube_diffuse_color(0.0, 0.0, 0.8);   // Diffuse color of the cube
            Vec3 ambient_color(0.8, 0.01, 0.01);      // Ambient color of the cube

            float cube_shininess = 32.0;              // Shininess factor for specular highlights

            // Apply Phong shading to calculate the pixel color
            return phong_shading(normal, light_direction, view_direction, ambient_color, cube_diffuse_color, cube_specular_color, cube_shininess);
        }
    }

    // No intersection, return background color (black)
    return Vec3(0.0, 0.0, 0.0);
}



// Save Image to PPM in text format (P3)
void save_image(const std::string& filename, Vec3* framebuffer, size_t width, size_t height) {
  std::ofstream file(filename, std::ios::out);
  file << "P3\n" << width << " " << height << "\n255\n";

  for (size_t i = 0; i < width * height; ++i) {
    int ir = static_cast<int>(255.99 * framebuffer[i].x);
    if (ir > 255) ir = 255;
    
    int ig = static_cast<int>(255.99 * framebuffer[i].y);
    if (ig > 255) ig = 255;
    
    int ib = static_cast<int>(255.99 * framebuffer[i].z);
    if (ib > 255) ib = 255;

    file << ir << " " << ig << " " << ib << "\n";
  }

  file.close();
}


// Animation function to move only the cube
void animate_cube(float animation_time, Vec3& cube_position, const Vec3& camera_origin, float min_y_boundary, float max_y_boundary) {
    // Move the cube only if it's within the specified range
    float cube_amplitude = 0.75f; // Adjust the amplitude of the Y-axis movement
    float cube_frequency = 0.75f * M_PI; // Adjust the frequency of the Y-axis movement
    
    // Calculate the new y-coordinate
    float new_y = cube_amplitude * sin(cube_frequency * animation_time) + camera_origin.y;
    
    // Check if the new y-coordinate is outside the specified range
    if (new_y < min_y_boundary) {
      new_y = max_y_boundary;
    }else if (new_y > max_y_boundary) {
      new_y = min_y_boundary;
    } 
        
    cube_position.y = new_y;
  
}


// Main animation function to move the light source
void animate_scene(float animation_time, Vec3& light_position, const Vec3& camera_origin) {
    // Light source animation: Move the light in a circular path
    float light_radius = 2.0f;
    float light_omega = 2.0f * M_PI; // Angular frequency (one complete rotation in 1 second)

    light_position.x = light_radius * cos(light_omega * animation_time) + camera_origin.x;
    light_position.y = camera_origin.y; // Keep it at the same height as the camera
    light_position.z = light_radius * sin(light_omega * animation_time) + camera_origin.z;
}


//Renders the scene by tracing rays through each pixel of the image.
void render(Vec3* framebuffer, size_t width, size_t height, size_t num_samples, float animation_time,
            Vec3 cube_position_local, float min_y_boundary, float max_y_boundary) {
    // Iterate over each pixel in the image
    for (size_t j = 0; j < height; ++j) {
        for (size_t i = 0; i < width; ++i) {
            size_t index = j * width + i;

            Vec3 pixel_color(0, 0, 0); // Initialize pixel color

            // Set the seed for the random number generator
            srand(1984 + index);

            // Iterate over each sample within the pixel
            for (size_t s = 0; s < num_samples; ++s) {
                // Calculate random offsets for antialiasing
                float u = (i + static_cast<float>(rand()) / RAND_MAX) / width;
                float v = (j + static_cast<float>(rand()) / RAND_MAX) / height;

                Camera camera; // Create a new camera for each sample (not efficient)

                // Generate a ray for the current sample
                Ray ray = camera.get_ray(u, v);

                Vec3 light_position;

                // Animate only the cube independently
                animate_cube(animation_time, cube_position_local, camera.origin, min_y_boundary, max_y_boundary);

                // Animate the light source and the rest of the scene
                animate_scene(animation_time, light_position, camera.origin);

                // Calculate the color for the current sample and add it to the pixel color
                pixel_color += color(ray, light_position, cube_position_local);
            }

            // Average the color over all samples to reduce noise
            pixel_color /= static_cast<float>(num_samples);

            // Set the pixel color in the framebuffer
            framebuffer[index] = pixel_color;
        }
    }
}