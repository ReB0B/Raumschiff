#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cmath> // For sin and cos functions

// GLM for matrix operations
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// Vertex Shader Source for the model
const char* vertexShaderSource = R"glsl(
    #version 330 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aNormal;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    out vec3 FragPos;  
    out vec3 Normal;  

    void main() {
        FragPos = vec3(model * vec4(aPos, 1.0));  
        Normal = mat3(transpose(inverse(model))) * aNormal;  

        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
)glsl";

// Fragment Shader Source for the model
const char* fragmentShaderSource = R"glsl(
    #version 330 core
    out vec4 FragColor;

    in vec3 FragPos;  
    in vec3 Normal;  

    // Light and material properties
    uniform vec3 lightPos; 
    uniform vec3 viewPos; 
    uniform vec3 lightColor;
    uniform vec3 objectColor;

    void main() {
        // Ambient
        float ambientStrength = 0.1;
        vec3 ambient = ambientStrength * lightColor;
          
        // Diffuse 
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);  
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;
        
        // Specular
        float specularStrength = 0.5;
        vec3 viewDir = normalize(viewPos - FragPos);  
        vec3 reflectDir = reflect(-lightDir, norm);  
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * lightColor;  
            
        vec3 result = (ambient + diffuse + specular) * objectColor;
        FragColor = vec4(result, 1.0);
    }
)glsl";

// Vertex Shader Source for the axes
const char* axesVertexShaderSource = R"glsl(
    #version 330 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aColor;

    uniform mat4 view;
    uniform mat4 projection;

    out vec3 vertexColor;

    void main() {
        vertexColor = aColor;
        gl_Position = projection * view * vec4(aPos, 1.0);
    }
)glsl";

// Fragment Shader Source for the axes
const char* axesFragmentShaderSource = R"glsl(
    #version 330 core
    in vec3 vertexColor;
    out vec4 FragColor;

    void main() {
        FragColor = vec4(vertexColor, 1.0);
    }
)glsl";

// Struct to represent an asteroid
struct Asteroid {
    glm::vec3 position;
    glm::vec3 velocity;
};

// Global variables for rotation and movement
glm::vec3 modelPosition = glm::vec3(0.0f, 0.0f, 0.0f);
float rotationY = 0.0f; // Yaw rotation
const float rotationSpeed = 0.01f;
const float movementSpeed = 0.05f;

// Global variables for asteroids
std::vector<Asteroid> asteroids;  // Vector to hold asteroid instances
double startTime;
double nextSpawnTime = 0.0;
int totalSpawns = 0;
const int spawnInterval = 3;     // Seconds
const int totalDuration = 10;    // Seconds
const int maxSpawns = totalDuration / spawnInterval;

// Function prototypes
void processInput(GLFWwindow* window);
void checkGLError(const std::string& errorMessage);

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Set up OpenGL version and profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Needed for macOS
#endif

    // Create window
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "3D Model Loader with Asteroids", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Set current context
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = GL_TRUE; // Needed for core profile
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    checkGLError("GLEW initialization error");

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);

    // Build and compile shaders for the model
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    checkGLError("Vertex shader compilation error");

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    checkGLError("Fragment shader compilation error");

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    checkGLError("Shader program linking error");

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Build and compile shaders for the axes
    unsigned int axesVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(axesVertexShader, 1, &axesVertexShaderSource, NULL);
    glCompileShader(axesVertexShader);
    checkGLError("Axes vertex shader compilation error");

    unsigned int axesFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(axesFragmentShader, 1, &axesFragmentShaderSource, NULL);
    glCompileShader(axesFragmentShader);
    checkGLError("Axes fragment shader compilation error");

    unsigned int axesShaderProgram = glCreateProgram();
    glAttachShader(axesShaderProgram, axesVertexShader);
    glAttachShader(axesShaderProgram, axesFragmentShader);
    glLinkProgram(axesShaderProgram);
    checkGLError("Axes shader program linking error");

    glDeleteShader(axesVertexShader);
    glDeleteShader(axesFragmentShader);

    // Load the spaceship model
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    std::string inputfile = "./BlenderObjects/Spaceship2.obj"; // Replace with your .obj file path
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str());

    if (!warn.empty()) {
        std::cout << "WARN: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << "ERR: " << err << std::endl;
    }

    if (!ret) {
        std::cerr << "Failed to load .obj file!" << std::endl;
        return -1;
    }

    // Prepare vertex data for the spaceship
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];

            // Process per-face
            for (size_t v = 0; v < fv; v++) {
                // Access vertex data
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];

                tinyobj::real_t nx = 0;
                tinyobj::real_t ny = 0;
                tinyobj::real_t nz = 0;
                if (idx.normal_index >= 0) {
                    nx = attrib.normals[3 * idx.normal_index + 0];
                    ny = attrib.normals[3 * idx.normal_index + 1];
                    nz = attrib.normals[3 * idx.normal_index + 2];
                }

                // Append vertex data
                vertices.push_back(vx);
                vertices.push_back(vy);
                vertices.push_back(vz);
                vertices.push_back(nx);
                vertices.push_back(ny);
                vertices.push_back(nz);

                indices.push_back(indices.size());
            }
            index_offset += fv;
        }
    }

    // Setup buffers and arrays for the spaceship
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // Bind buffers for the spaceship
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    checkGLError("Spaceship vertex attribute setup error");

    // Prepare vertex data for the axes
    float axesVertices[] = {
        // Positions          // Colors
        // X-axis (Red)
        0.0f, 0.0f, 0.0f,     1.0f, 0.0f, 0.0f, // Origin
        10.0f, 0.0f, 0.0f,    1.0f, 0.0f, 0.0f, // Positive X direction

        // Y-axis (Green)
        0.0f, 0.0f, 0.0f,     0.0f, 1.0f, 0.0f, // Origin
        0.0f, 10.0f, 0.0f,    0.0f, 1.0f, 0.0f, // Positive Y direction

        // Z-axis (Blue)
        0.0f, 0.0f, 0.0f,     0.0f, 0.0f, 1.0f, // Origin
        0.0f, 0.0f, 10.0f,    0.0f, 0.0f, 1.0f  // Positive Z direction
    };

    // Generate buffers and arrays for the axes
    unsigned int axesVAO, axesVBO;
    glGenVertexArrays(1, &axesVAO);
    glGenBuffers(1, &axesVBO);

    // Bind and set up axes VAO and VBO
    glBindVertexArray(axesVAO);

    glBindBuffer(GL_ARRAY_BUFFER, axesVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(axesVertices), axesVertices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    checkGLError("Axes attribute setup error");

    // Load the asteroid model
    tinyobj::attrib_t asteroidAttrib;
    std::vector<tinyobj::shape_t> asteroidShapes;
    std::vector<tinyobj::material_t> asteroidMaterials;
    std::string asteroidWarn, asteroidErr;

    std::string asteroidInputfile = "./BlenderObjects/monkey.obj"; // Replace with your asteroid .obj file path
    bool asteroidRet = tinyobj::LoadObj(&asteroidAttrib, &asteroidShapes, &asteroidMaterials, &asteroidWarn, &asteroidErr, asteroidInputfile.c_str());

    if (!asteroidWarn.empty()) {
        std::cout << "WARN: " << asteroidWarn << std::endl;
    }

    if (!asteroidErr.empty()) {
        std::cerr << "ERR: " << asteroidErr << std::endl;
    }

    if (!asteroidRet) {
        std::cerr << "Failed to load asteroid .obj file!" << std::endl;
        return -1;
    }

    // Prepare vertex data for the asteroid
    std::vector<float> asteroidVertices;
    std::vector<unsigned int> asteroidIndices;
    for (size_t s = 0; s < asteroidShapes.size(); s++) {
        size_t index_offset = 0;
        for (size_t f = 0; f < asteroidShapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = asteroidShapes[s].mesh.num_face_vertices[f];

            // Process per-face
            for (size_t v = 0; v < fv; v++) {
                // Access vertex data
                tinyobj::index_t idx = asteroidShapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = asteroidAttrib.vertices[3 * idx.vertex_index + 0];
                tinyobj::real_t vy = asteroidAttrib.vertices[3 * idx.vertex_index + 1];
                tinyobj::real_t vz = asteroidAttrib.vertices[3 * idx.vertex_index + 2];

                tinyobj::real_t nx = 0;
                tinyobj::real_t ny = 0;
                tinyobj::real_t nz = 0;
                if (idx.normal_index >= 0) {
                    nx = asteroidAttrib.normals[3 * idx.normal_index + 0];
                    ny = asteroidAttrib.normals[3 * idx.normal_index + 1];
                    nz = asteroidAttrib.normals[3 * idx.normal_index + 2];
                }

                // Append vertex data
                asteroidVertices.push_back(vx);
                asteroidVertices.push_back(vy);
                asteroidVertices.push_back(vz);
                asteroidVertices.push_back(nx);
                asteroidVertices.push_back(ny);
                asteroidVertices.push_back(nz);

                asteroidIndices.push_back(asteroidIndices.size());
            }
            index_offset += fv;
        }
    }

    // Setup buffers and arrays for the asteroid
    unsigned int asteroidVBO, asteroidVAO, asteroidEBO;
    glGenVertexArrays(1, &asteroidVAO);
    glGenBuffers(1, &asteroidVBO);
    glGenBuffers(1, &asteroidEBO);

    // Bind buffers for the asteroid
    glBindVertexArray(asteroidVAO);

    glBindBuffer(GL_ARRAY_BUFFER, asteroidVBO);
    glBufferData(GL_ARRAY_BUFFER, asteroidVertices.size() * sizeof(float), asteroidVertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, asteroidEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, asteroidIndices.size() * sizeof(unsigned int), asteroidIndices.data(), GL_STATIC_DRAW);

    // Vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    checkGLError("Asteroid vertex attribute setup error");

    // Get uniform locations for the model shader
    unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
    unsigned int viewLoc  = glGetUniformLocation(shaderProgram, "view");
    unsigned int projLoc  = glGetUniformLocation(shaderProgram, "projection");

    // Timing variables
    startTime = glfwGetTime();
    double lastFrameTime = startTime;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Calculate deltaTime
        double currentFrameTime = glfwGetTime();
        double deltaTime = currentFrameTime - lastFrameTime;
        lastFrameTime = currentFrameTime;

        // Input
        processInput(window);

        // Automatically move the spaceship along positive x-axis
        modelPosition.x += movementSpeed;

        // Render
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Transformations for the spaceship
        glm::mat4 model = glm::mat4(1.0f);

        // Rotate to make Z-axis point up
        model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)); // Rotate around X-axis

        // Apply translation based on modelPosition
        model = glm::translate(model, modelPosition);

        // Camera settings
        glm::vec3 cameraOffset = glm::vec3(-30.0f, 0.0f, 15.0f); // Adjust offsets as needed
        glm::vec3 target = glm::vec3(modelPosition.x, 0.0f, 0.0f); // static y & z axis
        glm::vec3 cameraPos = cameraOffset + target;
        glm::vec3 up = glm::vec3(0.0f, 0.0f, 1.0f);
        glm::mat4 view = glm::lookAt(cameraPos, target, up);

        // Projection
        glm::mat4 projection = glm::perspective(glm::radians(45.0f),
                                                (float)SCR_WIDTH / (float)SCR_HEIGHT,
                                                0.1f, 100.0f);

        // Render the axes
        glUseProgram(axesShaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(axesShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(axesShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glBindVertexArray(axesVAO);

        // Optionally set line width
        glLineWidth(2.0f);

        // Draw the axes
        glDrawArrays(GL_LINES, 0, 6);

        // Spawn asteroids every 3 seconds for 10 seconds
        double elapsedTime = currentFrameTime - startTime;
        if (elapsedTime >= nextSpawnTime && totalSpawns < maxSpawns) {
            // Spawn new pair of asteroids ahead of the spaceship
            float xPos = modelPosition.x + 50.0f + totalSpawns * 10.0f; // Spaced along x-axis ahead of spaceship
            float yPos = modelPosition.y;
            float zOffset = 5.0f; // Offset from spaceship along z-axis

            // Create left asteroid
            Asteroid leftAsteroid;
            leftAsteroid.position = glm::vec3(xPos, yPos, modelPosition.z - zOffset);
            leftAsteroid.velocity = glm::vec3(-movementSpeed, 0.0f, 0.0f); // Moving towards negative x

            // Create right asteroid
            Asteroid rightAsteroid;
            rightAsteroid.position = glm::vec3(xPos, yPos, modelPosition.z + zOffset);
            rightAsteroid.velocity = glm::vec3(-movementSpeed, 0.0f, 0.0f); // Moving towards negative x

            asteroids.push_back(leftAsteroid);
            asteroids.push_back(rightAsteroid);

            // Update nextSpawnTime and totalSpawns
            nextSpawnTime += spawnInterval;
            totalSpawns++;
        }

        // Update asteroid positions
        for (auto& asteroid : asteroids) {
            asteroid.position += asteroid.velocity;
        }

        // Render the spaceship
        glUseProgram(shaderProgram);

        // Set uniforms for the model shader
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(viewLoc,  1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc,  1, GL_FALSE, glm::value_ptr(projection));

        // Update viewPos uniform
        glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(cameraPos));

        // Light and material properties
        glUniform3f(glGetUniformLocation(shaderProgram, "lightPos"), 50.0f, 50.0f, 50.0f);
        glUniform3f(glGetUniformLocation(shaderProgram, "lightColor"), 1.0f, 1.0f, 1.0f);
        glUniform3f(glGetUniformLocation(shaderProgram, "objectColor"), 0.6f, 0.6f, 0.6f);

        // Render the spaceship
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

        // Render the asteroids
        for (auto& asteroid : asteroids) {
            glm::mat4 asteroidModel = glm::mat4(1.0f);
            asteroidModel = glm::translate(asteroidModel, asteroid.position);

            // Rotate the asteroid by 90 degrees around the desired axis
            asteroidModel = glm::rotate(asteroidModel, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

            // Set the model matrix uniform for the asteroid
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(asteroidModel));

            // Render the asteroid
            glBindVertexArray(asteroidVAO);
            glDrawElements(GL_TRIANGLES, asteroidIndices.size(), GL_UNSIGNED_INT, 0);
        }

        // Swap buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up resources
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    glDeleteVertexArrays(1, &axesVAO);
    glDeleteBuffers(1, &axesVBO);

    glDeleteVertexArrays(1, &asteroidVAO);
    glDeleteBuffers(1, &asteroidVBO);
    glDeleteBuffers(1, &asteroidEBO);

    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow* window) {
    // Close window
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // Constrain movement along z-axis between -15.0f and +15.0f
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        if (modelPosition.z >= 7.5f){
            // setting left boundary to 15.0f
            return;
        }
        modelPosition.z += movementSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        if (modelPosition.z <= -7.5f){
            // setting right boundary to -15.0f
            return;
        }
        modelPosition.z -= movementSpeed;
    }
}

// Function to check for OpenGL errors
void checkGLError(const std::string& errorMessage) {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << errorMessage << ": OpenGL error: " << err << std::endl;
    }
}
