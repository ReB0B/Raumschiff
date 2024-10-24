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
// GLM random functionality
#include <glm/gtc/random.hpp>

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

// Vertex Shader Source for the post-processing shader
const char* postProcessingVertexShaderSource = R"glsl(
    #version 330 core
    layout(location = 0) in vec2 aPos;
    layout(location = 1) in vec2 aTexCoords;

    out vec2 TexCoords;

    void main() {
        TexCoords = aTexCoords;
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
)glsl";

// Fragment Shader Source for the post-processing shader with chromatic aberration and vignette
const char* postProcessingFragmentShaderSource = R"glsl(
    #version 330 core
    out vec4 FragColor;

    in vec2 TexCoords;

    uniform sampler2D screenTexture;
    uniform bool isSpeedBoostActive;

    void main() {
        vec2 uv = TexCoords;
        vec2 center = vec2(0.5, 0.5);

        // Calculate distance from center (normalized to range [0,1])
        float dist = distance(uv, center) / sqrt(0.5);

        // Vignette effect using smoothstep for smooth transition
        float vignette = smoothstep(1.0, 0.5, dist);

        vec3 color;
        if (isSpeedBoostActive) {
            // Chromatic aberration effect
            vec2 offset = (uv - center) * 0.02;
            float r = texture(screenTexture, uv + offset).r;
            float g = texture(screenTexture, uv).g;
            float b = texture(screenTexture, uv - offset).b;
            color = vec3(r, g, b);
        } else {
            color = texture(screenTexture, uv).rgb;
        }

        // Apply vignette effect
        color *= vignette;

        FragColor = vec4(color, 1.0);
    }
)glsl";

// Vertex Shader Source for the stars
const char* starVertexShaderSource = R"glsl(
    #version 330 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in float aPhase;

    out float vPhase;

    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        vPhase = aPhase;
        gl_Position = projection * view * vec4(aPos, 1.0);
        gl_PointSize = 2.0; // Adjust size as needed
    }
)glsl";

// Fragment Shader Source for the stars
const char* starFragmentShaderSource = R"glsl(
    #version 330 core
    in float vPhase;
    out vec4 FragColor;

    uniform float time;

    void main() {
        // Twinkling effect
        float brightness = 0.5 + 0.5 * sin(time * 5.0 + vPhase);
        FragColor = vec4(vec3(brightness), 1.0);
    }
)glsl";

// Struct to represent an asteroid
struct Asteroid {
    glm::vec3 position;
    glm::vec3 velocity;
};

// Struct to represent a star
struct Star {
    glm::vec3 position;
    float phase;
};

// Global variables for rotation and movement
glm::vec3 modelPosition = glm::vec3(0.0f, 0.0f, 0.0f);
float rotationY = 0.0f; // Yaw rotation
float movementSpeed = 0.05f;
const float baseSpeed = 0.05f;
const float speedBoostMultiplier = 2.0f;

// Global variables for speed boost
bool isSpeedBoostActive = false;

// Global variables for asteroids
std::vector<Asteroid> asteroids;  // Vector to hold asteroid instances
double startTime;
double nextSpawnTime = 0.0;
const int spawnInterval = 3;     // Seconds

// Variables for smooth camera transition
glm::vec3 currentCameraOffset = glm::vec3(-30.0f, 0.0f, 15.0f);
glm::vec3 normalCameraOffset = glm::vec3(-30.0f, 0.0f, 15.0f);
glm::vec3 speedBoostCameraOffset = glm::vec3(-50.0f, 0.0f, 25.0f);
float cameraTransitionSpeed = 5.0f; // Adjust for faster or slower transition

// Global variables for stars
std::vector<Star> stars;
unsigned int starsVAO, starsVBO;
glm::vec3 lastStarGenPosition = glm::vec3(0.0f);

// Function prototypes
void processInput(GLFWwindow* window);
void checkGLError(const std::string& errorMessage);
void generateStars(int numStars, float radius, glm::vec3 centerPosition);
void setupStars();

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
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "3D Model Loader with Stars, Vignette, and Chromatic Aberration", NULL, NULL);
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

    // Enable program point size
    glEnable(GL_PROGRAM_POINT_SIZE);

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

    // Build and compile shaders for post-processing
    unsigned int ppVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(ppVertexShader, 1, &postProcessingVertexShaderSource, NULL);
    glCompileShader(ppVertexShader);
    checkGLError("Post-processing vertex shader compilation error");

    unsigned int ppFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(ppFragmentShader, 1, &postProcessingFragmentShaderSource, NULL);
    glCompileShader(ppFragmentShader);
    checkGLError("Post-processing fragment shader compilation error");

    unsigned int ppShaderProgram = glCreateProgram();
    glAttachShader(ppShaderProgram, ppVertexShader);
    glAttachShader(ppShaderProgram, ppFragmentShader);
    glLinkProgram(ppShaderProgram);
    checkGLError("Post-processing shader program linking error");

    glDeleteShader(ppVertexShader);
    glDeleteShader(ppFragmentShader);

    // Build and compile shaders for the stars
    unsigned int starVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(starVertexShader, 1, &starVertexShaderSource, NULL);
    glCompileShader(starVertexShader);
    checkGLError("Star vertex shader compilation error");

    unsigned int starFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(starFragmentShader, 1, &starFragmentShaderSource, NULL);
    glCompileShader(starFragmentShader);
    checkGLError("Star fragment shader compilation error");

    unsigned int starShaderProgram = glCreateProgram();
    glAttachShader(starShaderProgram, starVertexShader);
    glAttachShader(starShaderProgram, starFragmentShader);
    glLinkProgram(starShaderProgram);
    checkGLError("Star shader program linking error");

    glDeleteShader(starVertexShader);
    glDeleteShader(starFragmentShader);

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

    // Setup framebuffer for post-processing
    unsigned int framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    // Create a texture to render to
    unsigned int textureColorbuffer;
    glGenTextures(1, &textureColorbuffer);
    glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // Attach texture to framebuffer
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);

    // Create a renderbuffer object for depth and stencil attachment
    unsigned int rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    // Use a single renderbuffer object for both depth and stencil attachments
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SCR_WIDTH, SCR_HEIGHT);
    // Attach renderbuffer object to framebuffer
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

    // Check if framebuffer is complete
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "Framebuffer not complete!" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Screen quad vertices for post-processing
    float quadVertices[] = {
        // Positions   // TexCoords
        -1.0f,  1.0f,  0.0f, 1.0f, // Top-left
        -1.0f, -1.0f,  0.0f, 0.0f, // Bottom-left
         1.0f, -1.0f,  1.0f, 0.0f, // Bottom-right

        -1.0f,  1.0f,  0.0f, 1.0f, // Top-left
         1.0f, -1.0f,  1.0f, 0.0f, // Bottom-right
         1.0f,  1.0f,  1.0f, 1.0f  // Top-right
    };

    // Setup screen VAO
    unsigned int quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);

    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // TexCoord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    checkGLError("Screen quad setup error");

    // Generate stars
    int numStars = 1000; // Number of stars
    float starFieldRadius = 500.0f; // Increased radius
    glm::vec3 cameraPos = currentCameraOffset + modelPosition; // Initial camera position
    generateStars(numStars, starFieldRadius, cameraPos);
    setupStars();
    lastStarGenPosition = cameraPos;

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

        // Update movement speed based on speed boost
        if (isSpeedBoostActive) {
            movementSpeed = baseSpeed * speedBoostMultiplier;
        } else {
            movementSpeed = baseSpeed;
        }

        // Automatically move the spaceship along positive x-axis
        modelPosition.x += movementSpeed;

        // Smooth camera transition
        glm::vec3 targetCameraOffset = isSpeedBoostActive ? speedBoostCameraOffset : normalCameraOffset;
        currentCameraOffset = glm::mix(currentCameraOffset, targetCameraOffset, cameraTransitionSpeed * (float)deltaTime);

        // Camera settings
        glm::vec3 target = glm::vec3(modelPosition.x, 0.0f, 0.0f); // static y & z axis
        cameraPos = currentCameraOffset + target;
        glm::vec3 up = glm::vec3(0.0f, 0.0f, 1.0f);
        glm::mat4 view = glm::lookAt(cameraPos, target, up);

        // Update star field if camera has moved significantly
        float distanceMoved = glm::length(cameraPos - lastStarGenPosition);
        if (distanceMoved > starFieldRadius / 2) {
            generateStars(numStars, starFieldRadius, cameraPos);
            setupStars();
            lastStarGenPosition = cameraPos;
        }

        // First pass: Render scene to framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glEnable(GL_DEPTH_TEST); // Enable depth testing

        // Clear the framebuffer's content
        glClearColor(0.0f, 0.0f, 0.1f, 1.0f); // Set background to dark blue (night sky)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Transformations for the spaceship
        glm::mat4 model = glm::mat4(1.0f);

        // Rotate to make Z-axis point up
        model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)); // Rotate around X-axis

        // Apply translation based on modelPosition
        model = glm::translate(model, modelPosition);

        // Projection
        glm::mat4 projection = glm::perspective(glm::radians(45.0f),
                                                (float)SCR_WIDTH / (float)SCR_HEIGHT,
                                                0.1f, 1000.0f);

        // Render the stars
        glUseProgram(starShaderProgram);

        // Pass view and projection matrices
        glUniformMatrix4fv(glGetUniformLocation(starShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(starShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        // Pass time uniform
        float timeValue = glfwGetTime();
        glUniform1f(glGetUniformLocation(starShaderProgram, "time"), timeValue);

        // Disable depth testing so stars are always in the background
        glDisable(GL_DEPTH_TEST);

        // Enable blending for smoother star appearance
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glBindVertexArray(starsVAO);
        glDrawArrays(GL_POINTS, 0, stars.size());
        glBindVertexArray(0);

        // Re-enable depth testing
        glEnable(GL_DEPTH_TEST);

        // Disable blending
        glDisable(GL_BLEND);

        // Render the axes
        glUseProgram(axesShaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(axesShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(axesShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glBindVertexArray(axesVAO);

        // Optionally set line width
        glLineWidth(2.0f);

        // Draw the axes
        glDrawArrays(GL_LINES, 0, 6);

        // Spawn asteroids at regular intervals
        double elapsedTime = currentFrameTime - startTime;
        if (elapsedTime >= nextSpawnTime) {
            // Spawn new pair of asteroids ahead of the spaceship
            float xPos = modelPosition.x + 50.0f; // Ahead of spaceship
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

            // Update nextSpawnTime
            nextSpawnTime += spawnInterval;
        }

        // Update asteroid positions
        for (auto& asteroid : asteroids) {
            asteroid.position += asteroid.velocity;
        }

        // Remove asteroids that are far behind the spaceship
        asteroids.erase(std::remove_if(asteroids.begin(), asteroids.end(),
            [&](const Asteroid& asteroid) {
                return asteroid.position.x < modelPosition.x - 50.0f; // Adjust the threshold as needed
            }), asteroids.end());

        // Render the spaceship
        glUseProgram(shaderProgram);

        // Set uniforms for the model shader
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(viewLoc,  1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc,  1, GL_FALSE, glm::value_ptr(projection));

        // Update viewPos uniform
        glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(cameraPos));

        // Light and material properties
        glUniform3f(glGetUniformLocation(shaderProgram, "lightPos"), modelPosition.x + 50.0f, 50.0f, 50.0f);
        glUniform3f(glGetUniformLocation(shaderProgram, "lightColor"), 1.0f, 1.0f, 1.0f);

        // Set the spaceship color to white
        glUniform3f(glGetUniformLocation(shaderProgram, "objectColor"), 1.0f, 1.0f, 1.0f);

        // Render the spaceship
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

        // Render the asteroids
        for (auto& asteroid : asteroids) {
            glm::mat4 asteroidModel = glm::mat4(1.0f);
            asteroidModel = glm::translate(asteroidModel, asteroid.position);

            // Rotate the asteroid by 90 degrees around the X-axis
            asteroidModel = glm::rotate(asteroidModel, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

            // Set the model matrix uniform for the asteroid
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(asteroidModel));

            // Set the asteroid color to light grey
            glUniform3f(glGetUniformLocation(shaderProgram, "objectColor"), 0.8f, 0.8f, 0.8f);

            // Render the asteroid
            glBindVertexArray(asteroidVAO);
            glDrawElements(GL_TRIANGLES, asteroidIndices.size(), GL_UNSIGNED_INT, 0);
        }

        // Second pass: Render to default framebuffer with post-processing
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDisable(GL_DEPTH_TEST); // Disable depth test so screen-space quad isn't discarded

        // Clear the default framebuffer's content
        glClearColor(0.0f, 0.0f, 0.1f, 1.0f); // Set background to dark blue (night sky)
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(ppShaderProgram);
        glBindVertexArray(quadVAO);
        glBindTexture(GL_TEXTURE_2D, textureColorbuffer); // Use the color attachment texture as the texture of the quad

        // Set uniform for speed boost effect
        glUniform1i(glGetUniformLocation(ppShaderProgram, "isSpeedBoostActive"), isSpeedBoostActive);

        // Draw the quad
        glDrawArrays(GL_TRIANGLES, 0, 6);

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

    glDeleteVertexArrays(1, &starsVAO);
    glDeleteBuffers(1, &starsVBO);

    // Delete framebuffer resources
    glDeleteFramebuffers(1, &framebuffer);
    glDeleteTextures(1, &textureColorbuffer);
    glDeleteRenderbuffers(1, &rbo);

    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow* window) {
    // Close window
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // Constrain movement along z-axis between -7.5f and +7.5f
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        if (modelPosition.z <= -7.5f){
            // setting left boundary to -7.5f
            return;
        }
        modelPosition.z -= movementSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        if (modelPosition.z >= 7.5f){
            // setting right boundary to 7.5f
            return;
        }
        modelPosition.z += movementSpeed;
    }

    // Check for speed boost activation (Shift key)
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
        isSpeedBoostActive = true;
    } else {
        isSpeedBoostActive = false;
    }
}

// Function to check for OpenGL errors
void checkGLError(const std::string& errorMessage) {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << errorMessage << ": OpenGL error: " << err << std::endl;
    }
}

// Function to generate star positions
void generateStars(int numStars, float radius, glm::vec3 centerPosition) {
    stars.clear();
    for (int i = 0; i < numStars; ++i) {
        // Random direction
        float theta = glm::linearRand(0.0f, glm::two_pi<float>());
        float phi = glm::linearRand(0.0f, glm::pi<float>());

        // Random radius within a range to create a shell
        float r = glm::linearRand(radius * 0.9f, radius);

        // Spherical to Cartesian conversion
        float x = r * sin(phi) * cos(theta);
        float y = r * sin(phi) * sin(theta);
        float z = r * cos(phi);

        // Random phase for twinkling variation
        float phase = glm::linearRand(0.0f, glm::two_pi<float>());

        // Shift stars to be around the centerPosition
        stars.push_back({glm::vec3(x, y, z) + centerPosition, phase});
    }
}

// Function to setup stars VAO and VBO
void setupStars() {
    struct StarVertex {
        glm::vec3 position;
        float phase;
    };

    // Convert stars to StarVertex vector
    std::vector<StarVertex> starVertices;
    for (const auto& star : stars) {
        starVertices.push_back({star.position, star.phase});
    }

    glGenVertexArrays(1, &starsVAO);
    glGenBuffers(1, &starsVBO);

    glBindVertexArray(starsVAO);
    glBindBuffer(GL_ARRAY_BUFFER, starsVBO);
    glBufferData(GL_ARRAY_BUFFER, starVertices.size() * sizeof(StarVertex), &starVertices[0], GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(StarVertex), (void*)0);
    glEnableVertexAttribArray(0);

    // Phase attribute
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(StarVertex), (void*)offsetof(StarVertex, phase));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}
