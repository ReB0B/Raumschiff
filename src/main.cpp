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

// Global variables for rotation and movement
glm::vec3 modelPosition = glm::vec3(0.0f, 0.0f, 0.0f);
float rotationY = 0.0f; // Yaw rotation
const float rotationSpeed = 0.01f;
const float movementSpeed = 0.05f;

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
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "3D Model Loader with Axes Visualization", NULL, NULL);
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

    // Load .obj file
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

    // Prepare vertex data for the model
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

    // Setup buffers and arrays for the model
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // Bind buffers for the model
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

    checkGLError("Vertex attribute setup error");

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

    // Get uniform locations for the model shader
    unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
    unsigned int viewLoc  = glGetUniformLocation(shaderProgram, "view");
    unsigned int projLoc  = glGetUniformLocation(shaderProgram, "projection");

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Input
        processInput(window);

        // Render
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Transformations for the model
        glm::mat4 model = glm::mat4(1.0f);

        // Rotate to make Z-axis point up
        model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)); // Rotate around X-axis

        // Apply translation based on modelPosition
        model = glm::translate(model, modelPosition);

        // Apply rotation around new Y-axis (previously Z-axis)
        model = glm::rotate(model, rotationY, glm::vec3(0.0f, 0.0f, 1.0f));

        // Camera settings
        glm::vec3 cameraOffset = glm::vec3(30.0f, 0.0f, 15.0f); // Adjust offsets as needed
        glm::vec3 cameraPos = modelPosition + cameraOffset;
        glm::vec3 target = modelPosition;
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

        // Render the model
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

        // Render the model
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

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

    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow* window) {
    // Close window
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // Calculate forward vector based on current rotation
    glm::vec3 forward = glm::vec3(-cos(rotationY), -sin(rotationY), 0.0f);
    glm::vec3 right = glm::vec3(-forward.y, forward.x, 0.0f); // Right vector is perpendicular to forward

    // Forward/backward movement
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        modelPosition += forward * movementSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        modelPosition -= forward * movementSpeed;
    }

    // Left/right strafing movement
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        modelPosition -= right * movementSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        modelPosition += right * movementSpeed;
    }
}




// Function to check for OpenGL errors
void checkGLError(const std::string& errorMessage) {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << errorMessage << ": OpenGL error: " << err << std::endl;
    }
}