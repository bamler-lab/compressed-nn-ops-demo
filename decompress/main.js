function run() {
    const gl = getGl("canvas")
    let vertexShader = compileShader(gl, "vertex-shader");
    let fragmentShader = compileShader(gl, "fragment-shader");
    let program = createProgram(gl, vertexShader, fragmentShader);

    gl.useProgram(program);

    // Write the positions of vertices to a vertex shader
    var n = initVertexBuffers(gl, program);
    if (n < 0) {
        console.log('Failed to set the positions of the vertices');
        return;
    }

    // Clear canvas
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Draw
    gl.drawArrays(gl.TRIANGLES, 0, n);
}

function initVertexBuffers(gl, program) {
    // Vertices
    var dim = 2;
    var vertices = new Float32Array([
        -1., 1., 1., 1., 1., -1., // Triangle 1
        -1., 1., 1., -1., -1., -1. // Triangle 2 
    ]);

    // Fragment color
    var rgba = [0.0, 1, 0.0, 1.0];

    // Create a buffer object
    var vertexBuffer = gl.createBuffer();
    if (!vertexBuffer) {
        console.log('Failed to create the buffer object');
        return -1;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    // Assign the vertices in buffer object to aPosition variable
    var aPosition = gl.getAttribLocation(program, 'aPosition');
    if (aPosition < 0) {
        console.log('Failed to get the storage location of aPosition');
        return -1;
    }
    gl.vertexAttribPointer(aPosition, dim, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(aPosition);

    // // Assign the color to u_FragColor variable
    // var u_FragColor = gl.getUniformLocation(program, 'u_FragColor');
    // if (u_FragColor < 0) {
    //     console.log('Failed to get the storage location of u_FragColor');
    //     return -1;
    // }
    // gl.uniform4fv(u_FragColor, rgba);

    // Return number of vertices
    return vertices.length / dim;
}

/**
 * Obtains the GL context.
 *
 * @param {string} canvasId The id of the canvas tag.
 * @return {!WebGLRenderingContext} The WebGL rendering context.
 */
function getGl(canvasId) {
    return document.getElementById(canvasId).getContext('webgl');
}

/**
 * Creates and compiles a shader.
 *
 * @param {!WebGLRenderingContext} gl The WebGL Context.
 * @param {string} scriptId The id of the script tag.
 * @return {!WebGLShader} The shader.
 */
function compileShader(gl, scriptId) {
    let shaderScript = document.getElementById(scriptId);
    if (!shaderScript) {
        throw "unknown script element: " + scriptId;
    }

    let shaderType;
    switch (shaderScript.type) {
        case "x-shader/x-vertex":
            shaderType = gl.VERTEX_SHADER;
            break;
        case "x-shader/x-fragment":
            shaderType = gl.FRAGMENT_SHADER;
            break;
        default:
            throw "shader type not set";
    }

    let shader = gl.createShader(shaderType);
    gl.shaderSource(shader, shaderScript.text);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw "could not compile shader: " + gl.getShaderInfoLog(shader);
    }

    return shader;
}

/**
 * Creates a program from 2 shaders.
 *
 * @param {!WebGLRenderingContext} gl The WebGL context.
 * @param {!WebGLShader} vertexShader A vertex shader.
 * @param {!WebGLShader} fragmentShader A fragment shader.
 * @return {!WebGLProgram} A program.
 */
function createProgram(gl, vertexShader, fragmentShader) {
    let program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw "program failed to link: " + gl.getProgramInfoLog(program);
    }

    return program;
};
