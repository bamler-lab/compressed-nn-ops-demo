function run() {
    const gl = getGl("canvas")
    const vertexShader = compileShader(gl, "vertex-shader");
    const fragmentShader = compileShader(gl, "fragment-shader");
    const program = createProgram(gl, vertexShader, fragmentShader);

    gl.useProgram(program);
    const srcTextureLocation = gl.getUniformLocation(program, "srcTexture");
    console.log({ srcTextureLocation });

    const srcTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, srcTexture);
    const level = 0;

    {
        const srcTextureHeight = 256;
        const srcTextureWidth = 256;
        // define size and format of level 0
        const internalFormat = gl.RGBA; // TODO: try R32UI (with format RED_INTEGER	and type UNSIGNED_BYTE)
        const border = 0;
        const format = gl.RGBA;
        const type = gl.UNSIGNED_BYTE;
        const data = new Uint8Array(srcTextureHeight * srcTextureWidth * 4);
        let i = 0;
        for (let y = 0; y != srcTextureHeight; ++y) {
            for (let x = 0; x != srcTextureWidth; ++x) {
                data[i] = 0;
                data[i + 1] = x;
                data[i + 2] = y;
                data[i + 3] = 255;
                i += 4;
            }
        }
        gl.texImage2D(gl.TEXTURE_2D, level, internalFormat,
            srcTextureWidth, srcTextureHeight, border,
            format, type, data);

        // set the filtering so we don't need mips
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); // TODO: maybe better: gl.NEAREST
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }

    gl.uniform1i(srcTextureLocation, 0);

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
 * @return {!WebGL2RenderingContext} The WebGL rendering context.
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
    const shaderScript = document.getElementById(scriptId);
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

    const shader = gl.createShader(shaderType);
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
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw "program failed to link: " + gl.getProgramInfoLog(program);
    }

    return program;
};
