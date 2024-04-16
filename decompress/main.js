async function run() {
    const canvas = document.getElementById("canvas");
    const gl = getGl(canvas);
    const inputDataTexture = await createInputDataTexture(canvas, gl);
    const lookupTableTexture = await createLookupTable(canvas, gl, inputDataTexture);
    await decodeMatrix(canvas, gl, inputDataTexture, lookupTableTexture);

    // const vertexShader = compileShader(gl, "vertex-shader");
    // const fragmentShader = compileShader(gl, "fragment-shader");
    // const program = createProgram(gl, vertexShader, fragmentShader);
    // gl.useProgram(program);
    // const num_vertices = initVertexBuffers(gl, program);
    // gl.drawArrays(gl.TRIANGLES, 0, num_vertices);
}

async function createInputDataTexture(canvas, gl) {
    const response = fetch("mock-matrix.bin");
    const buf = await (await response).arrayBuffer();
    const data = new Uint16Array(buf);

    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // Upload the texture to the GPU:
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16UI, 512, 655, 0, gl.RED_INTEGER, gl.UNSIGNED_SHORT, data);

    // can't filter integer textures
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

    return texture;
}

async function createLookupTable(canvas, gl, inputDataTexture) {
    // BEGIN SETUP (has to be done only once)
    const vertexShader = compileShader(gl, "vertex-shader");
    const fragmentShader = compileShader(gl, "create-lookup-shader");
    const program = createProgram(gl, vertexShader, fragmentShader);

    const compressedDataLocation = gl.getUniformLocation(program, "compressedData");
    const lookupTableLocation = gl.getUniformLocation(program, "lookupTable");
    // END SETUP

    canvas.height = 8;
    await new Promise(setTimeout)
    gl.viewport(0, 0, 512, 8);

    gl.useProgram(program);
    // set which texture units to render with.
    gl.uniform1i(compressedDataLocation, 0);  // texture unit 0
    gl.uniform1i(lookupTableLocation, 1);  // texture unit 1

    // Set each texture unit to use a particular texture.
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, inputDataTexture);
    gl.activeTexture(gl.TEXTURE1);
    const targetTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, targetTexture);

    const level = 0;
    const internalFormat = gl.R8UI;
    const border = 0;
    const format = gl.RED_INTEGER;
    const type = gl.UNSIGNED_BYTE;
    // Upload the (still empty) texture to the GPU:
    gl.texImage2D(gl.TEXTURE_2D, level, internalFormat,
        512, 8, border,
        format, type, null);

    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);

    gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, targetTexture, level);


    let num_vertices = initVertexBuffers(gl, program);
    gl.drawArrays(gl.TRIANGLES, 0, num_vertices);

    // let rawBuffer = new ArrayBuffer(512 * 8 * 4 * 4);
    // let output = new Uint32Array(rawBuffer);
    // gl.readPixels(0, 0, 512, 8, gl.RGBA_INTEGER, gl.UNSIGNED_INT, output);
    // console.log(output.filter((_val, i) => i % 4 == 0));

    return targetTexture;
};

async function decodeMatrix(canvas, gl, inputDataTexture, lookupTableTexture) {
    // BEGIN SETUP (has to be done only once)
    const vertexShader = compileShader(gl, "vertex-shader");
    const fragmentShader = compileShader(gl, "decode-shader");
    const program = createProgram(gl, vertexShader, fragmentShader);

    const compressedDataLocation = gl.getUniformLocation(program, "compressedData");
    const lookupTableLocation = gl.getUniformLocation(program, "lookupTable");
    const saltLocation = gl.getUniformLocation(program, "salt");
    // END SETUP

    canvas.height = 2;
    await new Promise(setTimeout)
    gl.viewport(0, 0, 512, 2);

    gl.useProgram(program);
    // set which texture units to render with.
    gl.uniform1i(compressedDataLocation, 0);  // texture unit 0
    gl.uniform1i(lookupTableLocation, 1);  // texture unit 1

    // Upload existing textures (TODO: is this necessary again?)
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, inputDataTexture);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, lookupTableTexture);

    // Bind new texture.
    gl.activeTexture(gl.TEXTURE2);
    const targetTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, targetTexture);

    const level = 0;
    const internalFormat = gl.R32UI;
    const border = 0;
    const format = gl.RED_INTEGER;
    const type = gl.UNSIGNED_INT;
    // Upload the (still empty) texture to the GPU:
    gl.texImage2D(gl.TEXTURE_2D, level, internalFormat,
        512, 2, border, format, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, targetTexture, level);

    let num_vertices = initVertexBuffers(gl, program);

    await new Promise((resolve) => setTimeout(resolve, 5000));
    console.log("start");
    await new Promise((resolve) => setTimeout(resolve, 500));

    const start = new Date();
    for (let i = 0; i != 10000; ++i) {
        gl.uniform1ui(saltLocation, i);
        gl.drawArrays(gl.TRIANGLES, 0, num_vertices);
    }
    const end = new Date();
    console.log("duration: " + (end - start))

    let rawBuffer = new ArrayBuffer(1024 * 4 * 4);
    let output = new Uint32Array(rawBuffer);
    gl.readPixels(0, 0, 512, 2, gl.RGBA_INTEGER, gl.UNSIGNED_INT, output);
    console.log(output.filter((_val, i) => i % 4 == 0));
    document.write(output.filter((_val, i) => i % 4 == 0));

    return targetTexture;
};

function initVertexBuffers(gl, program) {
    var vertices = new Float32Array([
        -1., 1., 1., 1., 1., -1., // Triangle 1
        -1., 1., 1., -1., -1., -1. // Triangle 2 
    ]);

    var vertexBuffer = gl.createBuffer();
    if (!vertexBuffer) {
        throw 'Failed to create the buffer object';
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    var aPosition = gl.getAttribLocation(program, 'aPosition');
    if (aPosition < 0) {
        throw 'Failed to get the storage location of aPosition';
    }
    gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(aPosition);

    return vertices.length / 2; // number of vertices
}

/**
 * Obtains the GL context.
 *
 * @param {canvas} HTMLCanvasElement The canvas element.
 * @return {!WebGL2RenderingContext} The WebGL rendering context.
 */
function getGl(canvas) {
    return canvas.getContext('webgl2');
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
