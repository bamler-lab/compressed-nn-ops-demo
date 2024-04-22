async function run() {
    const canvas = document.getElementById("canvas");
    const gl = getGl(canvas);
    const vertexShader = compileShader(gl, "vertex-shader");

    const response = await fetch("100-compressed-matrices.bin");
    const buf = await response.arrayBuffer();
    const data = new Uint16Array(buf);

    const coderHeads = createTexture(gl, gl.RG16UI, 1024, 1024, gl.RG_INTEGER, gl.UNSIGNED_SHORT, data.subarray(0, 1024 * 1024 * 2));
    const compressedData = createTexture(gl, gl.R16UI, 1024, 1024, gl.RED_INTEGER, gl.UNSIGNED_SHORT, data.subarray(1024 * 1024 * 2, 1024 * 1024 * 3));
    const lookupTable = createTexture(gl, gl.R16UI, 4096, 1, gl.RED_INTEGER, gl.BYTE, null);

    const offsetTexture = createOffsetTexture(gl);
    const partpartsumsTexture = createEmptySumsTexture(gl, 1024, 1024);
    const partsumsTexture = createEmptySumsTexture(gl, 128, 1024);
    const rowsumsTexture = createEmptySumsTexture(gl, 32, 1024);
    const sumsumspartTexture = createEmptySumsTexture(gl, 1, 1024);
    const sumsumsTexture = createEmptySumsTexture(gl, 1, 32);

    let { vertexBuffer: vertexBuffer1, vertices: vertices1 } = createVertexBuffer(gl);
    const framebuffer1 = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer1);

    const createPartpartsums = new GlProgram(gl, vertexShader, "partpartsum-shader", ["input_data"]);
    const createPartsums = new GlProgram(gl, vertexShader, "partsum-shader", ["input_data"]);
    const createRowsums = new GlProgram(gl, vertexShader, "rowsum-shader", ["input_data"]);
    const createSumsumspart = new GlProgram(gl, vertexShader, "sumsumpart-shader", ["input_data"]);
    const createSumsums = new GlProgram(gl, vertexShader, "sumsum-shader", ["input_data"]);

    await new Promise(resolve => setTimeout(resolve, 1000));

    let start1 = performance.now();
    for (let i = 0; i != 1000; ++i) {
        createPartpartsums.run(gl, [offsetTexture], partpartsumsTexture, 1024, 1024, vertexBuffer1, vertices1);
        createPartsums.run(gl, [partpartsumsTexture], partsumsTexture, 128, 1024, vertexBuffer1, vertices1);
        createRowsums.run(gl, [partsumsTexture], rowsumsTexture, 32, 1024, vertexBuffer1, vertices1);
        createSumsumspart.run(gl, [rowsumsTexture], sumsumspartTexture, 1, 1024, vertexBuffer1, vertices1);
        createSumsums.run(gl, [sumsumspartTexture], sumsumsTexture, 1, 32, vertexBuffer1, vertices1);
    }

    rawBuffer1 = new ArrayBuffer(32 * 1 * 4 * 4);
    output1 = new Uint32Array(rawBuffer1);
    gl.readPixels(0, 0, 1, 32, gl.RGBA_INTEGER, gl.UNSIGNED_INT, output1);
    console.log(output1.filter((_val, i) => i % 4 == 0));

    await new Promise(resolve => setTimeout(resolve, 0));
    let end1 = performance.now();
    console.log('duration: ' + (end1 - start1))

    await new Promise(resolve => setTimeout(resolve, 1000));

    start1 = performance.now();
    for (let i = 0; i != 1000; ++i) {
        createPartpartsums.run(gl, [offsetTexture], partpartsumsTexture, 1024, 1024, vertexBuffer1, vertices1);
        createPartsums.run(gl, [partpartsumsTexture], partsumsTexture, 128, 1024, vertexBuffer1, vertices1);
        createRowsums.run(gl, [partsumsTexture], rowsumsTexture, 32, 1024, vertexBuffer1, vertices1);
        createSumsumspart.run(gl, [rowsumsTexture], sumsumspartTexture, 1, 1024, vertexBuffer1, vertices1);
        createSumsums.run(gl, [sumsumspartTexture], sumsumsTexture, 1, 32, vertexBuffer1, vertices1);
    }

    rawBuffer1 = new ArrayBuffer(32 * 1 * 4 * 4);
    output1 = new Uint32Array(rawBuffer1);
    gl.readPixels(0, 0, 1, 32, gl.RGBA_INTEGER, gl.UNSIGNED_INT, output1);
    console.log(output1.filter((_val, i) => i % 4 == 0));

    await new Promise(resolve => setTimeout(resolve, 0));
    end1 = performance.now();
    console.log('duration: ' + (end1 - start1))

    return;

    const compressedMatrixTexture = await createInputDataTexture(gl);
    let inputVectorTexture = createVectorTexture(gl);
    let outputVectorTexture = createVectorTexture(gl);
    let { vertexBuffer, vertices } = createVertexBuffer(gl);

    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);

    const createLookupTable = new GlProgram(gl, vertexShader, "create-lookup-shader", ["compressedMatrix"]);
    const compressedMatrixVectorProduct = new GlProgram(
        gl, vertexShader, "compressed-mat-vec-mul-shader", ["compressedMatrix", "lookupTable", "vector"]);

    await new Promise(resolve => setTimeout(resolve, 1000));

    let start = performance.now();
    for (let i = 0; i != 1000; ++i) {
        createLookupTable.run(gl, [compressedMatrixTexture], lookupTableTexture, 4096, 1, vertexBuffer, vertices);
        compressedMatrixVectorProduct.run(
            gl, [compressedMatrixTexture, lookupTableTexture, inputVectorTexture],
            outputVectorTexture, 1024, 1, vertexBuffer, vertices);
        let tmp = inputVectorTexture;
        inputVectorTexture = outputVectorTexture;
        outputVectorTexture = tmp;
    }

    let rawBuffer = new ArrayBuffer(1024 * 1 * 4 * 4);
    let output = new Int32Array(rawBuffer);
    gl.readPixels(0, 0, 1024, 1, gl.RGBA_INTEGER, gl.INT, output);
    console.log(output.filter((_val, i) => i % 4 == 0));

    await new Promise(resolve => setTimeout(resolve, 0));
    let end = performance.now();
    console.log('duration: ' + (end - start))

    await new Promise(resolve => setTimeout(resolve, 1000));

    start = performance.now();
    for (let i = 0; i != 1000; ++i) {
        createLookupTable.run(gl, [compressedMatrixTexture], lookupTableTexture, 4096, 1, vertexBuffer, vertices);
        compressedMatrixVectorProduct.run(
            gl, [compressedMatrixTexture, lookupTableTexture, inputVectorTexture],
            outputVectorTexture, 1024, 1, vertexBuffer, vertices);
        let tmp = inputVectorTexture;
        inputVectorTexture = outputVectorTexture;
        outputVectorTexture = tmp;
    }

    gl.readPixels(0, 0, 1024, 1, gl.RGBA_INTEGER, gl.INT, output);
    console.log(output.filter((_val, i) => i % 4 == 0));

    await new Promise(resolve => setTimeout(resolve, 0));
    end = performance.now();
    console.log('duration: ' + (end - start))
}

function createTexture(gl, internal_format, width, height, format, type, pixels) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // Upload the texture to the GPU:
    gl.texImage2D(gl.TEXTURE_2D, 0, internal_format, width, height, 0, format, type, pixels);

    // can't filter integer textures
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

    return texture;
}


class GlProgram {
    constructor(gl, vertexShader, fragmentShaderId, textureNames) {
        const fragmentShader = compileShader(gl, fragmentShaderId);
        this.program = createProgram(gl, vertexShader, fragmentShader);
        this.textureLocations = textureNames.map(name => gl.getUniformLocation(this.program, name))
        this.widthLocation = gl.getUniformLocation(this.program, "width");
        this.heightLocation = gl.getUniformLocation(this.program, "height");
        this.vertexPositionLocation = gl.getUniformLocation(this.program, "vertex_position");
    }

    run(gl, textures, targetTexture, width, height, vertexBuffer, vertices) {
        gl.viewport(0, 0, width, height);
        gl.useProgram(this.program);

        gl.uniform1i(this.widthLocation, width);
        gl.uniform1i(this.heightLocation, height);
        gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
        gl.vertexAttribPointer(this.vertexPositionLocation, 2, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.vertexPositionLocation);

        const gl_textures = [gl.TEXTURE0, gl.TEXTURE1, gl.TEXTURE2, gl.TEXTURE3, gl.TEXTURE4];
        for (let i = 0; i != textures.length; ++i) {
            gl.uniform1i(this.textureLocations[i], i);
            gl.activeTexture(gl_textures[i]);
            gl.bindTexture(gl.TEXTURE_2D, textures[i]);
        }

        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, targetTexture, 0);

        gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
}

function createOffsetTexture(gl) {
    const data = new Uint8Array(Array.from({ length: 1024 * 1024 }, (_, i) => i % 100));
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // Upload the texture to the GPU:
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8UI, 1024, 1024, 0, gl.RED_INTEGER, gl.UNSIGNED_BYTE, data);

    // can't filter integer textures
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

    return texture;
}

function createEmptySumsTexture(gl, width, height) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // Upload the texture to the GPU:
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16UI, width, height, 0, gl.RED_INTEGER, gl.UNSIGNED_SHORT, null);

    // can't filter integer textures
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

    return texture;
}


function createEmptyLookupTableTexture(gl) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);

    // Upload the (still empty) texture to the GPU:
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16UI, 1024, 1024, 0, gl.RED_INTEGER, gl.UNSIGNED_SHORT, null);

    // can't filter integer textures
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

    return texture;
}

async function createInputDataTexture(gl) {
    const response = await fetch("mock-matrix.bin");
    const buf = await response.arrayBuffer();
    const data = new Uint16Array(buf);

    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // Upload the texture to the GPU:
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16UI, 1024, 328, 0, gl.RED_INTEGER, gl.UNSIGNED_SHORT, data);

    // can't filter integer textures
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

    return texture;
}

function createVectorTexture(gl) {
    const data = new Int16Array(Array.from({ length: 1024 }, (_, i) => i));
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // Upload the texture to the GPU:
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16I, 1024, 1, 0, gl.RED_INTEGER, gl.SHORT, data);

    // can't filter integer textures
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

    return texture;
}

function createVertexBuffer(gl) {
    let vertices = new Float32Array([
        -1., 1., 1., 1., 1., -1., // Triangle 1
        -1., 1., 1., -1., -1., -1. // Triangle 2 
    ]);
    let vertexBuffer = gl.createBuffer();
    return { vertices, vertexBuffer };
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
