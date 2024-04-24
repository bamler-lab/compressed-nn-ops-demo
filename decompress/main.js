async function run() {
    const canvas = document.getElementById("canvas");
    const gl = getGl(canvas);
    const vertexShader = compileShader(gl, "vertex-shader");

    const response = await fetch("100-compressed-matrices.bin");
    const buf = await response.arrayBuffer();
    const data = new Uint16Array(buf);

    const coderHeadsIn = createTexture(gl, gl.RG16UI, 1024, 1024, gl.RG_INTEGER, gl.UNSIGNED_SHORT, data.subarray(0, 1024 * 1024 * 2));
    const coderHeadsOut = createTexture(gl, gl.RG16UI, 1024, 1024, gl.RG_INTEGER, gl.UNSIGNED_SHORT, null);
    const compressedData = createTexture(gl, gl.R16UI, 1024, 1024, gl.RED_INTEGER, gl.UNSIGNED_SHORT, data.subarray(1 + 2 * 1024 * 1024, 1 + 3 * 1024 * 1024));
    const lookupTable = createTexture(gl, gl.R8UI, 4096, 1, gl.RED_INTEGER, gl.UNSIGNED_BYTE, null);

    const coderOffsets1 = createTexture(gl, gl.R8UI, 1024, 1024, gl.RED_INTEGER, gl.UNSIGNED_BYTE, null);
    const coderOffsets2 = createTexture(gl, gl.R8UI, 256, 1024, gl.RED_INTEGER, gl.UNSIGNED_BYTE, null);
    const coderOffsets3 = createTexture(gl, gl.R16UI, 32, 1024, gl.RED_INTEGER, gl.UNSIGNED_SHORT, null);
    const coderOffsets4 = createTexture(gl, gl.R16UI, 1, 1024, gl.RED_INTEGER, gl.UNSIGNED_SHORT, null);
    const coderOffsets5 = createTexture(gl, gl.R32UI, 1, 32, gl.RED_INTEGER, gl.UNSIGNED_INT, null);

    const matrix0 = createTexture(gl, gl.R16UI, 1024, 1024, gl.RED_INTEGER, gl.UNSIGNED_SHORT, null);

    let { vertexBuffer, vertices } = createVertexBuffer(gl);
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);

    const createLookup = new GlProgram(gl, vertexShader, "create-lookup-shader", ["compressedData"]);
    const decode = new GlProgram(gl, vertexShader, "decode-shader", ["coderHeads", "compressedData", "lookupTable", "coderOffsets1", "coderOffsets2", "coderOffsets3", "coderOffsets4", "coderOffsets5"]);

    const sumOffsets0 = new GlProgram(gl, vertexShader, "sumOffsets0-shader", ["coderHeads"]);
    const sumOffsets1 = new GlProgram(gl, vertexShader, "sumOffsets1-shader", ["sumOffsets0"]);
    const sumOffsets2 = new GlProgram(gl, vertexShader, "sumOffsets2-shader", ["sumOffsets1"]);
    const sumOffsets3 = new GlProgram(gl, vertexShader, "sumOffsets3-shader", ["sumOffsets2"]);
    const sumOffsets4 = new GlProgram(gl, vertexShader, "sumOffsets4-shader", ["sumOffsets3"]);

    // await new Promise(resolve => setTimeout(resolve, 1000));

    let start = performance.now();
    for (let i = 0; i != 1; ++i) {
        createLookup.run(gl, [compressedData], [lookupTable], 4096, 1, vertexBuffer, vertices);
        decode.run(gl, [coderHeadsIn, compressedData, lookupTable, coderOffsets1, coderOffsets2, coderOffsets3, coderOffsets4, coderOffsets5], [matrix0, coderHeadsOut], 1024, 1024, vertexBuffer, vertices);
        sumOffsets0.run(gl, [coderHeadsOut], [coderOffsets1], 1024, 1024, vertexBuffer, vertices);
        sumOffsets1.run(gl, [coderOffsets1], [coderOffsets2], 256, 1024, vertexBuffer, vertices);
        sumOffsets2.run(gl, [coderOffsets2], [coderOffsets3], 32, 1024, vertexBuffer, vertices);
        sumOffsets3.run(gl, [coderOffsets3], [coderOffsets4], 1, 1024, vertexBuffer, vertices);
        sumOffsets4.run(gl, [coderOffsets4], [coderOffsets5], 1, 32, vertexBuffer, vertices);
    }

    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, coderOffsets5, 0);
    let rawBuffer = new ArrayBuffer(1 * 32 * 4 * 4);
    let output = new Uint32Array(rawBuffer);
    gl.readPixels(0, 0, 1, 32, gl.RGBA_INTEGER, gl.UNSIGNED_INT, output);
    console.log(output.filter((_val, i) => i % 4 == 0));
    console.log({ sum: output.filter((_val, i) => i % 4 == 0).reduce((s, x) => s + x, 0) });

    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, matrix0, 0);
    rawBuffer = new ArrayBuffer(1024 * 1024 * 4 * 4);
    output = new Uint32Array(rawBuffer);
    gl.readPixels(0, 0, 1024, 1024, gl.RGBA_INTEGER, gl.UNSIGNED_INT, output);
    console.log(output.filter((_val, i) => i % 4 == 0));

    await new Promise(resolve => setTimeout(resolve, 0));
    let end = performance.now();
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

    run(gl, inputTextures, outputTextures, width, height, vertexBuffer, vertices) {
        gl.viewport(0, 0, width, height);
        gl.useProgram(this.program);

        gl.uniform1i(this.widthLocation, width);
        gl.uniform1i(this.heightLocation, height);
        gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
        gl.vertexAttribPointer(this.vertexPositionLocation, 2, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.vertexPositionLocation);

        // WebGL is guaranteed to have at least 8 texture units.
        const gl_textures = [gl.TEXTURE0, gl.TEXTURE1, gl.TEXTURE2, gl.TEXTURE3, gl.TEXTURE4, gl.TEXTURE5, gl.TEXTURE6, gl.TEXTURE7];
        for (let i = 0; i != inputTextures.length; ++i) {
            gl.uniform1i(this.textureLocations[i], i);
            gl.activeTexture(gl_textures[i]);
            gl.bindTexture(gl.TEXTURE_2D, inputTextures[i]);
        }

        // Not sure how many color attachments we can rely on.
        const gl_color_attachments = [gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2, gl.COLOR_ATTACHMENT3];
        for (let i = 0; i != outputTextures.length; ++i) {
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl_color_attachments[i], gl.TEXTURE_2D, outputTextures[i], 0);
            // TODO: maybe use gl.DRAW_FRAMEBUFFER
        }

        gl.drawBuffers(gl_color_attachments.slice(0, outputTextures.length));
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        for (let i = 0; i != outputTextures.length; ++i) {
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl_color_attachments[i], gl.TEXTURE_2D, null, 0);
        }
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
